use std::{iter, marker::PhantomData};

use halo2_curves::ff::PrimeField;
use itertools::{izip, Itertools};

use crate::{
    backend::lookup::lasso::DecomposableTable,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{inner_product, split_bits, split_by_chunk_bits},
        expression::Expression,
    },
};

#[derive(Clone, Debug)]
pub struct AndTable<F>(PhantomData<F>);

impl<F> AndTable<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

/// T[X || Y] = T_1[X_1 || Y_1] + T_2[X_2 || Y_2] * 2^8 + ... + T_8[X_8 || Y_8] * 2^56
impl<F: PrimeField> DecomposableTable<F> for AndTable<F> {
    fn num_memories(&self) -> usize {
        8
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        let memory_size = 1 << 16;
        let mut evals = vec![];
        (0..memory_size).for_each(|i| {
            let (lhs, rhs) = split_bits(i, 8);
            let result = F::from((lhs & rhs) as u64);
            evals.push(result)
        });
        vec![MultilinearPolynomial::new(evals)]
    }

    fn chunk_bits(&self) -> Vec<usize> {
        vec![16; 8]
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        assert!(index_bits.len() % 2 == 0);
        let chunk_bits = self
            .chunk_bits()
            .iter()
            .map(|chunk_bits| chunk_bits / 2)
            .collect_vec();
        let (lhs, rhs) = index_bits.split_at(index_bits.len() / 2);
        izip!(
            split_by_chunk_bits(lhs, &chunk_bits),
            split_by_chunk_bits(rhs, &chunk_bits)
        )
        .map(|(chunked_lhs_bits, chunked_rhs_bits)| {
            iter::empty()
                .chain(chunked_lhs_bits)
                .chain(chunked_rhs_bits)
                .collect_vec()
        })
        .collect_vec()
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        Expression::DistributePowers(expressions, Box::new(Expression::Constant(F::from(1 << 8))))
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        let weight = F::from(1 << 8);
        inner_product(
            operands,
            iter::successors(Some(F::ONE), |power_of_weight| {
                Some(*power_of_weight * weight)
            })
            .take(operands.len())
            .collect_vec()
            .iter(),
        )
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        memory_index
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        0
    }
}

#[cfg(test)]
mod test {
    use std::{iter, array};

    use super::AndTable;
    use crate::{
        backend::{
            hyperplonk::{HyperPlonk, prover::instance_polys, util::Permutation},
            test::run_plonkish_backend, lookup::lasso::DecomposableTable, PlonkishCircuitInfo, PlonkishCircuit, mock::MockCircuit,
        },
        pcs::{
            multilinear::{
                Gemini, MultilinearBrakedown, MultilinearHyrax, MultilinearIpa, MultilinearKzg,
                Zeromorph,
            },
            univariate::UnivariateKzg,
        },
        util::{
            code::BrakedownSpec6, hash::Keccak256, test::{seeded_std_rng, rand_vec, rand_idx},
            transcript::Keccak256Transcript, arithmetic::{usize_from_bits_le, fe_to_bits_le}, expression::{Query, Rotation, Expression},
        }, poly::Polynomial,
    };
    use halo2_curves::{
        bn256::{self, Bn256},
        grumpkin, ff::PrimeField,
    };
    use itertools::Itertools;
    use num_integer::Integer;
    use rand::RngCore;

    fn rand_vanilla_plonk_with_lasso_lookup_circuit<F: PrimeField>(
        num_vars: usize,
        table: Box<dyn DecomposableTable<F>>,
        mut preprocess_rng: impl RngCore,
        mut witness_rng: impl RngCore,
    ) -> (PlonkishCircuitInfo<F>, impl PlonkishCircuit<F>) {
        let size = 1 << num_vars;
        let mut polys = [(); 13].map(|_| vec![F::ZERO; size]);
    
        let [t_l, t_r, t_o] = [(); 3].map(|_| {
            iter::empty()
                .chain([F::ZERO, F::ZERO])
                .chain(iter::repeat_with(|| F::random(&mut preprocess_rng)))
                .take(size)
                .collect_vec()
        });
        polys[7] = t_l;
        polys[8] = t_r;
        polys[9] = t_o;
    
        let instances = rand_vec(num_vars, &mut witness_rng);
        polys[0] = instance_polys(num_vars, [&instances])[0].evals().to_vec();
    
        let mut permutation = Permutation::default();
        for poly in [10, 11, 12] {
            permutation.copy((poly, 1), (poly, 1));
        }
        for idx in 0..size - 1 {
            let use_copy = preprocess_rng.next_u32().is_even() && idx > 1;
            let [w_l, w_r, w_o] = if use_copy {
                let [l_copy_idx, r_copy_idx] = [(); 2].map(|_| {
                    (
                        rand_idx(10..13, &mut preprocess_rng),
                        rand_idx(1..idx, &mut preprocess_rng),
                    )
                });
                permutation.copy(l_copy_idx, (10, idx));
                permutation.copy(r_copy_idx, (11, idx));
                let w_l = polys[l_copy_idx.0][l_copy_idx.1];
                let w_r = polys[r_copy_idx.0][r_copy_idx.1];
                let w_o = F::from(
                    (usize_from_bits_le(&fe_to_bits_le(w_l)) & usize_from_bits_le(&fe_to_bits_le(w_r)))
                        as u64,
                );
                [w_l, w_r, w_o]
            } else {
                let [w_l, w_r] = [(); 2].map(|_| witness_rng.next_u64());
                let w_o = w_l & w_r;
                [F::from(w_l), F::from(w_r), F::from(w_o)]
            };
    
            let values = vec![(10, w_l), (11, w_r), (12, w_o)];
            for (poly, value) in values {
                polys[poly][idx] = value;
            }
        }
        let [_, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o] = polys;
        let circuit_info = vanilla_plonk_with_lasso_lookup_circuit_info(
            num_vars,
            instances.len(),
            [q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o],
            table,
            permutation.into_cycles(),
        );
        (
            circuit_info,
            MockCircuit::new(vec![instances], vec![w_l, w_r, w_o]),
        )
    }

    fn vanilla_plonk_with_lasso_lookup_circuit_info<F: PrimeField>(
        num_vars: usize,
        num_instances: usize,
        preprocess_polys: [Vec<F>; 9],
        table: Box<dyn DecomposableTable<F>>,
        permutations: Vec<Vec<(usize, usize)>>,
    ) -> PlonkishCircuitInfo<F> {
        let [pi, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o] =
            &array::from_fn(|poly| Query::new(poly, Rotation::cur())).map(Expression::<F>::Polynomial);
        let lasso_lookup_input = w_o.clone();
        let lasso_lookup_indices = Expression::DistributePowers(
            vec![w_l.clone(), w_r.clone()],
            Box::new(Expression::Constant(F::from_u128(1 << 64))),
        );
        let chunk_bits = table.chunk_bits();
        let num_vars = chunk_bits.iter().chain([&num_vars]).max().unwrap();
        PlonkishCircuitInfo {
            k: *num_vars,
            num_instances: vec![num_instances],
            preprocess_polys: preprocess_polys.to_vec(),
            num_witness_polys: vec![3],
            num_challenges: vec![0],
            constraints: vec![],
            lookups: vec![vec![]],
            lasso_lookup: vec![(lasso_lookup_input, lasso_lookup_indices, table)],
            permutations,
            max_degree: Some(4),
        }
    }

    macro_rules! test {
        ($name:ident, $f:ty, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<$name _hyperplonk_vanilla_plonk_with_lasso_lookup>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let table = Box::new(AndTable::<$f>::new());
                        rand_vanilla_plonk_with_lasso_lookup_circuit(num_vars, table, seeded_std_rng(), seeded_std_rng())
                    });
                }
            }
        };
        ($name:ident, $f:ty, $pcs:ty) => {
            test!($name, $f, $pcs, 16..17);
        };
    }

    test!(brakedown, bn256::Fr, MultilinearBrakedown<bn256::Fr, Keccak256, BrakedownSpec6>);
    test!(
        hyrax,
        grumpkin::Fr,
        MultilinearHyrax<grumpkin::G1Affine>,
        5..16
    );
    test!(ipa, grumpkin::Fr, MultilinearIpa<grumpkin::G1Affine>);
    test!(kzg, bn256::Fr, MultilinearKzg<Bn256>);
    test!(gemini_kzg, bn256::Fr, Gemini<UnivariateKzg<Bn256>>);
    test!(zeromorph_kzg, bn256::Fr, Zeromorph<UnivariateKzg<Bn256>>);
}
