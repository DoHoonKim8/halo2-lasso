use std::{iter, marker::PhantomData};

use halo2_curves::ff::PrimeField;
use itertools::{izip, Itertools};

use crate::{
    backend::lookup::lasso::DecomposableTable,
    poly::multilinear::{MultilinearPolynomial, MultilinearPolynomialTerms, PolyExpr::*},
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

    fn subtable_polys_terms(&self) -> Vec<MultilinearPolynomialTerms<F>> {
        let init = Prod(vec![Var(0), Var(8)]);
        let mut terms = vec![init];
        (1..8).for_each(|i| {
            let coeff = Const(F::from(1 << i));
            let x = Var(i);
            let y = Var(i + 8);
            let term = Prod(vec![coeff, x, y]);
            terms.push(term);
        });
        vec![MultilinearPolynomialTerms::new(16, Sum(terms))]
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

    fn memory_to_subtable_index(&self, _memory_index: usize) -> usize {
        0
    }
}

#[cfg(test)]
mod test {
    use std::array;

    use super::AndTable;
    use crate::{
        backend::{
            hyperplonk::{prover::instance_polys, util::Permutation, HyperPlonk},
            lookup::lasso::DecomposableTable,
            mock::MockCircuit,
            test::run_plonkish_backend,
            PlonkishCircuit, PlonkishCircuitInfo,
        },
        pcs::{
            multilinear::{
                Gemini, MultilinearBrakedown, MultilinearHyrax, MultilinearIpa, MultilinearKzg,
                Zeromorph,
            },
            univariate::UnivariateKzg,
        },
        poly::Polynomial,
        util::{
            arithmetic::{fe_to_bits_le, usize_from_bits_le},
            code::BrakedownSpec6,
            expression::{Expression, Query, Rotation},
            hash::Keccak256,
            test::{rand_idx, rand_vec, seeded_std_rng},
            transcript::Keccak256Transcript,
        },
    };
    use halo2_curves::{
        bn256::{self, Bn256},
        ff::PrimeField,
        grumpkin,
    };
    use num_integer::Integer;
    use rand::RngCore;

    fn rand_lasso_lookup_circuit<F: PrimeField>(
        num_vars: usize,
        table: Box<dyn DecomposableTable<F>>,
        mut preprocess_rng: impl RngCore,
        mut witness_rng: impl RngCore,
    ) -> (PlonkishCircuitInfo<F>, impl PlonkishCircuit<F>) {
        let size = 1 << num_vars;
        let mut polys = [(); 9].map(|_| vec![F::ZERO; size]);

        let instances = rand_vec(num_vars, &mut witness_rng);
        polys[0] = instance_polys(num_vars, [&instances])[0].evals().to_vec();

        let mut permutation = Permutation::default();
        for poly in [6, 7, 8] {
            permutation.copy((poly, 1), (poly, 1));
        }
        for idx in 0..size - 1 {
            let use_copy = preprocess_rng.next_u32().is_even() && idx > 1;
            let [w_l, w_r, w_o] = if use_copy {
                let [l_copy_idx, r_copy_idx] = [(); 2].map(|_| {
                    (
                        rand_idx(6..9, &mut preprocess_rng),
                        rand_idx(1..idx, &mut preprocess_rng),
                    )
                });
                permutation.copy(l_copy_idx, (6, idx));
                permutation.copy(r_copy_idx, (7, idx));
                let w_l = polys[l_copy_idx.0][l_copy_idx.1];
                let w_r = polys[r_copy_idx.0][r_copy_idx.1];
                let w_o = usize_from_bits_le(&fe_to_bits_le(w_l))
                    & usize_from_bits_le(&fe_to_bits_le(w_r));
                let w_o = F::from(w_o as u64);
                [w_l, w_r, w_o]
            } else {
                let [w_l, w_r] = [(); 2].map(|_| witness_rng.next_u64());
                let w_o = w_l & w_r;
                [F::from(w_l), F::from(w_r), F::from(w_o)]
            };

            let q_c = F::random(&mut preprocess_rng);
            let values = if preprocess_rng.next_u32().is_even() {
                vec![
                    (1, F::ONE),
                    (2, F::ONE),
                    (4, -F::ONE),
                    (5, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_o),
                ]
            } else {
                vec![
                    (3, F::ONE),
                    (4, -F::ONE),
                    (5, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_o),
                ]
            };
            for (poly, value) in values {
                polys[poly][idx] = value;
            }
        }
        let [_, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o] = polys;
        let circuit_info = lasso_lookup_circuit_info(
            num_vars,
            instances.len(),
            [q_l, q_r, q_m, q_o, q_c],
            table,
            permutation.into_cycles(),
        );
        (
            circuit_info,
            MockCircuit::new(vec![instances], vec![w_l, w_r, w_o]),
        )
    }

    fn lasso_lookup_circuit_info<F: PrimeField>(
        num_vars: usize,
        num_instances: usize,
        preprocess_polys: [Vec<F>; 5],
        table: Box<dyn DecomposableTable<F>>,
        permutations: Vec<Vec<(usize, usize)>>,
    ) -> PlonkishCircuitInfo<F> {
        let [_, _, _, _, _, _, w_l, w_r, w_o] =
            &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                .map(Expression::<F>::Polynomial);
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
            lasso_lookup: Some((lasso_lookup_input, lasso_lookup_indices, table)),
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
                        rand_lasso_lookup_circuit(num_vars, table, seeded_std_rng(), seeded_std_rng())
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
