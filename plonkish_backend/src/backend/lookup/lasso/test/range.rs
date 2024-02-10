use std::{iter, marker::PhantomData};

use halo2_curves::ff::PrimeField;
use itertools::Itertools;

use crate::{
    backend::lookup::lasso::DecomposableTable,
    poly::multilinear::{MultilinearPolynomial, MultilinearPolynomialTerms, PolyExpr::*},
    util::{
        arithmetic::{div_ceil, inner_product},
        expression::Expression,
    },
};

#[derive(Clone, Debug)]
pub struct RangeTable<F, const NUM_BITS: usize, const LIMB_BITS: usize>(PhantomData<F>);

impl<F, const NUM_BITS: usize, const LIMB_BITS: usize> RangeTable<F, NUM_BITS, LIMB_BITS> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: PrimeField, const NUM_BITS: usize, const LIMB_BITS: usize> DecomposableTable<F>
    for RangeTable<F, NUM_BITS, LIMB_BITS>
{
    fn chunk_bits(&self) -> Vec<usize> {
        let remainder_bits = if NUM_BITS % LIMB_BITS != 0 {
            vec![NUM_BITS % LIMB_BITS]
        } else {
            vec![]
        };
        iter::repeat(LIMB_BITS)
            .take(NUM_BITS / LIMB_BITS)
            .chain(remainder_bits)
            .collect_vec()
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        Expression::DistributePowers(
            expressions,
            Box::new(Expression::Constant(F::from(1 << LIMB_BITS))),
        )
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        let weight = F::from(1 << LIMB_BITS);
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

    fn num_memories(&self) -> usize {
        div_ceil(NUM_BITS, LIMB_BITS)
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        index_bits.chunks(LIMB_BITS).map(Vec::from).collect_vec()
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        let mut evals = vec![];
        (0..1 << LIMB_BITS).for_each(|i| evals.push(F::from(i)));
        let limb_subtable_poly = MultilinearPolynomial::new(evals);
        if NUM_BITS % LIMB_BITS != 0 {
            let remainder = NUM_BITS % LIMB_BITS;
            let mut evals = vec![];
            (0..1 << remainder).for_each(|i| {
                evals.push(F::from(i));
            });
            let rem_subtable_poly = MultilinearPolynomial::new(evals);
            vec![limb_subtable_poly, rem_subtable_poly]
        } else {
            vec![limb_subtable_poly]
        }
    }

    fn subtable_polys_terms(&self) -> Vec<MultilinearPolynomialTerms<F>> {
        let limb_init = Var(0);
        let mut limb_terms = vec![limb_init];
        (1..LIMB_BITS).for_each(|i| {
            let coeff = Pow(Box::new(Const(F::from(2))), i as u32);
            let x = Var(i);
            let term = Prod(vec![coeff, x]);
            limb_terms.push(term);
        });
        let limb_subtable_poly = MultilinearPolynomialTerms::new(LIMB_BITS, Sum(limb_terms));
        if NUM_BITS % LIMB_BITS == 0 {
            vec![limb_subtable_poly]
        } else {
            let remainder = NUM_BITS % LIMB_BITS;
            let rem_init = Var(0);
            let mut rem_terms = vec![rem_init];
            (1..remainder).for_each(|i| {
                let coeff = Pow(Box::new(Const(F::from(2))), i as u32);
                let x = Var(i);
                let term = Prod(vec![coeff, x]);
                rem_terms.push(term);
            });
            vec![
                limb_subtable_poly,
                MultilinearPolynomialTerms::new(remainder, Sum(rem_terms)),
            ]
        }
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        memory_index
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        if NUM_BITS % LIMB_BITS != 0 && memory_index == NUM_BITS / LIMB_BITS {
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod test {
    use std::array;

    use super::RangeTable;
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

    fn rand_vanilla_plonk_with_lasso_lookup_circuit<F: PrimeField>(
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
            let w_l = if preprocess_rng.next_u32().is_even() && idx > 1 {
                let l_copy_idx = (6, rand_idx(1..idx, &mut preprocess_rng));
                permutation.copy(l_copy_idx, (6, idx));
                polys[l_copy_idx.0][l_copy_idx.1]
            } else {
                let value = witness_rng.next_u64() as usize;
                F::from_u128(value.pow(2) as u128)
            };
            let w_r = F::from(witness_rng.next_u64());
            let q_c = F::random(&mut preprocess_rng);
            let values = if preprocess_rng.next_u32().is_even() {
                vec![
                    (1, F::ONE),
                    (2, F::ONE),
                    (4, -F::ONE),
                    (5, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_l + w_r + q_c + polys[0][idx]),
                ]
            } else {
                vec![
                    (3, F::ONE),
                    (4, -F::ONE),
                    (5, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_l * w_r + q_c + polys[0][idx]),
                ]
            };
            for (poly, value) in values {
                polys[poly][idx] = value;
            }
        }
        let [_, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o] = polys;
        let circuit_info = vanilla_plonk_with_lasso_lookup_circuit_info(
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

    fn vanilla_plonk_with_lasso_lookup_circuit_info<F: PrimeField>(
        num_vars: usize,
        num_instances: usize,
        preprocess_polys: [Vec<F>; 5],
        table: Box<dyn DecomposableTable<F>>,
        permutations: Vec<Vec<(usize, usize)>>,
    ) -> PlonkishCircuitInfo<F> {
        let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o] =
            &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                .map(Expression::<F>::Polynomial);
        let lasso_lookup_indices = w_l.clone();
        let lasso_lookup_output = w_l.clone();
        let chunk_bits = table.chunk_bits();
        let num_vars = chunk_bits.iter().chain([&num_vars]).max().unwrap();
        PlonkishCircuitInfo {
            k: *num_vars,
            num_instances: vec![num_instances],
            preprocess_polys: preprocess_polys.to_vec(),
            num_witness_polys: vec![3],
            num_challenges: vec![0],
            constraints: vec![q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi],
            lookups: vec![vec![]],
            lasso_lookup: Some((lasso_lookup_output, lasso_lookup_indices, table)),
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
                        let table = Box::new(RangeTable::<$f, 128, 16>::new());
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
