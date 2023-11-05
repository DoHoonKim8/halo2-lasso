use std::{iter, marker::PhantomData};

use halo2_curves::ff::PrimeField;
use itertools::Itertools;

use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{inner_product, split_bits},
        expression::Expression,
    },
};

use super::DecomposableTable;

#[derive(Clone, Debug)]
pub struct AndTable<F>(PhantomData<F>);

impl<F> AndTable<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: PrimeField> DecomposableTable<F> for AndTable<F> {
    fn num_chunks(&self) -> usize {
        4
    }

    fn num_memories(&self) -> usize {
        4
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
        vec![16, 16, 16, 16]
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        Expression::DistributePowers(
            expressions,
            Box::new(Expression::Constant(F::from(2 << 16))),
        )
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        inner_product(
            operands,
            iter::successors(Some(F::ONE), |power_of_two| Some(power_of_two.double()))
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
