use std::{iter, marker::PhantomData};

use halo2_curves::ff::PrimeField;
use itertools::{izip, Itertools};

use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{inner_product, split_bits, split_by_chunk_bits},
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

/// T[X || Y] = T_1[X_1 || Y_1] + T_2[X_2 || Y_2] * 2^8 + ... + T_8[X_8 || Y_8] * 2^56
impl<F: PrimeField> DecomposableTable<F> for AndTable<F> {
    fn num_chunks(&self) -> usize {
        8
    }

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
