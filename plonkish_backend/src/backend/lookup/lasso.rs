use std::{fmt::Debug, marker::PhantomData};

use halo2_curves::ff::{Field, PrimeField};

use crate::{
    pcs::PolynomialCommitmentScheme, poly::multilinear::MultilinearPolynomial,
    util::expression::Expression,
};

pub mod memory_checking;
pub mod prover;
pub mod test;
pub mod verifier;

pub trait Subtable<F: PrimeField> {
    fn evaluate(point: &[F]) -> F;
}

/// This is a trait that contains information about decomposable table to which
/// backend prover and verifier can ask
pub trait DecomposableTable<F: PrimeField>: Debug + Sync + DecomposableTableClone<F> {
    fn num_memories(&self) -> usize;

    /// Returns multilinear extension polynomials of each subtable
    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>>;

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, operands: &[F]) -> F;

    /// Returns the size of bits for each chunk.
    /// Each chunk can have different bits.
    fn chunk_bits(&self) -> Vec<usize>;

    /// Returns the indices of each subtable lookups
    /// The length of `index_bits` is same as actual bit length of table index
    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>>;

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize;

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize;
}

pub trait DecomposableTableClone<F> {
    fn clone_box(&self) -> Box<dyn DecomposableTable<F>>;
}

impl<T, F: PrimeField> DecomposableTableClone<F> for T
where
    T: DecomposableTable<F> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn DecomposableTable<F>> {
        Box::new(self.clone())
    }
}

impl<F> Clone for Box<dyn DecomposableTable<F>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone, Debug)]
pub struct GeneralizedLasso<F: Field, Pcs: PolynomialCommitmentScheme<F>>(
    PhantomData<F>,
    PhantomData<Pcs>,
);
