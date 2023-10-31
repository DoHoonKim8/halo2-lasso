use std::fmt::Debug;

use halo2_curves::ff::PrimeField;
use halo2_proofs::plonk::ConstraintSystem;

use crate::{poly::multilinear::MultilinearPolynomial, util::expression::Expression};

/// This is a trait that decomposable tables provide implementations for.
/// This will be converted into `DecomposableTable`
pub trait SubtableStrategy<
    F: PrimeField,
    const TABLE_SIZE: usize,
    const NUM_CHUNKS: usize,
    const NUM_MEMORIES: usize,
>
{
    /// This is a configuration object that stores subtables
    type Config: Clone;

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, config: Self::Config) -> Expression<F>;
}

/// This is a trait that contains information about decomposable table to which
/// backend prover and verifier can ask
pub trait DecomposableTable<F: PrimeField>: Clone + Debug + Sync {
    const NUM_CHUNKS: usize;
    const NUM_MEMORIES: usize;

    /// Returns multilinear extension polynomials of each subtable
    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>>;

    fn combine_lookup_expressions(&self, expressions: &[&Expression<F>]) -> Expression<F>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, operands: &[F]) -> F;

    /// Returns the size of bits for each chunk.
    /// Each chunk can have different bits.
    fn chunk_bits(&self) -> [usize; Self::NUM_CHUNKS];

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize;

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize;
}
