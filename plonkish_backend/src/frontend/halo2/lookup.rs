use halo2_curves::ff::PrimeField;
use halo2_proofs::plonk::ConstraintSystem;

use crate::util::expression::Expression;

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
