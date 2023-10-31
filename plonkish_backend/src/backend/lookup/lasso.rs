use std::{collections::HashSet, fmt::Debug, iter, marker::PhantomData};

use halo2_curves::ff::{Field, PrimeField};
use itertools::Itertools;

use crate::{
    backend::lookup::lasso::prover::Surge,
    pcs::{CommitmentChunk, Evaluation, PolynomialCommitmentScheme},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::BooleanHypercube,
        expression::{CommonPolynomial, Expression},
        parallel::parallelize,
        transcript::{FieldTranscriptRead, TranscriptWrite},
    },
    Error,
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
    fn num_chunks(&self) -> usize;

    fn num_memories(&self) -> usize;

    /// Returns multilinear extension polynomials of each subtable
    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>>;

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, operands: &[F]) -> F;

    /// Returns the size of bits for each chunk.
    /// Each chunk can have different bits.
    fn chunk_bits(&self) -> Vec<usize>;

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
pub struct Lasso<
    F: Field + PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
> {
    _marker1: PhantomData<F>,
    _marker2: PhantomData<Pcs>,
}

impl<
        F: Field + PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    > Lasso<F, Pcs>
{
    pub fn lookup_polys(
        polys: &[&MultilinearPolynomial<F>],
        lookups: &Vec<(&Expression<F>, &Expression<F>)>,
    ) -> Vec<(MultilinearPolynomial<F>, MultilinearPolynomial<F>)> {
        let num_vars = polys[0].num_vars();
        let expression = lookups
            .iter()
            .map(|(input, index)| *input + *index)
            .sum::<Expression<_>>();
        let lagranges = {
            let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
            expression
                .used_langrange()
                .into_iter()
                .map(|i| (i, bh[i.rem_euclid(1 << num_vars) as usize]))
                .collect::<HashSet<_>>()
        };
        lookups
            .iter()
            .map(|lookup| Self::lookup_poly(lookup, &lagranges, polys))
            .collect()
    }

    fn lookup_poly(
        lookup: &(&Expression<F>, &Expression<F>),
        lagranges: &HashSet<(i32, usize)>,
        polys: &[&MultilinearPolynomial<F>],
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let num_vars = polys[0].num_vars();
        let bh = BooleanHypercube::new(num_vars);

        let evaluate = |expression: &Expression<F>| {
            let mut evals = vec![F::ZERO; 1 << num_vars];
            parallelize(&mut evals, |(evals, start)| {
                for (b, eval) in (start..).zip(evals) {
                    *eval = expression.evaluate(
                        &|constant| constant,
                        &|common_poly| match common_poly {
                            CommonPolynomial::Identity => F::from(b as u64),
                            CommonPolynomial::Lagrange(i) => {
                                if lagranges.contains(&(i, b)) {
                                    F::ONE
                                } else {
                                    F::ZERO
                                }
                            }
                            CommonPolynomial::EqXY(_) => unreachable!(),
                        },
                        &|query| polys[query.poly()][bh.rotate(b, query.rotation())],
                        &|_| unreachable!(),
                        &|value| -value,
                        &|lhs, rhs| lhs + &rhs,
                        &|lhs, rhs| lhs * &rhs,
                        &|value, scalar| value * &scalar,
                    );
                }
            });
            MultilinearPolynomial::new(evals)
        };

        let (input, index) = lookup;
        (evaluate(input), evaluate(index))
    }
}

#[derive(Clone, Debug)]
pub struct GeneralizedLasso<F: Field, Pcs: PolynomialCommitmentScheme<F>>(
    PhantomData<F>,
    PhantomData<Pcs>,
);
