use std::fmt::Debug;

use halo2_curves::ff::Field;

use crate::{
    pcs::{CommitmentChunk, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{expression::Expression, transcript::TranscriptWrite},
    Error,
};

pub mod lasso;
pub mod logup;

pub struct MVLookupStrategyOutput<
    F: Field,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
> {
    polys: Vec<Vec<MultilinearPolynomial<F>>>,
    comms: Vec<Vec<Pcs::Commitment>>,
}

impl<F: Field, Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>>
    MVLookupStrategyOutput<F, Pcs>
{
    pub fn polys(&self) -> Vec<MultilinearPolynomial<F>> {
        self.polys.concat()
    }

    pub fn comms(&self) -> Vec<Pcs::Commitment> {
        self.comms.concat()
    }
}

pub trait MVLookupStrategy<F: Field>: Clone + Debug {
    type Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>;

    fn preprocess(
        lookups: &[Vec<(Expression<F>, Expression<F>)>],
        polys: &[&MultilinearPolynomial<F>],
        challenges: &mut Vec<F>,
    ) -> Result<Vec<[MultilinearPolynomial<F>; 2]>, Error>;

    fn commit(
        pp: &<Self::Pcs as PolynomialCommitmentScheme<F>>::ProverParam,
        lookup_polys: &[[MultilinearPolynomial<F>; 2]],
        challenges: &mut Vec<F>,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
    ) -> Result<MVLookupStrategyOutput<F, Self::Pcs>, Error>;
}
