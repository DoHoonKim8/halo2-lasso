pub mod prover;
pub mod verifier;

use std::iter;

use halo2_curves::ff::PrimeField;
use itertools::Itertools;
pub use prover::MemoryCheckingProver;

use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{arithmetic::inner_product, transcript::FieldTranscriptRead},
    Error,
};

#[derive(Clone, Debug)]
struct MemoryGKR<F: PrimeField> {
    init: MultilinearPolynomial<F>,
    read: MultilinearPolynomial<F>,
    write: MultilinearPolynomial<F>,
    final_read: MultilinearPolynomial<F>,
}

impl<F: PrimeField> MemoryGKR<F> {
    pub fn new(
        init: MultilinearPolynomial<F>,
        read: MultilinearPolynomial<F>,
        write: MultilinearPolynomial<F>,
        final_read: MultilinearPolynomial<F>,
    ) -> Self {
        Self {
            init,
            read,
            write,
            final_read,
        }
    }
}
