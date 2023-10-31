pub mod prover;
pub mod verifier;

use std::iter;

use halo2_curves::ff::PrimeField;
use itertools::Itertools;
pub use prover::MemoryCheckingProver;
use rayon::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};

use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{arithmetic::inner_product, transcript::FieldTranscriptRead},
    Error,
};

#[derive(Clone, Debug)]
pub struct Chunk<F> {
    chunk_index: usize,
    chunk_bits: usize,
    memory: Vec<Memory<F>>,
}

impl<F: PrimeField> Chunk<F> {
    pub fn chunk_polys_index(&self, offset: usize, num_chunks: usize) -> Vec<usize> {
        let dim_poly_index = offset + 1 + self.chunk_index;
        let read_ts_poly_index = offset + 1 + num_chunks + self.chunk_index;
        let final_cts_poly_index = offset + 1 + 2 * num_chunks + self.chunk_index;
        vec![dim_poly_index, read_ts_poly_index, final_cts_poly_index]
    }

    pub fn new(chunk_index: usize, chunk_bits: usize, memory: Memory<F>) -> Self {
        Self {
            chunk_index,
            chunk_bits,
            memory: vec![memory],
        }
    }

    pub fn num_memories(&self) -> usize {
        self.memory.len()
    }

    pub fn chunk_bits(&self) -> usize {
        self.chunk_bits
    }

    pub fn add_memory(&mut self, memory: Memory<F>) {
        self.memory.push(memory);
    }

    pub fn memory_indices(&self) -> Vec<usize> {
        self.memory.iter().map(|memory| memory.memory_index).collect_vec()
    }

    /// check the following relations:
    /// - $read(x) == hash(dim(x), E(x), read_ts(x))$
    /// - $write(x) == hash(dim(x), E(x), read_ts(x) + 1)$
    /// - $init(y) == hash(y, T(y), 0)$
    /// - $final_read(y) == hash(y, T(y), final_cts(x))$
    pub fn verify_memories(
        &self,
        read_xs: &[F],
        write_xs: &[F],
        init_ys: &[F],
        final_read_ys: &[F],
        y: &[F],
        gamma: &F,
        tau: &F,
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(F, F, F, Vec<F>), Error> {
        let hash = |a: &F, v: &F, t: &F| -> F { *a + *v * gamma + *t * gamma.square() - tau };
        let [dim_x, read_ts_poly_x, final_cts_poly_y] =
            transcript.read_field_elements(3)?.try_into().unwrap();
        let e_poly_xs = transcript.read_field_elements(self.num_memories())?;
        self.memory.iter().enumerate().for_each(|(i, memory)| {
            assert_eq!(read_xs[i], hash(&dim_x, &e_poly_xs[i], &read_ts_poly_x));

            assert_eq!(
                write_xs[i],
                hash(&dim_x, &e_poly_xs[i], &(read_ts_poly_x + F::ONE))
            );

            let id_poly_y = inner_product(
                iter::successors(Some(F::ONE), |power_of_two| Some(power_of_two.double()))
                    .take(y.len())
                    .collect_vec()
                    .iter(),
                y,
            );

            let subtable_poly_y = memory.subtable_poly.evaluate(y);

            assert_eq!(init_ys[i], hash(&id_poly_y, &subtable_poly_y, &F::ZERO));

            assert_eq!(
                final_read_ys[i],
                hash(&id_poly_y, &subtable_poly_y, &final_cts_poly_y)
            );
        });
        Ok((dim_x, read_ts_poly_x, final_cts_poly_y, e_poly_xs))
    }
}

#[derive(Clone, Debug)]
pub struct Memory<F> {
    memory_index: usize,
    subtable_poly: MultilinearPolynomial<F>,
}

impl<F> Memory<F> {
    pub fn new(memory_index: usize, subtable_poly: MultilinearPolynomial<F>) -> Self {
        Self {
            memory_index,
            subtable_poly,
        }
    }
}

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
