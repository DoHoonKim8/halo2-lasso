use std::{iter, marker::PhantomData};

use halo2_curves::ff::PrimeField;
use itertools::{chain, Itertools};

use crate::{
    pcs::Evaluation,
    piop::gkr::verify_grand_product,
    poly::multilinear::MultilinearPolynomialTerms,
    util::{arithmetic::inner_product, transcript::FieldTranscriptRead},
    Error,
};

#[derive(Clone, Debug)]
pub(in crate::backend::lookup::lasso) struct Chunk<F> {
    chunk_index: usize,
    chunk_bits: usize,
    pub(crate) memory: Vec<Memory<F>>,
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
        self.memory
            .iter()
            .map(|memory| memory.memory_index)
            .collect_vec()
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
        hash: impl Fn(&F, &F, &F) -> F,
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(F, F, F, Vec<F>), Error> {
        let [dim_x, read_ts_poly_x, final_cts_poly_y] =
            transcript.read_field_elements(3)?.try_into().unwrap();
        let e_poly_xs = transcript.read_field_elements(self.num_memories())?;
        let id_poly_y = inner_product(
            iter::successors(Some(F::ONE), |power_of_two| Some(power_of_two.double()))
                .take(y.len())
                .collect_vec()
                .iter(),
            y,
        );
        self.memory.iter().enumerate().for_each(|(i, memory)| {
            assert_eq!(read_xs[i], hash(&dim_x, &e_poly_xs[i], &read_ts_poly_x));
            assert_eq!(
                write_xs[i],
                hash(&dim_x, &e_poly_xs[i], &(read_ts_poly_x + F::ONE))
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
pub(in crate::backend::lookup::lasso) struct Memory<F> {
    memory_index: usize,
    subtable_poly: MultilinearPolynomialTerms<F>,
}

impl<F> Memory<F> {
    pub fn new(memory_index: usize, subtable_poly: MultilinearPolynomialTerms<F>) -> Self {
        Self {
            memory_index,
            subtable_poly,
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::backend::lookup::lasso) struct MemoryCheckingVerifier<F: PrimeField> {
    /// chunks with the same bits size
    chunks: Vec<Chunk<F>>,
    _marker: PhantomData<F>,
}

impl<'a, F: PrimeField> MemoryCheckingVerifier<F> {
    pub fn new(chunks: Vec<Chunk<F>>) -> Self {
        Self {
            chunks,
            _marker: PhantomData,
        }
    }

    pub fn verify(
        &self,
        num_chunks: usize,
        num_reads: usize,
        polys_offset: usize,
        points_offset: usize,
        gamma: &F,
        tau: &F,
        lookup_opening_points: &mut Vec<Vec<F>>,
        lookup_opening_evals: &mut Vec<Evaluation<F>>,
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(), Error> {
        let num_memories: usize = self.chunks.iter().map(|chunk| chunk.num_memories()).sum();
        let memory_bits = self.chunks[0].chunk_bits();
        let (read_write_xs, x) = verify_grand_product(
            num_reads,
            iter::repeat(None).take(2 * num_memories),
            transcript,
        )?;
        let (read_xs, write_xs) = read_write_xs.split_at(num_memories);

        let (init_final_read_ys, y) = verify_grand_product(
            memory_bits,
            iter::repeat(None).take(2 * num_memories),
            transcript,
        )?;
        let (init_ys, final_read_ys) = init_final_read_ys.split_at(num_memories);

        let hash = |a: &F, v: &F, t: &F| -> F { *a + *v * gamma + *t * gamma.square() - tau };
        let mut offset = 0;
        let (dim_xs, read_ts_poly_xs, final_cts_poly_ys, e_poly_xs) = self
            .chunks
            .iter()
            .map(|chunk| {
                let num_memories = chunk.num_memories();
                let result = chunk.verify_memories(
                    &read_xs[offset..offset + num_memories],
                    &write_xs[offset..offset + num_memories],
                    &init_ys[offset..offset + num_memories],
                    &final_read_ys[offset..offset + num_memories],
                    &y,
                    hash,
                    transcript,
                );
                offset += num_memories;
                result
            })
            .collect::<Result<Vec<(F, F, F, Vec<F>)>, Error>>()?
            .into_iter()
            .multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<Vec<_>>)>();

        self.opening_evals(
            num_chunks,
            polys_offset,
            points_offset,
            &lookup_opening_points,
            lookup_opening_evals,
            &dim_xs,
            &read_ts_poly_xs,
            &final_cts_poly_ys,
            &e_poly_xs.concat(),
        );
        lookup_opening_points.extend_from_slice(&[x, y]);

        Ok(())
    }

    fn opening_evals(
        &self,
        num_chunks: usize,
        polys_offset: usize,
        points_offset: usize,
        lookup_opening_points: &Vec<Vec<F>>,
        lookup_opening_evals: &mut Vec<Evaluation<F>>,
        dim_xs: &[F],
        read_ts_poly_xs: &[F],
        final_cts_poly_ys: &[F],
        e_poly_xs: &[F],
    ) {
        let x_offset = points_offset + lookup_opening_points.len();
        let y_offset = x_offset + 1;
        let (dim_xs, read_ts_poly_xs, final_cts_poly_ys) = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let chunk_polys_index = chunk.chunk_polys_index(polys_offset, num_chunks);
                (
                    Evaluation::new(chunk_polys_index[0], x_offset, dim_xs[i]),
                    Evaluation::new(chunk_polys_index[1], x_offset, read_ts_poly_xs[i]),
                    Evaluation::new(chunk_polys_index[2], y_offset, final_cts_poly_ys[i]),
                )
            })
            .multiunzip::<(Vec<Evaluation<F>>, Vec<Evaluation<F>>, Vec<Evaluation<F>>)>();

        let e_poly_offset = polys_offset + 1 + 3 * num_chunks;
        let e_poly_xs = self
            .chunks
            .iter()
            .flat_map(|chunk| chunk.memory_indices())
            .zip(e_poly_xs)
            .map(|(memory_index, &e_poly_x)| {
                Evaluation::new(e_poly_offset + memory_index, x_offset, e_poly_x)
            })
            .collect_vec();
        lookup_opening_evals.extend_from_slice(
            &chain!(dim_xs, read_ts_poly_xs, final_cts_poly_ys, e_poly_xs).collect_vec(),
        );
    }
}
