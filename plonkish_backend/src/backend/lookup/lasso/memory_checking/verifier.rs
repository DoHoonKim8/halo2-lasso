use std::{iter, marker::PhantomData};

use halo2_curves::ff::PrimeField;
use itertools::{Itertools, chain};

use crate::{piop::gkr::verify_grand_product, util::transcript::FieldTranscriptRead, Error, pcs::Evaluation};

use super::Chunk;

#[derive(Clone, Debug)]
pub struct MemoryCheckingVerifier<F: PrimeField> {
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

    pub fn verify_grand_product(
        &self,
        num_chunks: usize,
        num_reads: usize,
        polys_offset: usize,
        points_offset: usize,
        gamma: &F,
        tau: &F,
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
        let num_memories: usize = self.chunks.iter().map(|chunk| chunk.num_memories()).sum();
        let memory_size = self.chunks[0].chunk_bits();
        let (read_write_xs, x) = verify_grand_product(
            num_reads,
            iter::repeat(None).take(2 * num_memories),
            transcript,
        )?;
        let (read_xs, write_xs) = read_write_xs.split_at(num_memories);

        let (init_final_read_ys, y) = verify_grand_product(
            memory_size,
            iter::repeat(None).take(2 * num_memories),
            transcript,
        )?;
        let (init_ys, final_read_ys) = init_final_read_ys.split_at(num_memories);

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
                    gamma,
                    tau,
                    transcript,
                );
                offset += num_memories;
                result
            })
            .collect::<Result<Vec<(F, F, F, Vec<F>)>, Error>>()?
            .into_iter()
            .multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<Vec<_>>)>();

        let opening_evals = self.opening_evals(
            num_chunks,
            polys_offset,
            points_offset,
            &dim_xs,
            &read_ts_poly_xs,
            &final_cts_poly_ys,
            &e_poly_xs.concat()
        ).collect_vec();

        Ok((vec![x, y], opening_evals))
    }

    fn opening_evals(
        &self,
        num_chunks: usize,
        polys_offset: usize,
        points_offset: usize,
        dim_xs: &[F],
        read_ts_poly_xs: &[F],
        final_cts_poly_ys: &[F],
        e_poly_xs: &[F],
    ) -> impl Iterator<Item = Evaluation<F>> {
        let (dim_xs, read_ts_poly_xs, final_cts_poly_xs) = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let chunk_polys_index = chunk.chunk_polys_index(polys_offset, num_chunks);
                (
                    Evaluation::new(chunk_polys_index[0], points_offset, dim_xs[i]),
                    Evaluation::new(chunk_polys_index[1], points_offset, read_ts_poly_xs[i]),
                    Evaluation::new(chunk_polys_index[2], points_offset + 1, final_cts_poly_ys[i]),
                )
            })
            .multiunzip::<(
                Vec<Evaluation<F>>,
                Vec<Evaluation<F>>,
                Vec<Evaluation<F>>,
            )>();

        let e_poly_offset = polys_offset + 1 + 3 * num_chunks;
        let e_poly_xs = self
            .chunks
            .iter()
            .flat_map(|chunk| chunk.memory_indices())
            .zip(e_poly_xs)
            .map(|(memory_index, &e_poly_x)| {
                Evaluation::new(
                    e_poly_offset + memory_index,
                    points_offset,
                    e_poly_x,
                )
            })
            .collect_vec();
        chain!(
            dim_xs,
            read_ts_poly_xs,
            final_cts_poly_xs,
            e_poly_xs
        )
    }
}
