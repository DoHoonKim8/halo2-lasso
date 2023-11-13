use std::iter;

use halo2_curves::ff::PrimeField;
use itertools::{chain, Itertools};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    backend::lookup::lasso::prover::Chunk, pcs::Evaluation, piop::gkr::prove_grand_product,
    poly::multilinear::MultilinearPolynomial, util::transcript::FieldTranscriptWrite, Error,
};

use super::MemoryGKR;

pub struct MemoryCheckingProver<'a, F: PrimeField> {
    /// chunks with the same bits size
    chunks: Vec<Chunk<'a, F>>,
    /// GKR initial polynomials for each memory
    memories: Vec<MemoryGKR<F>>,
}

impl<'a, F: PrimeField> MemoryCheckingProver<'a, F> {
    // T_1[dim_1(x)], ..., T_k[dim_1(x)],
    // ...
    // T_{\alpha-k+1}[dim_c(x)], ..., T_{\alpha}[dim_c(x)]
    pub fn new(chunks: Vec<Chunk<'a, F>>, tau: &F, gamma: &F) -> Self {
        let num_reads = chunks[0].num_reads();
        let memory_size = 1 << chunks[0].chunk_bits();

        let hash = |a: &F, v: &F, t: &F| -> F { *a + *v * gamma + *t * gamma.square() - tau };

        let memories_gkr: Vec<MemoryGKR<F>> = (0..chunks.len())
            .into_par_iter()
            .flat_map(|i| {
                let chunk = &chunks[i];
                let chunk_polys = chunk.chunk_polys().collect_vec();
                let (dim, read_ts_poly, final_cts_poly) =
                    (chunk_polys[0], chunk_polys[1], chunk_polys[2]);
                chunk
                    .memories()
                    .map(|memory| {
                        let memory_polys = memory.polys().collect_vec();
                        let (subtable_poly, e_poly) = (memory_polys[0], memory_polys[1]);
                        let mut init = vec![];
                        let mut read = vec![];
                        let mut write = vec![];
                        let mut final_read = vec![];
                        (0..memory_size).for_each(|i| {
                            init.push(hash(&F::from(i as u64), &subtable_poly[i], &F::ZERO));
                            final_read.push(hash(
                                &F::from(i as u64),
                                &subtable_poly[i],
                                &final_cts_poly[i],
                            ));
                        });
                        (0..num_reads).for_each(|i| {
                            read.push(hash(&dim[i], &e_poly[i], &read_ts_poly[i]));
                            write.push(hash(&dim[i], &e_poly[i], &(read_ts_poly[i] + F::ONE)));
                        });
                        MemoryGKR::new(
                            MultilinearPolynomial::new(init),
                            MultilinearPolynomial::new(read),
                            MultilinearPolynomial::new(write),
                            MultilinearPolynomial::new(final_read),
                        )
                    })
                    .collect_vec()
            })
            .collect();

        Self {
            chunks,
            memories: memories_gkr,
        }
    }

    fn inits(&self) -> impl Iterator<Item = &MultilinearPolynomial<F>> {
        self.memories.iter().map(|memory| &memory.init)
    }

    fn reads(&self) -> impl Iterator<Item = &MultilinearPolynomial<F>> {
        self.memories.iter().map(|memory| &memory.read)
    }

    fn writes(&self) -> impl Iterator<Item = &MultilinearPolynomial<F>> {
        self.memories.iter().map(|memory| &memory.write)
    }

    fn final_reads(&self) -> impl Iterator<Item = &MultilinearPolynomial<F>> {
        self.memories.iter().map(|memory| &memory.final_read)
    }

    fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            &MultilinearPolynomial<F>,
            &MultilinearPolynomial<F>,
            &MultilinearPolynomial<F>,
            &MultilinearPolynomial<F>,
        ),
    > {
        self.memories.iter().map(|memory| {
            (
                &memory.init,
                &memory.read,
                &memory.write,
                &memory.final_read,
            )
        })
    }

    pub fn claimed_v_0s(&self) -> impl IntoIterator<Item = Vec<Option<F>>> {
        let (claimed_read_0s, claimed_write_0s, claimed_init_0s, claimed_final_read_0s) = self
            .iter()
            .map(|(init, read, write, final_read)| {
                let claimed_init_0 = init.iter().product();
                let claimed_read_0 = read.iter().product();
                let claimed_write_0 = write.iter().product();
                let claimed_final_read_0 = final_read.iter().product();

                // sanity check
                assert_eq!(
                    claimed_init_0 * claimed_write_0,
                    claimed_read_0 * claimed_final_read_0
                );
                (
                    Some(claimed_read_0),
                    Some(claimed_write_0),
                    Some(claimed_init_0),
                    Some(claimed_final_read_0),
                )
            })
            .multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>)>();
        chain!([
            claimed_read_0s,
            claimed_write_0s,
            claimed_init_0s,
            claimed_final_read_0s
        ])
    }

    pub fn prove(
        &mut self,
        points_offset: usize,
        lookup_opening_points: &mut Vec<Vec<F>>,
        lookup_opening_evals: &mut Vec<Evaluation<F>>,
        transcript: &mut impl FieldTranscriptWrite<F>,
    ) -> Result<(), Error> {
        let (_, x) = prove_grand_product(
            iter::repeat(None).take(self.memories.len() * 2),
            chain!(self.reads(), self.writes()),
            transcript,
        )?;

        let (_, y) = prove_grand_product(
            iter::repeat(None).take(self.memories.len() * 2),
            chain!(self.inits(), self.final_reads()),
            transcript,
        )?;
        let x_offset = points_offset + lookup_opening_points.len();
        let y_offset = x_offset + 1;
        let (dim_xs, read_ts_poly_xs, final_cts_poly_ys, e_poly_xs) = self
            .chunks
            .iter()
            .map(|chunk| {
                let chunk_poly_evals = chunk.chunk_poly_evals(&x, &y);
                let e_poly_xs = chunk.e_poly_evals(&x);
                transcript.write_field_elements(&chunk_poly_evals).unwrap();
                transcript.write_field_elements(&e_poly_xs).unwrap();

                (
                    Evaluation::new(chunk.dim.offset, x_offset, chunk_poly_evals[0]),
                    Evaluation::new(chunk.read_ts_poly.offset, x_offset, chunk_poly_evals[1]),
                    Evaluation::new(chunk.final_cts_poly.offset, y_offset, chunk_poly_evals[2]),
                    chunk
                        .memories()
                        .enumerate()
                        .map(|(i, memory)| {
                            Evaluation::new(memory.e_poly.offset, x_offset, e_poly_xs[i])
                        })
                        .collect_vec(),
                )
            })
            .multiunzip::<(
                Vec<Evaluation<F>>,
                Vec<Evaluation<F>>,
                Vec<Evaluation<F>>,
                Vec<Vec<Evaluation<F>>>,
            )>();

        lookup_opening_points.extend_from_slice(&[x, y]);
        let opening_evals = chain!(
            dim_xs,
            read_ts_poly_xs,
            final_cts_poly_ys,
            e_poly_xs.concat()
        )
        .collect_vec();
        lookup_opening_evals.extend_from_slice(&opening_evals);

        Ok(())
    }
}
