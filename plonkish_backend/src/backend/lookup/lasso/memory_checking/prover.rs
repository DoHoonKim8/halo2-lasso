use std::iter;

use halo2_curves::ff::PrimeField;
use itertools::{chain, Itertools};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    backend::lookup::lasso::prover::Chunk,
    pcs::Evaluation,
    piop::gkr::prove_grand_product,
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::transcript::FieldTranscriptWrite,
    Error,
};

use super::MemoryGKR;

#[derive(Clone)]
pub struct MemoryCheckingProver<'a, F: PrimeField> {
    /// chunks with the same bits size
    chunks: Vec<Chunk<'a, F>>,
    /// GKR initial polynomials for each memory
    memories: Vec<MemoryGKR<F>>,
    /// random point at which `read_write` polynomials opened
    x: Vec<F>,
    /// random point at which `init_final_read` polynomials opened
    y: Vec<F>,
}

// e_polys -> x (Lasso Sumcheck)
// dims, e_polys, read_ts_polys -> x (for each MemoryChecking)
// final_cts_polys -> y (for each MemoryChecking)

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
            x: vec![],
            y: vec![],
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

    pub fn prove_grand_product(
        &mut self,
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

        self.chunks.iter().for_each(|chunk| {
            let chunk_poly_evals = chunk.chunk_poly_evals(&x, &y);
            let e_poly_xs = chunk.e_poly_evals(&x);
            transcript.write_field_elements(&chunk_poly_evals).unwrap();
            transcript.write_field_elements(&e_poly_xs).unwrap();
        });

        self.x = x;
        self.y = y;

        Ok(())
    }

    pub fn opening_points(&self) -> impl Iterator<Item = Vec<F>> {
        chain!([self.x.clone(), self.y.clone()])
    }

    pub fn opening_evals(
        &self,
        num_chunks: usize,
        polys_offset: usize,
        points_offset: usize,
    ) -> impl Iterator<Item = Evaluation<F>> {
        let (dim_xs, read_ts_poly_xs, final_cts_poly_xs, e_poly_xs) = self
            .chunks
            .iter()
            .map(|chunk| {
                let chunk_poly_evals = chunk.chunk_poly_evals(&self.x, &self.y);
                let chunk_polys_index = chunk.chunk_polys_index(polys_offset, num_chunks);
                let e_poly_xs = chunk.e_poly_evals(&self.x);
                let e_polys_offset = polys_offset + 1 + 3 * num_chunks;
                (
                    Evaluation::new(chunk_polys_index[0], points_offset, chunk_poly_evals[0]),
                    Evaluation::new(chunk_polys_index[1], points_offset, chunk_poly_evals[1]),
                    Evaluation::new(chunk_polys_index[2], points_offset + 1, chunk_poly_evals[2]),
                    chunk
                        .memories()
                        .enumerate()
                        .map(|(i, memory)| {
                            Evaluation::new(
                                e_polys_offset + memory.memory_index(),
                                points_offset,
                                e_poly_xs[i],
                            )
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
        chain!(
            dim_xs,
            read_ts_poly_xs,
            final_cts_poly_xs,
            e_poly_xs.concat()
        )
    }
}
