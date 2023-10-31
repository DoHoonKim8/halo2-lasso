use std::{
    collections::HashMap,
    iter::{self, repeat},
};

use halo2_curves::ff::{Field, PrimeField};
use itertools::{chain, izip, Itertools};

use crate::{
    pcs::{CommitmentChunk, Evaluation, PolynomialCommitmentScheme},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck as _, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::transcript::TranscriptWrite,
    Error,
};

use super::{memory_checking::MemoryCheckingProver, DecomposableTable, Lasso};

mod surge;

pub use surge::Surge;

type SumCheck<F> = ClassicSumCheck<EvaluationsProver<F>>;

pub struct Point<F: PrimeField> {
    offset: usize,
    point: Vec<F>,
}

#[derive(Clone)]
pub struct Poly<'a, F: PrimeField> {
    offset: usize,
    poly: &'a MultilinearPolynomial<F>,
}

#[derive(Clone, Debug)]
pub struct Chunk<'a, F: PrimeField> {
    chunk_index: usize,
    dim: &'a MultilinearPolynomial<F>,
    read_ts_poly: &'a MultilinearPolynomial<F>,
    final_cts_poly: &'a MultilinearPolynomial<F>,
    memories: Vec<Memory<'a, F>>,
}

impl<'a, F: PrimeField> Chunk<'a, F> {
    fn new(
        chunk_index: usize,
        dim: &'a MultilinearPolynomial<F>,
        read_ts_poly: &'a MultilinearPolynomial<F>,
        final_cts_poly: &'a MultilinearPolynomial<F>,
        memory: Memory<'a, F>,
    ) -> Self {
        // sanity check
        assert_eq!(dim.num_vars(), read_ts_poly.num_vars());

        Self {
            chunk_index,
            dim,
            read_ts_poly,
            final_cts_poly,
            memories: vec![memory],
        }
    }

    pub fn chunk_polys_index(&self, offset: usize, num_chunks: usize) -> Vec<usize> {
        let dim_poly_index = offset + 1 + self.chunk_index;
        let read_ts_poly_index = offset + 1 + num_chunks + self.chunk_index;
        let final_cts_poly_index = offset + 1 + 2 * num_chunks + self.chunk_index;
        vec![dim_poly_index, read_ts_poly_index, final_cts_poly_index]
    }

    pub fn chunk_index(&self) -> usize {
        self.chunk_index
    }

    pub fn chunk_bits(&self) -> usize {
        self.final_cts_poly.num_vars()
    }

    pub fn num_reads(&self) -> usize {
        1 << self.dim.num_vars()
    }

    pub fn chunk_polys(&self) -> impl Iterator<Item = &'a MultilinearPolynomial<F>> {
        chain!([self.dim, self.read_ts_poly, self.final_cts_poly])
    }

    pub fn chunk_poly_evals(&self, x: &[F], y: &[F]) -> Vec<F> {
        vec![
            self.dim.evaluate(x),
            self.read_ts_poly.evaluate(x),
            self.final_cts_poly.evaluate(y),
        ]
    }

    pub fn e_poly_evals(&self, x: &[F]) -> Vec<F> {
        self.memories
            .iter()
            .map(|memory| memory.e_poly.evaluate(x))
            .collect_vec()
    }

    pub(super) fn memories(&self) -> impl Iterator<Item = &Memory<'a, F>> {
        self.memories.iter()
    }

    pub(super) fn add_memory(&mut self, memory: Memory<'a, F>) {
        // sanity check
        let chunk_bits = self.chunk_bits();
        let num_reads = self.num_reads();
        assert_eq!(chunk_bits, memory.subtable_poly.num_vars());
        assert_eq!(num_reads, memory.e_poly.num_vars());

        self.memories.push(memory);
    }
}

#[derive(Clone, Debug)]
pub(super) struct Memory<'a, F: PrimeField> {
    memory_index: usize,
    subtable_poly: &'a MultilinearPolynomial<F>,
    e_poly: &'a MultilinearPolynomial<F>,
}

impl<'a, F: PrimeField> Memory<'a, F> {
    fn new(
        memory_index: usize,
        subtable_poly: &'a MultilinearPolynomial<F>,
        e_poly: &'a MultilinearPolynomial<F>,
    ) -> Self {
        Self {
            memory_index,
            subtable_poly,
            e_poly,
        }
    }

    pub fn memory_index(&self) -> usize {
        self.memory_index
    }

    pub fn e_poly(&self) -> &'a MultilinearPolynomial<F> {
        self.e_poly
    }

    pub fn polys(&self) -> impl Iterator<Item = &'a MultilinearPolynomial<F>> {
        chain!([self.subtable_poly, self.e_poly])
    }
}

pub struct LassoProver<
    F: Field + PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
> {
    // Remove this
    scheme: Lasso<F, Pcs>,
}

impl<
        F: Field + PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    > LassoProver<F, Pcs>
{
    pub fn e_polys(
        subtable_polys: &[&MultilinearPolynomial<F>],
        table: &Box<dyn DecomposableTable<F>>,
        nz: &Vec<&[usize]>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let num_chunks = table.num_chunks();
        let num_memories = table.num_memories();
        assert_eq!(nz.len(), num_chunks);
        let num_reads = nz[0].len();
        (0..num_memories)
            .map(|i| {
                let mut e_poly = Vec::with_capacity(num_reads);
                let subtable_poly = subtable_polys[table.memory_to_subtable_index(i)];
                let nz = nz[table.memory_to_chunk_index(i)];
                (0..num_reads).for_each(|j| {
                    e_poly.push(subtable_poly[nz[j]]);
                });
                MultilinearPolynomial::new(e_poly)
            })
            .collect_vec()
    }

    pub fn chunks<'a>(
        table: &Box<dyn DecomposableTable<F>>,
        subtable_polys: &'a [&MultilinearPolynomial<F>],
        e_polys: &'a [MultilinearPolynomial<F>],
        dims: &'a [MultilinearPolynomial<F>],
        read_ts_polys: &'a [MultilinearPolynomial<F>],
        final_cts_polys: &'a [MultilinearPolynomial<F>],
    ) -> Vec<Chunk<'a, F>> {
        // key: chunk index, value: chunk
        let mut chunk_map: HashMap<usize, Chunk<'a, F>> = HashMap::new();

        let num_memories = table.num_memories();
        let memories = (0..num_memories).map(|memory_index| {
            let subtable_poly = subtable_polys[table.memory_to_subtable_index(memory_index)];
            Memory::new(memory_index, subtable_poly, &e_polys[memory_index])
        });
        memories.enumerate().for_each(|(memory_index, memory)| {
            let chunk_index = table.memory_to_chunk_index(memory_index);
            if let Some(_) = chunk_map.get(&chunk_index) {
                chunk_map.entry(chunk_index).and_modify(|chunk| {
                    chunk.add_memory(memory);
                });
            } else {
                let dim = &dims[chunk_index];
                let read_ts_poly = &read_ts_polys[chunk_index];
                let final_cts_poly = &final_cts_polys[chunk_index];
                chunk_map.insert(
                    chunk_index,
                    Chunk::new(chunk_index, dim, read_ts_poly, final_cts_poly, memory),
                );
            }
        });

        // sanity check
        {
            let num_chunks = table.num_chunks();
            assert_eq!(chunk_map.len(), num_chunks);
        }

        let mut chunks = chunk_map.into_iter().collect_vec();
        chunks.sort_by_key(|(chunk_index, _)| *chunk_index);
        chunks.into_iter().map(|(_, chunk)| chunk).collect_vec()
    }

    pub fn prepare_memory_checking<'a>(
        table: &Box<dyn DecomposableTable<F>>,
        subtable_polys: &'a [&MultilinearPolynomial<F>],
        e_polys: &'a [MultilinearPolynomial<F>],
        dims: &'a [MultilinearPolynomial<F>],
        read_ts_polys: &'a [MultilinearPolynomial<F>],
        final_cts_polys: &'a [MultilinearPolynomial<F>],
        gamma: &F,
        tau: &F,
    ) -> Vec<MemoryCheckingProver<'a, F>> {
        let chunks = Self::chunks(
            table,
            subtable_polys,
            e_polys,
            dims,
            read_ts_polys,
            final_cts_polys,
        );
        let chunk_bits = table.chunk_bits();
        // key: chunk bits, value: chunks
        let mut chunk_map: HashMap<usize, Vec<Chunk<'a, F>>> = HashMap::new();

        chunks.iter().enumerate().for_each(|(chunk_index, chunk)| {
            let chunk_bits = chunk_bits[chunk_index];
            if let Some(_) = chunk_map.get(&chunk_bits) {
                chunk_map.entry(chunk_bits).and_modify(|chunks| {
                    chunks.push(chunk.clone());
                });
            } else {
                chunk_map.insert(chunk_bits, vec![chunk.clone()]);
            }
        });

        chunk_map
            .into_iter()
            .map(|(_, chunks)| MemoryCheckingProver::new(chunks, tau, gamma))
            .collect_vec()
    }
}
