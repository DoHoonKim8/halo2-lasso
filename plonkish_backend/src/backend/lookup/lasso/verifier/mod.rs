use std::{collections::HashMap, marker::PhantomData};

use halo2_curves::ff::{Field, PrimeField};
use itertools::Itertools;

use crate::{
    pcs::{Evaluation, PolynomialCommitmentScheme},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck,
    },
    poly::multilinear::MultilinearPolynomial,
    util::transcript::FieldTranscriptRead,
    Error,
};

use super::{
    memory_checking::{verifier::MemoryCheckingVerifier, Chunk, Memory},
    prover::Surge,
    DecomposableTable,
};

pub struct LassoVerifier<
    F: Field + PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
>(PhantomData<F>, PhantomData<Pcs>);

impl<
        F: Field + PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    > LassoVerifier<F, Pcs>
{
    pub fn verify_sum_check(
        table: &Box<dyn DecomposableTable<F>>,
        num_vars: usize,
        polys_offset: usize,
        points_offset: usize,
        r: &[F],
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
        let expression = Surge::<F, Pcs>::sum_check_expression(&table);
        let claim = transcript.read_field_element()?;
        let (eval, x) = ClassicSumCheck::<EvaluationsProver<_>>::verify(
            &(),
            num_vars,
            expression.degree(),
            claim,
            transcript,
        )?;
        let points = vec![r.to_vec(), x];
        let pcs_query = Surge::<F, Pcs>::pcs_query(&expression, 0);
        let e_polys_offset = polys_offset + 1 + table.num_chunks() * 3;
        let evals = pcs_query
            .iter()
            .map(|query| {
                let value = transcript.read_field_element().unwrap();
                Evaluation::new(e_polys_offset + query.poly(), points_offset + 1, value)
            })
            .chain([Evaluation::new(polys_offset, points_offset, claim)])
            .collect_vec();

        Ok((points, evals))
    }

    fn chunks(table: &Box<dyn DecomposableTable<F>>) -> Vec<Chunk<F>> {
        let num_memories = table.num_memories();
        let chunk_bits = table.chunk_bits();
        let subtable_polys = table.subtable_polys();
        // key: chunk index, value: chunk
        let mut chunk_map: HashMap<usize, Chunk<F>> = HashMap::new();
        (0..num_memories).for_each(|memory_index| {
            let chunk_index = table.memory_to_chunk_index(memory_index);
            let chunk_bits = chunk_bits[chunk_index];
            let subtable_poly = &subtable_polys[table.memory_to_subtable_index(memory_index)];
            let memory = Memory::new(memory_index, subtable_poly.clone());
            if let Some(_) = chunk_map.get(&chunk_index) {
                chunk_map.entry(chunk_index).and_modify(|chunk| {
                    chunk.add_memory(memory);
                });
            } else {
                chunk_map.insert(chunk_index, Chunk::new(chunk_index, chunk_bits, memory));
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
    ) -> Vec<MemoryCheckingVerifier<F>> {
        let chunks = Self::chunks(table);
        let chunk_bits = table.chunk_bits();
        // key: chunk_bits, value: chunks
        let mut chunk_map = HashMap::<usize, Vec<Chunk<F>>>::new();
        chunks
            .into_iter()
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let chunk_bits = chunk_bits[chunk_index];
                if let Some(_) = chunk_map.get(&chunk_bits) {
                    chunk_map.entry(chunk_bits).and_modify(|chunks| {
                        chunks.push(chunk);
                    });
                } else {
                    chunk_map.insert(chunk_bits, vec![chunk]);
                }
            });
        chunk_map
            .into_iter()
            .map(|(_, chunks)| MemoryCheckingVerifier::new(chunks))
            .collect_vec()
    }
}
