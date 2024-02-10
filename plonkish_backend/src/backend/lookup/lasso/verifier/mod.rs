use std::{collections::HashMap, iter, marker::PhantomData};

use halo2_curves::ff::{Field, PrimeField};
use itertools::Itertools;

use crate::{
    pcs::{Evaluation, PolynomialCommitmentScheme},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        evaluate, SumCheck,
    },
    poly::multilinear::MultilinearPolynomial,
    util::transcript::{FieldTranscriptRead, TranscriptRead},
    Error,
};

use super::{
    memory_checking::verifier::{Chunk, Memory, MemoryCheckingVerifier},
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
    pub fn read_commitments(
        vp: &Pcs::VerifierParam,
        table: &Box<dyn DecomposableTable<F>>,
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
    ) -> Result<Vec<Pcs::Commitment>, Error> {
        // read input_comm, dim_comms
        let num_chunks = table.chunk_bits().len();
        let num_memories = table.num_memories();
        let input_comm = Pcs::read_commitment(vp, transcript)?;
        let dim_comms = Pcs::read_commitments(vp, num_chunks, transcript)?;

        // read read_ts_comms & final_cts_comms & e_comms
        let read_ts_comms = Pcs::read_commitments(vp, num_chunks, transcript)?;
        let final_cts_comms = Pcs::read_commitments(vp, num_chunks, transcript)?;
        let e_comms = Pcs::read_commitments(vp, num_memories, transcript)?;
        Ok(iter::empty()
            .chain(vec![input_comm])
            .chain(dim_comms)
            .chain(read_ts_comms)
            .chain(final_cts_comms)
            .chain(e_comms)
            .collect_vec())
    }

    pub fn verify_sum_check(
        table: &Box<dyn DecomposableTable<F>>,
        num_vars: usize,
        polys_offset: usize,
        points_offset: usize,
        lookup_opening_points: &mut Vec<Vec<F>>,
        lookup_opening_evals: &mut Vec<Evaluation<F>>,
        r: &[F],
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(), Error> {
        let expression = Surge::<F, Pcs>::sum_check_expression(&table);
        let claim = transcript.read_field_element()?;
        let (x_eval, x) = ClassicSumCheck::<EvaluationsProver<F>>::verify(
            &(),
            num_vars,
            expression.degree(),
            claim,
            transcript,
        )?;
        lookup_opening_points.extend_from_slice(&[r.to_vec(), x.clone()]);

        let pcs_query = expression.used_query();
        let evals = pcs_query
            .into_iter()
            .map(|query| {
                let value = transcript.read_field_element().unwrap();
                (query, value)
            })
            .collect();
        if evaluate(&expression, num_vars, &evals, &[], &[r], &x) != x_eval {
            return Err(Error::InvalidSnark(
                "Unmatched between Lasso sum_check output and query evaluation".to_string(),
            ));
        }
        let e_polys_offset = polys_offset + 1 + table.chunk_bits().len() * 3;
        let evals = evals
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
            .map(|(query, value)| {
                Evaluation::new(e_polys_offset + query.poly(), points_offset + 1, value)
            })
            .chain([Evaluation::new(polys_offset, points_offset, claim)])
            .collect_vec();
        lookup_opening_evals.extend_from_slice(&evals);
        Ok(())
    }

    fn chunks(table: &Box<dyn DecomposableTable<F>>) -> Vec<Chunk<F>> {
        let num_memories = table.num_memories();
        let chunk_bits = table.chunk_bits();
        let s_new = table.subtable_polys_terms();
        // key: chunk index, value: chunk
        let mut chunk_map: HashMap<usize, Chunk<F>> = HashMap::new();
        (0..num_memories).for_each(|memory_index| {
            let chunk_index = table.memory_to_chunk_index(memory_index);
            let chunk_bits = chunk_bits[chunk_index];
            let s = &s_new[table.memory_to_subtable_index(memory_index)];
            let memory = Memory::new(memory_index, s.clone());
            if chunk_map.get(&chunk_index).is_some() {
                chunk_map.entry(chunk_index).and_modify(|chunk| {
                    chunk.add_memory(memory);
                });
            } else {
                chunk_map.insert(chunk_index, Chunk::new(chunk_index, chunk_bits, memory));
            }
        });

        // sanity check
        {
            let num_chunks = table.chunk_bits().len();
            assert_eq!(chunk_map.len(), num_chunks);
        }

        let mut chunks = chunk_map.into_iter().collect_vec();
        chunks.sort_by_key(|(chunk_index, _)| *chunk_index);
        chunks.into_iter().map(|(_, chunk)| chunk).collect_vec()
    }

    fn prepare_memory_checking(
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
                if chunk_map.get(&chunk_bits).is_some() {
                    chunk_map.entry(chunk_bits).and_modify(|chunks| {
                        chunks.push(chunk);
                    });
                } else {
                    chunk_map.insert(chunk_bits, vec![chunk]);
                }
            });
        chunk_map
            .into_iter()
            .enumerate()
            .map(|(_, (_, chunks))| MemoryCheckingVerifier::new(chunks))
            .collect_vec()
    }

    pub fn memory_checking(
        num_reads: usize,
        polys_offset: usize,
        points_offset: usize,
        lookup_opening_points: &mut Vec<Vec<F>>,
        lookup_opening_evals: &mut Vec<Evaluation<F>>,
        table: &Box<dyn DecomposableTable<F>>,
        gamma: &F,
        tau: &F,
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(), Error> {
        let memory_checking = Self::prepare_memory_checking(table);
        memory_checking
            .iter()
            .map(|memory_checking| {
                memory_checking.verify(
                    table.chunk_bits().len(),
                    num_reads,
                    polys_offset,
                    points_offset,
                    &gamma,
                    &tau,
                    lookup_opening_points,
                    lookup_opening_evals,
                    transcript,
                )
            })
            .collect::<Result<Vec<()>, Error>>()?;
        Ok(())
    }
}
