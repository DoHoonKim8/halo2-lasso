use crate::{
    backend::{
        hyperplonk::{
            preprocessor::{batch_size, compose, permutation_polys},
            prover::{instance_polys, permutation_z_polys, prove_zero_check},
            verifier::{pcs_query, points, verify_zero_check},
        },
        lookup::lasso::verifier::LassoVerifier,
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::{PolynomialCommitmentScheme, Evaluation},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, BooleanHypercube, PrimeField},
        end_timer,
        expression::{Expression, Query},
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{fmt::Debug, hash::Hash, iter, marker::PhantomData};

use super::lookup::lasso::{
    prover::{LassoProver, Surge},
    DecomposableTable, Lasso,
};

pub(crate) mod preprocessor;
pub(crate) mod prover;
pub(crate) mod verifier;

#[cfg(any(test, feature = "benchmark"))]
pub mod util;

#[derive(Clone, Debug)]
pub struct HyperPlonk<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug)]
pub struct HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::ProverParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    pub(crate) lasso_lookups: Vec<(Expression<F>, Expression<F>, Box<dyn DecomposableTable<F>>)>,
    pub(crate) lookup_polys_offset: usize,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_polys: Vec<MultilinearPolynomial<F>>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
    pub(crate) permutation_comms: Vec<Pcs::Commitment>,
}

#[derive(Clone, Debug)]
pub struct HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::VerifierParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) lasso_tables: Vec<Box<dyn DecomposableTable<F>>>,
    pub(crate) lookup_polys_offset: usize,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_comms: Vec<(usize, Pcs::Commitment)>,
}

impl<F, Pcs> PlonkishBackend<F> for HyperPlonk<Pcs>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    type Pcs = Pcs;
    type ProverParam = HyperPlonkProverParam<F, Pcs>;
    type VerifierParam = HyperPlonkVerifierParam<F, Pcs>;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<Pcs::Param, Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        Pcs::setup(poly_size, batch_size, rng)
    }

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;

        // Compute preprocesses comms
        let preprocess_polys = circuit_info
            .preprocess_polys
            .iter()
            .cloned()
            .map(MultilinearPolynomial::new)
            .collect_vec();
        let preprocess_comms = Pcs::batch_commit(&pcs_pp, &preprocess_polys)?;

        // Compute permutation polys and comms
        let permutation_polys = permutation_polys(
            num_vars,
            &circuit_info.permutation_polys(),
            &circuit_info.permutations,
        );
        let permutation_comms = Pcs::batch_commit(&pcs_pp, &permutation_polys)?;

        // Compose `VirtualPolynomialInfo`
        let (num_permutation_z_polys, expression) = compose(circuit_info);
        let lookup_polys_offset = circuit_info.num_instances.len()
            + preprocess_polys.len()
            + circuit_info.num_witness_polys.iter().sum::<usize>()
            + permutation_polys.len()
            + num_permutation_z_polys;

        let vp = HyperPlonkVerifierParam {
            pcs: pcs_vp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            lasso_tables: circuit_info
                .lasso_lookups
                .iter()
                .map(|(_, _, table)| table.clone())
                .collect_vec(),
            lookup_polys_offset,
            num_permutation_z_polys,
            num_vars,
            expression: expression.clone(),
            preprocess_comms: preprocess_comms.clone(),
            permutation_comms: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_comms.clone())
                .collect(),
        };
        let pp = HyperPlonkProverParam {
            pcs: pcs_pp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            lookups: circuit_info.lookups.clone(),
            lasso_lookups: circuit_info.lasso_lookups.clone(),
            lookup_polys_offset,
            num_permutation_z_polys,
            num_vars,
            expression,
            preprocess_polys,
            preprocess_comms,
            permutation_polys: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_polys)
                .collect(),
            permutation_comms,
        };
        Ok((pp, vp))
    }

    fn prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys(pp.num_vars, instances)
        };

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4);
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)?
                .into_iter()
                .map(MultilinearPolynomial::new)
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = iter::empty()
            .chain(instance_polys.iter())
            .chain(pp.preprocess_polys.iter())
            .chain(witness_polys.iter())
            .collect_vec();

        let (lookups, tables) = pp
            .lasso_lookups
            .iter()
            .map(|(input, index, table)| ((input, index), table))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let lookup_polys = Lasso::<F, Pcs>::lookup_polys(&polys, &lookups);
        let (lookup_input_polys, lookup_nz_polys) =
            lookup_polys.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

        let lookup_input_poly = &lookup_input_polys[0];
        let lookup_nz_poly = &lookup_nz_polys[0];
        let table = tables[0];
        let num_vars = lookup_input_poly.num_vars();
        // why this is 3??
        let lookup_points_offset = 3;

        // commit to input_poly
        let lookup_input_comm = Pcs::commit_and_write(&pp.pcs, lookup_input_poly, transcript)?;

        // get surge and dims
        let mut surge = Surge::<F, Pcs>::new();

        // commit to dims
        let dims = surge.commit(&table, lookup_nz_poly);
        let dim_comms = Pcs::batch_commit_and_write(&pp.pcs, &dims, transcript)?;

        // Round n
        // squeeze `r`
        let r = transcript.squeeze_challenges(num_vars);

        // get subtable_polys
        let subtable_polys = table.subtable_polys();
        let subtable_polys = subtable_polys.iter().collect_vec();
        let subtable_polys = subtable_polys.as_slice();

        // get e_polys & read_ts_polys & final_cts_polys
        let e_polys = {
            let nz = surge.nz();
            LassoProver::<F, Pcs>::e_polys(subtable_polys, &table, &nz)
        };
        let (read_ts_polys, final_cts_polys) = surge.counter_polys(&table);

        // commit to read_ts_polys & final_cts_polys & e_polys
        let read_ts_comms = Pcs::batch_commit_and_write(&pp.pcs, &read_ts_polys, transcript)?;
        let final_cts_comms = Pcs::batch_commit_and_write(&pp.pcs, &final_cts_polys, transcript)?;
        let e_comms = Pcs::batch_commit_and_write(&pp.pcs, e_polys.as_slice(), transcript)?;

        // Lasso Sumcheck
        let (lookup_points, lookup_evals) = Surge::<F, Pcs>::prove_sum_check(
            &table,
            lookup_input_poly,
            e_polys.as_slice(),
            &r,
            num_vars,
            pp.lookup_polys_offset,
            lookup_points_offset,
            transcript,
        )?;

        // squeeze memory checking challenges -> we will reuse beta, gamma for memory checking of Lasso
        // Round n+1
        let [beta, gamma] = transcript.squeeze_challenges(2).try_into().unwrap();

        // memory_checking
        let mut memory_checking = LassoProver::<F, Pcs>::prepare_memory_checking(
            &table,
            &subtable_polys,
            &e_polys,
            &dims,
            &read_ts_polys,
            &final_cts_polys,
            &beta,
            &gamma,
        );

        memory_checking
            .iter_mut()
            .map(|memory_checking| memory_checking.prove_grand_product(transcript))
            .collect::<Result<Vec<_>, Error>>()?;

        // for each memory_checking, prepare dims, e_polys, read_ts_polys and `x`
        // for each memory_checking, prepare final_cts_polys and `y`
        let mem_check_opening_points = memory_checking
            .iter()
            .flat_map(|memory_checking| memory_checking.opening_points())
            .collect_vec();

        let mem_check_opening_evals = memory_checking
            .iter()
            .enumerate()
            .flat_map(|(index, memory_checking)| {
                memory_checking.opening_evals(
                    table.num_chunks(),
                    pp.lookup_polys_offset,
                    lookup_points_offset + 2 + 2 * index,
                )
            })
            .collect_vec();

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys = permutation_z_polys(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        let permutation_z_comms =
            Pcs::batch_commit_and_write(&pp.pcs, permutation_z_polys.iter(), transcript)?;

        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let polys = iter::empty()
            .chain(polys)
            .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
            .chain(permutation_z_polys.iter())
            .collect_vec();
        challenges.extend([beta, gamma, alpha]);
        let (points, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        // PCS open
        let polys = iter::empty()
            .chain(polys)
            .chain([lookup_input_poly])
            .chain(dims.iter())
            .chain(read_ts_polys.iter())
            .chain(final_cts_polys.iter())
            .chain(e_polys.iter());
        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(pp.num_instances.len()))
            .chain(&pp.preprocess_comms)
            .chain(&witness_comms)
            .chain(&pp.permutation_comms)
            .chain(&permutation_z_comms)
            .chain([&lookup_input_comm])
            .chain(dim_comms.iter())
            .chain(read_ts_comms.iter())
            .chain(final_cts_comms.iter())
            .chain(e_comms.iter())
            .collect_vec();
        let points = iter::empty()
            .chain(points)
            .chain(lookup_points)
            .chain(mem_check_opening_points)
            .collect_vec();
        let evals = iter::empty()
            .chain(evals)
            .chain(lookup_evals)
            .chain(mem_check_opening_evals)
            .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        Pcs::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;
        end_timer(timer);
        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let num_witness_polys = vp.num_witness_polys.iter().sum();
        let mut witness_comms = Vec::with_capacity(num_witness_polys);
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        let lookup_points_offset = 3;
        // read input_comm, dim_comms
        let input_comm = Pcs::read_commitment(&vp.pcs, transcript)?;
        let lasso_lookup_tables = &vp.lasso_tables;
        let lookup_table = &lasso_lookup_tables[0];
        let num_chunks = lookup_table.num_chunks();
        let num_memories = lookup_table.num_memories();
        let dim_comms = Pcs::read_commitments(&vp.pcs, num_chunks, transcript)?;

        // Round n
        let r = transcript.squeeze_challenges(vp.num_vars);

        // read read_ts_comms & final_cts_comms & e_comms
        let read_ts_comms = Pcs::read_commitments(&vp.pcs, num_chunks, transcript)?;
        let final_cts_comms = Pcs::read_commitments(&vp.pcs, num_chunks, transcript)?;
        let e_comms = Pcs::read_commitments(&vp.pcs, num_memories, transcript)?;

        let (lookup_points, lookup_evals) = LassoVerifier::<F, Pcs>::verify_sum_check(
            lookup_table,
            vp.num_vars,
            vp.lookup_polys_offset,
            lookup_points_offset,
            &r,
            transcript,
        )?;

        // Round n+1

        let [beta, gamma] = transcript.squeeze_challenges(2).try_into().unwrap();

        // memory checking
        let memory_checking = LassoVerifier::<F, Pcs>::prepare_memory_checking(lookup_table);
        let (mem_check_opening_points, mem_check_opening_evals) =
            memory_checking
                .iter()
                .enumerate()
                .map(|(index, memory_checking)| {
                    memory_checking.verify_grand_product(
                        lookup_table.num_chunks(),
                        vp.num_vars,
                        vp.lookup_polys_offset,
                        lookup_points_offset + 2 + 2 * index,
                        &beta,
                        &gamma,
                        transcript
                    )
                })
                .collect::<Result<Vec<(Vec<Vec<F>>, Vec<Evaluation<F>>)>, Error>>()?
                .into_iter()
                .unzip::<_, _, Vec<_>, Vec<_>>();

        let permutation_z_comms =
            Pcs::read_commitments(&vp.pcs, vp.num_permutation_z_polys, transcript)?;
        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        challenges.extend([beta, gamma, alpha]);
        let (points, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        // PCS verify
        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(vp.num_instances.len()))
            .chain(&vp.preprocess_comms)
            .chain(&witness_comms)
            .chain(vp.permutation_comms.iter().map(|(_, comm)| comm))
            .chain(&permutation_z_comms)
            .chain([&input_comm])
            .chain(dim_comms.iter())
            .chain(read_ts_comms.iter())
            .chain(final_cts_comms.iter())
            .chain(e_comms.iter())
            .collect_vec();
        let points = iter::empty()
            .chain(points)
            .chain(lookup_points)
            .chain(mem_check_opening_points.concat())
            .collect_vec();
        let evals = iter::empty()
            .chain(evals)
            .chain(lookup_evals)
            .chain(mem_check_opening_evals.concat())
            .collect_vec();
        Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;

        Ok(())
    }
}

impl<Pcs> WitnessEncoding for HyperPlonk<Pcs> {
    fn row_mapping(k: usize) -> Vec<usize> {
        BooleanHypercube::new(k).iter().skip(1).chain([0]).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{
            hyperplonk::{
                util::{
                    rand_vanilla_plonk_circuit, rand_vanilla_plonk_with_lasso_lookup_circuit,
                    rand_vanilla_plonk_with_lookup_circuit,
                },
                HyperPlonk,
            },
            test::run_plonkish_backend,
        },
        pcs::{
            multilinear::{
                Gemini, MultilinearBrakedown, MultilinearHyrax, MultilinearIpa, MultilinearKzg,
                Zeromorph,
            },
            univariate::UnivariateKzg,
        },
        util::{
            code::BrakedownSpec6, hash::Keccak256, test::seeded_std_rng,
            transcript::Keccak256Transcript,
        },
    };
    use halo2_curves::{
        bn256::{self, Bn256},
        grumpkin,
    };

    macro_rules! tests {
        ($name:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<$name _hyperplonk_vanilla_plonk>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_circuit(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }

                #[test]
                fn [<$name _hyperplonk_vanilla_plonk_with_lookup>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_with_lookup_circuit(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }

                #[test]
                fn [<$name _hyperplonk_vanilla_plonk_with_lasso_lookup>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_with_lasso_lookup_circuit(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }
            }
        };
        ($name:ident, $pcs:ty) => {
            tests!($name, $pcs, 15..16);
        };
    }

    tests!(brakedown, MultilinearBrakedown<bn256::Fr, Keccak256, BrakedownSpec6>);
    tests!(hyrax, MultilinearHyrax<grumpkin::G1Affine>, 5..16);
    tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    tests!(kzg, MultilinearKzg<Bn256>);
    tests!(gemini_kzg, Gemini<UnivariateKzg<Bn256>>);
    tests!(zeromorph_kzg, Zeromorph<UnivariateKzg<Bn256>>);
}
