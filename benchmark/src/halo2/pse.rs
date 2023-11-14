pub use aggregation::AggregationCircuit;
pub use range::RangeCircuit;
pub use sha256::Sha256Circuit;

mod aggregation {
    use halo2_proofs::{
        circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
        plonk::{create_proof, keygen_pk, keygen_vk, Circuit, ConstraintSystem, Error},
        poly::{
            commitment::{Params, ParamsProver},
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::ProverGWC,
            },
        },
    };
    use itertools::Itertools;
    use plonkish_backend::{
        frontend::halo2::{circuit::VanillaPlonk, CircuitExt},
        halo2_curves::{
            ff::PrimeField,
            ff::{FromUniformBytes, WithSmallOrderMulGroup},
            pairing::Engine,
            serde::SerdeObject,
            CurveAffine,
        },
        util::arithmetic::MultiMillerLoop,
    };
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use snark_verifier::{
        loader::{
            self,
            halo2::{
                halo2_wrong_ecc::{self, integer::rns::Rns, maingate::*, EccConfig},
                Scalar,
            },
            native::NativeLoader,
        },
        pcs::{
            kzg::{self, *},
            AccumulationDecider, AccumulationScheme, AccumulationSchemeProver,
        },
        system::halo2::{compile, transcript, Config},
        util::arithmetic::fe_to_limbs,
        verifier::{self, plonk::PlonkProtocol, SnarkVerifier},
    };
    use std::rc::Rc;

    const LIMBS: usize = 4;
    const BITS: usize = 68;

    type KzgAs<M> = kzg::KzgAs<M, Gwc19>;
    type KzgSvk<M> = KzgSuccinctVerifyingKey<<M as Engine>::G1Affine>;
    type KzgDk<M> = KzgDecidingKey<M>;
    type PlonkSuccinctVerifier<M> = verifier::plonk::PlonkSuccinctVerifier<KzgAs<M>>;

    const T: usize = 5;
    const RATE: usize = 4;
    const R_F: usize = 8;
    const R_P: usize = 60;

    type EccChip<C> = halo2_wrong_ecc::BaseFieldEccChip<C, LIMBS, BITS>;
    type Halo2Loader<'a, C> = loader::halo2::Halo2Loader<'a, C, EccChip<C>>;
    type PoseidonTranscript<C, S> =
        transcript::halo2::PoseidonTranscript<C, NativeLoader, S, T, RATE, R_F, R_P>;

    #[derive(Clone)]
    struct Snark<C: CurveAffine> {
        protocol: PlonkProtocol<C>,
        instances: Vec<Vec<C::Scalar>>,
        proof: Vec<u8>,
    }

    impl<C: CurveAffine> Snark<C> {
        fn new(protocol: PlonkProtocol<C>, instances: Vec<Vec<C::Scalar>>, proof: Vec<u8>) -> Self {
            Self {
                protocol,
                instances,
                proof,
            }
        }
    }

    impl<C: CurveAffine> From<Snark<C>> for SnarkWitness<C> {
        fn from(snark: Snark<C>) -> Self {
            Self {
                protocol: snark.protocol,
                instances: Value::known(snark.instances),
                proof: Value::known(snark.proof),
            }
        }
    }

    #[derive(Clone)]
    struct SnarkWitness<C: CurveAffine> {
        protocol: PlonkProtocol<C>,
        instances: Value<Vec<Vec<C::Scalar>>>,
        proof: Value<Vec<u8>>,
    }

    impl<C: CurveAffine> SnarkWitness<C> {
        fn proof(&self) -> Value<&[u8]> {
            self.proof.as_ref().map(Vec::as_slice)
        }

        fn loaded_instances<'b>(
            &self,
            loader: &Rc<Halo2Loader<'b, C>>,
        ) -> Vec<Vec<Scalar<'b, C, EccChip<C>>>> {
            self.instances
                .as_ref()
                .transpose_vec(self.protocol.num_instance.len())
                .into_iter()
                .zip(self.protocol.num_instance.iter())
                .map(|(instances, num_instance)| {
                    instances
                        .transpose_vec(*num_instance)
                        .into_iter()
                        .map(|instance| loader.assign_scalar(instance.copied()))
                        .collect_vec()
                })
                .collect_vec()
        }
    }

    #[derive(Clone)]
    pub struct AggregationConfig {
        main_gate_config: MainGateConfig,
        range_config: RangeConfig,
    }

    impl AggregationConfig {
        fn configure<C: CurveAffine>(meta: &mut ConstraintSystem<C::Scalar>) -> Self {
            let main_gate_config = MainGate::<C::Scalar>::configure(meta);
            let range_config = RangeChip::<C::Scalar>::configure(
                meta,
                &main_gate_config,
                vec![BITS / LIMBS],
                Rns::<C::Base, C::Scalar, LIMBS, BITS>::construct().overflow_lengths(),
            );
            Self {
                main_gate_config,
                range_config,
            }
        }

        fn main_gate<F: PrimeField>(&self) -> MainGate<F> {
            MainGate::new(self.main_gate_config.clone())
        }

        fn range_chip<F: PrimeField>(&self) -> RangeChip<F> {
            RangeChip::new(self.range_config.clone())
        }

        fn ecc_chip<C: CurveAffine>(&self) -> EccChip<C> {
            EccChip::new(EccConfig::new(
                self.range_config.clone(),
                self.main_gate_config.clone(),
            ))
        }

        fn load_table<F: PrimeField>(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
            self.range_chip().load_table(layouter)
        }

        #[allow(clippy::type_complexity)]
        fn aggregate<'a, M: MultiMillerLoop>(
            &self,
            layouter: &mut impl Layouter<M::Scalar>,
            svk: &KzgSvk<M>,
            snarks: impl IntoIterator<Item = &'a SnarkWitness<M::G1Affine>>,
        ) -> Result<Vec<AssignedCell<M::Scalar, M::Scalar>>, Error>
        where
            M::Scalar: FromUniformBytes<64>,
            M::G1Affine: SerdeObject,
            M::G2Affine: SerdeObject,
        {
            type PoseidonTranscript<'a, C, S> = transcript::halo2::PoseidonTranscript<
                C,
                Rc<Halo2Loader<'a, C>>,
                S,
                T,
                RATE,
                R_F,
                R_P,
            >;
            let snarks = snarks.into_iter().collect_vec();
            layouter.assign_region(
                || "Aggregate snarks",
                |region| {
                    let ctx = RegionCtx::new(region, 0);

                    let ecc_chip = self.ecc_chip::<M::G1Affine>();
                    let loader = Halo2Loader::new(ecc_chip, ctx);

                    let accumulators = snarks
                        .iter()
                        .map(|snark| {
                            let protocol = snark.protocol.loaded(&loader);
                            let instances = snark.loaded_instances(&loader);
                            let mut transcript = PoseidonTranscript::new(&loader, snark.proof());
                            let proof = PlonkSuccinctVerifier::<M>::read_proof(
                                svk,
                                &protocol,
                                &instances,
                                &mut transcript,
                            )
                            .unwrap();
                            PlonkSuccinctVerifier::verify(svk, &protocol, &instances, &proof)
                                .unwrap()
                        })
                        .collect_vec();

                    let accumulator = {
                        let as_vk = Default::default();
                        let as_proof = Vec::new();
                        let accumulators = accumulators.into_iter().flatten().collect_vec();
                        let mut transcript =
                            PoseidonTranscript::new(&loader, Value::known(as_proof.as_slice()));
                        let proof =
                            KzgAs::<M>::read_proof(&as_vk, &accumulators, &mut transcript).unwrap();
                        KzgAs::<M>::verify(&as_vk, &accumulators, &proof).unwrap()
                    };

                    let accumulator_limbs = [accumulator.lhs, accumulator.rhs]
                        .iter()
                        .map(|ec_point| {
                            loader.ecc_chip().assign_ec_point_to_limbs(
                                &mut loader.ctx_mut(),
                                ec_point.assigned(),
                            )
                        })
                        .collect::<Result<Vec<_>, Error>>()?
                        .into_iter()
                        .flatten()
                        .collect();

                    Ok(accumulator_limbs)
                },
            )
        }
    }

    fn aggregate<'a, M: MultiMillerLoop>(
        param: &ParamsKZG<M>,
        snarks: impl IntoIterator<Item = &'a Snark<M::G1Affine>>,
    ) -> Option<[M::Scalar; 4 * LIMBS]>
    where
        M::Scalar: FromUniformBytes<64>,
        M::G1Affine: SerdeObject,
        M::G2Affine: SerdeObject,
    {
        let svk = KzgSvk::<M>::new(param.get_g()[0]);
        let dk = KzgDk::new(svk, param.g2(), param.s_g2());

        let accumulators = snarks
            .into_iter()
            .map(|snark| {
                let mut transcript = PoseidonTranscript::new(snark.proof.as_slice());
                let proof = PlonkSuccinctVerifier::<M>::read_proof(
                    &svk,
                    &snark.protocol,
                    &snark.instances,
                    &mut transcript,
                )?;
                PlonkSuccinctVerifier::verify(&svk, &snark.protocol, &snark.instances, &proof)
            })
            .try_collect::<_, Vec<_>, _>()
            .ok()?
            .into_iter()
            .flatten()
            .collect_vec();

        let accumulator = {
            let as_pk = Default::default();
            let rng = StdRng::from_seed(Default::default());
            let mut transcript = PoseidonTranscript::new(Vec::new());
            let accumulator =
                KzgAs::<M>::create_proof(&as_pk, &accumulators, &mut transcript, rng).ok()?;
            assert!(transcript.finalize().is_empty());
            accumulator
        };
        KzgAs::<M>::decide(&dk, accumulator.clone()).ok()?;

        let KzgAccumulator { lhs, rhs } = accumulator;
        let accumulator_limbs = [lhs, rhs]
            .into_iter()
            .flat_map(|point| {
                let coordinates = point.coordinates().unwrap();
                [*coordinates.x(), *coordinates.y()]
            })
            .flat_map(fe_to_limbs::<_, _, LIMBS, BITS>)
            .collect_vec()
            .try_into()
            .unwrap();
        Some(accumulator_limbs)
    }

    pub struct AggregationCircuit<M: MultiMillerLoop> {
        svk: KzgSvk<M>,
        snarks: Vec<SnarkWitness<M::G1Affine>>,
        instances: Vec<M::Scalar>,
    }

    impl<M: MultiMillerLoop> Circuit<M::Scalar> for AggregationCircuit<M>
    where
        M::Scalar: FromUniformBytes<64>,
        M::G1Affine: SerdeObject,
        M::G2Affine: SerdeObject,
    {
        type Config = AggregationConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            unimplemented!()
        }

        fn configure(meta: &mut ConstraintSystem<M::Scalar>) -> Self::Config {
            AggregationConfig::configure::<M::G1Affine>(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<M::Scalar>,
        ) -> Result<(), Error> {
            config.load_table(&mut layouter)?;

            let main_gate = config.main_gate();
            for (row, limb) in config
                .aggregate::<M>(&mut layouter, &self.svk, &self.snarks)?
                .into_iter()
                .enumerate()
            {
                main_gate.expose_public(layouter.namespace(|| ""), limb, row)?;
            }

            Ok(())
        }
    }

    impl<M: MultiMillerLoop> CircuitExt<M::Scalar> for AggregationCircuit<M>
    where
        M::Scalar: FromUniformBytes<64> + WithSmallOrderMulGroup<3>,
        M::G1Affine: SerdeObject,
        M::G2Affine: SerdeObject,
    {
        fn rand(k: usize, mut rng: impl RngCore) -> Self {
            let param = ParamsKZG::<M>::setup(4, &mut rng);
            let snark = {
                let circuit = VanillaPlonk::rand(param.k() as usize, &mut rng);
                let vk = keygen_vk::<_, _, _, true>(&param, &circuit).unwrap();
                let pk = keygen_pk::<_, _, _, true>(&param, vk, &circuit).unwrap();
                let protocol = compile(
                    &param,
                    pk.get_vk(),
                    Config::kzg().with_num_instance(vec![1]),
                );
                let instances = circuit.instances();
                let proof = {
                    let mut transcript = PoseidonTranscript::new(Vec::new());
                    create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, _, _, true>(
                        &param,
                        &pk,
                        &[circuit],
                        &[&instances.iter().map(Vec::as_slice).collect_vec()],
                        &mut rng,
                        &mut transcript,
                    )
                    .unwrap();
                    transcript.finalize()
                };
                Snark::new(protocol, instances, proof)
            };
            let snarks = vec![snark; (1 << k) / 1000000];
            let accumulator_limbs = aggregate(&param, &snarks).unwrap();
            Self {
                svk: KzgSvk::<M>::new(param.get_g()[0]),
                snarks: snarks.into_iter().map_into().collect(),
                instances: accumulator_limbs.to_vec(),
            }
        }

        fn instances(&self) -> Vec<Vec<M::Scalar>> {
            vec![self.instances.clone()]
        }
    }
}

// Modified from https://github.com/celer-network/halo2/tree/KZG-bench-sha256
mod sha256 {
    use halo2_gadgets::sha256::{BlockWord, Sha256, Table16Chip, Table16Config};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use plonkish_backend::{frontend::halo2::CircuitExt, halo2_curves::bn256::Fr};
    use rand::RngCore;

    const INPUT_2: [BlockWord; 16 * 2] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 2];
    const INPUT_3: [BlockWord; 16 * 3] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 3];
    const INPUT_5: [BlockWord; 16 * 5] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 5];
    const INPUT_9: [BlockWord; 16 * 9] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 9];
    const INPUT_17: [BlockWord; 16 * 17] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 17];
    const INPUT_33: [BlockWord; 16 * 33] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 33];
    const INPUT_65: [BlockWord; 16 * 65] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 65];
    const INPUT_129: [BlockWord; 16 * 129] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 129];
    const INPUT_257: [BlockWord; 16 * 257] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 257];
    const INPUT_513: [BlockWord; 16 * 513] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 513];
    const INPUT_1025: [BlockWord; 16 * 1025] =
        [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 1025];

    #[derive(Default)]
    pub struct Sha256Circuit {
        input_size: usize,
    }

    impl Circuit<Fr> for Sha256Circuit {
        type Config = Table16Config;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            unimplemented!()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            Table16Chip::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            Table16Chip::load(config.clone(), &mut layouter)?;
            let chip = Table16Chip::construct(config);
            let input = match self.input_size {
                2 => INPUT_2.as_slice(),
                3 => INPUT_3.as_slice(),
                5 => INPUT_5.as_slice(),
                9 => INPUT_9.as_slice(),
                17 => INPUT_17.as_slice(),
                33 => INPUT_33.as_slice(),
                65 => INPUT_65.as_slice(),
                129 => INPUT_129.as_slice(),
                257 => INPUT_257.as_slice(),
                513 => INPUT_513.as_slice(),
                1025 => INPUT_1025.as_slice(),
                _ => panic!("Unexpected input_size: {}", self.input_size),
            };
            Sha256::digest(chip, layouter, input).map(|_| ())
        }
    }

    impl CircuitExt<Fr> for Sha256Circuit {
        fn rand(k: usize, _: impl RngCore) -> Self {
            assert!(k > 16);
            let input_size = if k > 22 {
                1025
            } else {
                [33, 65, 129, 257, 513, 1025][k - 17]
            };
            Self { input_size }
        }

        fn instances(&self) -> Vec<Vec<Fr>> {
            Vec::new()
        }
    }
}

mod range {
    use halo2_proofs::circuit::floor_planner::V1;
    use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
    use halo2_proofs::halo2curves::ff::PrimeField;
    use halo2_proofs::plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, Selector,
    };
    use halo2_proofs::poly::Rotation;
    use itertools::Itertools;
    use plonkish_backend::util::arithmetic::{fe_to_bits_le, usize_from_bits_le};
    use plonkish_backend::{frontend::halo2::CircuitExt, halo2_curves::bn256::Fr};
    use rand::RngCore;

    #[derive(Clone, Debug)]
    pub struct RangeCircuitConfig {
        pub(crate) a1: Column<Advice>,
        pub(crate) a2: Column<Advice>,
        pub(crate) a3: Column<Advice>,
        pub(crate) a4: Column<Advice>,
        pub(crate) a5: Column<Advice>,
        pub(crate) a6: Column<Advice>,
        pub(crate) a7: Column<Advice>,
        pub(crate) a8: Column<Advice>,
        pub(crate) w: Column<Advice>,

        pub(crate) t: Column<Fixed>,
        pub(crate) s_gate: Selector,
        pub(crate) s_lookup: Selector,
    }

    pub struct RangeCircuit {
        inputs: Vec<Fr>,
    }

    impl Circuit<Fr> for RangeCircuit {
        type Config = RangeCircuitConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self { inputs: vec![] }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a1 = meta.advice_column();
            let a2 = meta.advice_column();
            let a3 = meta.advice_column();
            let a4 = meta.advice_column();
            let a5 = meta.advice_column();
            let a6 = meta.advice_column();
            let a7 = meta.advice_column();
            let a8 = meta.advice_column();
            let w = meta.advice_column();
            let s_gate = meta.selector();

            let t = meta.fixed_column();
            let s_lookup = meta.complex_selector();

            meta.create_gate("composition", |meta| {
                let a1 = meta.query_advice(a1, Rotation::cur());
                let a2 = meta.query_advice(a2, Rotation::cur());
                let a3 = meta.query_advice(a3, Rotation::cur());
                let a4 = meta.query_advice(a4, Rotation::cur());
                let a5 = meta.query_advice(a5, Rotation::cur());
                let a6 = meta.query_advice(a6, Rotation::cur());
                let a7 = meta.query_advice(a7, Rotation::cur());
                let a8 = meta.query_advice(a8, Rotation::cur());

                let w = meta.query_advice(w, Rotation::cur());
                let s_gate = meta.query_selector(s_gate);

                let composed = [a2, a3, a4, a5, a6, a7, a8]
                    .into_iter()
                    .enumerate()
                    .fold(a1, |acc, (i, expr)| {
                        acc + expr * Expression::Constant(Fr::from(1 << (16 * (i + 1))))
                    });
                vec![s_gate * (composed - w)]
            });

            for col in [a1, a2, a3, a4, a5, a6, a7, a8] {
                meta.lookup_any("", |meta| {
                    let selector = meta.query_selector(s_lookup);
                    let value = meta.query_advice(col, Rotation::cur());
                    let t = meta.query_fixed(t, Rotation::cur());
                    vec![(selector * value, t)]
                });
            }

            RangeCircuitConfig {
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                w,
                s_gate,
                t,
                s_lookup,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "",
                |mut region| {
                    self.inputs.iter().enumerate().for_each(|(i, input)| {
                        let input_bits = &fe_to_bits_le(input.clone())[..128];
                        assert_eq!(
                            Fr::from_u128(usize_from_bits_le(input_bits) as u128),
                            *input
                        );
                        config.s_gate.enable(&mut region, i).unwrap();
                        config.s_lookup.enable(&mut region, i).unwrap();
                        region
                            .assign_advice(|| "", config.w, i, || Value::known(*input))
                            .unwrap();
                        [
                            config.a1, config.a2, config.a3, config.a4, config.a5, config.a6,
                            config.a7, config.a8,
                        ]
                        .iter()
                        .zip(input_bits.chunks(16))
                        .map(|(col, limb)| {
                            region.assign_advice(
                                || "",
                                *col,
                                i,
                                || Value::known(Fr::from(usize_from_bits_le(limb) as u64)),
                            )
                        })
                        .collect::<Result<Vec<_>, Error>>()
                        .unwrap();
                    });

                    Ok(())
                },
            )?;

            // load table
            layouter.assign_region(
                || "",
                |mut region| {
                    let mut offset = 0;
                    let table_values: Vec<Fr> = (0..1 << 16).map(|e| Fr::from(e as u64)).collect();
                    for value in table_values.iter() {
                        region.assign_fixed(|| "", config.t, offset, || Value::known(*value))?;
                        offset += 1;
                    }
                    Ok(())
                },
            )?;

            Ok(())
        }
    }

    impl CircuitExt<Fr> for RangeCircuit {
        fn rand(k: usize, mut rng: impl RngCore) -> Self {
            let inputs = vec![(); 1 << k]
                .iter()
                .map(|_| Fr::from_u128((rng.next_u64() as usize).pow(2) as u128))
                .collect_vec();
            Self { inputs }
        }

        fn instances(&self) -> Vec<Vec<Fr>> {
            vec![]
        }
    }
}
