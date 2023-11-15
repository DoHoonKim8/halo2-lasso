pub use range::range_circuit;

mod range {
    use std::{array, iter};

    use halo2_proofs::halo2curves::ff::PrimeField;
    use plonkish_backend::{
        backend::{
            hyperplonk::util::Permutation,
            lookup::lasso::{test::range::RangeTable, DecomposableTable},
            mock::MockCircuit,
            PlonkishCircuit, PlonkishCircuitInfo,
        },
        halo2_curves::bn256::Fr,
        util::expression::{Expression, Query, Rotation},
    };
    use rand::RngCore;

    pub fn range_circuit(
        num_vars: usize,
        mut witness_rng: impl RngCore,
    ) -> (PlonkishCircuitInfo<Fr>, impl PlonkishCircuit<Fr>) {
        let size = 1 << num_vars;
        let mut polys = [(); 2].map(|_| vec![Fr::zero(); size]);

        let instances = vec![];
        polys[0] = instances;

        let mut permutation = Permutation::default();
        // dummy permutation
        permutation.copy((1, 1), (1, 1));

        for idx in 0..size - 1 {
            let w = Fr::from_u128((witness_rng.next_u64() as usize).pow(2) as u128);
            polys[1][idx] = w;
        }

        let [_, w] = polys;
        let circuit_info = range_circuit_info(
            num_vars,
            permutation.into_cycles(),
        );
        (circuit_info, MockCircuit::new(vec![vec![]], vec![w]))
    }

    fn range_circuit_info(
        num_vars: usize,
        permutations: Vec<Vec<(usize, usize)>>,
    ) -> PlonkishCircuitInfo<Fr> {
        let [_, w] = &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
            .map(Expression::<Fr>::Polynomial);
        let lasso_lookup_input = w.clone();
        let lasso_lookup_indices = w.clone();
        let range_table = Box::new(RangeTable::<Fr, 128, 16>::new());
        let chunk_bits = range_table.chunk_bits();
        let max_poly_size = iter::empty()
            .chain([&num_vars])
            .chain(chunk_bits.iter())
            .max()
            .unwrap();
        PlonkishCircuitInfo {
            k: *max_poly_size,
            num_vars,
            num_instances: vec![0],
            preprocess_polys: vec![],
            num_witness_polys: vec![1],
            num_challenges: vec![0],
            constraints: vec![],
            lookups: vec![vec![]],
            lasso_lookup: Some((lasso_lookup_input, lasso_lookup_indices, range_table)),
            permutations,
            max_degree: Some(4),
        }
    }
}
