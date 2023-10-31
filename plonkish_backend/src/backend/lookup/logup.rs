use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    iter,
    marker::PhantomData,
};

use halo2_curves::ff::{BatchInvert, Field, PrimeField};
use itertools::Itertools;

use crate::{
    pcs::{CommitmentChunk, PolynomialCommitmentScheme},
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, powers, sum, BooleanHypercube},
        end_timer,
        expression::{CommonPolynomial, Expression},
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        start_timer,
        transcript::TranscriptWrite,
    },
    Error,
};

use super::{MVLookupStrategy, MVLookupStrategyOutput};

#[derive(Clone, Debug)]
pub struct LogUp<F: Field, Pcs: PolynomialCommitmentScheme<F>>(PhantomData<F>, PhantomData<Pcs>);

impl<F: Field + PrimeField, Pcs: PolynomialCommitmentScheme<F>> LogUp<F, Pcs> {
    pub fn lookup_compressed_polys(
        lookups: &[Vec<(Expression<F>, Expression<F>)>],
        polys: &[&MultilinearPolynomial<F>],
        challenges: &[F],
        betas: &[F],
    ) -> Vec<[MultilinearPolynomial<F>; 2]> {
        if lookups.is_empty() {
            return Default::default();
        }

        let num_vars = polys[0].num_vars();
        let expression = lookups
            .iter()
            .flat_map(|lookup| lookup.iter().map(|(input, table)| (input + table)))
            .sum::<Expression<_>>();
        let lagranges = {
            let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
            expression
                .used_langrange()
                .into_iter()
                .map(|i| (i, bh[i.rem_euclid(1 << num_vars) as usize]))
                .collect::<HashSet<_>>()
        };
        lookups
            .iter()
            .map(|lookup| {
                Self::lookup_compressed_poly(lookup, &lagranges, polys, challenges, betas)
            })
            .collect()
    }

    pub fn lookup_compressed_poly(
        lookup: &[(Expression<F>, Expression<F>)],
        lagranges: &HashSet<(i32, usize)>,
        polys: &[&MultilinearPolynomial<F>],
        challenges: &[F],
        betas: &[F],
    ) -> [MultilinearPolynomial<F>; 2] {
        let num_vars = polys[0].num_vars();
        let bh = BooleanHypercube::new(num_vars);
        let compress = |expressions: &[&Expression<F>]| {
            betas
                .iter()
                .copied()
                .zip(expressions.iter().map(|expression| {
                    let mut compressed = vec![F::ZERO; 1 << num_vars];
                    parallelize(&mut compressed, |(compressed, start)| {
                        for (b, compressed) in (start..).zip(compressed) {
                            *compressed = expression.evaluate(
                                &|constant| constant,
                                &|common_poly| match common_poly {
                                    CommonPolynomial::Identity => F::from(b as u64),
                                    CommonPolynomial::Lagrange(i) => {
                                        if lagranges.contains(&(i, b)) {
                                            F::ONE
                                        } else {
                                            F::ZERO
                                        }
                                    }
                                    CommonPolynomial::EqXY(_) => unreachable!(),
                                },
                                &|query| polys[query.poly()][bh.rotate(b, query.rotation())],
                                &|challenge| challenges[challenge],
                                &|value| -value,
                                &|lhs, rhs| lhs + &rhs,
                                &|lhs, rhs| lhs * &rhs,
                                &|value, scalar| value * &scalar,
                            );
                        }
                    });
                    MultilinearPolynomial::new(compressed)
                }))
                .sum::<MultilinearPolynomial<_>>()
        };

        let (inputs, tables) = lookup
            .iter()
            .map(|(input, table)| (input, table))
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let timer = start_timer(|| "compressed_input_poly");
        let compressed_input_poly = compress(&inputs);
        end_timer(timer);

        let timer = start_timer(|| "compressed_table_poly");
        let compressed_table_poly = compress(&tables);
        end_timer(timer);

        [compressed_input_poly, compressed_table_poly]
    }
}

impl<F: Field + PrimeField + Hash, Pcs: PolynomialCommitmentScheme<F>> LogUp<F, Pcs> {
    pub(crate) fn lookup_m_polys(
        compressed_polys: &[[MultilinearPolynomial<F>; 2]],
    ) -> Result<Vec<MultilinearPolynomial<F>>, Error> {
        compressed_polys
            .iter()
            .map(|compressed_polys| Self::lookup_m_poly(compressed_polys))
            .try_collect()
    }

    pub(super) fn lookup_m_poly(
        compressed_polys: &[MultilinearPolynomial<F>; 2],
    ) -> Result<MultilinearPolynomial<F>, Error> {
        let [input, table] = compressed_polys;

        let counts = {
            let indice_map = table.iter().zip(0..).collect::<HashMap<_, usize>>();

            let chunk_size = div_ceil(input.evals().len(), num_threads());
            let num_chunks = div_ceil(input.evals().len(), chunk_size);
            let mut counts = vec![HashMap::new(); num_chunks];
            let mut valids = vec![true; num_chunks];
            parallelize_iter(
                counts
                    .iter_mut()
                    .zip(valids.iter_mut())
                    .zip((0..).step_by(chunk_size)),
                |((count, valid), start)| {
                    for input in input[start..].iter().take(chunk_size) {
                        if let Some(idx) = indice_map.get(input) {
                            count
                                .entry(*idx)
                                .and_modify(|count| *count += 1)
                                .or_insert(1);
                        } else {
                            *valid = false;
                            break;
                        }
                    }
                },
            );
            if valids.iter().any(|valid| !valid) {
                return Err(Error::InvalidSnark("Invalid lookup input".to_string()));
            }
            counts
        };

        let mut m = vec![0; 1 << input.num_vars()];
        for (idx, count) in counts.into_iter().flatten() {
            m[idx] += count;
        }
        let m = par_map_collect(m, |count| match count {
            0 => F::ZERO,
            1 => F::ONE,
            count => F::from(count),
        });
        Ok(MultilinearPolynomial::new(m))
    }

    pub(super) fn lookup_h_polys(
        compressed_polys: &[[MultilinearPolynomial<F>; 2]],
        m_polys: &[MultilinearPolynomial<F>],
        gamma: &F,
    ) -> Vec<MultilinearPolynomial<F>> {
        compressed_polys
            .iter()
            .zip(m_polys.iter())
            .map(|(compressed_polys, m_poly)| Self::lookup_h_poly(compressed_polys, m_poly, gamma))
            .collect()
    }

    pub(super) fn lookup_h_poly(
        compressed_polys: &[MultilinearPolynomial<F>; 2],
        m_poly: &MultilinearPolynomial<F>,
        gamma: &F,
    ) -> MultilinearPolynomial<F> {
        let [input, table] = compressed_polys;
        let mut h_input = vec![F::ZERO; 1 << input.num_vars()];
        let mut h_table = vec![F::ZERO; 1 << input.num_vars()];

        parallelize(&mut h_input, |(h_input, start)| {
            for (h_input, input) in h_input.iter_mut().zip(input[start..].iter()) {
                *h_input = *gamma + input;
            }
        });
        parallelize(&mut h_table, |(h_table, start)| {
            for (h_table, table) in h_table.iter_mut().zip(table[start..].iter()) {
                *h_table = *gamma + table;
            }
        });

        let chunk_size = div_ceil(2 * h_input.len(), num_threads());
        parallelize_iter(
            iter::empty()
                .chain(h_input.chunks_mut(chunk_size))
                .chain(h_table.chunks_mut(chunk_size)),
            |h| {
                h.iter_mut().batch_invert();
            },
        );

        parallelize(&mut h_input, |(h_input, start)| {
            for (h_input, (h_table, m)) in h_input
                .iter_mut()
                .zip(h_table[start..].iter().zip(m_poly[start..].iter()))
            {
                *h_input -= *h_table * m;
            }
        });

        if cfg!(feature = "sanity-check") {
            assert_eq!(sum::<F>(&h_input), F::ZERO);
        }

        MultilinearPolynomial::new(h_input)
    }
}

impl<
        F: Field + PrimeField + Hash,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    > MVLookupStrategy<F> for LogUp<F, Pcs>
{
    type Pcs = Pcs;

    fn preprocess(
        lookups: &[Vec<(Expression<F>, Expression<F>)>],
        polys: &[&MultilinearPolynomial<F>],
        challenges: &mut Vec<F>,
    ) -> Result<Vec<[MultilinearPolynomial<F>; 2]>, Error> {
        let timer = start_timer(|| format!("lookup_compressed_polys-{}", lookups.len()));
        let lookup_compressed_polys = {
            let beta = challenges.last().unwrap();
            let max_lookup_width = lookups.iter().map(Vec::len).max().unwrap_or_default();
            let betas = powers(*beta).take(max_lookup_width).collect_vec();
            Self::lookup_compressed_polys(lookups, &polys, &challenges, &betas)
        };
        end_timer(timer);
        Ok(lookup_compressed_polys)
    }

    fn commit(
        pp: &Pcs::ProverParam,
        lookup_compressed_polys: &[[MultilinearPolynomial<F>; 2]],
        challenges: &mut Vec<F>,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
    ) -> Result<MVLookupStrategyOutput<F, Self::Pcs>, crate::Error> {
        let timer = start_timer(|| format!("lookup_m_polys-{}", lookup_compressed_polys.len()));
        let lookup_m_polys = Self::lookup_m_polys(&lookup_compressed_polys)?;
        end_timer(timer);

        let lookup_m_comms = Pcs::batch_commit_and_write(&pp, &lookup_m_polys, transcript)?;

        let gamma = transcript.squeeze_challenge();
        challenges.extend([gamma]);

        let timer = start_timer(|| format!("lookup_h_polys-{}", lookup_compressed_polys.len()));
        let lookup_h_polys =
            Self::lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);
        end_timer(timer);

        let lookup_h_comms = Pcs::batch_commit_and_write(&pp, &lookup_h_polys, transcript)?;

        let mut polys = Vec::with_capacity(2 * lookup_compressed_polys.len());
        polys.extend([lookup_m_polys, lookup_h_polys]);
        let mut comms = Vec::with_capacity(lookup_m_comms.len() + lookup_h_comms.len());
        comms.extend([lookup_m_comms, lookup_h_comms]);
        Ok(MVLookupStrategyOutput { polys, comms })
    }
}
