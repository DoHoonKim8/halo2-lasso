use std::{array, iter};

use halo2_curves::ff::PrimeField;
use itertools::{chain, izip, Itertools};

use crate::{
    piop::{
        gkr::{err_unmatched_sum_check_output, eval_by_query},
        sum_check::{
            classic::{ClassicSumCheck, EvaluationsProver},
            evaluate, SumCheck as _, VirtualPolynomial,
        },
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, inner_product, powers},
        expression::{Expression, Query, Rotation},
        parallel::{num_threads, parallelize_iter},
        transcript::{FieldTranscriptRead, FieldTranscriptWrite},
    },
    Error,
};

type SumCheck<F> = ClassicSumCheck<EvaluationsProver<F>>;

struct Layer<F> {
    v_l: MultilinearPolynomial<F>,
    v_r: MultilinearPolynomial<F>,
}

impl<F> From<[Vec<F>; 2]> for Layer<F> {
    fn from(values: [Vec<F>; 2]) -> Self {
        let [v_l, v_r] = values.map(MultilinearPolynomial::new);
        Self { v_l, v_r }
    }
}

impl<F: PrimeField> Layer<F> {
    fn bottom(v: &&MultilinearPolynomial<F>) -> Self {
        let mid = v.evals().len() >> 1;
        [&v[..mid], &v[mid..]].map(ToOwned::to_owned).into()
    }

    fn num_vars(&self) -> usize {
        self.v_l.num_vars()
    }

    fn polys(&self) -> [&MultilinearPolynomial<F>; 2] {
        [&self.v_l, &self.v_r]
    }

    fn poly_chunks(&self, chunk_size: usize) -> impl Iterator<Item = (&[F], &[F])> {
        let [v_l, v_r] = self.polys().map(|poly| poly.evals().chunks(chunk_size));
        izip!(v_l, v_r)
    }

    fn up(&self) -> Self {
        assert!(self.num_vars() != 0);

        let len = 1 << self.num_vars();
        let chunk_size = div_ceil(len, num_threads()).next_power_of_two();

        let mut outputs: [_; 2] = array::from_fn(|_| vec![F::ZERO; len >> 1]);
        let (v_up_l, v_up_r) = outputs.split_at_mut(1);

        parallelize_iter(
            izip!(
                chain![v_up_l, v_up_r].flat_map(|v_up| v_up.chunks_mut(chunk_size)),
                self.poly_chunks(chunk_size),
            ),
            |(v_up, (v_l, v_r))| {
                izip!(v_up, v_l, v_r).for_each(|(v_up, v_l, v_r)| {
                    *v_up = *v_l * *v_r;
                })
            },
        );

        outputs.into()
    }
}

pub fn prove_grand_product<'a, F: PrimeField>(
    claimed_v_0s: impl IntoIterator<Item = Option<F>>,
    vs: impl IntoIterator<Item = &'a MultilinearPolynomial<F>>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<F>, Vec<F>), Error> {
    let claimed_v_0s = claimed_v_0s.into_iter().collect_vec();
    let vs = vs.into_iter().collect_vec();
    let num_batching = claimed_v_0s.len();

    assert!(num_batching != 0);
    assert_eq!(num_batching, vs.len());
    for poly in &vs {
        assert_eq!(poly.num_vars(), vs[0].num_vars());
    }

    let bottom_layers = vs.iter().map(Layer::bottom).collect_vec();
    let layers = iter::successors(bottom_layers.into(), |layers| {
        (layers[0].num_vars() > 0).then(|| layers.iter().map(Layer::up).collect())
    })
    .collect_vec();

    let claimed_v_0s = {
        let v_0s = chain![layers.last().unwrap()]
            .map(|layer| {
                let [v_l, v_r] = layer.polys().map(|poly| poly[0]);
                v_l * v_r
            })
            .collect_vec();

        let mut hash_to_transcript = |claimed: Vec<_>, computed: Vec<_>| {
            izip!(claimed, computed)
                .map(|(claimed, computed)| match claimed {
                    Some(claimed) => {
                        if cfg!(feature = "sanity-check") {
                            assert_eq!(claimed, computed)
                        }
                        transcript.common_field_element(&computed).map(|_| computed)
                    }
                    None => transcript.write_field_element(&computed).map(|_| computed),
                })
                .try_collect::<_, Vec<_>, _>()
        };

        hash_to_transcript(claimed_v_0s, v_0s)?
    };

    let expression = sum_check_expression(num_batching);

    let (v_xs, x) =
        layers
            .iter()
            .rev()
            .fold(Ok((claimed_v_0s, Vec::new())), |result, layers| {
                let (claimed_v_ys, y) = result?;

                let num_vars = layers[0].num_vars();
                let polys = layers.iter().flat_map(|layer| layer.polys());

                let (mut x, evals) = if num_vars == 0 {
                    (vec![], polys.map(|poly| poly[0]).collect_vec())
                } else {
                    let gamma = transcript.squeeze_challenge();

                    let (x, evals) = {
                        let claim = sum_check_claim(&claimed_v_ys, gamma);
                        SumCheck::prove(
                            &(),
                            num_vars,
                            VirtualPolynomial::new(&expression, polys, &[gamma], &[y]),
                            claim,
                            transcript,
                        )?
                    };

                    (x, evals)
                };

                transcript.write_field_elements(&evals)?;

                let mu = transcript.squeeze_challenge();

                let v_xs = layer_down_claim(&evals, mu);
                x.push(mu);

                Ok((v_xs, x))
            })?;

    if cfg!(feature = "sanity-check") {
        izip!(vs, &v_xs).for_each(|(poly, eval)| assert_eq!(poly.evaluate(&x), *eval));
    }

    Ok((v_xs, x))
}

pub fn verify_grand_product<F: PrimeField>(
    num_vars: usize,
    claimed_v_0s: impl IntoIterator<Item = Option<F>>,
    transcript: &mut impl FieldTranscriptRead<F>,
) -> Result<(Vec<F>, Vec<F>), Error> {
    let claimed_v_0s = claimed_v_0s.into_iter().collect_vec();
    let num_batching = claimed_v_0s.len();

    assert!(num_batching != 0);
    let claimed_v_0s = {
        claimed_v_0s
            .into_iter()
            .map(|claimed| match claimed {
                Some(claimed) => transcript.common_field_element(&claimed).map(|_| claimed),
                None => transcript.read_field_element(),
            })
            .try_collect::<_, Vec<_>, _>()?
    };

    let expression = sum_check_expression(num_batching);

    let (v_xs, x) = (0..num_vars).fold(Ok((claimed_v_0s, Vec::new())), |result, num_vars| {
        let (claimed_v_ys, y) = result?;

        let (mut x, evals) = if num_vars == 0 {
            let evals = transcript.read_field_elements(2 * num_batching)?;

            for (claimed_v, (&v_l, &v_r)) in izip!(claimed_v_ys, evals.iter().tuples()) {
                if claimed_v != v_l * v_r {
                    return Err(err_unmatched_sum_check_output());
                }
            }

            (Vec::new(), evals)
        } else {
            let gamma = transcript.squeeze_challenge();

            let (x_eval, x) = {
                let claim = sum_check_claim(&claimed_v_ys, gamma);
                SumCheck::verify(&(), num_vars, expression.degree(), claim, transcript)?
            };

            let evals = transcript.read_field_elements(2 * num_batching)?;

            let eval_by_query = eval_by_query(&evals);
            if x_eval != evaluate(&expression, num_vars, &eval_by_query, &[gamma], &[&y], &x) {
                return Err(err_unmatched_sum_check_output());
            }

            (x, evals)
        };

        let mu = transcript.squeeze_challenge();

        let v_xs = layer_down_claim(&evals, mu);
        x.push(mu);

        Ok((v_xs, x))
    })?;

    Ok((v_xs, x))
}

fn sum_check_expression<F: PrimeField>(num_batching: usize) -> Expression<F> {
    let exprs = &(0..2 * num_batching)
        .map(|idx| Expression::<F>::Polynomial(Query::new(idx, Rotation::cur())))
        .tuples()
        .map(|(ref v_l, ref v_r)| v_l * v_r)
        .collect_vec();
    let eq_xy = &Expression::eq_xy(0);
    let gamma = &Expression::Challenge(0);
    Expression::distribute_powers(exprs, gamma) * eq_xy
}

fn sum_check_claim<F: PrimeField>(claimed_v_ys: &[F], gamma: F) -> F {
    inner_product(
        claimed_v_ys,
        &powers(gamma).take(claimed_v_ys.len()).collect_vec(),
    )
}

fn layer_down_claim<F: PrimeField>(evals: &[F], mu: F) -> Vec<F> {
    evals
        .iter()
        .tuples()
        .map(|(&v_l, &v_r)| v_l + mu * (v_r - v_l))
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use std::iter;

    use itertools::{chain, Itertools};

    use crate::{
        piop::gkr::{prove_grand_product, verify_grand_product},
        poly::multilinear::MultilinearPolynomial,
        util::{
            izip_eq,
            test::{rand_vec, seeded_std_rng},
            transcript::{InMemoryTranscript, Keccak256Transcript},
        },
    };
    use halo2_curves::bn256::Fr;

    #[test]
    fn grand_product_test() {
        let num_batching = 4;
        for num_vars in 1..16 {
            let mut rng = seeded_std_rng();

            let vs = iter::repeat_with(|| rand_vec(1 << num_vars, &mut rng))
                .map(MultilinearPolynomial::new)
                .take(num_batching)
                .collect_vec();
            let v_0s = vec![None; num_batching];

            let proof = {
                let mut transcript = Keccak256Transcript::new(());
                prove_grand_product::<Fr>(v_0s.to_vec(), vs.iter(), &mut transcript).unwrap();
                transcript.into_proof()
            };

            let result = {
                let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
                verify_grand_product::<Fr>(num_vars, v_0s.to_vec(), &mut transcript)
            };
            assert_eq!(result.as_ref().map(|_| ()), Ok(()));

            let (v_xs, x) = result.unwrap();
            for (poly, eval) in izip_eq!(chain![vs], chain![v_xs]) {
                assert_eq!(poly.evaluate(&x), eval);
            }
        }
    }
}
