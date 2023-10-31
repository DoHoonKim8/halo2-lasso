use crate::{
    backend::{
        hyperplonk::{
            verifier::{pcs_query, point_offset, points},
            HyperPlonk,
        },
        WitnessEncoding,
    },
    pcs::Evaluation,
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, steps_by, sum, BatchInvert, BooleanHypercube, PrimeField},
        end_timer,
        expression::{CommonPolynomial, Expression, Rotation},
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        start_timer,
        transcript::FieldTranscriptWrite,
        Itertools,
    },
    Error,
};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    iter,
};

pub(crate) fn instance_polys<'a, F: PrimeField>(
    num_vars: usize,
    instances: impl IntoIterator<Item = impl IntoIterator<Item = &'a F>>,
) -> Vec<MultilinearPolynomial<F>> {
    let row_mapping = HyperPlonk::<()>::row_mapping(num_vars);
    instances
        .into_iter()
        .map(|instances| {
            let mut poly = vec![F::ZERO; 1 << num_vars];
            for (b, instance) in row_mapping.iter().zip(instances.into_iter()) {
                poly[*b] = *instance;
            }
            poly
        })
        .map(MultilinearPolynomial::new)
        .collect()
}

pub(crate) fn permutation_z_polys<F: PrimeField>(
    num_chunks: usize,
    permutation_polys: &[(usize, MultilinearPolynomial<F>)],
    polys: &[&MultilinearPolynomial<F>],
    beta: &F,
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    if permutation_polys.is_empty() {
        return Vec::new();
    }

    let chunk_size = div_ceil(permutation_polys.len(), num_chunks);
    let num_vars = polys[0].num_vars();

    let timer = start_timer(|| "products");
    let products = permutation_polys
        .chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, permutation_polys)| {
            let mut product = vec![F::ONE; 1 << num_vars];

            for (poly, permutation_poly) in permutation_polys.iter() {
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), permutation) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(permutation_poly[start..].iter())
                    {
                        *product *= (*beta * permutation) + gamma + value;
                    }
                });
            }

            parallelize(&mut product, |(product, _)| {
                product.iter_mut().batch_invert();
            });

            for ((poly, _), idx) in permutation_polys.iter().zip(chunk_idx * chunk_size..) {
                let id_offset = idx << num_vars;
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), beta_id) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(steps_by(F::from((id_offset + start) as u64) * beta, *beta))
                    {
                        *product *= beta_id + gamma + value;
                    }
                });
            }

            product
        })
        .collect_vec();
    end_timer(timer);

    let timer = start_timer(|| "z_polys");
    let z = iter::empty()
        .chain(iter::repeat(F::ZERO).take(num_chunks))
        .chain(Some(F::ONE))
        .chain(
            BooleanHypercube::new(num_vars)
                .iter()
                .skip(1)
                .flat_map(|b| iter::repeat(b).take(num_chunks))
                .zip(products.iter().cycle())
                .scan(F::ONE, |state, (b, product)| {
                    *state *= &product[b];
                    Some(*state)
                }),
        )
        .take(num_chunks << num_vars)
        .collect_vec();

    if cfg!(feature = "sanity-check") {
        let b_last = BooleanHypercube::new(num_vars).iter().last().unwrap();
        assert_eq!(
            *z.last().unwrap() * products.last().unwrap()[b_last],
            F::ONE
        );
    }

    drop(products);
    end_timer(timer);

    let _timer = start_timer(|| "into_bh_order");
    let nth_map = BooleanHypercube::new(num_vars)
        .nth_map()
        .into_iter()
        .map(|b| num_chunks * b)
        .collect_vec();
    (0..num_chunks)
        .map(|offset| MultilinearPolynomial::new(par_map_collect(&nth_map, |b| z[offset + b])))
        .collect()
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_zero_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    prove_sum_check(
        num_instance_poly,
        expression,
        F::ZERO,
        polys,
        challenges,
        y,
        transcript,
    )
}

#[allow(clippy::type_complexity)]
pub(crate) fn prove_sum_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    sum: F,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    let num_vars = polys[0].num_vars();
    let ys = [y];
    let virtual_poly = VirtualPolynomial::new(expression, polys.to_vec(), &challenges, &ys);
    let (x, evals) = ClassicSumCheck::<EvaluationsProver<_>>::prove(
        &(),
        num_vars,
        virtual_poly,
        sum,
        transcript,
    )?;

    let pcs_query = pcs_query(expression, num_instance_poly);
    let point_offset = point_offset(&pcs_query);

    let timer = start_timer(|| format!("evals-{}", pcs_query.len()));
    let evals = pcs_query
        .iter()
        .flat_map(|query| {
            (point_offset[&query.rotation()]..)
                .zip(if query.rotation() == Rotation::cur() {
                    vec![evals[query.poly()]]
                } else {
                    polys[query.poly()].evaluate_for_rotation(&x, query.rotation())
                })
                .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
        })
        .collect_vec();
    end_timer(timer);

    transcript.write_field_elements(evals.iter().map(Evaluation::value))?;

    Ok((points(&pcs_query, &x), evals))
}
