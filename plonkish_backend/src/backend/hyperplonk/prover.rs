use crate::{
    backend::{
        hyperplonk::{
            verifier::{pcs_query, point_offset, points},
            HyperPlonk,
        },
        lookup::lasso::prover::LassoProver,
        WitnessEncoding,
    },
    pcs::{Evaluation, PolynomialCommitmentScheme},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, steps_by, BatchInvert, BooleanHypercube, PrimeField},
        end_timer,
        expression::{Expression, Rotation},
        parallel::{par_map_collect, parallelize},
        start_timer,
        transcript::{FieldTranscriptWrite, TranscriptWrite},
        Itertools,
    },
    Error,
};
use std::iter;

use super::HyperPlonkProverParam;

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

pub(super) fn prove_lookup<
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
>(
    pp: &HyperPlonkProverParam<F, Pcs>,
    polys: &[&MultilinearPolynomial<F>],
    transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
) -> Result<
    (
        Vec<MultilinearPolynomial<F>>,
        Vec<Pcs::Commitment>,
        Vec<Vec<F>>,
        Vec<Evaluation<F>>,
        Vec<F>,
    ),
    Error,
> {
    if pp.lasso_lookup.is_none() {
        return Ok((vec![], vec![], vec![], vec![], vec![]));
    }
    let lasso_lookup = pp.lasso_lookup.as_ref().unwrap();
    let (lookup, table) = ((&lasso_lookup.0, &lasso_lookup.1), &lasso_lookup.2);
    let (lookup_input_poly, lookup_nz_poly) = LassoProver::<F, Pcs>::lookup_poly(&lookup, &polys);

    let num_vars = lookup_input_poly.num_vars();

    // get subtable_polys
    let subtable_polys = table.subtable_polys();
    let subtable_polys = subtable_polys.iter().collect_vec();
    let subtable_polys = subtable_polys.as_slice();

    let (lookup_polys, lookup_comms) = LassoProver::<F, Pcs>::commit(
        &pp.pcs,
        pp.lookup_polys_offset,
        &table,
        subtable_polys,
        lookup_input_poly,
        &lookup_nz_poly,
        transcript,
    )?;

    // Round n
    // squeeze `r`
    let r = transcript.squeeze_challenges(num_vars);

    let (input_poly, dims, read_ts_polys, final_cts_polys, e_polys) = (
        &lookup_polys[0][0],
        &lookup_polys[1],
        &lookup_polys[2],
        &lookup_polys[3],
        &lookup_polys[4],
    );
    // Lasso Sumcheck
    let (lookup_points, lookup_evals) = LassoProver::<F, Pcs>::prove_sum_check(
        pp.lookup_points_offset,
        &table,
        input_poly,
        &e_polys.iter().collect_vec(),
        &r,
        num_vars,
        transcript,
    )?;

    // squeeze memory checking challenges -> we will reuse beta, gamma for memory checking of Lasso
    // Round n+1
    let [beta, gamma] = transcript.squeeze_challenges(2).try_into().unwrap();

    // memory_checking
    let (mem_check_opening_points, mem_check_opening_evals) =
        LassoProver::<F, Pcs>::memory_checking(
            pp.lookup_points_offset,
            table,
            subtable_polys,
            dims,
            read_ts_polys,
            final_cts_polys,
            e_polys,
            &beta,
            &gamma,
            transcript,
        )?;

    let lookup_polys = lookup_polys
        .into_iter()
        .flat_map(|lookup_polys| lookup_polys.into_iter().map(|poly| poly.poly).collect_vec())
        .collect_vec();
    let lookup_comms = lookup_comms.concat();
    let lookup_opening_points = iter::empty()
        .chain(lookup_points)
        .chain(mem_check_opening_points)
        .collect_vec();
    let lookup_evals = iter::empty()
        .chain(lookup_evals)
        .chain(mem_check_opening_evals)
        .collect_vec();
    Ok((
        lookup_polys,
        lookup_comms,
        lookup_opening_points,
        lookup_evals,
        vec![beta, gamma],
    ))
}
