mod fractional_sum_check;
mod grand_product;

use std::collections::HashMap;

pub use fractional_sum_check::{prove_fractional_sum_check, verify_fractional_sum_check};
pub use grand_product::{prove_grand_product, verify_grand_product};
use halo2_curves::ff::PrimeField;
use itertools::izip;

use crate::{
    util::expression::{Query, Rotation},
    Error,
};

fn eval_by_query<F: PrimeField>(evals: &[F]) -> HashMap<Query, F> {
    izip!(
        (0..).map(|idx| Query::new(idx, Rotation::cur())),
        evals.iter().cloned()
    )
    .collect()
}

fn err_unmatched_sum_check_output() -> Error {
    Error::InvalidSumcheck("Unmatched between sum_check output and query evaluation".to_string())
}
