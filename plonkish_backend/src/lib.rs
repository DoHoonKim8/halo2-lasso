#![allow(clippy::op_ref)]
#![feature(generic_const_exprs)]
#![feature(map_first_last)]
#![feature(int_log)]

pub mod backend;
pub mod frontend;
pub mod pcs;
pub mod piop;
pub mod poly;
pub mod util;

pub use halo2_curves;

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    InvalidSumcheck(String),
    InvalidPcsParam(String),
    InvalidPcsOpen(String),
    InvalidSnark(String),
    Serialization(String),
    Transcript(std::io::ErrorKind, String),
}
