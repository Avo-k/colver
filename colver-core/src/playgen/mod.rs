//! Playgen: autoregressive game-continuation model ("world sampler" for IS-DD).
//!
//! A causal transformer over tokenized auctions + plays, trained by teacher
//! forcing on self-play games. Sampling a continuation to the end of the deal
//! reveals a full hidden-hand assignment, i.e. a determinized world drawn from
//! the (approximate) posterior p(hands | observed public history).

pub mod tokens;

#[cfg(feature = "rand")]
pub mod infer;

#[cfg(feature = "dmc_train")]
pub mod model;

#[cfg(feature = "dmc_train")]
pub mod gpu;
