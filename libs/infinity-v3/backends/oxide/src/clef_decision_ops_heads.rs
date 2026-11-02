// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Clef routing/field/prior/feature launches over cached Work buffers.

#![allow(clippy::too_many_arguments)]

use crate::{arg, blocks, clef_decision_head::ClefKernels, launch, product, u32_size};
use cuda_core::Stream;
use infinity_v3_backend_core::BackendError;

pub(crate) fn query_compose(
    kernels: &ClefKernels,
    stream: &Stream,
    question: u64,
    context: u64,
    lexical: u64,
    field_of_option: u64,
    out: u64,
    options: usize,
    width: usize,
) -> Result<(), BackendError> {
    let mut question = question;
    let mut context = context;
    let mut lexical = lexical;
    let mut field_of_option = field_of_option;
    let mut out = out;
    let mut options = u32_size(options)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.query_compose,
        stream,
        blocks(product(options as usize, width as usize)?)?,
        0,
        &mut [
            arg(&mut question),
            arg(&mut context),
            arg(&mut lexical),
            arg(&mut field_of_option),
            arg(&mut out),
            arg(&mut options),
            arg(&mut width),
        ],
    )
}

pub(crate) fn route_scores(
    kernels: &ClefKernels,
    stream: &Stream,
    options: u64,
    base: u64,
    field_of_option: u64,
    out: u64,
    rows: usize,
    width: usize,
    fields: usize,
    scale: f32,
) -> Result<(), BackendError> {
    let mut options = options;
    let mut base = base;
    let mut field_of_option = field_of_option;
    let mut out = out;
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    let mut fields = u32_size(fields)?;
    let mut scale = scale;
    launch(
        &kernels.route_scores,
        stream,
        (rows, 1, 1),
        0,
        &mut [
            arg(&mut options),
            arg(&mut base),
            arg(&mut field_of_option),
            arg(&mut out),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut fields),
            arg(&mut scale),
        ],
    )
}

pub(crate) fn softmax_sum(
    kernels: &ClefKernels,
    stream: &Stream,
    option: u64,
    scores: u64,
    field_option_offsets: u64,
    out: u64,
    options: usize,
    width: usize,
    fields: usize,
) -> Result<(), BackendError> {
    let mut option = option;
    let mut scores = scores;
    let mut field_option_offsets = field_option_offsets;
    let mut out = out;
    let mut options = u32_size(options)?;
    let mut width = u32_size(width)?;
    let mut fields = u32_size(fields)?;
    launch(
        &kernels.softmax_sum,
        stream,
        (fields, 1, 1),
        0,
        &mut [
            arg(&mut option),
            arg(&mut scores),
            arg(&mut field_option_offsets),
            arg(&mut out),
            arg(&mut options),
            arg(&mut width),
            arg(&mut fields),
        ],
    )
}

pub(crate) fn field_compose(
    kernels: &ClefKernels,
    stream: &Stream,
    base: u64,
    summary: u64,
    global: u64,
    types: u64,
    field_request: u64,
    field_kind: u64,
    out: u64,
    fields: usize,
    width: usize,
) -> Result<(), BackendError> {
    let mut base = base;
    let mut summary = summary;
    let mut global = global;
    let mut types = types;
    let mut field_request = field_request;
    let mut field_kind = field_kind;
    let mut out = out;
    let mut fields = u32_size(fields)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.field_compose,
        stream,
        blocks(product(fields as usize, width as usize)?)?,
        0,
        &mut [
            arg(&mut base),
            arg(&mut summary),
            arg(&mut global),
            arg(&mut types),
            arg(&mut field_request),
            arg(&mut field_kind),
            arg(&mut out),
            arg(&mut fields),
            arg(&mut width),
        ],
    )
}

pub(crate) fn anchor(
    kernels: &ClefKernels,
    stream: &Stream,
    question: u64,
    global: u64,
    field_request: u64,
    out: u64,
    fields: usize,
    hidden: usize,
) -> Result<(), BackendError> {
    let mut question = question;
    let mut global = global;
    let mut field_request = field_request;
    let mut out = out;
    let mut fields = u32_size(fields)?;
    let mut hidden = u32_size(hidden)?;
    launch(
        &kernels.anchor,
        stream,
        (fields, 1, 1),
        0,
        &mut [
            arg(&mut question),
            arg(&mut global),
            arg(&mut field_request),
            arg(&mut out),
            arg(&mut fields),
            arg(&mut hidden),
        ],
    )
}

pub(crate) fn prior(
    kernels: &ClefKernels,
    stream: &Stream,
    lexical: u64,
    anchors: u64,
    field_of_option: u64,
    out: u64,
    options: usize,
    hidden: usize,
    scale: f32,
) -> Result<(), BackendError> {
    let mut lexical = lexical;
    let mut anchors = anchors;
    let mut field_of_option = field_of_option;
    let mut out = out;
    let mut options = u32_size(options)?;
    let mut hidden = u32_size(hidden)?;
    let mut scale = scale;
    launch(
        &kernels.prior,
        stream,
        (options, 1, 1),
        0,
        &mut [
            arg(&mut lexical),
            arg(&mut anchors),
            arg(&mut field_of_option),
            arg(&mut out),
            arg(&mut options),
            arg(&mut hidden),
            arg(&mut scale),
        ],
    )
}

pub(crate) fn features(
    kernels: &ClefKernels,
    stream: &Stream,
    field: u64,
    option: u64,
    field_of_option: u64,
    out: u64,
    cosine: u64,
    options: usize,
    width: usize,
) -> Result<(), BackendError> {
    let mut field = field;
    let mut option = option;
    let mut field_of_option = field_of_option;
    let mut out = out;
    let mut cosine = cosine;
    let mut options = u32_size(options)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.features,
        stream,
        (options, 1, 1),
        0,
        &mut [
            arg(&mut field),
            arg(&mut option),
            arg(&mut field_of_option),
            arg(&mut out),
            arg(&mut cosine),
            arg(&mut options),
            arg(&mut width),
        ],
    )
}

pub(crate) fn logits(
    kernels: &ClefKernels,
    stream: &Stream,
    prior: u64,
    cosine: u64,
    residual: u64,
    out: u64,
    options: usize,
    joint_scale: f32,
    gate: f32,
) -> Result<(), BackendError> {
    let mut prior = prior;
    let mut cosine = cosine;
    let mut residual = residual;
    let mut out = out;
    let mut options = u32_size(options)?;
    let mut joint_scale = joint_scale;
    let mut gate = gate;
    launch(
        &kernels.logits,
        stream,
        blocks(options as usize)?,
        0,
        &mut [
            arg(&mut prior),
            arg(&mut cosine),
            arg(&mut residual),
            arg(&mut out),
            arg(&mut options),
            arg(&mut joint_scale),
            arg(&mut gate),
        ],
    )
}
