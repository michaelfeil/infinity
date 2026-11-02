// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Capacity-backed Clef device scratch owned by the caller's cached Work.
//!
//! Nothing here is allocated per request. A cold Work miss creates the fixed
//! capacity once; every later request only rewrites the single metadata tail
//! and reuses these BF16/F32 owners. The outer Work/Engine completion fence is
//! the only owner authority. If completion is uncertain, Drop retains every
//! device owner rather than freeing storage a failed stream may still read.

use crate::{
    clef_decision::{infer, MAX_CLEF_OPTIONS, MAX_CLEF_OPTION_TOKENS},
    product,
    tensor::Bf16,
};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::{
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    sync::Arc,
};

/// Fixed request-independent geometry for one cached Work arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClefCapacity {
    tokens: usize,
    requests: usize,
    fields: usize,
    options: usize,
    option_tokens: usize,
    hidden: usize,
    width: usize,
    ffn: usize,
    metadata_words: usize,
}

impl ClefCapacity {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        tokens: usize,
        requests: usize,
        fields: usize,
        options: usize,
        option_tokens: usize,
        hidden: usize,
        width: usize,
        ffn: usize,
    ) -> Result<Self, BackendError> {
        if tokens == 0 || requests == 0 || fields == 0 || hidden == 0 || width == 0 || ffn == 0 {
            return Err(infer(
                "Clef capacity requires positive token/request/field/width extents",
            ));
        }
        if fields > crate::clef_decision::MAX_CLEF_FIELDS
            || options > MAX_CLEF_OPTIONS
            || option_tokens > MAX_CLEF_OPTION_TOKENS
        {
            return Err(infer(
                "Clef capacity exceeds the Native field/option bounds",
            ));
        }
        let metadata_words = [
            requests.checked_add(1),
            Some(requests),
            requests.checked_add(1),
            requests.checked_add(1),
            fields.checked_add(1),
            Some(fields),
            Some(fields),
            Some(options),
            fields.checked_mul(2),
            Some(fields),
            options.checked_mul(2),
            Some(options),
            options.checked_add(1),
            Some(options),
            Some(option_tokens),
        ]
        .into_iter()
        .try_fold(0usize, |total, value| {
            let value = value.ok_or_else(|| infer("Clef metadata capacity overflow"))?;
            total
                .checked_add(value)
                .ok_or_else(|| infer("Clef metadata capacity overflow"))
        })?;
        Ok(Self {
            tokens,
            requests,
            fields,
            options,
            option_tokens,
            hidden,
            width,
            ffn,
            metadata_words,
        })
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }
    pub(crate) fn requests(&self) -> usize {
        self.requests
    }
    pub(crate) fn fields(&self) -> usize {
        self.fields
    }
    pub(crate) fn options(&self) -> usize {
        self.options
    }
    pub(crate) fn option_tokens(&self) -> usize {
        self.option_tokens
    }
    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }
    pub(crate) fn width(&self) -> usize {
        self.width
    }
    pub(crate) fn ffn(&self) -> usize {
        self.ffn
    }
    pub(crate) fn metadata_words(&self) -> usize {
        self.metadata_words
    }

    fn rows(&self) -> Result<usize, BackendError> {
        Ok(self.tokens.max(self.options).max(self.fields))
    }

    fn bf16_words(&self) -> Result<usize, BackendError> {
        let rows = self.rows()?;
        let values = [
            product(self.tokens, self.hidden)?,
            product(self.tokens, self.width)?,
            product(self.fields, self.hidden)?,
            product(self.fields, self.width)?,
            product(self.fields, self.width)?,
            product(self.requests, self.hidden)?,
            product(self.fields, self.hidden)?,
            product(self.options, self.hidden)?,
            product(self.option_tokens, self.hidden)?,
            product(self.options, self.hidden)?,
            product(self.options, self.width)?,
            product(self.options, self.width)?,
            product(self.options, self.width)?,
            product(rows, self.width)?,
            product(rows, self.width)?,
            product(rows, self.width)?,
            product(product(rows, self.width)?, 3)?,
            product(rows, self.width)?,
            product(rows, self.width)?,
            product(rows, self.width)?,
            product(rows, self.width)?,
            product(rows, self.width)?,
            product(rows, self.ffn)?,
            product(self.tokens, self.width)?,
            product(self.fields, self.width)?,
            product(self.fields, self.width)?,
            product(self.requests, self.width)?,
            product(self.fields, self.width)?,
            product(self.fields, self.width)?,
            product(self.fields, self.width)?,
            product(self.options, self.width)?,
            product(product(self.options, self.width)?, 4)?,
            self.options,
            self.options,
            self.options,
            self.options,
        ];
        values.into_iter().try_fold(0usize, |total, value| {
            total
                .checked_add(value)
                .ok_or_else(|| infer("Clef BF16 scratch size overflow"))
        })
    }

    pub(crate) fn device_bytes(&self) -> Result<usize, BackendError> {
        let bf16 = self
            .bf16_words()?
            .checked_mul(std::mem::size_of::<Bf16>())
            .ok_or_else(|| infer("Clef BF16 byte count overflow"))?;
        bf16.checked_add(
            self.options
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| infer("Clef logits byte count overflow"))?,
        )
        .and_then(|bytes| {
            self.metadata_words
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|metadata| bytes.checked_add(metadata))
        })
        .ok_or_else(|| infer("Clef workspace byte count overflow"))
    }
}

/// Every reusable scratch owner. Fields are deliberately explicit so the
/// executor cannot silently allocate a request-shaped substitute.
pub(crate) struct ClefScratch {
    pub(crate) hidden_norm: DeviceBuffer<Bf16>,
    pub(crate) memory: DeviceBuffer<Bf16>,
    pub(crate) question_mean: DeviceBuffer<Bf16>,
    pub(crate) question_base: DeviceBuffer<Bf16>,
    pub(crate) question_query: DeviceBuffer<Bf16>,
    pub(crate) global_rows: DeviceBuffer<Bf16>,
    pub(crate) anchors: DeviceBuffer<Bf16>,
    pub(crate) option_context: DeviceBuffer<Bf16>,
    pub(crate) lexical_rows: DeviceBuffer<Bf16>,
    pub(crate) lexical_mean: DeviceBuffer<Bf16>,
    pub(crate) context_projection: DeviceBuffer<Bf16>,
    pub(crate) lexical_projection: DeviceBuffer<Bf16>,
    pub(crate) queries: DeviceBuffer<Bf16>,
    pub(crate) state: DeviceBuffer<Bf16>,
    pub(crate) next: DeviceBuffer<Bf16>,
    pub(crate) norm: DeviceBuffer<Bf16>,
    pub(crate) qkv: DeviceBuffer<Bf16>,
    pub(crate) query: DeviceBuffer<Bf16>,
    pub(crate) key: DeviceBuffer<Bf16>,
    pub(crate) value: DeviceBuffer<Bf16>,
    pub(crate) attention: DeviceBuffer<Bf16>,
    pub(crate) branch: DeviceBuffer<Bf16>,
    pub(crate) ffn: DeviceBuffer<Bf16>,
    pub(crate) memory_norm: DeviceBuffer<Bf16>,
    pub(crate) summaries: DeviceBuffer<Bf16>,
    pub(crate) summary_norm: DeviceBuffer<Bf16>,
    pub(crate) global_projection: DeviceBuffer<Bf16>,
    pub(crate) field_state: DeviceBuffer<Bf16>,
    pub(crate) field_next: DeviceBuffer<Bf16>,
    pub(crate) field_norm: DeviceBuffer<Bf16>,
    pub(crate) option_norm: DeviceBuffer<Bf16>,
    pub(crate) residual_features: DeviceBuffer<Bf16>,
    pub(crate) scores: DeviceBuffer<Bf16>,
    pub(crate) prior: DeviceBuffer<Bf16>,
    pub(crate) cosine: DeviceBuffer<Bf16>,
    pub(crate) residual: DeviceBuffer<Bf16>,
    pub(crate) logits_f32: DeviceBuffer<f32>,
    pub(crate) metadata: DeviceBuffer<u32>,
    pub(crate) capacity: ClefCapacity,
}

pub(crate) struct ClefWorkspace {
    inner: ManuallyDrop<ClefScratch>,
    drained: bool,
    stream: Arc<Stream>,
}

impl ClefWorkspace {
    pub(crate) fn new(stream: &Arc<Stream>, capacity: ClefCapacity) -> Result<Self, BackendError> {
        let rows = capacity.rows()?;
        let alloc = |elements: usize| -> Result<DeviceBuffer<Bf16>, BackendError> {
            // SAFETY: every logical element is written before any read.
            unsafe { DeviceBuffer::uninitialized_async(stream, elements) }.map_err(infer)
        };
        let alloc_f32 = |elements: usize| -> Result<DeviceBuffer<f32>, BackendError> {
            // SAFETY: every logical element is written before readback.
            unsafe { DeviceBuffer::uninitialized_async(stream, elements) }.map_err(infer)
        };
        let scratch = ClefScratch {
            hidden_norm: alloc(product(capacity.tokens, capacity.hidden)?)?,
            memory: alloc(product(capacity.tokens, capacity.width)?)?,
            question_mean: alloc(product(capacity.fields, capacity.hidden)?)?,
            question_base: alloc(product(capacity.fields, capacity.width)?)?,
            question_query: alloc(product(capacity.fields, capacity.width)?)?,
            global_rows: alloc(product(capacity.requests, capacity.hidden)?)?,
            anchors: alloc(product(capacity.fields, capacity.hidden)?)?,
            option_context: alloc(product(capacity.options, capacity.hidden)?)?,
            lexical_rows: alloc(product(capacity.option_tokens, capacity.hidden)?)?,
            lexical_mean: alloc(product(capacity.options, capacity.hidden)?)?,
            context_projection: alloc(product(capacity.options, capacity.width)?)?,
            lexical_projection: alloc(product(capacity.options, capacity.width)?)?,
            queries: alloc(product(capacity.options, capacity.width)?)?,
            state: alloc(product(rows, capacity.width)?)?,
            next: alloc(product(rows, capacity.width)?)?,
            norm: alloc(product(rows, capacity.width)?)?,
            qkv: alloc(product(product(rows, capacity.width)?, 3)?)?,
            query: alloc(product(rows, capacity.width)?)?,
            key: alloc(product(rows, capacity.width)?)?,
            value: alloc(product(rows, capacity.width)?)?,
            attention: alloc(product(rows, capacity.width)?)?,
            branch: alloc(product(rows, capacity.width)?)?,
            ffn: alloc(product(rows, capacity.ffn)?)?,
            memory_norm: alloc(product(capacity.tokens, capacity.width)?)?,
            summaries: alloc(product(capacity.fields, capacity.width)?)?,
            summary_norm: alloc(product(capacity.fields, capacity.width)?)?,
            global_projection: alloc(product(capacity.requests, capacity.width)?)?,
            field_state: alloc(product(capacity.fields, capacity.width)?)?,
            field_next: alloc(product(capacity.fields, capacity.width)?)?,
            field_norm: alloc(product(capacity.fields, capacity.width)?)?,
            option_norm: alloc(product(capacity.options, capacity.width)?)?,
            residual_features: alloc(product(product(capacity.options, capacity.width)?, 4)?)?,
            scores: alloc(capacity.options)?,
            prior: alloc(capacity.options)?,
            cosine: alloc(capacity.options)?,
            residual: alloc(capacity.options)?,
            logits_f32: alloc_f32(capacity.options)?,
            metadata: DeviceBuffer::zeroed(stream, capacity.metadata_words).map_err(infer)?,
            capacity,
        };
        Ok(Self {
            inner: ManuallyDrop::new(scratch),
            drained: false,
            stream: stream.clone(),
        })
    }

    pub(crate) fn mark_drained(&mut self) {
        self.drained = true;
    }
}

impl Deref for ClefWorkspace {
    type Target = ClefScratch;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl DerefMut for ClefWorkspace {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Drop for ClefWorkspace {
    fn drop(&mut self) {
        if !self.drained && self.stream.synchronize().is_err() {
            // Completion is uncertain: retain every device owner.
            let scratch = unsafe { ManuallyDrop::take(&mut self.inner) };
            std::mem::forget(scratch);
            return;
        }
        // SAFETY: the stream drained just now or the outer Work already proved
        // completion before dropping the same fence.
        unsafe { ManuallyDrop::drop(&mut self.inner) };
    }
}
