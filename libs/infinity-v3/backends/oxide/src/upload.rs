// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Bounded load-time staging into final device weight layouts.
use crate::{
    model::WeightSource,
    start,
    tensor::{Bf16, Element, F16},
};
use cuda_core::{DeviceBuffer, DeviceCopy, Event, PinnedHostBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
}
impl sealed::Sealed for Bf16 {}
impl sealed::Sealed for F16 {}

/// Cold-load destination for a bounded 16-bit weight upload. Implementations
/// are closed to BF16 and FP16, so no inference path branches on a dtype.
pub(crate) trait UploadElement:
    sealed::Sealed + Element<Storage = Self> + DeviceCopy + 'static
{
    fn read(
        source: &WeightSource,
        offset_elements: usize,
        out: &mut [u16],
    ) -> Result<(), BackendError>;
}

impl UploadElement for Bf16 {
    fn read(
        source: &WeightSource,
        offset_elements: usize,
        out: &mut [u16],
    ) -> Result<(), BackendError> {
        source.read_bf16(offset_elements, out)
    }
}

impl UploadElement for F16 {
    fn read(
        source: &WeightSource,
        offset_elements: usize,
        out: &mut [u16],
    ) -> Result<(), BackendError> {
        source.read_f16(offset_elements, out)
    }
}

// Two slots, one MiB pinned in total, independent of shard/model size.
const ELEMENTS: usize = 1 << 18;
struct Slot {
    host: PinnedHostBuffer<u16>,
    complete: Event,
    pending: bool,
}
pub(crate) struct WeightUploader {
    stream: Arc<Stream>,
    slots: Option<[Slot; 2]>,
    next: usize,
}
impl WeightUploader {
    pub(crate) fn new(stream: Arc<Stream>) -> Result<Self, BackendError> {
        let slot = || -> Result<Slot, BackendError> {
            Ok(Slot {
                host: PinnedHostBuffer::zeroed(stream.context(), ELEMENTS).map_err(start)?,
                complete: stream.context().new_event().map_err(start)?,
                pending: false,
            })
        };
        Ok(Self {
            slots: Some([slot()?, slot()?]),
            stream,
            next: 0,
        })
    }

    pub(crate) fn upload(
        &mut self,
        sources: &[&WeightSource],
    ) -> Result<DeviceBuffer<Bf16>, BackendError> {
        self.upload_typed::<Bf16>(sources)
    }

    pub(crate) fn upload_typed<E: UploadElement>(
        &mut self,
        sources: &[&WeightSource],
    ) -> Result<DeviceBuffer<E>, BackendError> {
        let count = sources.iter().try_fold(0usize, |n, source| {
            source
                .shape()
                .iter()
                .try_fold(1usize, |size, &dim| size.checked_mul(dim))
                .and_then(|size| n.checked_add(size))
                .ok_or_else(|| start("Weight size overflow"))
        })?;
        count
            .checked_mul(2)
            .ok_or_else(|| start("Weight byte size overflow"))?;
        // SAFETY: every byte is populated on this stream before readiness. No
        // inference can access this not-yet-published allocation.
        let buffer =
            unsafe { DeviceBuffer::uninitialized_async(&self.stream, count) }.map_err(start)?;
        if let Err(error) = self.populate::<E>(&buffer, sources) {
            if self.stream.synchronize().is_err() {
                // Preserve allocation/context when completion is uncertain.
                // The failed backend is never published to requests.
                std::mem::forget(buffer);
            }
            return Err(error);
        }
        Ok(buffer)
    }

    fn populate<E: UploadElement>(
        &mut self,
        buffer: &DeviceBuffer<E>,
        sources: &[&WeightSource],
    ) -> Result<(), BackendError> {
        let mut destination = 0usize;
        for source in sources {
            let mut position = 0usize;
            while position < source.len() {
                let length = (source.len() - position).min(ELEMENTS);
                let slot = self
                    .slots
                    .as_mut()
                    .and_then(|slots| slots.get_mut(self.next))
                    .ok_or_else(|| start("Missing weight staging slot"))?;
                if slot.pending {
                    slot.complete.synchronize().map_err(start)?;
                    slot.pending = false;
                }
                let out = slot
                    .host
                    .as_mut_slice()
                    .get_mut(..length)
                    .ok_or_else(|| start("Weight staging range overflow"))?;
                E::read(source, position, out)?;
                let offset = destination
                    .checked_mul(2)
                    .ok_or_else(|| start("Weight offset overflow"))?;
                let ptr = buffer
                    .cu_deviceptr()
                    .checked_add(u64::try_from(offset).map_err(start)?)
                    .ok_or_else(|| start("Weight address overflow"))?;
                // SAFETY: ranges are checked; the pinned slot stays owned and
                // unmodified until its recorded event or final stream drain.
                unsafe {
                    cuda_core::memory::memcpy_htod_async(
                        ptr,
                        slot.host.as_ptr(),
                        length * 2,
                        self.stream.cu_stream(),
                    )
                }
                .map_err(start)?;
                slot.complete.record(&self.stream).map_err(start)?;
                slot.pending = true;
                self.next = (self.next + 1) % 2;
                position += length;
                destination += length;
            }
        }
        Ok(())
    }
    pub(crate) fn finish(&self) -> Result<(), BackendError> {
        self.stream.synchronize().map_err(start)
    }
}
impl Drop for WeightUploader {
    fn drop(&mut self) {
        if self.stream.synchronize().is_err() {
            if let Some(slots) = self.slots.take() {
                std::mem::forget(slots);
            }
        }
    }
}
