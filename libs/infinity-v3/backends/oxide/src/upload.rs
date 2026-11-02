// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Bounded load-time staging into final device weight layouts.
use crate::{
    gemma2_media_bridge::device_matches,
    model::WeightSource,
    start,
    tensor::{Bf16, Element, ScalarFormat, F16},
};
use cuda_core::{DeviceBuffer, DeviceCopy, Event, PinnedHostBuffer, Stream};
use infinity_v3_backend_core::{
    media::{ColdChunk, ColdUploadPlan, MediaDtype},
    BackendError,
};
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

    pub(crate) fn stream(&self) -> &Arc<Stream> {
        &self.stream
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

    /// Stream a pure media cold-upload plan through the existing two pinned
    /// slots. `fill` runs on the CPU and returns before the H2D copy; the slot
    /// is not reused until its recorded event completes. No whole-CPU copy is
    /// allocated here. The caller must call `finish()` (or drop the uploader)
    /// before publishing the returned buffer on another stream.
    #[allow(dead_code)] // Bound by the family executor once real media weights load.
    pub(crate) fn upload_plan<E: UploadElement>(
        &mut self,
        plan: &ColdUploadPlan,
        mut fill: impl FnMut(&ColdChunk, &mut [u8]) -> Result<(), BackendError>,
    ) -> Result<DeviceBuffer<E>, BackendError> {
        let expected = match E::FORMAT {
            ScalarFormat::Bf16 => MediaDtype::Bf16,
            ScalarFormat::F16 => MediaDtype::F16,
            other => return Err(start(format!("Unsupported media upload dtype {other:?}"))),
        };
        if plan.tensors().is_empty() {
            return Err(start("Empty media cold upload plan"));
        }
        let ordinal = self.stream.context().ordinal();
        for tensor in plan.tensors() {
            if tensor.dtype() != expected {
                return Err(start("Media upload plan dtype does not match destination"));
            }
            if !device_matches(tensor.device(), ordinal) {
                return Err(start("Media upload plan targets a different CUDA ordinal"));
            }
        }
        let bytes = plan.total_bytes();
        if !bytes.is_multiple_of(2) {
            return Err(start("Media upload plan byte size is not 16-bit aligned"));
        }
        let count = bytes / 2;
        // SAFETY: every non-padding byte is populated on this stream before the
        // caller publishes the returned buffer.
        let buffer =
            unsafe { DeviceBuffer::uninitialized_async(&self.stream, count) }.map_err(start)?;
        let result = {
            let Self {
                stream,
                slots,
                next,
            } = self;
            let slots = slots
                .as_mut()
                .ok_or_else(|| start("Missing media staging slots"))?;
            let (left, right) = slots.split_at_mut(1);
            let mut adapters = [
                RealSlot {
                    slot: &mut left[0],
                    stream,
                    buffer_ptr: buffer.cu_deviceptr(),
                },
                RealSlot {
                    slot: &mut right[0],
                    stream,
                    buffer_ptr: buffer.cu_deviceptr(),
                },
            ];
            fill_plan(&mut adapters, next, plan, ELEMENTS * 2, &mut fill)
        };
        if let Err(error) = result {
            if self.stream.synchronize().is_err() {
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
trait StagingSlot {
    fn wait(&mut self) -> Result<(), BackendError>;
    fn host_mut(&mut self, bytes: usize) -> Result<&mut [u8], BackendError>;
    fn enqueue(&mut self, destination_offset: usize, bytes: usize) -> Result<(), BackendError>;
    fn record(&mut self) -> Result<(), BackendError>;
}

#[allow(dead_code)] // Used by upload_plan's real staging adapter.
struct RealSlot<'a> {
    slot: &'a mut Slot,
    stream: &'a Stream,
    buffer_ptr: u64,
}

impl StagingSlot for RealSlot<'_> {
    fn wait(&mut self) -> Result<(), BackendError> {
        if self.slot.pending {
            self.slot.complete.synchronize().map_err(start)?;
            self.slot.pending = false;
        }
        Ok(())
    }

    fn host_mut(&mut self, bytes: usize) -> Result<&mut [u8], BackendError> {
        if self.slot.pending {
            return Err(start("Media staging slot is still pending"));
        }
        let host = self.slot.host.as_mut_slice();
        let host_bytes = host
            .len()
            .checked_mul(2)
            .ok_or_else(|| start("Media staging range overflow"))?;
        if bytes > host_bytes {
            return Err(start("Media staging range overflow"));
        }
        // SAFETY: the pinned buffer is initialized and exclusively borrowed; the
        // byte view is bounded by its declared capacity.
        Ok(unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) })
    }

    fn enqueue(&mut self, destination_offset: usize, bytes: usize) -> Result<(), BackendError> {
        let ptr = self
            .buffer_ptr
            .checked_add(u64::try_from(destination_offset).map_err(start)?)
            .ok_or_else(|| start("Media upload address overflow"))?;
        // SAFETY: ranges were checked by ColdUploadPlan and host_mut; the slot is
        // pinned and remains unmodified until its event is recorded and waited.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                ptr,
                self.slot.host.as_ptr(),
                bytes,
                self.stream.cu_stream(),
            )
        }
        .map_err(start)
    }

    fn record(&mut self) -> Result<(), BackendError> {
        self.slot.complete.record(self.stream).map_err(start)?;
        self.slot.pending = true;
        Ok(())
    }
}

fn fill_plan<S: StagingSlot>(
    slots: &mut [S; 2],
    next: &mut usize,
    plan: &ColdUploadPlan,
    chunk_bytes: usize,
    fill: &mut impl FnMut(&ColdChunk, &mut [u8]) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    for chunk in plan.chunks(chunk_bytes).map_err(start)? {
        let chunk = chunk.map_err(start)?;
        let slot = slots
            .get_mut(*next)
            .ok_or_else(|| start("Missing media staging slot"))?;
        slot.wait()?;
        let host = slot.host_mut(chunk.bytes)?;
        fill(&chunk, host)?;
        slot.enqueue(chunk.destination_offset, chunk.bytes)?;
        slot.record()?;
        *next = (*next + 1) % 2;
    }
    Ok(())
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

#[cfg(test)]
mod media_plan_tests {
    use super::*;
    use infinity_v3_backend_core::media::MediaDevice;

    struct FakeSlot {
        host: Vec<u8>,
        pending: bool,
        waits: usize,
        writes: Vec<Vec<u8>>,
    }

    impl FakeSlot {
        fn new() -> Self {
            Self {
                host: vec![0; 8],
                pending: false,
                waits: 0,
                writes: Vec::new(),
            }
        }
    }

    impl StagingSlot for FakeSlot {
        fn wait(&mut self) -> Result<(), BackendError> {
            self.waits += 1;
            self.pending = false;
            Ok(())
        }

        fn host_mut(&mut self, bytes: usize) -> Result<&mut [u8], BackendError> {
            assert!(!self.pending, "pending staging slot was overwritten");
            Ok(&mut self.host[..bytes])
        }

        fn enqueue(
            &mut self,
            _destination_offset: usize,
            bytes: usize,
        ) -> Result<(), BackendError> {
            assert!(!self.pending);
            self.writes.push(self.host[..bytes].to_vec());
            self.pending = true;
            Ok(())
        }

        fn record(&mut self) -> Result<(), BackendError> {
            Ok(())
        }
    }

    #[test]
    fn plan_device_ordinal_is_checked() {
        assert!(device_matches(MediaDevice::Cuda { index: 1 }, 1));
        assert!(!device_matches(MediaDevice::Cuda { index: 1 }, 0));
        assert!(!device_matches(MediaDevice::Cpu, 0));
    }

    #[test]
    fn pending_slot_is_waited_before_reuse() {
        let mut plan = ColdUploadPlan::new(4).unwrap();
        plan.push(
            "first",
            4,
            1,
            MediaDtype::F32,
            MediaDevice::Cuda { index: 0 },
        )
        .unwrap();
        plan.push(
            "second",
            1,
            1,
            MediaDtype::F32,
            MediaDevice::Cuda { index: 0 },
        )
        .unwrap();
        let mut slots = [FakeSlot::new(), FakeSlot::new()];
        let mut next = 0;
        fill_plan(&mut slots, &mut next, &plan, 4, &mut |chunk, out| {
            out.fill(chunk.tensor_index as u8);
            Ok(())
        })
        .unwrap();
        assert!(slots.iter().any(|slot| slot.waits >= 2));
        assert_eq!(slots[0].writes.len() + slots[1].writes.len(), 5);
        assert_eq!(next, 1);
        for slot in &mut slots {
            slot.wait().unwrap();
        }
        assert!(slots.iter().all(|slot| !slot.pending));
    }
}
