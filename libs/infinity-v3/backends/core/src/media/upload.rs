//! Pure cold-load sizing and chunk planning.
//!
//! This module owns no host scratch, device buffer, lease, or completion event.
//! An asynchronous uploader must consume [`ColdUploadPlan::chunks`] through its
//! own leased staging slots and wait for each slot's event before reuse. No
//! callback in this module implies completion.

use super::{MediaDevice, MediaDtype, MediaError};

/// Destination tensors are naturally aligned for typed views and vector copies.
const COLD_ALIGNMENT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdChunk {
    pub tensor_index: usize,
    pub source_offset: usize,
    pub destination_offset: usize,
    pub bytes: usize,
}

impl ColdChunk {
    pub fn source_range(&self) -> std::ops::Range<usize> {
        self.source_offset..self.source_offset.saturating_add(self.bytes)
    }

    pub fn destination_range(&self) -> std::ops::Range<usize> {
        self.destination_offset..self.destination_offset.saturating_add(self.bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdTensor {
    name: String,
    rows: usize,
    columns: usize,
    dtype: MediaDtype,
    device: MediaDevice,
    source_bytes: usize,
    destination_offset: usize,
}

impl ColdTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn dtype(&self) -> MediaDtype {
        self.dtype
    }

    pub fn device(&self) -> MediaDevice {
        self.device
    }

    pub fn source_bytes(&self) -> usize {
        self.source_bytes
    }

    pub fn destination_offset(&self) -> usize {
        self.destination_offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdUploadPlan {
    default_chunk_bytes: usize,
    total_bytes: usize,
    padding_bytes: usize,
    tensors: Vec<ColdTensor>,
}

impl ColdUploadPlan {
    pub fn new(default_chunk_bytes: usize) -> Result<Self, MediaError> {
        if default_chunk_bytes == 0 {
            return Err(MediaError::ColdChunkSize {
                requested: default_chunk_bytes,
            });
        }
        Ok(Self {
            default_chunk_bytes,
            total_bytes: 0,
            padding_bytes: 0,
            tensors: Vec::new(),
        })
    }

    /// Append one source tensor to the aligned destination stream. Source bytes
    /// are unchanged; dtype conversion remains in the existing weight uploader.
    pub fn push(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        dtype: MediaDtype,
        device: MediaDevice,
    ) -> Result<usize, MediaError> {
        if name.is_empty() {
            return Err(MediaError::Empty("cold upload tensor name"));
        }
        if rows == 0 || columns == 0 {
            return Err(MediaError::ZeroDimension("cold upload tensor"));
        }
        let elements = rows
            .checked_mul(columns)
            .ok_or(MediaError::DimensionOverflow("cold upload tensor"))?;
        let source_bytes = dtype.checked_bytes(elements)?;
        let alignment = COLD_ALIGNMENT.max(dtype.storage_bytes());
        let destination_offset = align_up(self.total_bytes, alignment)?;
        let padding = destination_offset - self.total_bytes;
        let total_bytes = destination_offset
            .checked_add(source_bytes)
            .ok_or(MediaError::ByteOverflow("cold upload destination"))?;
        self.tensors
            .try_reserve(1)
            .map_err(|_| MediaError::Overflow("cold upload tensor allocation"))?;
        self.tensors.push(ColdTensor {
            name: name.to_owned(),
            rows,
            columns,
            dtype,
            device,
            source_bytes,
            destination_offset,
        });
        self.total_bytes = total_bytes;
        self.padding_bytes = self
            .padding_bytes
            .checked_add(padding)
            .ok_or(MediaError::ByteOverflow("cold upload padding"))?;
        Ok(self.tensors.len() - 1)
    }

    pub fn default_chunk_bytes(&self) -> usize {
        self.default_chunk_bytes
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn padding_bytes(&self) -> usize {
        self.padding_bytes
    }

    pub fn tensors(&self) -> &[ColdTensor] {
        &self.tensors
    }

    /// Allocate-free chunk iteration. Each chunk belongs to exactly one tensor
    /// and carries checked source/destination ranges. The caller remains
    /// responsible for staging leases and completion before reusing host bytes.
    pub fn chunks(&self, chunk_bytes: usize) -> Result<ColdUploadChunks<'_>, MediaError> {
        if chunk_bytes == 0 {
            return Err(MediaError::ColdChunkSize {
                requested: chunk_bytes,
            });
        }
        for tensor in &self.tensors {
            if chunk_bytes < tensor.dtype.storage_bytes() {
                return Err(MediaError::ColdChunkSize {
                    requested: chunk_bytes,
                });
            }
        }
        Ok(ColdUploadChunks {
            tensors: &self.tensors,
            chunk_bytes,
            tensor_index: 0,
            next_element: 0,
            finished: false,
        })
    }
}

/// Fallible, allocation-free iterator over all planned tensors.
#[derive(Debug)]
pub struct ColdUploadChunks<'a> {
    tensors: &'a [ColdTensor],
    chunk_bytes: usize,
    tensor_index: usize,
    next_element: usize,
    finished: bool,
}

impl Iterator for ColdUploadChunks<'_> {
    type Item = Result<ColdChunk, MediaError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let tensor = self.tensors.get(self.tensor_index)?;
            let bytes_per_element = tensor.dtype.storage_bytes();
            let elements = match tensor.rows.checked_mul(tensor.columns) {
                Some(elements) => elements,
                None => {
                    self.finished = true;
                    return Some(Err(MediaError::DimensionOverflow("cold upload tensor")));
                }
            };
            if self.next_element >= elements {
                self.tensor_index = match self.tensor_index.checked_add(1) {
                    Some(index) => index,
                    None => {
                        self.finished = true;
                        return Some(Err(MediaError::Overflow("cold upload tensor index")));
                    }
                };
                self.next_element = 0;
                continue;
            }
            let chunk_elements = (self.chunk_bytes / bytes_per_element).max(1);
            let count = chunk_elements.min(elements - self.next_element);
            let source_offset = match self
                .next_element
                .checked_mul(bytes_per_element)
                .ok_or(MediaError::ByteOverflow("cold upload source"))
            {
                Ok(offset) => offset,
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            };
            let bytes = match count.checked_mul(bytes_per_element) {
                Some(bytes) => bytes,
                None => {
                    self.finished = true;
                    return Some(Err(MediaError::ByteOverflow("cold upload chunk")));
                }
            };
            match source_offset.checked_add(bytes) {
                Some(end) if end <= tensor.source_bytes => {}
                _ => {
                    self.finished = true;
                    return Some(Err(MediaError::ColdSourceRange {
                        tensor: tensor.name.clone(),
                        offset: source_offset,
                        bytes,
                        len: tensor.source_bytes,
                    }));
                }
            };
            let destination_offset = match tensor.destination_offset.checked_add(source_offset) {
                Some(offset) => offset,
                None => {
                    self.finished = true;
                    return Some(Err(MediaError::ByteOverflow("cold upload destination")));
                }
            };
            self.next_element = match self.next_element.checked_add(count) {
                Some(next) => next,
                None => {
                    self.finished = true;
                    return Some(Err(MediaError::Overflow("cold upload element index")));
                }
            };
            return Some(Ok(ColdChunk {
                tensor_index: self.tensor_index,
                source_offset,
                destination_offset,
                bytes,
            }));
        }
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, MediaError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(MediaError::Alignment(alignment));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(MediaError::ByteOverflow("cold upload alignment"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huge_plan_is_chunked_without_allocating_or_streaming_payloads() {
        let mut plan = ColdUploadPlan::new(1024).unwrap();
        // 4 GiB of logical F32 data plus 2 MiB BF16, with no payload allocation.
        plan.push(
            "vision.patch_embed.proj.weight",
            1_000_000,
            1024,
            MediaDtype::F32,
            MediaDevice::Cuda { index: 0 },
        )
        .unwrap();
        plan.push(
            "vision.pos_embed.weight",
            1024,
            1024,
            MediaDtype::Bf16,
            MediaDevice::Cuda { index: 0 },
        )
        .unwrap();
        assert_eq!(plan.total_bytes(), 4_096_000_000 + 2_097_152);
        assert_eq!(plan.padding_bytes(), 0);
        assert_eq!(plan.tensors()[1].destination_offset(), 4_096_000_000);

        let chunks: Vec<_> = plan.chunks(256).unwrap().collect::<Result<_, _>>().unwrap();
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.bytes <= 256));
        assert_eq!(
            chunks.iter().map(|chunk| chunk.bytes).sum::<usize>(),
            4_096_000_000 + 2_097_152
        );
        assert_eq!(chunks.first().unwrap().source_range(), 0..256);
        assert_eq!(chunks.first().unwrap().destination_range(), 0..256);
        assert_eq!(chunks.last().unwrap().tensor_index, 1);
    }

    #[test]
    fn destination_alignment_inserts_checked_padding() {
        let mut plan = ColdUploadPlan::new(8).unwrap();
        plan.push("first", 3, 1, MediaDtype::F16, MediaDevice::Cpu)
            .unwrap();
        plan.push("second", 2, 1, MediaDtype::F32, MediaDevice::Cpu)
            .unwrap();
        assert_eq!(plan.tensors()[0].destination_offset(), 0);
        assert_eq!(plan.tensors()[1].destination_offset(), 16);
        assert_eq!(plan.padding_bytes(), 10);
        assert_eq!(plan.total_bytes(), 24);
        assert!(plan
            .tensors()
            .iter()
            .all(|tensor| tensor.destination_offset() % 16 == 0));
    }

    #[test]
    fn malformed_plans_and_chunk_sizes_are_typed() {
        assert_eq!(
            ColdUploadPlan::new(0),
            Err(MediaError::ColdChunkSize { requested: 0 })
        );
        let mut plan = ColdUploadPlan::new(8).unwrap();
        assert!(matches!(
            plan.push("", 1, 1, MediaDtype::F32, MediaDevice::Cpu),
            Err(MediaError::Empty(_))
        ));
        assert!(matches!(
            plan.push("bad", 0, 1, MediaDtype::F32, MediaDevice::Cpu),
            Err(MediaError::ZeroDimension(_))
        ));
        assert!(matches!(
            plan.push("overflow", usize::MAX, 2, MediaDtype::F32, MediaDevice::Cpu),
            Err(MediaError::DimensionOverflow(_))
        ));
        plan.push("ok", 2, 2, MediaDtype::F32, MediaDevice::Cpu)
            .unwrap();
        assert_eq!(
            plan.chunks(0).unwrap_err(),
            MediaError::ColdChunkSize { requested: 0 }
        );
        let mut bf16 = ColdUploadPlan::new(1).unwrap();
        bf16.push("bf16", 1, 1, MediaDtype::Bf16, MediaDevice::Cpu)
            .unwrap();
        assert_eq!(
            bf16.chunks(1).unwrap_err(),
            MediaError::ColdChunkSize { requested: 1 }
        );
    }
}
