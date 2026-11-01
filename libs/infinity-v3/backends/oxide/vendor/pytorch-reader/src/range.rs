//! Bounded materialization of row-major tensor ranges.
//!
//! A range read never asks a storage for more than [`MAX_BOUNDED_WINDOW`] bytes at once.
//! Contiguous storage runs are coalesced and streamed in chunks; holes between strided
//! elements are read separately instead of being spanned as one unbounded window.

use crate::pickle_reader::is_contiguous_stride;
use crate::storage::{MAX_BOUNDED_WINDOW, StorageSource};
use std::io;

/// A logical range mapped onto one storage, read in bounded windows.
pub(crate) struct BoundedRange<'a> {
    source: &'a StorageSource,
    key: &'a str,
    shape: &'a [usize],
    stride: &'a [usize],
    storage_offset: usize,
    element_size: usize,
    num_elements: usize,
}

impl<'a> BoundedRange<'a> {
    pub(crate) fn new(
        source: &'a StorageSource,
        key: &'a str,
        shape: &'a [usize],
        stride: &'a [usize],
        storage_offset: usize,
        element_size: usize,
        num_elements: usize,
    ) -> Self {
        Self {
            source,
            key,
            shape,
            stride,
            storage_offset,
            element_size,
            num_elements,
        }
    }

    /// Read logical elements `[start, start + count)` in row-major order.
    pub(crate) fn read(&self, start: usize, count: usize) -> io::Result<Vec<u8>> {
        let end = start.checked_add(count).ok_or_else(|| {
            input_error(format!(
                "Range start {start} and count {count} overflow usize"
            ))
        })?;
        if end > self.num_elements {
            return Err(input_error(format!(
                "Range [{start}, {end}) is outside tensor with {} elements",
                self.num_elements
            )));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let out_len = count
            .checked_mul(self.element_size)
            .ok_or_else(|| data_error("Range byte length overflows usize"))?;
        let mut out = Vec::with_capacity(out_len);
        if is_contiguous_stride(self.shape, self.stride) {
            let first = self
                .storage_offset
                .checked_add(start)
                .ok_or_else(|| data_error("Range storage offset overflows usize"))?;
            self.append_run(first, count, &mut out)?;
        } else {
            self.gather(start, end, &mut out)?;
        }
        Ok(out)
    }

    /// Storage index of one logical element.
    fn storage_index(&self, linear: usize) -> io::Result<usize> {
        let mut remaining = linear;
        let mut index = self.storage_offset;
        for (&dim, &step) in self.shape.iter().zip(self.stride).rev() {
            if dim == 0 {
                return Err(data_error("Non-empty range has a zero-sized dimension"));
            }
            let coordinate = remaining % dim;
            remaining /= dim;
            let delta = coordinate
                .checked_mul(step)
                .ok_or_else(|| data_error("Range stride calculation overflows usize"))?;
            index = index
                .checked_add(delta)
                .ok_or_else(|| data_error("Range storage index overflows usize"))?;
        }
        Ok(index)
    }

    /// Append `elements` contiguous storage elements starting at `storage_start`.
    fn append_run(
        &self,
        storage_start: usize,
        elements: usize,
        out: &mut Vec<u8>,
    ) -> io::Result<()> {
        let chunk_elements = MAX_BOUNDED_WINDOW
            .checked_div(self.element_size)
            .filter(|&chunk| chunk > 0)
            .unwrap_or(1);
        let mut done = 0usize;
        while done < elements {
            let take = (elements - done).min(chunk_elements);
            let element_start = storage_start
                .checked_add(done)
                .ok_or_else(|| data_error("Range storage index overflows usize"))?;
            let start_byte = element_start
                .checked_mul(self.element_size)
                .ok_or_else(|| data_error("Range byte offset overflows usize"))?;
            let len_byte = take
                .checked_mul(self.element_size)
                .ok_or_else(|| data_error("Range byte length overflows usize"))?;
            let (data, skipped) = self.read_window(start_byte, len_byte)?;
            let available = skipped
                .checked_add(data.len())
                .and_then(|bytes| bytes.checked_div(self.element_size))
                .ok_or_else(|| data_error("Storage length overflows usize"))?;
            let needed_end = element_start
                .checked_add(take)
                .ok_or_else(|| data_error("Range storage extent overflows usize"))?;
            if needed_end > available {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "Range needs {needed_end} elements from storage '{}', but only {available} are available",
                        self.key
                    ),
                ));
            }
            let offset = start_byte
                .checked_sub(skipped)
                .ok_or_else(|| data_error("Storage read began past the requested range"))?;
            let end = offset
                .checked_add(len_byte)
                .ok_or_else(|| data_error("Range byte extent overflows usize"))?;
            let window = data
                .get(offset..end)
                .ok_or_else(|| data_error("Storage window lies outside the bytes returned"))?;
            out.extend_from_slice(window);
            done += take;
        }
        Ok(())
    }

    /// Gather a strided logical range, coalescing only actual contiguous runs.
    fn gather(&self, start: usize, end: usize, out: &mut Vec<u8>) -> io::Result<()> {
        let mut last_element = vec![0u8; self.element_size];
        let mut previous = None;
        let mut linear = start;
        while linear < end {
            let index = self.storage_index(linear)?;
            if previous == Some(index) {
                out.extend_from_slice(&last_element);
                linear = linear
                    .checked_add(1)
                    .ok_or_else(|| data_error("Range logical index overflows usize"))?;
                continue;
            }
            let mut run_end = linear
                .checked_add(1)
                .ok_or_else(|| data_error("Range logical index overflows usize"))?;
            let mut next_index = index
                .checked_add(1)
                .ok_or_else(|| data_error("Range storage index overflows usize"))?;
            while run_end < end {
                let candidate = self.storage_index(run_end)?;
                if candidate != next_index {
                    break;
                }
                next_index = candidate
                    .checked_add(1)
                    .ok_or_else(|| data_error("Range storage index overflows usize"))?;
                run_end = run_end
                    .checked_add(1)
                    .ok_or_else(|| data_error("Range logical index overflows usize"))?;
            }
            let run_len = run_end - linear;
            self.append_run(index, run_len, out)?;
            let last_start = out
                .len()
                .checked_sub(self.element_size)
                .ok_or_else(|| data_error("Range produced no last element"))?;
            let last = out
                .get(last_start..)
                .ok_or_else(|| data_error("Range output is shorter than one element"))?;
            last_element.clear();
            last_element.extend_from_slice(last);
            previous = Some(
                index
                    .checked_add(run_len)
                    .and_then(|last| last.checked_sub(1))
                    .ok_or_else(|| data_error("Range storage index overflows usize"))?,
            );
            linear = run_end;
        }
        Ok(())
    }

    fn read_window(&self, start: usize, max_len: usize) -> io::Result<(Vec<u8>, usize)> {
        self.source.read(self.key, start, max_len).map_err(|err| {
            let kind = match err.kind() {
                kind if err.raw_os_error().is_some() => kind,
                io::ErrorKind::UnexpectedEof => io::ErrorKind::UnexpectedEof,
                _ => io::ErrorKind::InvalidData,
            };
            io::Error::new(
                kind,
                format!(
                    "Failed to read storage '{}' at byte {start} for tensor with shape {:?}: {err}",
                    self.key, self.shape
                ),
            )
        })
    }
}

fn data_error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn input_error(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
