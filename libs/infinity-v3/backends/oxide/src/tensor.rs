// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Bounded, allocation-free tensor metadata and borrowed device views.
//!
//! Layouts are dense row-major or checked strided descriptors in element units.
//! Views validate metadata and addresses; they do not enqueue work, synchronize,
//! retain the context, or copy device storage.

use cuda_core::{Context, DeviceBuffer, DeviceCopy};
use std::{fmt, mem::size_of, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    DimensionOverflow,
    AddressOverflow,
    BufferTooSmall,
    StorageOverflow,
    Overlapping,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DimensionOverflow => "tensor dimension product overflows usize",
            Self::AddressOverflow => "tensor address span overflows usize or u64",
            Self::BufferTooSmall => "tensor view exceeds its device buffer",
            Self::StorageOverflow => "tensor storage byte count overflows usize",
            Self::Overlapping => "mutable tensor view requires nonoverlapping row-major strides",
        })
    }
}

impl std::error::Error for TensorError {}

/// Storage encodings descriptors may name. `F32`, `F16`, `Bf16`, and `E4M3Fn` have
/// `Element` markers; all other formats remain metadata-only here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarFormat {
    F32,
    F16,
    Bf16,
    F8E4M3FN,
    F8E5M2,
    F8E8M0FNU,
    F8E5M3FNU,
    F4E2M1,
    I4,
}

impl ScalarFormat {
    /// Checked packed or unpacked storage size. Sub-byte formats round up, so
    /// odd logical counts occupy a final partial byte.
    pub const fn storage_bytes(self, logical_count: usize) -> Result<usize, TensorError> {
        let bytes = match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::F8E4M3FN | Self::F8E5M2 | Self::F8E8M0FNU | Self::F8E5M3FNU => 1,
            Self::F4E2M1 | Self::I4 => return Ok(logical_count / 2 + logical_count % 2),
        };
        match logical_count.checked_mul(bytes) {
            Some(n) => Ok(n),
            None => Err(TensorError::StorageOverflow),
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Sealed scalar marker over byte-addressable device storage.
pub trait Element: sealed::Sealed + Copy + 'static {
    type Storage: DeviceCopy;
    const FORMAT: ScalarFormat;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bf16(u16);
impl Bf16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}
// SAFETY: transparent u16 storage, every bit pattern valid, no ownership.
unsafe impl DeviceCopy for Bf16 {}
impl sealed::Sealed for Bf16 {}
impl Element for Bf16 {
    type Storage = Bf16;
    const FORMAT: ScalarFormat = ScalarFormat::Bf16;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F16(u16);
impl F16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}
// SAFETY: transparent u16 storage, every bit pattern valid, no ownership.
unsafe impl DeviceCopy for F16 {}
impl sealed::Sealed for F16 {}
impl Element for F16 {
    type Storage = F16;
    const FORMAT: ScalarFormat = ScalarFormat::F16;
}

/// One E4M3FN encoded byte. This is internal quantized matrix storage;
/// serving/model output dtypes remain independently constrained to F16/BF16.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct E4M3Fn(u8);
impl E4M3Fn {
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
    pub const fn to_bits(self) -> u8 {
        self.0
    }
}
// SAFETY: transparent u8 storage; all bit patterns are valid storage values.
unsafe impl DeviceCopy for E4M3Fn {}
impl sealed::Sealed for E4M3Fn {}
impl Element for E4M3Fn {
    type Storage = E4M3Fn;
    const FORMAT: ScalarFormat = ScalarFormat::F8E4M3FN;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct F32;
impl sealed::Sealed for F32 {}
impl Element for F32 {
    type Storage = f32;
    const FORMAT: ScalarFormat = ScalarFormat::F32;
}

/// Dense row-major or strided tensor metadata in element units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout<const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
    num_elements: usize,
    span_elements: usize,
}

impl<const N: usize> Layout<N> {
    /// Checked dense row-major layout. A zero extent anywhere makes the element
    /// count and span zero without multiplying unrelated extents.
    pub fn contiguous(shape: [usize; N]) -> Result<Self, TensorError> {
        let num_elements = checked_product(&shape)?;
        let strides = row_major_strides(&shape, num_elements)?;
        let span_elements = checked_span(&shape, &strides, num_elements)?;
        Ok(Self {
            shape,
            strides,
            num_elements,
            span_elements,
        })
    }

    /// Checked strided layout. Strides are nonnegative element steps; zero is
    /// allowed for read-only broadcasts.
    pub fn strided(shape: [usize; N], strides: [usize; N]) -> Result<Self, TensorError> {
        let num_elements = checked_product(&shape)?;
        let span_elements = checked_span(&shape, &strides, num_elements)?;
        Ok(Self {
            shape,
            strides,
            num_elements,
            span_elements,
        })
    }

    #[inline]
    pub const fn shape(&self) -> &[usize; N] {
        &self.shape
    }
    #[inline]
    pub const fn strides(&self) -> &[usize; N] {
        &self.strides
    }
    #[inline]
    pub const fn num_elements(&self) -> usize {
        self.num_elements
    }
    #[inline]
    pub const fn span_elements(&self) -> usize {
        self.span_elements
    }

    /// True when addresses are dense row-major. Unit extents ignore their
    /// stride, and an empty layout is contiguous because it addresses nothing.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        if self.num_elements == 0 {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..N).rev() {
            if self.shape[i] > 1 && self.strides[i] != expected {
                return false;
            }
            expected = match expected.checked_mul(self.shape[i]) {
                Some(next) => next,
                None => return false,
            };
        }
        true
    }

    /// Allows row padding while rejecting overlapping or broadcast writes.
    /// Permuted mutable layouts are deliberately outside this first interface.
    #[inline]
    pub fn is_nonoverlapping_row_major(&self) -> bool {
        if self.num_elements == 0 {
            return true;
        }
        let mut span = 1usize;
        for (&extent, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if extent <= 1 {
                continue;
            }
            if stride < span {
                return false;
            }
            span = match (extent - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(span))
            {
                Some(n) => n,
                None => return false,
            };
        }
        true
    }
}

fn checked_product<const N: usize>(shape: &[usize; N]) -> Result<usize, TensorError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let mut product = 1usize;
    for &extent in shape {
        product = product
            .checked_mul(extent)
            .ok_or(TensorError::DimensionOverflow)?;
    }
    Ok(product)
}

fn row_major_strides<const N: usize>(
    shape: &[usize; N],
    num_elements: usize,
) -> Result<[usize; N], TensorError> {
    let mut strides = [1usize; N];
    let mut trailing = 1usize;
    for i in (0..N).rev() {
        strides[i] = trailing;
        trailing = match trailing.checked_mul(shape[i]) {
            Some(next) => next,
            None if num_elements == 0 => 0,
            None => return Err(TensorError::DimensionOverflow),
        };
    }
    Ok(strides)
}

fn checked_span<const N: usize>(
    shape: &[usize; N],
    strides: &[usize; N],
    num_elements: usize,
) -> Result<usize, TensorError> {
    if num_elements == 0 {
        return Ok(0);
    }
    let mut last = 0usize;
    for i in 0..N {
        let extent_minus_one = shape[i]
            .checked_sub(1)
            .ok_or(TensorError::DimensionOverflow)?;
        let term = strides[i]
            .checked_mul(extent_minus_one)
            .ok_or(TensorError::AddressOverflow)?;
        last = last.checked_add(term).ok_or(TensorError::AddressOverflow)?;
    }
    last.checked_add(1).ok_or(TensorError::AddressOverflow)
}

fn checked_view_address(
    base: u64,
    offset_elements: usize,
    span_elements: usize,
    capacity_elements: usize,
    element_bytes: usize,
) -> Result<u64, TensorError> {
    let end = offset_elements
        .checked_add(span_elements)
        .ok_or(TensorError::AddressOverflow)?;
    if end > capacity_elements {
        return Err(TensorError::BufferTooSmall);
    }
    let offset_bytes = offset_elements
        .checked_mul(element_bytes)
        .ok_or(TensorError::AddressOverflow)?;
    let span_bytes = span_elements
        .checked_mul(element_bytes)
        .ok_or(TensorError::AddressOverflow)?;
    let end_bytes = offset_bytes
        .checked_add(span_bytes)
        .ok_or(TensorError::AddressOverflow)?;
    let start = u64::try_from(offset_bytes).map_err(|_| TensorError::AddressOverflow)?;
    let end = u64::try_from(end_bytes).map_err(|_| TensorError::AddressOverflow)?;
    let start = base
        .checked_add(start)
        .ok_or(TensorError::AddressOverflow)?;
    base.checked_add(end).ok_or(TensorError::AddressOverflow)?;
    Ok(start)
}

/// Read-only borrowed view. The lifetime keeps the buffer valid for synchronous
/// use; it does not retain storage through asynchronous work. An enqueue remains
/// unsafe until the caller owns completion and outlives it.
///
/// FP16 storage cannot be silently bound as BF16:
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use infinity_backend_oxide::tensor::{Bf16, F16, Layout, TensorRef};
/// fn wrong_format(buffer: &DeviceBuffer<F16>, layout: Layout<1>) {
///     let _ = TensorRef::<Bf16, 1>::new(buffer, layout);
/// }
/// ```
#[derive(Clone, Copy)]
pub struct TensorRef<'a, E: Element, const N: usize> {
    buffer: &'a DeviceBuffer<E::Storage>,
    layout: Layout<N>,
    ptr: u64,
}

impl<'a, E: Element, const N: usize> TensorRef<'a, E, N> {
    #[inline]
    pub fn new(
        buffer: &'a DeviceBuffer<E::Storage>,
        layout: Layout<N>,
    ) -> Result<Self, TensorError> {
        Self::with_offset(buffer, 0, layout)
    }

    /// Validates the element offset, layout span, buffer capacity, and address
    /// arithmetic without a driver call or an additional owner.
    #[inline]
    pub fn with_offset(
        buffer: &'a DeviceBuffer<E::Storage>,
        offset_elements: usize,
        layout: Layout<N>,
    ) -> Result<Self, TensorError> {
        let base = buffer.cu_deviceptr();
        let ptr = checked_view_address(
            base,
            offset_elements,
            layout.span_elements(),
            buffer.len(),
            size_of::<E::Storage>(),
        )?;
        Ok(Self {
            buffer,
            layout,
            ptr,
        })
    }

    #[inline]
    pub const fn layout(&self) -> &Layout<N> {
        &self.layout
    }
    #[inline]
    pub const fn device_ptr(&self) -> u64 {
        self.ptr
    }
    #[inline]
    pub fn context(&self) -> &Arc<Context> {
        self.buffer.context()
    }
}

/// Exclusive borrowed view. Mutable views currently require a nonoverlapping
/// row-major layout (padding allowed), so broadcast writes are rejected. The same
/// asynchronous-lifetime caution as `TensorRef` applies.
pub struct TensorMut<'a, E: Element, const N: usize> {
    buffer: &'a mut DeviceBuffer<E::Storage>,
    layout: Layout<N>,
    ptr: u64,
}

impl<'a, E: Element, const N: usize> TensorMut<'a, E, N> {
    #[inline]
    pub fn new(
        buffer: &'a mut DeviceBuffer<E::Storage>,
        layout: Layout<N>,
    ) -> Result<Self, TensorError> {
        Self::with_offset(buffer, 0, layout)
    }

    /// Validates row-major nonoverlap plus offset, capacity, and address
    /// arithmetic without manufacturing any shared-buffer access.
    #[inline]
    pub fn with_offset(
        buffer: &'a mut DeviceBuffer<E::Storage>,
        offset_elements: usize,
        layout: Layout<N>,
    ) -> Result<Self, TensorError> {
        if !layout.is_nonoverlapping_row_major() {
            return Err(TensorError::Overlapping);
        }
        let base = buffer.cu_deviceptr();
        let ptr = checked_view_address(
            base,
            offset_elements,
            layout.span_elements(),
            buffer.len(),
            size_of::<E::Storage>(),
        )?;
        Ok(Self {
            buffer,
            layout,
            ptr,
        })
    }

    #[inline]
    pub const fn layout(&self) -> &Layout<N> {
        &self.layout
    }
    #[inline]
    pub const fn device_ptr(&self) -> u64 {
        self.ptr
    }
    #[inline]
    pub fn context(&self) -> &Arc<Context> {
        self.buffer.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_padded_layouts_report_exact_metadata() {
        let dense = Layout::contiguous([2, 3, 4]).unwrap();
        assert!(dense.is_nonoverlapping_row_major());
        assert!(Layout::strided([3, 5], [8, 1])
            .unwrap()
            .is_nonoverlapping_row_major());
        assert!(!Layout::strided([3, 5], [4, 1])
            .unwrap()
            .is_nonoverlapping_row_major());
        assert!(!Layout::strided([3, 5], [0, 1])
            .unwrap()
            .is_nonoverlapping_row_major());
        assert_eq!((dense.shape(), dense.strides()), (&[2, 3, 4], &[12, 4, 1]));
        assert_eq!((dense.num_elements(), dense.span_elements()), (24, 24));
        assert!(dense.is_contiguous());
        let padded = Layout::strided([2, 3], [8, 2]).unwrap();
        assert_eq!(
            (
                padded.strides(),
                padded.num_elements(),
                padded.span_elements()
            ),
            (&[8, 2], 6, 13)
        );
        assert!(!padded.is_contiguous());
    }

    #[test]
    fn zero_extents_span_nothing_without_multiplying_huge_neighbors() {
        let leading = Layout::contiguous([0, usize::MAX, usize::MAX]).unwrap();
        assert_eq!((leading.num_elements(), leading.span_elements()), (0, 0));
        assert!(leading.is_contiguous());
        let strided = Layout::strided([usize::MAX, 0, usize::MAX], [usize::MAX; 3]).unwrap();
        assert_eq!((strided.num_elements(), strided.span_elements()), (0, 0));
    }

    #[test]
    fn scalar_and_broadcast_layouts_stay_bounded() {
        let scalar = Layout::<0>::contiguous([]).unwrap();
        assert_eq!((scalar.num_elements(), scalar.span_elements()), (1, 1));
        assert!(scalar.is_contiguous());
        let broadcast = Layout::strided([4, 3], [0, 1]).unwrap();
        assert_eq!(
            (broadcast.num_elements(), broadcast.span_elements()),
            (12, 3)
        );
        assert!(!broadcast.is_contiguous());
        let singleton = Layout::strided([1], [usize::MAX]).unwrap();
        assert_eq!(
            (singleton.span_elements(), singleton.is_contiguous()),
            (1, true)
        );
    }

    #[test]
    fn product_and_address_overflows_are_rejected() {
        assert_eq!(
            Layout::contiguous([usize::MAX, 2]),
            Err(TensorError::DimensionOverflow)
        );
        assert_eq!(
            Layout::strided([2], [usize::MAX]),
            Err(TensorError::AddressOverflow)
        );
        assert_eq!(
            Layout::strided([usize::MAX], [2]),
            Err(TensorError::AddressOverflow)
        );
    }

    #[test]
    fn packed_storage_rounds_odd_logical_counts() {
        assert_eq!(ScalarFormat::F4E2M1.storage_bytes(0), Ok(0));
        assert_eq!(ScalarFormat::F4E2M1.storage_bytes(1), Ok(1));
        assert_eq!(ScalarFormat::F4E2M1.storage_bytes(5), Ok(3));
        assert_eq!(ScalarFormat::I4.storage_bytes(6), Ok(3));
        assert_eq!(
            (
                ScalarFormat::F8E4M3FN.storage_bytes(5),
                ScalarFormat::Bf16.storage_bytes(3)
            ),
            (Ok(5), Ok(6))
        );
        assert_eq!(
            ScalarFormat::F32.storage_bytes(usize::MAX),
            Err(TensorError::StorageOverflow)
        );
    }

    #[test]
    fn view_bounds_reject_capacity_and_address_overflow() {
        assert_eq!(
            checked_view_address(0x1000, 0, 4, 3, 2),
            Err(TensorError::BufferTooSmall)
        );
        assert_eq!(
            checked_view_address(0x1000, 2, 1, 2, 2),
            Err(TensorError::BufferTooSmall)
        );
        assert_eq!(
            checked_view_address(0x1000, usize::MAX, 2, usize::MAX, 1),
            Err(TensorError::AddressOverflow)
        );
        assert_eq!(
            checked_view_address(u64::MAX - 1, 1, 1, 2, 2),
            Err(TensorError::AddressOverflow)
        );
        assert_eq!(checked_view_address(0x1000, 2, 1, 3, 2), Ok(0x1004));
    }

    #[test]
    fn elements_preserve_storage_and_format_contract() {
        assert_eq!(
            (
                size_of::<<Bf16 as Element>::Storage>(),
                size_of::<<F16 as Element>::Storage>(),
                size_of::<<F32 as Element>::Storage>()
            ),
            (2, 2, 4)
        );
        assert_eq!(
            (Bf16::FORMAT, F16::FORMAT, F32::FORMAT),
            (ScalarFormat::Bf16, ScalarFormat::F16, ScalarFormat::F32)
        );
    }
}
