//! Scalar, device, and checked two-dimensional layout descriptors for media rows.

use super::MediaError;

/// Storage formats admitted by the media foundation. Model weights may use either
/// 16-bit format; patch/feature payloads stay F32 until a tower selects its dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaDtype {
    F32,
    F16,
    Bf16,
}

impl MediaDtype {
    pub const fn storage_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
        }
    }

    /// Admit a checkpoint dtype without introducing a parser for formats the
    /// foundation cannot store. Unsupported future formats fail as a typed error.
    pub fn from_checkpoint_name(name: &str) -> Result<Self, MediaError> {
        match name {
            "F32" | "f32" => Ok(Self::F32),
            "F16" | "f16" => Ok(Self::F16),
            "BF16" | "bf16" => Ok(Self::Bf16),
            other => Err(MediaError::UnsupportedDtype {
                name: other.to_owned(),
            }),
        }
    }

    pub(crate) fn checked_bytes(self, elements: usize) -> Result<usize, MediaError> {
        elements
            .checked_mul(self.storage_bytes())
            .ok_or(MediaError::ByteOverflow("media tensor"))
    }
}

/// Destination device named by a layout. No driver call is made here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaDevice {
    Cpu,
    Cuda { index: u32 },
}

impl MediaDevice {
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }
}

/// A checked, nonempty two-dimensional row region in element units.
///
/// The foundation accepts contiguous rows and padded rows (a larger row stride).
/// Transposed or broadcast media layouts are rejected before any backend view is
/// constructed, because the native media ABI describes `[rows, columns]` matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLayout {
    shape: [usize; 2],
    strides: [usize; 2],
    offset_elements: usize,
    dtype: MediaDtype,
    device: MediaDevice,
    span_elements: usize,
    span_bytes: usize,
    total_bytes: usize,
}

impl MediaLayout {
    pub fn row_major(
        rows: usize,
        columns: usize,
        dtype: MediaDtype,
        device: MediaDevice,
    ) -> Result<Self, MediaError> {
        Self::strided(rows, columns, columns, 1, 0, dtype, device)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn strided(
        rows: usize,
        columns: usize,
        row_stride: usize,
        column_stride: usize,
        offset_elements: usize,
        dtype: MediaDtype,
        device: MediaDevice,
    ) -> Result<Self, MediaError> {
        if rows == 0 || columns == 0 {
            return Err(MediaError::ZeroDimension("media tensor"));
        }
        if row_stride == 0 || column_stride == 0 {
            return Err(MediaError::InvalidStride("media strides must be positive"));
        }
        let logical_columns = columns
            .checked_mul(column_stride)
            .ok_or(MediaError::LayoutOverflow("media column span"))?;
        if rows > 1 && row_stride < logical_columns {
            return Err(MediaError::InvalidStride(
                "media rows overlap or are not row-major",
            ));
        }
        let last_row = rows
            .checked_sub(1)
            .and_then(|row| row.checked_mul(row_stride))
            .ok_or(MediaError::LayoutOverflow("media row stride"))?;
        let last_column = columns
            .checked_sub(1)
            .and_then(|column| column.checked_mul(column_stride))
            .ok_or(MediaError::LayoutOverflow("media column stride"))?;
        let span_elements = last_row
            .checked_add(last_column)
            .and_then(|span| span.checked_add(1))
            .ok_or(MediaError::LayoutOverflow("media layout span"))?;
        let total_elements = offset_elements
            .checked_add(span_elements)
            .ok_or(MediaError::LayoutOverflow("media layout offset"))?;
        let span_bytes = dtype.checked_bytes(span_elements)?;
        let total_bytes = dtype.checked_bytes(total_elements)?;
        Ok(Self {
            shape: [rows, columns],
            strides: [row_stride, column_stride],
            offset_elements,
            dtype,
            device,
            span_elements,
            span_bytes,
            total_bytes,
        })
    }

    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub const fn rows(&self) -> usize {
        self.shape[0]
    }

    pub const fn columns(&self) -> usize {
        self.shape[1]
    }

    pub const fn strides(&self) -> [usize; 2] {
        self.strides
    }

    pub const fn offset_elements(&self) -> usize {
        self.offset_elements
    }

    pub const fn dtype(&self) -> MediaDtype {
        self.dtype
    }

    pub const fn device(&self) -> MediaDevice {
        self.device
    }

    pub const fn span_elements(&self) -> usize {
        self.span_elements
    }

    /// Bytes occupied only by addressed elements, excluding the prefix offset.
    pub const fn span_bytes(&self) -> usize {
        self.span_bytes
    }

    /// Bytes from the backing base through the final addressed element.
    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn is_contiguous_row_major(&self) -> bool {
        self.strides[1] == 1 && (self.rows() == 1 || self.strides[0] == self.columns())
    }

    /// Validate that a borrowed payload contains every addressed element.
    pub fn validate_capacity(&self, capacity_elements: usize) -> Result<(), MediaError> {
        let required = self
            .offset_elements()
            .checked_add(self.span_elements())
            .ok_or(MediaError::LayoutOverflow("media layout capacity"))?;
        if required > capacity_elements {
            return Err(MediaError::LayoutCapacity {
                required,
                capacity: capacity_elements,
            });
        }
        Ok(())
    }

    /// Checked element index for a logical `(row, column)` coordinate.
    pub fn element_offset(&self, row: usize, column: usize) -> Result<usize, MediaError> {
        if row >= self.rows() || column >= self.columns() {
            return Err(MediaError::SelectionIndex {
                index: row.max(column),
                limit: if row >= self.rows() {
                    self.rows()
                } else {
                    self.columns()
                },
            });
        }
        row.checked_mul(self.strides[0])
            .and_then(|offset| offset.checked_add(column.checked_mul(self.strides[1])?))
            .and_then(|offset| offset.checked_add(self.offset_elements))
            .ok_or(MediaError::LayoutOverflow("media element offset"))
    }

    /// Borrow one row only when column elements are contiguous.
    pub fn row_slice<'a>(&self, data: &'a [f32], row: usize) -> Option<&'a [f32]> {
        if self.dtype() != MediaDtype::F32 || self.strides[1] != 1 || row >= self.rows() {
            return None;
        }
        let start = self.element_offset(row, 0).ok()?;
        let end = start.checked_add(self.columns())?;
        data.get(start..end)
    }
}

/// Constant-cost borrowed F32 host view with its checked layout.
#[derive(Debug, Clone, Copy)]
pub struct F32View<'a> {
    data: &'a [f32],
    layout: MediaLayout,
}

impl<'a> F32View<'a> {
    pub fn new(data: &'a [f32], layout: MediaLayout) -> Result<Self, MediaError> {
        if layout.dtype() != MediaDtype::F32 {
            return Err(MediaError::UnsupportedDtype {
                name: layout.dtype().name().to_owned(),
            });
        }
        if !layout.device().is_cpu() {
            return Err(MediaError::HostViewDevice);
        }
        layout.validate_capacity(data.len())?;
        Ok(Self { data, layout })
    }

    pub const fn layout(&self) -> MediaLayout {
        self.layout
    }

    pub const fn data(&self) -> &'a [f32] {
        self.data
    }

    /// Pointer to the logical first element, not the backing-slice base.
    pub const fn as_ptr(&self) -> *const f32 {
        // `F32View::new` already checked `offset_elements + span_elements <= data.len()`.
        unsafe { self.data.as_ptr().add(self.layout.offset_elements) }
    }

    /// Pointer to the backing slice base. Use this only when applying the
    /// checked `offset_elements()` yourself.
    pub const fn base_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Logical first element through the checked span.
    pub fn logical_data(&self) -> &'a [f32] {
        let start = self.layout.offset_elements;
        &self.data[start..start + self.layout.span_elements]
    }

    pub fn row(&self, row: usize) -> Option<&'a [f32]> {
        self.layout.row_slice(self.data, row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_and_padded_rows_have_checked_spans() {
        let dense = MediaLayout::row_major(2, 3, MediaDtype::F32, MediaDevice::Cpu).unwrap();
        assert_eq!(
            (dense.shape(), dense.strides(), dense.span_elements()),
            ([2, 3], [3, 1], 6)
        );
        assert_eq!((dense.span_bytes(), dense.total_bytes()), (24, 24));
        assert!(dense.is_contiguous_row_major());

        let padded =
            MediaLayout::strided(2, 3, 5, 1, 2, MediaDtype::F16, MediaDevice::Cpu).unwrap();
        assert_eq!(
            (
                padded.strides(),
                padded.offset_elements(),
                padded.span_elements()
            ),
            ([5, 1], 2, 8)
        );
        assert_eq!((padded.span_bytes(), padded.total_bytes()), (16, 20));
        assert!(!padded.is_contiguous_row_major());
        padded.validate_capacity(10).unwrap();
        assert_eq!(
            padded.validate_capacity(9),
            Err(MediaError::LayoutCapacity {
                required: 10,
                capacity: 9
            })
        );
    }

    #[test]
    fn invalid_strides_and_offsets_are_rejected() {
        assert!(matches!(
            MediaLayout::strided(2, 4, 3, 1, 0, MediaDtype::F32, MediaDevice::Cpu),
            Err(MediaError::InvalidStride(_))
        ));
        assert!(matches!(
            MediaLayout::strided(1, 4, 4, 0, 0, MediaDtype::F32, MediaDevice::Cpu),
            Err(MediaError::InvalidStride(_))
        ));
        assert!(matches!(
            MediaLayout::strided(2, 4, usize::MAX, 1, 0, MediaDtype::F32, MediaDevice::Cpu),
            Err(MediaError::LayoutOverflow(_))
        ));
        assert!(matches!(
            MediaLayout::row_major(0, 4, MediaDtype::F32, MediaDevice::Cpu),
            Err(MediaError::ZeroDimension(_))
        ));
    }

    #[test]
    fn f32_views_borrow_and_preserve_the_checked_layout() {
        let data = [9.0, 1.0, 2.0, 3.0, 8.0, 4.0, 5.0, 6.0];
        let layout =
            MediaLayout::strided(2, 3, 4, 1, 1, MediaDtype::F32, MediaDevice::Cpu).unwrap();
        let view = F32View::new(&data, layout).unwrap();
        assert_eq!(view.row(0), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(view.row(1), Some([4.0, 5.0, 6.0].as_slice()));
        assert_eq!(view.as_ptr(), data[1..].as_ptr());
        assert_eq!(view.base_ptr(), data.as_ptr());
        assert_eq!(view.logical_data(), &data[1..]);
        assert_eq!(layout.total_bytes(), data.len() * 4);

        let short = F32View::new(&data[..4], layout);
        assert!(matches!(short, Err(MediaError::LayoutCapacity { .. })));
        let cuda =
            MediaLayout::row_major(1, 1, MediaDtype::F32, MediaDevice::Cuda { index: 0 }).unwrap();
        assert_eq!(
            F32View::new(&data, cuda).unwrap_err(),
            MediaError::HostViewDevice
        );
        let bf16 = MediaLayout::row_major(1, 1, MediaDtype::Bf16, MediaDevice::Cpu).unwrap();
        assert!(matches!(
            F32View::new(&data, bf16),
            Err(MediaError::UnsupportedDtype { .. })
        ));
    }

    #[test]
    fn future_checkpoint_dtypes_fail_without_a_parser() {
        let future = ["F8_E4M3", "F8_E5M2", "F8_E8M0", "F4_E2M1"];
        for name in future {
            assert_eq!(
                MediaDtype::from_checkpoint_name(name),
                Err(MediaError::UnsupportedDtype {
                    name: name.to_owned()
                })
            );
        }
        assert_eq!(
            MediaDtype::from_checkpoint_name("BF16").unwrap(),
            MediaDtype::Bf16
        );
    }
}
