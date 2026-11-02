//! Binding from shared media layouts to existing oxide tensor views.
//!
//! The shared foundation validates descriptors on the host. This module only
//! adapts an already-uploaded `DeviceBuffer` to the existing checked
//! `TensorRef`/`TensorMut`; it allocates nothing, records no `Event`, submits no
//! work, and does not retain the buffer or context.
#![allow(dead_code)] // Bound by the family executor before this module is exposed.

use crate::tensor::{Element, Layout, ScalarFormat, TensorError, TensorMut, TensorRef};
use cuda_core::{Context, DeviceBuffer};
use infinity_v3_backend_core::media::{MediaDevice, MediaDtype, MediaLayout};
use std::{fmt, sync::Arc};

#[derive(Debug)]
pub enum MediaBridgeError {
    DType {
        layout: MediaDtype,
        element: ScalarFormat,
    },
    Device(MediaDevice),
    DeviceOrdinal {
        layout: u32,
        buffer: usize,
    },
    ContextMismatch,
    Tensor(TensorError),
}

impl fmt::Display for MediaBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DType { layout, element } => {
                write!(f, "media layout dtype {layout:?} cannot bind {element:?}")
            }
            Self::Device(device) => {
                write!(f, "media layout device {device:?} is not a device view")
            }
            Self::DeviceOrdinal { layout, buffer } => write!(
                f,
                "media layout CUDA ordinal {layout} does not match buffer ordinal {buffer}"
            ),
            Self::ContextMismatch => f.write_str("media layout context does not match buffer"),
            Self::Tensor(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for MediaBridgeError {}

impl From<TensorError> for MediaBridgeError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

pub const fn scalar_format(dtype: MediaDtype) -> ScalarFormat {
    match dtype {
        MediaDtype::F32 => ScalarFormat::F32,
        MediaDtype::F16 => ScalarFormat::F16,
        MediaDtype::Bf16 => ScalarFormat::Bf16,
    }
}

/// Pure ordinal rule used by both device views and CPU tests.
pub const fn device_matches(device: MediaDevice, ordinal: usize) -> bool {
    match device {
        MediaDevice::Cuda { index } => index as usize == ordinal,
        MediaDevice::Cpu => false,
    }
}

fn check_context(
    layout: &MediaLayout,
    buffer_context: &Arc<Context>,
    expected: &Arc<Context>,
) -> Result<(), MediaBridgeError> {
    let MediaDevice::Cuda { index } = layout.device() else {
        return Err(MediaBridgeError::Device(layout.device()));
    };
    if !device_matches(layout.device(), buffer_context.ordinal()) {
        return Err(MediaBridgeError::DeviceOrdinal {
            layout: index,
            buffer: buffer_context.ordinal(),
        });
    }
    if !Arc::ptr_eq(buffer_context, expected) {
        return Err(MediaBridgeError::ContextMismatch);
    }
    Ok(())
}

/// Convert the shape/strides into the existing checked 2-D oxide layout. The
/// element offset remains separate and is passed to `with_offset` below.
pub fn checked_layout(layout: &MediaLayout) -> Result<Layout<2>, MediaBridgeError> {
    Ok(Layout::strided(layout.shape(), layout.strides())?)
}

/// Offset and layout pair consumed by `TensorRef::with_offset`/`TensorMut`.
pub fn checked_subview(layout: &MediaLayout) -> Result<(usize, Layout<2>), MediaBridgeError> {
    Ok((layout.offset_elements(), checked_layout(layout)?))
}

/// Bind an uploaded media matrix as a constant-cost borrowed tensor subview.
///
/// The caller owns the buffer and completion. `expected` must be the same
/// context that allocated the buffer; this rejects same-ordinal foreign
/// contexts as well as a different CUDA ordinal.
pub fn warm_ref<'a, E: Element>(
    buffer: &'a DeviceBuffer<E::Storage>,
    layout: &MediaLayout,
    expected: &Arc<Context>,
) -> Result<TensorRef<'a, E, 2>, MediaBridgeError> {
    check_context(layout, buffer.context(), expected)?;
    if E::FORMAT != scalar_format(layout.dtype()) {
        return Err(MediaBridgeError::DType {
            layout: layout.dtype(),
            element: E::FORMAT,
        });
    }
    let (offset, view) = checked_subview(layout)?;
    Ok(TensorRef::with_offset(buffer, offset, view)?)
}

/// Exclusive counterpart of [`warm_ref`], also preserving the checked offset.
pub fn warm_mut<'a, E: Element>(
    buffer: &'a mut DeviceBuffer<E::Storage>,
    layout: &MediaLayout,
    expected: &Arc<Context>,
) -> Result<TensorMut<'a, E, 2>, MediaBridgeError> {
    check_context(layout, buffer.context(), expected)?;
    if E::FORMAT != scalar_format(layout.dtype()) {
        return Err(MediaBridgeError::DType {
            layout: layout.dtype(),
            element: E::FORMAT,
        });
    }
    let (offset, view) = checked_subview(layout)?;
    Ok(TensorMut::with_offset(buffer, offset, view)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::media::MediaDtype;

    #[test]
    fn shared_layout_converts_to_the_existing_checked_tensor_layout() {
        let layout = MediaLayout::strided(
            2,
            3,
            5,
            1,
            1,
            MediaDtype::F16,
            MediaDevice::Cuda { index: 0 },
        )
        .unwrap();
        let (offset, converted) = checked_subview(&layout).unwrap();
        assert_eq!(converted.shape(), &[2, 3]);
        assert_eq!(converted.strides(), &[5, 1]);
        assert_eq!(offset, 1);
        assert_eq!(layout.offset_elements(), 1);
        assert!(converted.is_nonoverlapping_row_major());
        assert_eq!(scalar_format(MediaDtype::Bf16), ScalarFormat::Bf16);
        assert_eq!(scalar_format(MediaDtype::F32), ScalarFormat::F32);
    }

    #[test]
    fn cuda_ordinal_rule_rejects_cpu_and_mismatched_devices() {
        assert!(device_matches(MediaDevice::Cuda { index: 1 }, 1));
        assert!(!device_matches(MediaDevice::Cuda { index: 1 }, 0));
        assert!(!device_matches(MediaDevice::Cpu, 0));
    }
}
