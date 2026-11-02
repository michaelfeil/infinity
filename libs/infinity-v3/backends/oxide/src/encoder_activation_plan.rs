// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold-checked, allocation-free pointwise activation plans.
//!
//! V2 models exactly two honest contracts:
//!
//! * [`BiasMode::None`]: activate a BF16 row that is already the Native linear
//!   output, including any linear bias. The operator does not add a bias.
//! * [`BiasMode::SeparateBf16`]: add an explicit separate BF16 bias, round that
//!   sum to BF16, then activate. This is a separate elementwise add, not
//!   `at::linear`, not a fused GEMM epilogue, and not a claim that the two are
//!   bit-identical.
//!
//! The V1 `ModelDtype`/`FusedFloat` claims are intentionally absent.

use crate::encoder_activation::Activation;
use crate::tensor::{Layout, TensorError};
use std::fmt;

const ELEMENT_BLOCK: usize = 256;
const ELEMENT_BYTES: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasMode {
    None,
    SeparateBf16,
}

impl BiasMode {
    pub const fn has_bias(self) -> bool {
        matches!(self, Self::SeparateBf16)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// F32 activation of the already-BF16 input; one BF16 write.
    SingleModelWrite,
    /// Explicit separate BF16 add rounded once, then F32 activation and one
    /// BF16 write.
    SeparateBiasThenSingleWrite,
}

impl BiasMode {
    pub const fn rounding(self) -> Rounding {
        match self {
            Self::None => Rounding::SingleModelWrite,
            Self::SeparateBf16 => Rounding::SeparateBiasThenSingleWrite,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    ZeroWidth,
    ShapeMismatch,
    StrideTooSmall,
    NonUnitInnerStride,
    OverlappingOutput,
    MissingBias,
    UnexpectedBias,
    BiasLayout,
    ExtentOverflow,
    GridOverflow,
    LayoutMismatch,
    ContextMismatch,
    Tensor(TensorError),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Self::Tensor(error) = self {
            return error.fmt(f);
        }
        f.write_str(match self {
            Self::ZeroWidth => "encoder activation width must be nonzero",
            Self::ShapeMismatch => "encoder activation input and output shapes disagree",
            Self::StrideTooSmall => "encoder activation row stride is smaller than its width",
            Self::NonUnitInnerStride => "encoder activation requires unit inner stride",
            Self::OverlappingOutput => "encoder activation output rows overlap",
            Self::MissingBias => "separate-bias mode requires a bias layout",
            Self::UnexpectedBias => "activation-only mode rejects a bias layout",
            Self::BiasLayout => "encoder activation bias must be one contiguous vector",
            Self::ExtentOverflow => "encoder activation extent exceeds its 32-bit ABI",
            Self::GridOverflow => "encoder activation launch grid exceeds CUDA limits",
            Self::LayoutMismatch => "bound view layout differs from the prepared plan",
            Self::ContextMismatch => "bound buffers and stream belong to different contexts",
            Self::Tensor(_) => "invalid encoder activation tensor layout",
        })
    }
}

impl std::error::Error for PlanError {}

impl From<TensorError> for PlanError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

fn u32_dim(value: usize) -> Result<u32, PlanError> {
    u32::try_from(value).map_err(|_| PlanError::ExtentOverflow)
}

fn checked_layout(shape: [usize; 2], strides: [usize; 2]) -> Result<Layout<2>, PlanError> {
    Layout::strided(shape, strides).map_err(Into::into)
}

fn checked_bias_layout(shape: [usize; 1], strides: [usize; 1]) -> Result<Layout<1>, PlanError> {
    Layout::strided(shape, strides).map_err(Into::into)
}

/// Immutable launch metadata. All fields are prevalidated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationPlan {
    activation: Activation,
    bias: BiasMode,
    rows: usize,
    width: usize,
    input: Layout<2>,
    output: Layout<2>,
    bias_layout: Option<Layout<1>>,
    elements: usize,
    grid: [u32; 3],
    block: [u32; 3],
}

impl ActivationPlan {
    /// Build an activation-only plan for BF16 rows that already include any
    /// linear bias.
    pub fn activation_only(
        activation: Activation,
        rows: usize,
        width: usize,
        input_row_stride: usize,
        output_row_stride: usize,
    ) -> Result<Self, PlanError> {
        Self::padded(
            activation,
            BiasMode::None,
            rows,
            width,
            input_row_stride,
            output_row_stride,
        )
    }

    /// Build an explicit separate-bias plan. This is not a fused linear path.
    pub fn separate_bf16_bias(
        activation: Activation,
        rows: usize,
        width: usize,
        input_row_stride: usize,
        output_row_stride: usize,
    ) -> Result<Self, PlanError> {
        Self::padded(
            activation,
            BiasMode::SeparateBf16,
            rows,
            width,
            input_row_stride,
            output_row_stride,
        )
    }

    fn padded(
        activation: Activation,
        bias: BiasMode,
        rows: usize,
        width: usize,
        input_row_stride: usize,
        output_row_stride: usize,
    ) -> Result<Self, PlanError> {
        let input = checked_layout([rows, width], [input_row_stride, 1])?;
        let output = checked_layout([rows, width], [output_row_stride, 1])?;
        let bias_layout = if bias.has_bias() {
            Some(checked_bias_layout([width], [1])?)
        } else {
            None
        };
        Self::from_layouts(activation, bias, input, output, bias_layout)
    }

    /// Build a plan from explicit checked layouts. This is the cold admission
    /// point used by callers that already carry a padded storage descriptor.
    pub fn from_layouts(
        activation: Activation,
        bias: BiasMode,
        input: Layout<2>,
        output: Layout<2>,
        bias_layout: Option<Layout<1>>,
    ) -> Result<Self, PlanError> {
        let [rows, width] = *input.shape();
        if width == 0 {
            return Err(PlanError::ZeroWidth);
        }
        if output.shape() != input.shape() {
            return Err(PlanError::ShapeMismatch);
        }
        let [input_stride, input_inner] = *input.strides();
        let [output_stride, output_inner] = *output.strides();
        if input_inner != 1 || output_inner != 1 {
            return Err(PlanError::NonUnitInnerStride);
        }
        if input_stride < width || output_stride < width {
            return Err(PlanError::StrideTooSmall);
        }
        if !output.is_nonoverlapping_row_major() {
            return Err(PlanError::OverlappingOutput);
        }
        match (bias, bias_layout) {
            (BiasMode::None, Some(_)) => return Err(PlanError::UnexpectedBias),
            (BiasMode::None, None) => {}
            (_, None) => return Err(PlanError::MissingBias),
            (_, Some(layout)) => {
                if layout.shape() != &[width] || layout.strides() != &[1] {
                    return Err(PlanError::BiasLayout);
                }
            }
        }
        let elements = input.num_elements();
        let _ = u32_dim(rows)?;
        let _ = u32_dim(width)?;
        let _ = u32_dim(input_stride)?;
        let _ = u32_dim(output_stride)?;
        let blocks = elements.div_ceil(ELEMENT_BLOCK);
        let grid_x = u32::try_from(blocks).map_err(|_| PlanError::GridOverflow)?;
        Ok(Self {
            activation,
            bias,
            rows,
            width,
            input,
            output,
            bias_layout,
            elements,
            grid: [grid_x, 1, 1],
            block: [ELEMENT_BLOCK as u32, 1, 1],
        })
    }

    #[inline]
    pub const fn activation(&self) -> Activation {
        self.activation
    }

    #[inline]
    pub const fn bias(&self) -> BiasMode {
        self.bias
    }

    #[inline]
    pub const fn rounding(&self) -> Rounding {
        self.bias.rounding()
    }

    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    #[inline]
    pub const fn input_layout(&self) -> &Layout<2> {
        &self.input
    }

    #[inline]
    pub const fn output_layout(&self) -> &Layout<2> {
        &self.output
    }

    #[inline]
    pub const fn bias_layout(&self) -> Option<&Layout<1>> {
        self.bias_layout.as_ref()
    }

    #[inline]
    pub const fn grid(&self) -> [u32; 3] {
        self.grid
    }

    #[inline]
    pub const fn block(&self) -> [u32; 3] {
        self.block
    }

    #[inline]
    pub const fn export(&self) -> &'static str {
        self.activation.export()
    }

    /// Pure validation shared by the safe `TensorRef`/`TensorMut` binding in the
    /// oxide adapter. A safe caller cannot reach a raw launch without a real
    /// device view whose capacity and address span have already been checked.
    pub fn validate_bound_layouts(
        &self,
        input: &Layout<2>,
        output: &Layout<2>,
        bias: Option<&Layout<1>>,
        same_context: bool,
    ) -> Result<(), PlanError> {
        if input != &self.input || output != &self.output || bias != self.bias_layout.as_ref() {
            return Err(PlanError::LayoutMismatch);
        }
        if !same_context {
            return Err(PlanError::ContextMismatch);
        }
        if self.bias.has_bias() != bias.is_some() {
            return Err(if self.bias.has_bias() {
                PlanError::MissingBias
            } else {
                PlanError::UnexpectedBias
            });
        }
        Ok(())
    }

    /// Required active bytes, used by the adapter's capacity checks.
    #[inline]
    pub fn input_bytes(&self) -> Option<u64> {
        u64::try_from(self.input.span_elements())
            .ok()
            .and_then(|elements| elements.checked_mul(ELEMENT_BYTES))
    }

    #[inline]
    pub fn output_bytes(&self) -> Option<u64> {
        u64::try_from(self.output.span_elements())
            .ok()
            .and_then(|elements| elements.checked_mul(ELEMENT_BYTES))
    }

    #[inline]
    pub fn bias_bytes(&self) -> Option<u64> {
        self.bias_layout.and_then(|layout| {
            u64::try_from(layout.span_elements())
                .ok()
                .and_then(|elements| elements.checked_mul(ELEMENT_BYTES))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(activation: Activation, bias: BiasMode) -> ActivationPlan {
        ActivationPlan::padded(activation, bias, 3, 5, 8, 7).unwrap()
    }

    #[test]
    fn padded_layouts_are_exact() {
        let p = plan(Activation::GeluTanh, BiasMode::SeparateBf16);
        assert_eq!(p.grid(), [1, 1, 1]);
        assert_eq!(p.block(), [256, 1, 1]);
        assert_eq!(p.input_layout().strides(), &[8, 1]);
        assert_eq!(p.output_layout().strides(), &[7, 1]);
        assert_eq!(p.bias_layout().unwrap().shape(), &[5]);
        assert_eq!(p.export(), Activation::GeluTanh.export());
        assert_eq!(p.rounding(), Rounding::SeparateBiasThenSingleWrite);
        assert_eq!(
            ActivationPlan::activation_only(Activation::Relu, 3, 5, 8, 7)
                .unwrap()
                .bias(),
            BiasMode::None
        );
    }

    #[test]
    fn bad_layouts_are_rejected_cold() {
        let input = Layout::strided([2, 5], [5, 1]).unwrap();
        let output = Layout::strided([2, 5], [4, 1]).unwrap();
        assert_eq!(
            ActivationPlan::from_layouts(Activation::Relu, BiasMode::None, input, output, None),
            Err(PlanError::StrideTooSmall)
        );
        let input = Layout::strided([2, 5], [10, 2]).unwrap();
        let output = Layout::strided([2, 5], [5, 1]).unwrap();
        assert_eq!(
            ActivationPlan::from_layouts(Activation::Relu, BiasMode::None, input, output, None),
            Err(PlanError::NonUnitInnerStride)
        );
        let input = Layout::strided([2, 5], [5, 1]).unwrap();
        let bias = Some(Layout::strided([4], [1]).unwrap());
        assert_eq!(
            ActivationPlan::from_layouts(
                Activation::Relu,
                BiasMode::SeparateBf16,
                input,
                output,
                bias
            ),
            Err(PlanError::BiasLayout)
        );
        assert_eq!(
            ActivationPlan::from_layouts(
                Activation::Relu,
                BiasMode::None,
                input,
                output,
                Some(Layout::strided([5], [1]).unwrap())
            ),
            Err(PlanError::UnexpectedBias)
        );
    }

    #[test]
    fn zero_width_and_zero_rows_are_checked() {
        let input = Layout::strided([0, 5], [8, 1]).unwrap();
        let output = Layout::strided([0, 5], [8, 1]).unwrap();
        let p = ActivationPlan::from_layouts(Activation::Silu, BiasMode::None, input, output, None)
            .unwrap();
        assert_eq!(p.grid(), [0, 1, 1]);
        let zero_width = Layout::strided([3, 0], [8, 1]).unwrap();
        assert_eq!(
            ActivationPlan::from_layouts(
                Activation::Silu,
                BiasMode::None,
                zero_width,
                zero_width,
                None
            ),
            Err(PlanError::ZeroWidth)
        );
    }

    #[test]
    fn bound_layout_validation_rejects_mismatch_and_context_mix() {
        let p = plan(Activation::Relu, BiasMode::SeparateBf16);
        let input = *p.input_layout();
        let output = *p.output_layout();
        let bias = *p.bias_layout().unwrap();
        assert!(p
            .validate_bound_layouts(&input, &output, Some(&bias), true)
            .is_ok());
        assert_eq!(
            p.validate_bound_layouts(&input, &output, Some(&bias), false),
            Err(PlanError::ContextMismatch)
        );
        assert_eq!(
            p.validate_bound_layouts(
                &Layout::strided([3, 5], [5, 1]).unwrap(),
                &output,
                Some(&bias),
                true
            ),
            Err(PlanError::LayoutMismatch)
        );
        assert_eq!(
            p.validate_bound_layouts(&input, &output, None, true),
            Err(PlanError::LayoutMismatch)
        );
        assert_eq!(p.input_bytes(), Some(((3 - 1) * 8 + 5) * 2));
        assert_eq!(p.output_bytes(), Some(((3 - 1) * 7 + 5) * 2));
        assert_eq!(p.bias_bytes(), Some(5 * 2));
    }
}
