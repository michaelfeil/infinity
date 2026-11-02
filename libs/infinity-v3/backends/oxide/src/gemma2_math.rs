// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Pure CPU admission and launch geometry for the eight EmbeddingGemma2
//! primitives. The plan constructors validate checked layouts; they never query
//! the driver, allocate device memory, enqueue work, or retain a context.
//!
//! Function resolution is the one cold driver-touching step:
//! [`PreparedGemma2::from_module`] calls `Module::load_function`, and the
//! retained `Function` values keep their `Arc<Module>` and `Arc<Context>` owner
//! alive. It runs only when a caller explicitly asks for the eight new exports.
//! Existing model startup and the generic kernel loader never look them up.

use crate::tensor::{Layout, TensorError};
use cuda_core::{Function, Module};
use half::bf16;
use std::{fmt, sync::Arc};

/// The exact eight exports appended by this leaf.
pub const EXPORTS: [&str; 8] = [
    "infinity_gemma2_norm_bf16_f32scale",
    "infinity_gemma2_residual_norm_bf16_f32scale",
    "infinity_gemma2_unit_norm_bf16",
    "infinity_gemma2_gated_gelu_bf16",
    "infinity_gemma2_ple_gate_bf16",
    "infinity_gemma2_scale_bf16",
    "infinity_gemma2_rope_tables_bf16",
    "infinity_gemma2_mean_pool_bf16",
];

/// Norm trees use exactly one thread per width lane.
pub const NORM_WIDTHS: [usize; 2] = [256, 512];
/// Mean pooling visits eight features per `blockIdx.y` tile.
pub const POOL_TILE: usize = 8;
/// Mean pooling clamps its reduction block to this width.
pub const MAX_POOL_THREADS: u32 = 1024;
const ELEMENT_BLOCK: usize = 256;
const MAX_GRID_X: u32 = i32::MAX as u32;
const MAX_GRID_Y: u32 = 65535;
const MAX_GRID_Z: u32 = 65535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma2Error {
    ZeroExtent,
    InvalidWidth,
    HeadRemainder,
    StrideTooSmall,
    ExtentOverflow,
    GridOverflow,
    InvalidEpsilon,
    InvalidScalar,
    InvalidLayer,
    InvalidSelection,
    MissingExport(&'static str),
}

impl fmt::Display for Gemma2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroExtent => f.write_str("Gemma2 operation has a zero extent"),
            Self::InvalidWidth => f.write_str("Gemma2 operation has an unsupported width"),
            Self::HeadRemainder => f.write_str("Gemma2 rows do not form whole attention heads"),
            Self::StrideTooSmall => f.write_str("Gemma2 token stride is smaller than one token"),
            Self::ExtentOverflow => f.write_str("Gemma2 element extent overflows its ABI"),
            Self::GridOverflow => f.write_str("Gemma2 launch grid exceeds CUDA limits"),
            Self::InvalidEpsilon => f.write_str("Gemma2 epsilon must be finite and positive"),
            Self::InvalidScalar => {
                f.write_str("Gemma2 scale must be finite and exactly BF16-rounded")
            }
            Self::InvalidLayer => f.write_str("Gemma2 PLE layer is outside the layer count"),
            Self::InvalidSelection => {
                f.write_str("Gemma2 mean selection has an invalid nonempty span")
            }
            Self::MissingExport(name) => write!(f, "Gemma2 module is missing {name}"),
        }
    }
}

impl std::error::Error for Gemma2Error {}

impl From<TensorError> for Gemma2Error {
    fn from(_: TensorError) -> Self {
        Self::ExtentOverflow
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchGeometry {
    grid: [u32; 3],
    block: [u32; 3],
}

impl LaunchGeometry {
    pub const fn new(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self { grid, block }
    }
    pub const fn grid(&self) -> [u32; 3] {
        self.grid
    }
    pub const fn block(&self) -> [u32; 3] {
        self.block
    }
}

fn positive(value: usize) -> Result<(), Gemma2Error> {
    if value == 0 {
        Err(Gemma2Error::ZeroExtent)
    } else {
        Ok(())
    }
}

fn dim(value: usize) -> Result<u32, Gemma2Error> {
    u32::try_from(value).map_err(|_| Gemma2Error::ExtentOverflow)
}

fn grid_x(value: usize) -> Result<u32, Gemma2Error> {
    let value = dim(value)?;
    if value > MAX_GRID_X {
        Err(Gemma2Error::GridOverflow)
    } else {
        Ok(value)
    }
}

fn grid_y(value: usize) -> Result<u32, Gemma2Error> {
    let value = dim(value)?;
    if value > MAX_GRID_Y {
        Err(Gemma2Error::GridOverflow)
    } else {
        Ok(value)
    }
}

fn grid_geometry(grid: [u32; 3], block: [u32; 3]) -> Result<LaunchGeometry, Gemma2Error> {
    if grid[0] > MAX_GRID_X || grid[1] > MAX_GRID_Y || grid[2] > MAX_GRID_Z {
        Err(Gemma2Error::GridOverflow)
    } else {
        Ok(LaunchGeometry::new(grid, block))
    }
}

fn elements(rows: usize, width: usize) -> Result<usize, Gemma2Error> {
    rows.checked_mul(width).ok_or(Gemma2Error::ExtentOverflow)
}

fn layout<const N: usize>(shape: [usize; N]) -> Result<Layout<N>, Gemma2Error> {
    Layout::contiguous(shape).map_err(Into::into)
}

fn epsilon(value: f32) -> Result<f32, Gemma2Error> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(Gemma2Error::InvalidEpsilon)
    }
}

/// True when an F32 value is exactly one widened BF16 scalar.
pub fn is_bf16_rounded_f32(value: f32) -> bool {
    value.is_finite() && bf16::from_f32(value).to_f32().to_bits() == value.to_bits()
}

fn bf16_scalar(value: f32) -> Result<f32, Gemma2Error> {
    if is_bf16_rounded_f32(value) {
        Ok(value)
    } else {
        Err(Gemma2Error::InvalidScalar)
    }
}

fn element_geometry(count: usize) -> Result<LaunchGeometry, Gemma2Error> {
    grid_geometry(
        [grid_x(count.div_ceil(ELEMENT_BLOCK))?, 1, 1],
        [ELEMENT_BLOCK as u32, 1, 1],
    )
}

/// One-row-per-CTA DivSqrt norm, shared by `norm` and `unit_norm`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormPlan {
    rows: u32,
    width: u32,
    heads: u32,
    token_stride: u64,
    eps: f32,
    input: Layout<3>,
    output: Layout<2>,
}

impl NormPlan {
    pub fn norm(
        rows: usize,
        width: usize,
        heads: usize,
        token_stride: usize,
        eps: f32,
    ) -> Result<Self, Gemma2Error> {
        Self::new(rows, width, heads, token_stride, eps)
    }

    pub fn unit_norm(
        rows: usize,
        width: usize,
        heads: usize,
        token_stride: usize,
        eps: f32,
    ) -> Result<Self, Gemma2Error> {
        Self::new(rows, width, heads, token_stride, eps)
    }

    fn new(
        rows: usize,
        width: usize,
        heads: usize,
        token_stride: usize,
        eps: f32,
    ) -> Result<Self, Gemma2Error> {
        positive(rows)?;
        positive(width)?;
        positive(heads)?;
        let grid_rows = grid_x(rows)?;
        if !NORM_WIDTHS.contains(&width) {
            return Err(Gemma2Error::InvalidWidth);
        }
        if !rows.is_multiple_of(heads) {
            return Err(Gemma2Error::HeadRemainder);
        }
        let token_width = elements(heads, width)?;
        if token_stride < token_width {
            return Err(Gemma2Error::StrideTooSmall);
        }
        let tokens = rows / heads;
        let input = Layout::strided([tokens, heads, width], [token_stride, width, 1])
            .map_err(|_| Gemma2Error::ExtentOverflow)?;
        Ok(Self {
            rows: grid_rows,
            width: dim(width)?,
            heads: dim(heads)?,
            token_stride: u64::try_from(token_stride).map_err(|_| Gemma2Error::ExtentOverflow)?,
            eps: epsilon(eps)?,
            input,
            output: layout([rows, width])?,
        })
    }

    pub const fn geometry(&self) -> LaunchGeometry {
        LaunchGeometry::new([self.rows, 1, 1], [self.width, 1, 1])
    }
    pub const fn rows(&self) -> u32 {
        self.rows
    }
    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn heads(&self) -> u32 {
        self.heads
    }
    pub const fn token_stride(&self) -> u64 {
        self.token_stride
    }
    pub const fn eps(&self) -> f32 {
        self.eps
    }
    pub const fn input_layout(&self) -> Layout<3> {
        self.input
    }
    pub const fn output_layout(&self) -> Layout<2> {
        self.output
    }
}

/// Residual DivSqrt norm with a contiguous width-512 residual add.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualPlan {
    rows: u32,
    width: u32,
    eps: f32,
    input: Layout<2>,
    output: Layout<2>,
}

impl ResidualPlan {
    pub fn new(rows: usize, width: usize, eps: f32) -> Result<Self, Gemma2Error> {
        positive(rows)?;
        positive(width)?;
        let grid_rows = grid_x(rows)?;
        if width != 512 {
            return Err(Gemma2Error::InvalidWidth);
        }
        Ok(Self {
            rows: grid_rows,
            width: dim(width)?,
            eps: epsilon(eps)?,
            input: layout([rows, width])?,
            output: layout([rows, width])?,
        })
    }
    pub const fn geometry(&self) -> LaunchGeometry {
        LaunchGeometry::new([self.rows, 1, 1], [self.width, 1, 1])
    }
    pub const fn rows(&self) -> u32 {
        self.rows
    }
    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn eps(&self) -> f32 {
        self.eps
    }
    pub const fn input_layout(&self) -> Layout<2> {
        self.input
    }
    pub const fn output_layout(&self) -> Layout<2> {
        self.output
    }
}

/// Gate-first packed MLP activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedPlan {
    rows: u32,
    width: u32,
    count: usize,
    input: Layout<3>,
    output: Layout<2>,
}

impl GatedPlan {
    pub fn new(rows: usize, width: usize) -> Result<Self, Gemma2Error> {
        positive(rows)?;
        positive(width)?;
        Ok(Self {
            rows: dim(rows)?,
            width: dim(width)?,
            count: elements(rows, width)?,
            input: layout([rows, 2, width])?,
            output: layout([rows, width])?,
        })
    }
    pub fn geometry(&self) -> Result<LaunchGeometry, Gemma2Error> {
        element_geometry(self.count)
    }
    pub const fn rows(&self) -> u32 {
        self.rows
    }
    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn count(&self) -> usize {
        self.count
    }
    pub const fn input_layout(&self) -> Layout<3> {
        self.input
    }
    pub const fn output_layout(&self) -> Layout<2> {
        self.output
    }
}

/// Per-layer PLE gate activation against the token's packed layer slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlePlan {
    rows: u32,
    width: u32,
    layers: u32,
    layer: u32,
    count: usize,
    gate: Layout<2>,
    ple: Layout<3>,
    output: Layout<2>,
}

impl PlePlan {
    pub fn new(
        rows: usize,
        width: usize,
        layers: usize,
        layer: usize,
    ) -> Result<Self, Gemma2Error> {
        positive(rows)?;
        positive(width)?;
        positive(layers)?;
        if layer >= layers {
            return Err(Gemma2Error::InvalidLayer);
        }
        Ok(Self {
            rows: dim(rows)?,
            width: dim(width)?,
            layers: dim(layers)?,
            layer: dim(layer)?,
            count: elements(rows, width)?,
            gate: layout([rows, width])?,
            ple: layout([rows, layers, width])?,
            output: layout([rows, width])?,
        })
    }
    pub fn geometry(&self) -> Result<LaunchGeometry, Gemma2Error> {
        element_geometry(self.count)
    }
    pub const fn rows(&self) -> u32 {
        self.rows
    }
    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn layers(&self) -> u32 {
        self.layers
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn gate_layout(&self) -> Layout<2> {
        self.gate
    }
    pub const fn ple_layout(&self) -> Layout<3> {
        self.ple
    }
    pub const fn output_layout(&self) -> Layout<2> {
        self.output
    }
    pub const fn ple_elements(&self) -> usize {
        self.ple.num_elements()
    }
}

/// Pointwise BF16 multiply by one already-rounded F32 scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalePlan {
    count: u32,
    scale: f32,
    input: Layout<1>,
    output: Layout<1>,
}

impl ScalePlan {
    pub fn new(count: usize, scale: f32) -> Result<Self, Gemma2Error> {
        positive(count)?;
        Ok(Self {
            count: dim(count)?,
            scale: bf16_scalar(scale)?,
            input: layout([count])?,
            output: layout([count])?,
        })
    }
    pub fn geometry(&self) -> Result<LaunchGeometry, Gemma2Error> {
        element_geometry(self.count as usize)
    }
    pub const fn count(&self) -> u32 {
        self.count
    }
    pub const fn scale(&self) -> f32 {
        self.scale
    }
    pub const fn input_layout(&self) -> Layout<1> {
        self.input
    }
    pub const fn output_layout(&self) -> Layout<1> {
        self.output
    }
}

/// Full-table RoPE initializer geometry. Outputs stay F32 but hold one BF16
/// round-trip of the libdevice sine/cosine value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopePlan {
    positions: u32,
    pairs: u32,
    count: usize,
    inverse: Layout<1>,
    cosine: Layout<2>,
    sine: Layout<2>,
}

impl RopePlan {
    pub fn new(positions: usize, pairs: usize) -> Result<Self, Gemma2Error> {
        positive(positions)?;
        positive(pairs)?;
        Ok(Self {
            positions: dim(positions)?,
            pairs: dim(pairs)?,
            count: elements(positions, pairs)?,
            inverse: layout([pairs])?,
            cosine: layout([positions, pairs])?,
            sine: layout([positions, pairs])?,
        })
    }
    pub fn geometry(&self) -> Result<LaunchGeometry, Gemma2Error> {
        element_geometry(self.count)
    }
    pub const fn positions(&self) -> u32 {
        self.positions
    }
    pub const fn pairs(&self) -> u32 {
        self.pairs
    }
    pub const fn full_table_elements(&self) -> usize {
        self.count
    }
    pub const fn inverse_layout(&self) -> Layout<1> {
        self.inverse
    }
    pub const fn cosine_layout(&self) -> Layout<2> {
        self.cosine
    }
    pub const fn sine_layout(&self) -> Layout<2> {
        self.sine
    }
}

/// Selected-span BF16 mean pooling widened to F32. Duplicate and unsorted
/// selections are valid and keep their output order. The reduction block is a
/// power of two in `POOL_TILE..=MAX_POOL_THREADS`, so every feature store lane
/// stays populated even when the longest selected span is shorter than a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeanPlan {
    count: u32,
    width: u32,
    feature_tiles: u32,
    threads: u32,
    input_rows: u32,
    hidden: Layout<2>,
    cu_seqlens: Layout<1>,
    selected: Layout<1>,
    output: Layout<2>,
}

impl MeanPlan {
    pub fn new(
        input_rows: usize,
        width: usize,
        cu_seqlens: &[u32],
        selected: &[u32],
    ) -> Result<Self, Gemma2Error> {
        positive(input_rows)?;
        positive(width)?;
        positive(selected.len())?;
        let count = grid_x(selected.len())?;
        let feature_tiles = grid_y(width.div_ceil(POOL_TILE))?;
        if cu_seqlens.len() < 2 {
            return Err(Gemma2Error::InvalidSelection);
        }
        let mut longest = 0usize;
        for &index in selected {
            let slot = usize::try_from(index).map_err(|_| Gemma2Error::InvalidSelection)?;
            let next = slot.checked_add(1).ok_or(Gemma2Error::InvalidSelection)?;
            let start = *cu_seqlens.get(slot).ok_or(Gemma2Error::InvalidSelection)?;
            let end = *cu_seqlens.get(next).ok_or(Gemma2Error::InvalidSelection)?;
            if end <= start {
                return Err(Gemma2Error::InvalidSelection);
            }
            let start = usize::try_from(start).map_err(|_| Gemma2Error::InvalidSelection)?;
            let end = usize::try_from(end).map_err(|_| Gemma2Error::InvalidSelection)?;
            if end > input_rows {
                return Err(Gemma2Error::InvalidSelection);
            }
            longest = longest.max(end - start);
        }
        let threads = longest
            .min(MAX_POOL_THREADS as usize)
            .next_power_of_two()
            .max(POOL_TILE);
        Ok(Self {
            count,
            width: dim(width)?,
            threads: dim(threads)?,
            input_rows: dim(input_rows)?,
            hidden: layout([input_rows, width])?,
            cu_seqlens: layout([cu_seqlens.len()])?,
            selected: layout([selected.len()])?,
            output: layout([selected.len(), width])?,
            feature_tiles,
        })
    }
    pub fn geometry(&self) -> Result<LaunchGeometry, Gemma2Error> {
        grid_geometry([self.count, self.feature_tiles, 1], [self.threads, 1, 1])
    }
    pub const fn count(&self) -> u32 {
        self.count
    }
    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn threads(&self) -> u32 {
        self.threads
    }
    pub const fn input_rows(&self) -> u32 {
        self.input_rows
    }
    pub const fn hidden_layout(&self) -> Layout<2> {
        self.hidden
    }
    pub const fn cu_seqlens_layout(&self) -> Layout<1> {
        self.cu_seqlens
    }
    pub const fn selected_layout(&self) -> Layout<1> {
        self.selected
    }
    pub const fn output_layout(&self) -> Layout<2> {
        self.output
    }
}

/// Explicit cold resolution of the eight exports. Each resolved `Function`
/// retains its owning `Arc<Module>` and therefore its `Arc<Context>` through
/// cuda-core. Only this resolver touches the driver; plan constructors remain
/// pure CPU validation, and old startup never invokes either path.
pub struct PreparedGemma2 {
    pub norm: Function,
    pub residual: Function,
    pub unit_norm: Function,
    pub gated: Function,
    pub ple: Function,
    pub scale: Function,
    pub rope: Function,
    pub mean: Function,
}

impl PreparedGemma2 {
    /// Resolves all eight exports from an already-open module. This is the only
    /// driver-touching preparation step; generic loading never calls it.
    pub fn from_module(module: &Arc<Module>) -> Result<Self, Gemma2Error> {
        Self::resolve(|name| module.load_function(name))
    }

    fn resolve<E>(mut load: impl FnMut(&str) -> Result<Function, E>) -> Result<Self, Gemma2Error> {
        let mut take =
            |name: &'static str| load(name).map_err(|_| Gemma2Error::MissingExport(name));
        Ok(Self {
            norm: take(EXPORTS[0])?,
            residual: take(EXPORTS[1])?,
            unit_norm: take(EXPORTS[2])?,
            gated: take(EXPORTS[3])?,
            ple: take(EXPORTS[4])?,
            scale: take(EXPORTS[5])?,
            rope: take(EXPORTS[6])?,
            mean: take(EXPORTS[7])?,
        })
    }
}

/// First required export absent from an old bundle's names.
pub fn missing_export<'a>(available: impl IntoIterator<Item = &'a str>) -> Option<&'static str> {
    let available: std::collections::BTreeSet<&str> = available.into_iter().collect();
    EXPORTS.into_iter().find(|name| !available.contains(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;

    const BASE_LEGACY_FINGERPRINT: &str =
        "cc49b1fce76e3d09441cdc4e5ea77ae4a210334ff386aef8adcb848b6016bd32";
    const GEMMA2_SOURCE_SHA256: &str =
        "a32363d7f4017e061b90dd118841b4cde4cff0df244cbc2942d149d23c45c82c";

    fn hex(bytes: &[u8]) -> String {
        let mut text = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            let _ = write!(text, "{byte:02x}");
        }
        text
    }

    fn build_symbols() -> std::collections::BTreeSet<&'static str> {
        include_str!("../device/build.sh")
            .split('"')
            .filter(|value| {
                value.starts_with("infinity_")
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            })
            .collect()
    }

    #[test]
    fn norm_widths_and_padded_head_stride_are_exact() {
        let dense = NormPlan::norm(8, 256, 4, 1024, 1e-6).unwrap();
        assert_eq!(dense.geometry().grid(), [8, 1, 1]);
        assert_eq!(dense.geometry().block(), [256, 1, 1]);
        assert_eq!(dense.input_layout().shape(), &[2, 4, 256]);
        assert_eq!(dense.input_layout().strides(), &[1024, 256, 1]);
        assert_eq!(dense.output_layout().shape(), &[8, 256]);
        assert_eq!(dense.token_stride(), 1024);

        let padded = NormPlan::unit_norm(8, 512, 4, 2560, 1e-6).unwrap();
        assert_eq!(padded.input_layout().strides(), &[2560, 512, 1]);
        assert_eq!(padded.geometry().block(), [512, 1, 1]);

        assert_eq!(
            NormPlan::norm(8, 128, 4, 512, 1e-6),
            Err(Gemma2Error::InvalidWidth)
        );
        assert_eq!(
            NormPlan::norm(8, 768, 4, 3072, 1e-6),
            Err(Gemma2Error::InvalidWidth)
        );
        assert_eq!(
            NormPlan::norm(7, 256, 4, 1024, 1e-6),
            Err(Gemma2Error::HeadRemainder)
        );
        assert_eq!(
            NormPlan::norm(8, 256, 4, 1000, 1e-6),
            Err(Gemma2Error::StrideTooSmall)
        );
    }

    #[test]
    fn residual_is_width_512_and_contiguous() {
        let plan = ResidualPlan::new(3, 512, 1e-6).unwrap();
        assert_eq!(plan.geometry().grid(), [3, 1, 1]);
        assert_eq!(plan.geometry().block(), [512, 1, 1]);
        assert_eq!(plan.input_layout().shape(), &[3, 512]);
        assert_eq!(plan.output_layout().strides(), &[512, 1]);
        assert_eq!(
            ResidualPlan::new(3, 256, 1e-6),
            Err(Gemma2Error::InvalidWidth)
        );
    }

    #[test]
    fn element_plans_round_counts_to_256_and_catch_overflow() {
        let gated = GatedPlan::new(3, 100).unwrap();
        assert_eq!(gated.geometry().unwrap().grid(), [2, 1, 1]);
        assert_eq!(gated.input_layout().shape(), &[3, 2, 100]);
        assert_eq!(gated.output_layout().shape(), &[3, 100]);

        let ple = PlePlan::new(3, 100, 2, 1).unwrap();
        assert_eq!(ple.geometry().unwrap().grid(), [2, 1, 1]);
        assert_eq!(ple.ple_layout().shape(), &[3, 2, 100]);
        assert_eq!(ple.ple_elements(), 600);

        let scale = ScalePlan::new(300, 0.5).unwrap();
        assert_eq!(scale.geometry().unwrap().grid(), [2, 1, 1]);

        let rope = RopePlan::new(17, 32).unwrap();
        assert_eq!(rope.full_table_elements(), 544);
        assert_eq!(rope.geometry().unwrap().grid(), [3, 1, 1]);
        assert_eq!(rope.cosine_layout().shape(), &[17, 32]);

        assert_eq!(
            GatedPlan::new(usize::MAX, 2),
            Err(Gemma2Error::ExtentOverflow)
        );
        assert_eq!(
            PlePlan::new(usize::MAX, 2, 2, 0),
            Err(Gemma2Error::ExtentOverflow)
        );
        assert_eq!(
            RopePlan::new(usize::MAX, 2),
            Err(Gemma2Error::ExtentOverflow)
        );
    }

    #[test]
    fn epsilon_and_scale_require_finite_exact_values() {
        for eps in [f32::NAN, f32::INFINITY, 0.0, -1e-6] {
            assert_eq!(
                NormPlan::norm(8, 256, 4, 1024, eps),
                Err(Gemma2Error::InvalidEpsilon)
            );
            assert_eq!(
                ResidualPlan::new(8, 512, eps),
                Err(Gemma2Error::InvalidEpsilon)
            );
        }
        for scale in [f32::NAN, f32::INFINITY, 1.000_1] {
            assert_eq!(ScalePlan::new(8, scale), Err(Gemma2Error::InvalidScalar));
        }
        assert!(ScalePlan::new(8, f32::from_bits(0x3f80_0000)).is_ok());
        let rounded = half::bf16::from_f32(0.1).to_f32();
        assert!(is_bf16_rounded_f32(rounded));
        assert!(!is_bf16_rounded_f32(0.1));
    }

    #[test]
    fn ple_layer_must_be_inside_the_declared_layers() {
        assert_eq!(PlePlan::new(4, 256, 4, 4), Err(Gemma2Error::InvalidLayer));
        assert_eq!(PlePlan::new(4, 256, 0, 0), Err(Gemma2Error::ZeroExtent));
        let last = PlePlan::new(4, 256, 4, 3).unwrap();
        assert_eq!(last.layer(), 3);
        assert_eq!(last.layers(), 4);
    }

    #[test]
    fn mean_accepts_duplicates_and_arbitrary_order() {
        let cu = [0, 3, 4, 8, 10, 15];
        let selected = [0, 2, 2, 4, 1, 3, 4];
        let plan = MeanPlan::new(15, 20, &cu, &selected).unwrap();
        assert_eq!(plan.count(), 7);
        assert_eq!(plan.threads(), 8);
        assert_eq!(plan.geometry().unwrap().grid(), [7, 3, 1]);
        assert_eq!(plan.geometry().unwrap().block(), [8, 1, 1]);
        assert_eq!(plan.output_layout().shape(), &[7, 20]);
    }

    #[test]
    fn mean_short_spans_keep_full_feature_tile_lanes() {
        for longest in 1..=4usize {
            for width in [9usize, 16, 17] {
                let cu = [0, longest as u32];
                let plan = MeanPlan::new(longest, width, &cu, &[0]).unwrap();
                let geometry = plan.geometry().unwrap();
                assert!(geometry.block()[0] >= POOL_TILE as u32);
                assert_eq!(geometry.block(), [POOL_TILE as u32, 1, 1]);
                assert_eq!(geometry.grid(), [1, width.div_ceil(POOL_TILE) as u32, 1]);
            }
        }

        let cu = [0, 1, 2, 3];
        let ordered = MeanPlan::new(3, 9, &cu, &[2, 0, 2, 1]).unwrap();
        assert_eq!(ordered.count(), 4);
        assert_eq!(ordered.geometry().unwrap().grid(), [4, 2, 1]);
        assert_eq!(ordered.threads(), POOL_TILE as u32);

        let long = MeanPlan::new(5000, 9, &[0, 5000], &[0]).unwrap();
        assert_eq!(long.threads(), MAX_POOL_THREADS);
        assert_eq!(long.geometry().unwrap().grid(), [1, 2, 1]);
    }

    #[test]
    fn mean_rejects_empty_and_out_of_range_spans() {
        assert_eq!(
            MeanPlan::new(15, 20, &[0, 1], &[]),
            Err(Gemma2Error::ZeroExtent)
        );
        assert_eq!(
            MeanPlan::new(15, 20, &[0], &[0]),
            Err(Gemma2Error::InvalidSelection)
        );
        assert_eq!(
            MeanPlan::new(15, 20, &[0, 1], &[1]),
            Err(Gemma2Error::InvalidSelection)
        );
        assert_eq!(
            MeanPlan::new(15, 20, &[0, 2, 2], &[1]),
            Err(Gemma2Error::InvalidSelection)
        );
        assert_eq!(
            MeanPlan::new(15, 20, &[0, 16], &[0]),
            Err(Gemma2Error::InvalidSelection)
        );
        assert_eq!(
            MeanPlan::new(0, 20, &[0, 1], &[0]),
            Err(Gemma2Error::ZeroExtent)
        );
    }

    #[test]
    fn mean_threads_are_power_of_two_and_clamped() {
        let wide = MeanPlan::new(5000, 8, &[0, 5000], &[0]).unwrap();
        assert_eq!(wide.threads(), MAX_POOL_THREADS);
        assert!(wide.threads().is_power_of_two());
        let long = MeanPlan::new(3080, 8, &[0, 1, 1024, 2048, 3073, 3080], &[1, 4]).unwrap();
        assert_eq!(long.threads(), 1024);
        assert!(long.threads() <= MAX_POOL_THREADS);
    }

    #[test]
    fn cuda_grid_limits_are_rejected_before_a_plan_is_valid() {
        assert_eq!(grid_x(i32::MAX as usize), Ok(i32::MAX as u32));
        assert_eq!(
            grid_x(i32::MAX as usize + 1),
            Err(Gemma2Error::GridOverflow)
        );
        assert_eq!(grid_x(usize::MAX), Err(Gemma2Error::ExtentOverflow));
        assert_eq!(grid_y(65535), Ok(65535));
        assert_eq!(grid_y(65536), Err(Gemma2Error::GridOverflow));
        assert_eq!(
            grid_geometry([i32::MAX as u32 + 1, 1, 1], [1, 1, 1]),
            Err(Gemma2Error::GridOverflow)
        );
        assert_eq!(
            grid_geometry([1, 65536, 1], [1, 1, 1]),
            Err(Gemma2Error::GridOverflow)
        );
        assert_eq!(
            NormPlan::norm(i32::MAX as usize + 1, 256, 1, 256, 1e-6),
            Err(Gemma2Error::GridOverflow)
        );
        assert_eq!(
            ResidualPlan::new(i32::MAX as usize + 1, 512, 1e-6),
            Err(Gemma2Error::GridOverflow)
        );

        let boundary_width = POOL_TILE * MAX_GRID_Y as usize;
        let boundary = MeanPlan::new(1, boundary_width, &[0, 1], &[0]).unwrap();
        assert_eq!(boundary.geometry().unwrap().grid(), [1, MAX_GRID_Y, 1]);
        assert_eq!(
            MeanPlan::new(1, boundary_width + 1, &[0, 1], &[0]),
            Err(Gemma2Error::GridOverflow)
        );
    }

    #[test]
    fn every_zero_extent_is_rejected_at_admission() {
        assert_eq!(
            NormPlan::norm(0, 256, 4, 1024, 1e-6),
            Err(Gemma2Error::ZeroExtent)
        );
        assert_eq!(
            NormPlan::norm(8, 0, 4, 0, 1e-6),
            Err(Gemma2Error::ZeroExtent)
        );
        assert_eq!(
            NormPlan::norm(8, 256, 0, 1024, 1e-6),
            Err(Gemma2Error::ZeroExtent)
        );
        assert_eq!(
            ResidualPlan::new(0, 512, 1e-6),
            Err(Gemma2Error::ZeroExtent)
        );
        assert_eq!(GatedPlan::new(0, 1), Err(Gemma2Error::ZeroExtent));
        assert_eq!(PlePlan::new(0, 256, 1, 0), Err(Gemma2Error::ZeroExtent));
        assert_eq!(ScalePlan::new(0, 0.5), Err(Gemma2Error::ZeroExtent));
        assert_eq!(RopePlan::new(0, 32), Err(Gemma2Error::ZeroExtent));
    }

    #[test]
    fn resolver_reports_a_typed_missing_new_export() {
        let error = PreparedGemma2::resolve::<&str>(|_| Err("absent")).err();
        assert_eq!(error, Some(Gemma2Error::MissingExport(EXPORTS[0])));

        assert_eq!(missing_export(EXPORTS), None);
        assert_eq!(
            missing_export(EXPORTS[..7].iter().copied()),
            Some(EXPORTS[7])
        );
        let old_names = build_symbols()
            .into_iter()
            .filter(|name| !name.starts_with("infinity_gemma2_"));
        assert_eq!(missing_export(old_names), Some(EXPORTS[0]));
    }

    #[test]
    fn source_inventory_keeps_base_78_with_additive_encoder_heads() {
        const HEAD_ADDITIONS: [&str; 5] = [
            "infinity_deberta_embedding_compose",
            "infinity_deberta_conv_768",
            "infinity_deberta_plain_norm_bf16_aten",
            "infinity_plain_layer_norm_768_bf16_aten",
            "infinity_splade_pool_bf16_f32",
        ];
        let symbols = build_symbols();
        let gemma3_count = symbols
            .iter()
            .filter(|name| name.starts_with("infinity_gemma3_"))
            .count();
        let media_count = symbols
            .iter()
            .filter(|name| name.starts_with("infinity_gemma2_media_"))
            .count();
        let gemma4_count = symbols
            .iter()
            .filter(|name| name.starts_with("infinity_gemma4_"))
            .count();
        // Keep the original 91-export base dynamic and add every declared
        // family leaf independently. Do not replace the base count with a
        // guessed grand total.
        assert_eq!(
            symbols.len(),
            91 + gemma3_count + media_count + gemma4_count
        );
        assert_eq!(media_count, 24);
        assert_eq!(gemma4_count, 20);
        let gemma2: Vec<_> = symbols
            .iter()
            .copied()
            .filter(|name| {
                name.starts_with("infinity_gemma2_") && !name.starts_with("infinity_gemma2_media_")
            })
            .collect();
        assert_eq!(gemma2.len(), EXPORTS.len());
        for name in EXPORTS {
            assert!(symbols.contains(name));
        }
        for name in HEAD_ADDITIONS {
            assert!(symbols.contains(name));
        }
        let legacy: Vec<_> = symbols
            .iter()
            .copied()
            .filter(|name| {
                !name.starts_with("infinity_gemma2_")
                    && !name.starts_with("infinity_gemma3_")
                    && !name.starts_with("infinity_gemma4_")
                    && !HEAD_ADDITIONS.contains(name)
            })
            .collect();
        assert_eq!(legacy.len(), 78);
        assert_eq!(
            hex(&Sha256::digest(legacy.join("\n").as_bytes())),
            BASE_LEGACY_FINGERPRINT
        );

        let main = include_str!("../device/src/main.rs");
        assert_eq!(
            main.matches("pub unsafe fn infinity_gemma2_").count()
                - main.matches("pub unsafe fn infinity_gemma2_media_").count(),
            EXPORTS.len()
        );
        assert_eq!(
            main.matches("pub unsafe fn infinity_gemma2_media_").count(),
            24
        );
        assert_eq!(main.matches("pub unsafe fn infinity_gemma4_").count(), 20);
        assert!(main.contains("mod gemma2;"));
        assert!(main.contains("mod gemma2_media;"));
        assert!(main.contains("mod gemma4;"));
        assert!(main.contains("mod gemma4_vision;"));
        assert!(main.contains("pub unsafe fn infinity_gemma4_router_scale_bf16_to_f32"));
        assert!(main.contains("pub unsafe fn infinity_gemma4_vision_pool_standardize_rows_bf16"));
        let kernel = include_str!("../device/src/gemma2.rs");
        assert_eq!(
            hex(&Sha256::digest(kernel.as_bytes())),
            GEMMA2_SOURCE_SHA256
        );
    }
}
