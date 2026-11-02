// SPDX-License-Identifier: Apache-2.0
//
// This device module is derived from the publicly available fast_kernels.cu
// source and is distributed under the Apache License, Version 2.0:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Rounded BF16 gated GELU for packed `[up(width), gate(width)]` rows.
//!
//! The gate lives in the second packed half. Every intermediate is rounded to
//! BF16; only the FMA and `tanhf` internals observe FP32 values. Inputs and
//! outputs are disjoint, and the host validates dimensions before launch.

use crate::{bf16_from_f32, f32_from_bf16, read, write};
use cuda_device::{float, thread};

/// Round one FP32 result through the native BF16 conversion.
#[inline(always)]
fn rounded_bf16(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

/// Apply rounded BF16 GELU to the gate half and multiply it by the up half.
/// Grid=ceil(rows*width/256), block=256; one scalar output per thread.
#[inline(always)]
pub(crate) unsafe fn gated_gelu_up_first(input: *const u16, out: *mut u16, rows: u32, width: u32) {
    if width == 0 {
        return;
    }
    let width = width as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= rows as usize * width {
        return;
    }
    let row = index / width;
    let column = index % width;
    let packed = row * width * 2 + column;
    let gate = unsafe { read(input, packed + width) };
    let up = unsafe { read(input, packed) };

    let square = rounded_bf16(gate * gate);
    let cube = rounded_bf16(square * gate);
    let alpha = rounded_bf16(float::fma_rn_f32(rounded_bf16(0.044715), cube, gate));
    let argument = rounded_bf16(rounded_bf16(0.797_884_6) * alpha);
    let tangent = rounded_bf16(libm::tanhf(argument));
    let half_gate = rounded_bf16(0.5 * gate);
    let plus = rounded_bf16(1.0 + tangent);
    let active = rounded_bf16(half_gate * plus);
    unsafe { write(out, index, active * up) };
}
