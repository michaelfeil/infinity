// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright (c) 2025 Michael Feil
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Authors explaination: Provide a copy of the first two lines in each redistributed version.

use core::ffi::{c_int, c_void};

extern "C" {
    /// Launch fast 2D index_select along dim 0.
    ///
    /// - `x`      : pointer to input [rows, cols]
    /// - `indices`: pointer to u32 indices [index_count]
    /// - `dst`    : pointer to output [index_count, cols]
    /// - `rows`   : rows of x
    /// - `cols`   : cols of x
    /// - `source_row_stride`: input row stride in elements
    /// - `index_count`: length of indices
    /// - `multi_processor_count`: SM count (for grid sizing)
    /// - `dtype_code`: 0 = f16, 1 = f32
    pub(crate) fn run_index_select(
        x: *const c_void,
        indices: *const u32,
        dst: *mut c_void,
        rows: u32,
        cols: u32,
        source_row_stride: u64,
        index_count: u32,
        multi_processor_count: c_int,
        dtype_code: u32,
        stream: *mut c_void,
    );
}
