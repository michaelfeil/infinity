// SPDX-License-Identifier: MIT
// Paged export order is the native paged-mode ABI. Fields are identical to
// profiles.inc. These are the noncausal d256 GQA4 profiles used by block
// suffix attention; SM90 exports them through its non-TMA paged path and
// SM100 through the dedicated hd256 page-128 path.
FA4_PAGED_PROFILE(qwen35, 16, 4, 256, 0, 0)
FA4_PAGED_PROFILE(wide_gqa6, 24, 4, 256, 0, 0)
