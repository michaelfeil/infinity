#!/usr/bin/env python3
"""CPU reference for the exact ATen 5c488 CUDA BF16 sum(dim=0) order.

Only the arithmetic order derived from Reduce.cuh is modelled:
  * FP32 additions with IEEE round-to-nearest-even,
  * one FP32->BF16 RNE projection after the whole sum,
  * the candidate's FP32 divide by T, then one BF16 RNE projection.

No approximations and no NumPy/Torch dependency. Run with:
  taskset -c 152-159 python3 reference_vectors.py
"""

import struct

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def widen(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]

def bf16_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if (bits & 0x7F80_0000) == 0x7F80_0000 and (bits & 0x007F_FFFF) != 0:
        return (((bits >> 16) & 0xFFFF) | 0x0040) & 0xFFFF
    retained_lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + retained_lsb) >> 16) & 0xFFFF

def add(a, b):
    return f32(a + b)

def thread_reduce(x, first_row, stride):
    """Reduce.cuh thread_reduce_impl: vt0=4 accumulators, then left fold."""
    total = len(x)
    acc = [0.0, 0.0, 0.0, 0.0]
    row = first_row
    while row + 3 * stride < total:
        acc[0] = add(acc[0], x[row])
        acc[1] = add(acc[1], x[row + stride])
        acc[2] = add(acc[2], x[row + 2 * stride])
        acc[3] = add(acc[3], x[row + 3 * stride])
        row += 4 * stride
    i = 0
    while i < 4 and row < total:
        acc[i] = add(acc[i], x[row])
        row += stride
        i += 1
    acc[0] = add(acc[0], acc[1])
    acc[0] = add(acc[0], acc[2])
    acc[0] = add(acc[0], acc[3])
    return acc[0]

def aten_sum_bits(x, ctas=1):
    """Exact per-feature schedule for T rows, contiguous [T,H], BF16."""
    total = len(x)
    if total < 64:
        # One thread owns the whole feature; output_mult.y != 0.
        return bf16_bits(thread_reduce(x, 0, 1))
    if ctas <= 1:
        # 64 <= T < 1021, or a hypothetical ctas_per_output == 1:
        # four warps split rows by threadIdx.y, then block_y halving tree.
        v = [thread_reduce(x, y, 4) for y in range(4)]
        return bf16_bits(add(add(v[0], v[2]), add(v[1], v[3])))
    # T >= 1021 with global reduction: C CTAs each produce q_cta,
    # then the last CTA folds q_cta by cta % 4 and applies block_y again.
    q = []
    for cta in range(ctas):
        v = [thread_reduce(x, y + 4 * cta, 4 * ctas) for y in range(4)]
        q.append(add(add(v[0], v[2]), add(v[1], v[3])))
    u = [0.0] * 4
    for y in range(4):
        cta = y
        while cta < ctas:
            u[y] = add(u[y], q[cta])
            cta += 4
    return bf16_bits(add(add(u[0], u[2]), add(u[1], u[3])))

def sequential_bits(x):
    total = 0.0
    for value in x:
        total = add(total, value)
    return bf16_bits(total)

def mean_bits(sum_bits, rows):
    return bf16_bits(f32(widen(sum_bits) / f32(float(rows))))

def check(name, rows, bit_values, ctas=1, expected=None):
    values = [widen(bits) for bits in bit_values]
    aten = aten_sum_bits(values, ctas)
    seq = sequential_bits(values)
    aten_mean = mean_bits(aten, rows)
    seq_mean = mean_bits(seq, rows)
    print(f"{name}: T={rows} C={ctas}")
    print(f"  ATen sum  0x{aten:04x}  mean 0x{aten_mean:04x}")
    print(f"  sequential sum 0x{seq:04x}  mean 0x{seq_mean:04x}")
    if expected is not None:
        assert (aten, seq, aten_mean, seq_mean) == expected
    return aten, seq, aten_mean, seq_mean

# The first four rows are identical in both orders; row 4 is the first
# tail row and lands in accumulator 0, so ATen pairs (x0+x4) before x1..x3.
# eps = 2^-26 is lost in 1.0 + eps but survives in the separate accumulator.
EPS = 0x3280  # BF16 2^-26
check("T=5 structured", 5,
      [0x3F80, EPS, 0xBF80, EPS, EPS],
      expected=(0x3280, 0x3300, 0x314D, 0x31CD))

# Sparse T=64 witness: ATen's block_y tree pairs y=0 with y=2 and y=1 with
# y=3, producing exactly zero; sequential order leaves two eps values.
check("T=64 sparse", 64,
      [0x3F80, 0xBF80, EPS, EPS] + [0] * 60,
      expected=(0x0000, 0x3300, 0x0000, 0x3000))

# Same sparse witness once the global-reduction path starts at T=1021.
# C=16 is the config on any device whose target_grid_size/grid.x >= 16,
# which holds for the listed H on current data-centre GPUs.
check("T=1021 global", 1021,
      [0x3F80, 0xBF80, EPS, EPS] + [0] * 1017,
      ctas=16,
      expected=(0x0000, 0x3300, 0x0000, 0x2E00))

# For T <= 4 the four-accumulator fold is row order, so both agree.
for rows in range(1, 5):
    sample = [0x3F80 + 3 * i for i in range(rows)]
    values = [widen(bits) for bits in sample]
    assert aten_sum_bits(values) == sequential_bits(values), rows
print("T=1..4: ATen order equals sequential order, as expected")
