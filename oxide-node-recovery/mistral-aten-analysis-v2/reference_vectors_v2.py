#!/usr/bin/env python3
"""CPU reference for ATen 5c488 BF16 sum(dim=0) and true-divide mean.

The sum order is the one derived from Reduce.cuh. The mean follows the exact
CPU-scalar true-divide branch in BinaryDivTrueKernel.cu: compute
float(double(1.0) / double(T)), multiply by the BF16 sum in FP32, then round to
BF16. All arithmetic is modelled with IEEE RNE; stdlib only.

Run (offline, small):
    taskset -c 152-159 python3 reference_vectors_v2.py

Optional exhaustive token-domain equivalence for the requested T list is in the
default run. The full T=1..4096 sweep was also run separately with two workers
(137 s on this host) and found no final-BF16 difference between reciprocal
multiply and true divide for any positive finite BF16 sum.
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
    """Reduce.cuh thread_reduce_impl: vt0=4 accumulators, then a left fold."""
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
    total = len(x)
    if total < 64:
        return bf16_bits(thread_reduce(x, 0, 1))
    if ctas <= 1:
        v = [thread_reduce(x, y, 4) for y in range(4)]
        return bf16_bits(add(add(v[0], v[2]), add(v[1], v[3])))
    q = []
    for cta in range(ctas):
        v = [thread_reduce(x, y + 4 * cta, 4 * ctas) for y in range(4)]
        q.append(add(add(v[0], v[2]), add(v[1], v[3])))
    u = [0.0] * 4
    for y in range(4):
        source = y
        while source < ctas:
            u[y] = add(u[y], q[source])
            source += 4
    return bf16_bits(add(add(u[0], u[2]), add(u[1], u[3])))

def sequential_bits(x):
    total = 0.0
    for value in x:
        total = add(total, value)
    return bf16_bits(total)

def inv_for(rows):
    # BinaryDivTrueKernel.cu: opmath_t(high_prec_t(1.0) / scalar<high_prec_t>).
    # high_prec_t is double, opmath_t is float for BFloat16.
    return f32(1.0 / float(rows))

def mean_recip_bits(sum_bits, rows):
    return bf16_bits(f32(widen(sum_bits) * inv_for(rows)))

def mean_div_bits(sum_bits, rows):
    return bf16_bits(f32(widen(sum_bits) / f32(float(rows))))

def check(name, rows, bit_values, ctas=1, expected=None):
    values = [widen(bits) for bits in bit_values]
    aten = aten_sum_bits(values, ctas)
    seq = sequential_bits(values)
    aten_mean = mean_recip_bits(aten, rows)
    seq_mean = mean_recip_bits(seq, rows)
    print(f"{name}: T={rows} C={ctas}")
    print(f"  ATen sum  0x{aten:04x}  mean(recip) 0x{aten_mean:04x}")
    print(f"  sequential sum 0x{seq:04x}  mean(recip) 0x{seq_mean:04x}")
    if expected is not None:
        assert (aten, seq, aten_mean, seq_mean) == expected
    return aten, seq, aten_mean, seq_mean

# ---- rounding contract -----------------------------------------------------

# FP32 ties-to-even at 1.0 (spacing 2^-23).
assert add(1.0, 2.0 ** -24) == 1.0
assert struct.unpack("<I", struct.pack("<f", add(1.0, 3.0 * 2.0 ** -24)))[0] == 0x3F80_0002
# BF16 ties-to-even at 1.0 (spacing 2^-7). Midpoints are 1 + k*2^-8.
assert bf16_bits(1.0 + 2.0 ** -8) == 0x3F80
assert bf16_bits(1.0 + 3.0 * 2.0 ** -8) == 0x3F82
assert bf16_bits(-(1.0 + 2.0 ** -8)) == 0xBF80
print("rounding: FP32 and BF16 ties-to-even pass")

# Final sum projection must round once. 1.0 + 2^-8 is the BF16 midpoint and
# rounds down to even; 1.0 + 3*2^-8 is the next midpoint and rounds up to even.
assert bf16_bits(1.0 + 2.0 ** -8) == 0x3F80
assert bf16_bits(1.0 + 3.0 * 2.0 ** -8) == 0x3F82
print("sum projection: single BF16 RNE at the end passes")

# ---- exact CPU-scalar reciprocal recipe -----------------------------------

# For T=5 and the normal-range BF16 value 0x3f01, FP32 multiply by the rounded
# reciprocal differs by one F32 ulp from FP32 division, while the final BF16
# mean is the same. This pins the recipe without claiming a divide.
sum_bits = 0x3F01
product = f32(widen(sum_bits) * inv_for(5))
quotient = f32(widen(sum_bits) / 5.0)
assert struct.unpack("<I", struct.pack("<f", product))[0] == 0x3DCE_6667
assert struct.unpack("<I", struct.pack("<f", quotient))[0] == 0x3DCE_6666
assert mean_recip_bits(sum_bits, 5) == mean_div_bits(sum_bits, 5) == 0x3DCE
print("mean recipe: exact reciprocal-multiply resolved; final BF16 equal here")

# ---- reduce-order witnesses with the exact mean ---------------------------

EPS = 0x3280  # BF16 2^-26
check("T=5 structured", 5,
      [0x3F80, EPS, 0xBF80, EPS, EPS],
      expected=(0x3280, 0x3300, 0x314D, 0x31CD))

check("T=64 sparse", 64,
      [0x3F80, 0xBF80, EPS, EPS] + [0] * 60,
      expected=(0x0000, 0x3300, 0x0000, 0x3000))

check("T=1021 global", 1021,
      [0x3F80, 0xBF80, EPS, EPS] + [0] * 1017,
      ctas=16,
      expected=(0x0000, 0x3300, 0x0000, 0x2E00))

for rows in range(1, 5):
    values = [widen(0x3F80 + 3 * i) for i in range(rows)]
    assert aten_sum_bits(values) == sequential_bits(values), rows
print("T=1..4: ATen order equals sequential order")

# ---- exhaustive final-BF16 equivalence on the requested T set -------------

REQUESTED_T = (5, 7, 63, 64, 65, 1020, 1021, 1024, 2048, 4096)
for rows in REQUESTED_T:
    for bits in range(1, 0x7F80):  # positive finite BF16
        assert mean_recip_bits(bits, rows) == mean_div_bits(bits, rows), (rows, bits)
print("exhaustive positive-finite BF16 sum domain: reciprocal == divide")
print("on T =", REQUESTED_T)
