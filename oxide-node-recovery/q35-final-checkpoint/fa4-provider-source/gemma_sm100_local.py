# SPDX-License-Identifier: MIT
"""Select the existing SM100 one-CTA local-mask kernel for Gemma d256.

The dedicated d256 two-CTA kernel explicitly forbids local attention. This
bounded source adaptation changes selection/planning only for local d256;
its native one-CTA mask/math implementation is retained without edits.
"""
import hashlib
from pathlib import Path


def patch_interface(path):
    path = Path(path)
    original = path.read_text()
    dedicated = 'use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256'
    if original.count(dedicated) != 2:
        raise RuntimeError('Expected exact pinned forward/backward d256 selectors')
    text = original.replace(dedicated, dedicated + ' and not local', 1)
    total_blocks = 'if arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256:'
    if text.count(total_blocks) != 1:
        raise RuntimeError('Expected exact pinned dedicated two-CTA block estimator')
    text = text.replace(total_blocks, total_blocks[:-1] + ' and not local:', 1)
    marker = '    m_block_size_effective = q_stage * tile_m'
    if text.count(marker) != 1:
        raise RuntimeError('Expected exact pinned forward query-stage plan')
    text = text.replace(marker, '''    # Generic d256 local: TMEM = 2*N + Dv = 384 columns; Qstage2
    # would require640 and exceed the SM100512-column allocation.
    if arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256 and local:
        assert tile_m == 128 and tile_n == 64 and num_splits == 1
        q_stage = 1

''' + marker, 1)
    path.write_text(text)
    return dict(before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=hashlib.sha256(text.encode()).hexdigest(),
                scope='SM100 local d256 only; existing generic1CTA mask unchanged',
                tile=[128, 64], q_stage=1, split_kv=False, pack_gqa=False,
                tmem_columns=384, tmem_limit=512,
                dedicated_two_CTA_assertion_unchanged=True)
