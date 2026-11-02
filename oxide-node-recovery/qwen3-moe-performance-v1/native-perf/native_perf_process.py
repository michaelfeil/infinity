#!/usr/bin/env python3
"""Paired LibTorch native timing process using the exact V8-qualified SDK.

One process = one native engine, one GPU0, no second engine. It runs the same
T7/ragged129/stable2k cases as the Oxide process, records first call separately,
ABA warm variable shapes, 20 warmups and 200 timed calls, and packs result bytes
inside the timed section. No model approximation is introduced.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

NATIVE = Path('/tmp/infinity-oxide-parity-20261101/qwen3-moe-native-capi-v1')
sys.path.insert(0, str(NATIVE))
import native_moe
import native_base
import native_batch as nb
import native_raw_probe as physical

QUIET_NAN = struct.pack('<I', 0x7FC00000)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_reference(metadata, batch, batch_sha256, layout, flat, outputs, canonical, first_us, samples, graph_stats):
    return {
        'schema': 'infinity-rust-raw-probe/v1',
        'backend': 'native-capi-timing-reference',
        'pool': metadata['pool'],
        'dtype': 'bfloat16',
        'physical_batch': batch,
        'batch_sha256': batch_sha256,
        'width': 2048,
        'layout': layout,
        'canonical': canonical,
        'outputs': outputs,
        'validated_only': False,
        'native_flat_sha256': hashlib.sha256(flat).hexdigest(),
        'first_call_us': first_us,
        'samples_us': samples,
        'graph_stats': graph_stats,
        'vendor_binding_proof': metadata.get('vendor_binding_proof'),
        'raw_options': metadata.get('raw_options'),
    }


def make_storage(layout):
    capacity = layout['physical_float_count']
    storage = nb.make_guarded_buffer(capacity)
    backing = (C.c_char * len(storage)).from_buffer(storage)
    output = C.cast(backing, C.POINTER(C.c_float))
    return capacity, storage, output


def run_once(library, handle, arrays, batch, layout, pool_code, capacity, storage, output):
    if capacity:
        storage[:capacity * 4] = QUIET_NAN * capacity
    storage[capacity * 4:] = nb.guard_bytes()
    started = time.perf_counter_ns()
    status = library.tei_forward(
        handle, arrays['ids'], arrays['types'], arrays['positions'],
        arrays['cumulative'], len(batch['cumulative_seq_lengths']) - 1,
        batch['max_length'], pool_code, arrays['pooled'], len(batch['pooled_indices']),
        arrays['raw'], len(batch['raw_indices']), output, capacity,
        None, None, 0, None, 0)
    if status != 0:
        detail = library.tei_error()
        kind = library.tei_last_error_kind()
        text = detail.decode() if detail else ''
        raise RuntimeError(f'tei_forward failed (status {status}, kind {kind}: {text})')
    flat = bytes(storage[:capacity * 4])
    nb.check_guards(storage, capacity)
    elapsed_us = (time.perf_counter_ns() - started) / 1000.0
    return elapsed_us, flat


def load_cases(path):
    doc = json.loads(Path(path).read_text())
    root = Path(path).parent
    cases = []
    for entry in doc['cases']:
        batch_path = root / entry['batch']
        digest, batch = nb.load_batch(batch_path)
        cases.append({'id': entry['id'], 'path': str(batch_path), 'batch': batch, 'batch_sha256': digest})
    return cases, doc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--pool', choices=['cls', 'mean', 'last-token'], default='mean')
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--aba-cycles', type=int, default=5)
    p.add_argument('--library', default=native_moe.SDK)
    p.add_argument('--helper', default=str(NATIVE / 'bench_helper.so'))
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f'refusing existing output directory {a.output}')
    if a.warmup < 0 or a.iterations <= 0:
        raise SystemExit('invalid warmup/iterations')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise SystemExit('Root GPU0 lease requires CUDA_VISIBLE_DEVICES=0')
    cases, manifest = load_cases(a.cases)
    if len(cases) < 2:
        raise SystemExit('ABA requires at least two cases')
    prep = argparse.Namespace(model=a.model, pool=a.pool, library=a.library, helper=a.helper)
    torch, base, library, bundle, metadata = native_moe.prepare(prep)
    handle = None
    a.output.mkdir(parents=True)
    try:
        handle = base.create_engine(library, bundle.native, metadata['raw_options'], 'cuda:0', 2, True)
        base.upload_weights(library, handle, bundle)
        base.check_status(library, library.tei_ready(handle), 'tei_ready')
        widths = native_moe.check_widths(library, handle)
        pool_code = native_moe.sc.POOLS[a.pool]
        prepared = []
        for case in cases:
            layout = nb.physical_layout(case['batch'], 2048)
            arrays = physical.build_call_arrays(case['batch'])
            capacity, storage, output = make_storage(layout)
            # First call timed separately.
            first_us, first_flat = run_once(library, handle, arrays, case['batch'], layout, pool_code, capacity, storage, output)
            outputs, facts = nb.canonical_outputs(case['batch'], layout, first_flat)
            first_hash = sha_bytes(first_flat)
            prepared.append({**case, 'layout': layout, 'arrays': arrays, 'capacity': capacity,
                             'storage': storage, 'output': output, 'first_us': first_us,
                             'first_flat': first_flat, 'first_hash': first_hash,
                             'outputs': outputs, 'canonical': facts})
        # ABA warm variable shape, separate from steady timing.
        aba = []
        for cycle in range(a.aba_cycles):
            for idx in (0, 1):
                c = prepared[idx]
                us, flat = run_once(library, handle, c['arrays'], c['batch'], c['layout'], pool_code, c['capacity'], c['storage'], c['output'])
                aba.append({'cycle': cycle, 'case': c['id'], 'elapsed_us': us, 'result_sha256': sha_bytes(flat)})
        graph_after_aba = physical.graph_counts(library, handle)
        results = []
        for c in prepared:
            warmup = []
            for _ in range(a.warmup):
                us, flat = run_once(library, handle, c['arrays'], c['batch'], c['layout'], pool_code, c['capacity'], c['storage'], c['output'])
                if sha_bytes(flat) != c['first_hash']:
                    raise RuntimeError(f"{c['id']} warmup output differs")
                warmup.append(us)
            timed = []
            for _ in range(a.iterations):
                us, flat = run_once(library, handle, c['arrays'], c['batch'], c['layout'], pool_code, c['capacity'], c['storage'], c['output'])
                if sha_bytes(flat) != c['first_hash']:
                    raise RuntimeError(f"{c['id']} timed output differs")
                timed.append(us)
            stats_after = physical.graph_counts(library, handle)
            report = build_reference(metadata, c['batch'], c['batch_sha256'], c['layout'], c['first_flat'], c['outputs'], c['canonical'], c['first_us'], timed, stats_after[0])
            path = a.output / (c['id'] + '-report.json')
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            ordered = sorted(timed)
            results.append({
                'id': c['id'], 'batch': c['path'], 'batch_sha256': c['batch_sha256'],
                'tokens': len(c['batch']['input_ids']), 'first_call_us': c['first_us'],
                'warmup_us': warmup, 'timed_us': timed, 'timed_count': len(timed),
                'p50_us': ordered[len(ordered)//2], 'p95_us': ordered[min(len(ordered)*95//100, len(ordered)-1)],
                'mean_us': sum(timed)/len(timed), 'result_sha256': c['first_hash'],
                'report': str(path), 'report_sha256': sha_file(path),
                'graph_stats_after': stats_after[0], 'graph_count_after': stats_after[1],
            })
        summary = {
            'schema': 'qwen3-moe-native-perf/v1', 'backend': 'native-libtorch',
            'model': str(a.model), 'pool': a.pool, 'device': 0, 'width': 2048,
            'cases_manifest': str(a.cases), 'cases_manifest_sha256': sha_file(a.cases),
            'warmup': a.warmup, 'iterations': a.iterations, 'aba_cycles': a.aba_cycles,
            'graphs': {'requested': bool(metadata['raw_options'].get('_cuda_graphs')),
                       'after_aba': graph_after_aba[0], 'after_aba_count': graph_after_aba[1]},
            'vendor_binding_proof': metadata.get('vendor_binding_proof'),
            'native_sdk': metadata.get('native_sdk'), 'helper': metadata.get('helper'),
            'raw_options': metadata.get('raw_options'), 'aba': aba, 'cases': results,
        }
        (a.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(a.output / 'summary.json')
    finally:
        if handle is not None:
            library.tei_destroy(handle)


if __name__ == '__main__':
    main()
