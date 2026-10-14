#!/usr/bin/env python3
"""Native Laya C-ABI latency/accuracy/profile matrix; requires pinned Torch and NumPy.

Example: python laya_performance.py --library libtei_torch.so --model MODEL
  --fixture original-inputs.json --output REPORT --iterations 100 --repeats 2
Run baseline and candidate on the same otherwise-idle GPU, without HTTP/tokenization.
"""
import argparse
import ctypes as C
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


class Config(C.Structure):
    _fields_ = [(k, C.c_int64) for k in
                ['hidden', 'heads', 'layers', 'intermediate', 'vocab', 'positions', 'types']] + [
                    ('epsilon', C.c_double), ('activation', C.c_int32)]


class Decision(C.Structure):
    _fields_ = [('kind', C.c_int32), ('question_type', C.c_int64),
                ('markers', C.c_void_p), ('marker_count', C.c_size_t),
                ('token_ids', C.c_void_p), ('token_count', C.c_size_t),
                ('fields', C.c_void_p), ('field_count', C.c_size_t)]


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(v, prefix + '.' + k if prefix else k)
    elif value is not None:
        yield prefix, value if isinstance(value, str) else json.dumps(value)


class Native:
    def __init__(self, library, model, graphs, transient_max_bytes=8 << 30, fp8=False):
        self.lib = lib = C.CDLL(str(library.resolve()))
        lib.tei_error.restype = C.c_char_p
        lib.tei_create.argtypes = [C.POINTER(Config), C.c_char_p, C.c_int32]
        lib.tei_create.restype = C.c_void_p
        lib.tei_option.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
        lib.tei_weight.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int32]
        for name in ['tei_ready', 'tei_destroy', 'tei_graph_count']:
            getattr(lib, name).argtypes = [C.c_void_p]
        lib.tei_graph_count.restype = C.c_int64
        lib.tei_decide.argtypes = [C.c_void_p] + [C.c_void_p] * 4 + [
            C.c_int64, C.c_int64, C.c_void_p, C.c_void_p, C.c_size_t,
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
        cfg = json.loads((model / 'encoder/config.json').read_text())
        conf = Config(cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_hidden_layers'],
                      cfg['intermediate_size'], cfg['vocab_size'], cfg['max_position_embeddings'],
                      cfg.get('type_vocab_size', 1), cfg['norm_eps'], 0)
        self.handle = lib.tei_create(C.byref(conf), b'cuda:0', 2)
        if not self.handle:
            raise RuntimeError(lib.tei_error().decode())
        for layer, key in [('full_attention', 'global_rope_theta'), ('sliding_attention', 'local_rope_theta')]:
            if key not in cfg:
                cfg[key] = cfg['rope_parameters'][layer]['rope_theta']
        cfg.update(_decision={'kind': 'laya', 'config': json.loads((model / 'rl_agent_config.json').read_text())},
                   _cuda_graphs=graphs, _cuda_graph_max_tokens=32768, _cuda_graph_entries=8,
                   _cuda_graph_max_bytes=2 << 30, _pooling='decision', _dense_count=0,
                   _cudnn_varlen=False, _fp8_dynamic=fp8, _transient_max_bytes=transient_max_bytes)
        for key, value in flatten(cfg):
            self.check(lib.tei_option(self.handle, key.encode(), value.encode()))
        file = model / 'model.safetensors'
        with file.open('rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as mm:
                n = int.from_bytes(mm[:8], 'little')
                for name, tensor in json.loads(mm[8:8+n]).items():
                    if name == '__metadata__':
                        continue
                    dtype = tensor['dtype']
                    if dtype not in ('F32', 'F16', 'BF16'):
                        raise ValueError(dtype)
                    array = np.ndarray(tensor['shape'], dtype={'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}[dtype],
                                       buffer=mm, offset=8+n+tensor['data_offsets'][0])
                    shape = np.asarray(tensor['shape'], np.int64)
                    self.check(lib.tei_weight(self.handle, name.encode(), array.ctypes.data,
                                             shape.ctypes.data, len(shape), {'F32': 0, 'F16': 1, 'BF16': 2}[dtype]))
                    del array
        self.check(lib.tei_ready(self.handle))

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.tei_error().decode())

    def prepare(self, sequences):
        lengths = [len(s['ids']) for s in sequences]
        ids = np.asarray([i for s in sequences for i in s['ids']], np.int64)
        positions = np.asarray([i for n in lengths for i in range(n)], np.int64)
        types = np.zeros_like(ids)
        cumulative = np.asarray([0] + list(np.cumsum(lengths)), np.int32)
        markers = [np.asarray(s['markers'], np.int64) for s in sequences]
        metadata = (Decision * len(sequences))(*[
            Decision(0, s['question_type'], m.ctypes.data, len(m), None, 0, None, 0)
            for s, m in zip(sequences, markers)])
        logits = np.zeros(sum(len(m) for m in markers), np.float32)
        actions = np.zeros(len(sequences), np.float32)

        def forward():
            # Keep all buffers, including marker storage, alive through the synchronous ABI call.
            _ = markers
            self.check(self.lib.tei_decide(self.handle, ids.ctypes.data, types.ctypes.data,
                positions.ctypes.data, cumulative.ctypes.data, len(sequences), max(lengths),
                metadata, logits.ctypes.data, len(logits), actions.ctypes.data, None, None, 0, None, 0))
            if not np.isfinite(logits).all() or not np.isfinite(actions).all():
                raise AssertionError('nonfinite decision output')
            return logits, actions
        return forward

    def close(self):
        self.lib.tei_destroy(self.handle)


def cases(fixture, extended=False):
    original = json.loads(fixture.read_text())['cases']
    seed_rows = original[0]['sequences']
    result = []
    for length in [100, 200, 300, 400, 500]:
        for batch in [1, 2, 4, 8, 16, 32]:
            sequences = []
            for i in range(batch):
                source = seed_rows[i % len(seed_rows)]
                body = source['ids'][1:-1]
                ids = [source['ids'][0]] + (body * ((length + len(body)) // len(body)))[:length-2] + [source['ids'][-1]]
                # Shape-controlled inputs, not claimed as semantic publisher prompts.
                count = [2, 5, 16, 32][i % 4]
                markers = np.linspace(2, length-2, count).astype(int).tolist()
                sequences.append(dict(ids=ids, markers=markers, question_type=i % 3))
            result.append(dict(name=f'length{length}-batch{batch}', sequences=sequences, controlled=True))
    for batch in [2, 8, 32]:
        seq = []
        for i in range(batch):
            base = result[(i % 5)*6]['sequences'][0]
            seq.append(dict(base, question_type=i % 3))
        result.append(dict(name=f'ragged100-500-batch{batch}', sequences=seq, controlled=True))
    if extended:
        for batch, total in [(20, 5500), (40, 12000)]:
            rows = [seed_rows[i % len(seed_rows)] for i in range(batch)]
            lengths = [len(s['ids']) for s in rows]
            difference = total - sum(lengths)
            for i in range(abs(difference)):
                lengths[i % batch] += 1 if difference > 0 else -1
            sequences = []
            for source, length in zip(rows, lengths):
                assert max(source['markers']) < length - 1
                body = source['ids'][1:-1]
                ids = [source['ids'][0]] + (body * ((length + len(body)) // len(body)))[:length-2] + [source['ids'][-1]]
                sequences.append(dict(source, ids=ids))
            result.append(dict(name=f'target-batch{batch}-tokens{total}', sequences=sequences, controlled=True))
        for length in [100, 200, 300, 400, 500]:
            template = next(c for c in result if c['name'] == f'length{length}-batch32')
            for batch in [40, 48, 64]:
                result.append(dict(name=f'length{length}-batch{batch}',
                                   sequences=[template['sequences'][i % 32] for i in range(batch)], controlled=True))
    return result + [dict(c, controlled=False) for c in original]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--graphs', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--case', default='')
    p.add_argument('--extended', action='store_true', help='Include exact latency targets and batches 40/48/64')
    p.add_argument('--extended-only', action='store_true', help='Measure only added larger batches and exact targets')
    p.add_argument('--transient-max-bytes', type=int, default=8 << 30,
                   help='Native transient allocation admission budget (increase for larger shape studies)')
    p.add_argument('--tune-gemm', type=int, default=0,
                   help='Experimentally tune this many cuBLASLt heuristic candidates before measuring')
    p.add_argument('--fp8', action='store_true', help='Evaluate existing dynamic FP8 encoder MLP; changes precision')
    args = p.parse_args()
    assert args.iterations >= 20 and args.repeats >= 1
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.tune_gemm:
        torch.cuda.tunable.set_filename(str(args.output / 'gemm-tuning.csv'))
        torch.cuda.tunable.set_cublaslt_requested_algo_count(args.tune_gemm)
        torch.cuda.tunable.set_max_tuning_duration(30)
        torch.cuda.tunable.set_max_tuning_iterations(100)
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
    rng = random.Random(1927)
    workload = cases(args.fixture, args.extended or args.extended_only)
    if args.extended_only:
        workload = [c for c in workload if c['name'].startswith('target-') or
                    c['name'].split('-')[-1] in ('batch40', 'batch48', 'batch64')]
    if args.case:
        workload = [c for c in workload if c['name'] == args.case]
        assert workload, args.case
    (args.output / 'inputs.json').write_text(json.dumps(workload))
    manifest = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cpu_affinity=sorted(os.sched_getaffinity(0)),
                    library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                    model_metadata=json.loads((args.model / 'checkpoint_metadata.json').read_text()),
                    graphs=args.graphs, iterations=args.iterations, repeats=args.repeats,
                    transient_max_bytes=args.transient_max_bytes,
                    gemm_heuristic_candidates=args.tune_gemm,
                    fp8_mlp=args.fp8,
                    measurement='synchronous native C ABI; includes input upload, encoder, head and host outputs; excludes HTTP/tokenization; BF16; exact packed tokens')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    engine = Native(args.library, args.model, args.graphs, args.transient_max_bytes, args.fp8)
    rows = []
    try:
        for repeat in range(args.repeats):
            order = list(workload)
            rng.shuffle(order)
            for case in order:
                forward = engine.prepare(case['sequences'])
                ref_logits, ref_actions = [v.copy() for v in forward()]
                for _ in range(20):
                    forward()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for _ in range(args.iterations):
                    start = time.perf_counter_ns()
                    logits, actions = forward()
                    samples.append((time.perf_counter_ns()-start)/1e6)
                assert np.array_equal(logits, ref_logits) and np.array_equal(actions, ref_actions)
                row = dict(case=case['name'], repeat=repeat, lengths=[len(s['ids']) for s in case['sequences']],
                           samples_ms=samples, p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples,95)),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                           graph_count=engine.lib.tei_graph_count(engine.handle), logits=ref_logits.tolist(), actions=ref_actions.tolist())
                row['tokens_per_second'] = sum(row['lengths']) * 1000 / row['p50_ms']
                row['questions_per_second'] = len(row['lengths']) * 1000 / row['p50_ms']
                rows.append(row)
                (args.output / 'results.json').write_text(json.dumps(rows, indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ('samples_ms','logits','actions')}), flush=True)
        if args.profile:
            for name in ['length100-batch1','length300-batch8','length500-batch32',
                         'target-batch20-tokens5500', 'target-batch40-tokens12000']:
                case = next((c for c in workload if c['name'] == name), None)
                if case is None:
                    continue
                forward = engine.prepare(case['sequences'])
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    for _ in range(5):
                        forward()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output / (name+'-trace.json')))
                operators = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                                  self_cuda_us=e.self_device_time_total, total_cuda_us=e.device_time_total, shapes=e.input_shapes)
                             for e in prof.key_averages(group_by_input_shape=True)]
                (args.output / (name+'-operators.json')).write_text(json.dumps(operators, indent=2))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
