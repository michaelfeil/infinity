#!/usr/bin/env python3
"""Compare native packed image decisions to full-sequence Transformers 5.19.0.

Uses a tiny random hybrid/vision checkpoint by default; --model selects local
trained weights. No model code from the Hub is executed. See the LibTorch guide.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation


class Config(C.Structure):
    _fields_ = [(name, C.c_int64) for name in [
        'hidden', 'heads', 'layers', 'intermediate', 'vocab', 'positions', 'types'
    ]] + [('epsilon', C.c_double), ('activation', C.c_int32)]


class Decision(C.Structure):
    _fields_ = [('kind', C.c_int32), ('question_type', C.c_int64),
                ('markers', C.c_void_p), ('marker_count', C.c_size_t),
                ('token_ids', C.c_void_p), ('token_count', C.c_size_t),
                ('fields', C.c_void_p), ('field_count', C.c_size_t)]


class Image(C.Structure):
    _fields_ = [('pixels', C.c_void_p), ('rows', C.c_int64), ('patch_dim', C.c_int64),
                ('grid', C.c_int64 * 3), ('merge_size', C.c_int64),
                ('token_start', C.c_int64), ('token_count', C.c_int64),
                ('sequence_start', C.c_int64), ('feature_key', C.c_uint8 * 32),
                ('has_feature_key', C.c_uint8)]


def flatten(value, prefix=''):
    if isinstance(value, (dict, list)):
        for key, item in (value.items() if isinstance(value, dict) else enumerate(value)):
            yield from flatten(item, f'{prefix}.{key}' if prefix else str(key))
    elif value is not None:
        yield prefix, value if isinstance(value, str) else json.dumps(value)


class Native:
    def __init__(self, library, model, device, dtype):
        self.lib = lib = C.CDLL(str(library.resolve()))
        lib.tei_error.restype = C.c_char_p
        lib.tei_create.argtypes = [C.POINTER(Config), C.c_char_p, C.c_int32]
        lib.tei_create.restype = C.c_void_p
        lib.tei_option.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
        lib.tei_weight.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int32]
        lib.tei_ready.argtypes = lib.tei_destroy.argtypes = [C.c_void_p]
        lib.tei_decide.argtypes = [C.c_void_p] + [C.c_void_p] * 4 + [
            C.c_int64, C.c_int64, C.c_void_p, C.c_void_p, C.c_size_t,
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
        cfg = model.config.to_dict()
        text = cfg['text_config']
        conf = Config(text['hidden_size'], text['num_attention_heads'], text['num_hidden_layers'],
                      text['intermediate_size'], text['vocab_size'], text['max_position_embeddings'], 1,
                      text['rms_norm_eps'], 0)
        self.handle = lib.tei_create(C.byref(conf), device.encode(), 0 if dtype == 'float32' else 2)
        if not self.handle:
            raise RuntimeError(lib.tei_error().decode())
        cfg.update(_decision={'kind': 'option_tokens', 'config': {}},
                   _decision_attention_mode='block_bidirectional', _pooling='decision', _dense_count=0)
        for key, value in flatten(cfg):
            self.check(lib.tei_option(self.handle, key.encode(), value.encode()))
        for name, value in model.state_dict().items():
            tensor = value.detach().cpu().contiguous()
            shape = np.asarray(tensor.shape, np.int64)
            self.check(lib.tei_weight(self.handle, name.encode(), tensor.data_ptr(), shape.ctypes.data,
                                     len(shape), {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[tensor.dtype]))
        self.check(lib.tei_ready(self.handle))

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.tei_error().decode())

    def prepare(self, cases):
        ids = np.concatenate([case['ids'] for case in cases]).astype(np.int64)
        axes = np.concatenate([case['axes'] for case in cases], axis=1).astype(np.int64)
        positions = axes[0].copy()
        types = np.zeros_like(ids)
        cumulative = np.asarray([0] + list(np.cumsum([len(case['ids']) for case in cases])), np.int32)
        labels = np.asarray([11, 12, 13], np.int64)
        metadata = (Decision * len(cases))(*[
            Decision(4, case['prefix'], None, 0, labels.ctypes.data, len(labels), None, 0) for case in cases])
        pixels = [image['pixels'] for case in cases for image in case['images']]
        descriptors = []
        for row, case in enumerate(cases):
            for image in case['images']:
                descriptors.append(Image(image['pixels'].ctypes.data, *image['pixels'].shape,
                    (C.c_int64 * 3)(*image['grid']), image['merge'], cumulative[row] + image['start'],
                    np.prod(image['grid']) // image['merge'] ** 2, cumulative[row], (C.c_uint8 * 32)(), 0))
        images = (Image * len(descriptors))(*descriptors)
        logits = np.zeros(len(cases) * len(labels), np.float32)
        actions = np.zeros(len(cases), np.float32)

        def forward():
            # Retain every borrowed buffer until the synchronous native call returns.
            _ = pixels, labels
            self.check(self.lib.tei_decide(self.handle, ids.ctypes.data, types.ctypes.data,
                positions.ctypes.data, cumulative.ctypes.data, len(cases), max(np.diff(cumulative)),
                metadata, logits.ctypes.data, len(logits), actions.ctypes.data,
                axes.ctypes.data if len(images) or not np.array_equal(axes, np.tile(positions, (3, 1))) else None, images, len(images), None, 0))
            assert np.isfinite(logits).all()
            return logits.reshape(len(cases), -1).copy()
        return forward

    def close(self):
        self.lib.tei_destroy(self.handle)


def cases_for(config, image_grid=4):
    """Real visual spans/rotary geometry, shared pixels, ragged suffixes and mixed rows."""
    vision = config.vision_config
    merge = vision.spatial_merge_size
    patch_dim = 3 * vision.temporal_patch_size * vision.patch_size ** 2
    rng = np.random.default_rng(19)
    def case(grids, suffix, zero=False):
        ids, axes, images = [4], [[0], [0], [0]], []
        position = 1
        for height, width in grids:
            ids.append(config.vision_start_token_id)
            for axis in axes:
                axis.append(position)
            position += 1
            start = len(ids)
            for h in range(height // merge):
                for w in range(width // merge):
                    ids.append(config.image_token_id)
                    for axis, value in zip(axes, [position, position + h, position + w]):
                        axis.append(value)
            grid = [1, height, width]
            pixels = rng.normal(0, .2, (height * width, patch_dim)).astype(np.float32)
            if zero:
                pixels.fill(0)
            images.append(dict(start=start, grid=grid, merge=merge, pixels=pixels))
            position += max(height, width) // merge
            ids.append(config.vision_end_token_id)
            for axis in axes:
                axis.append(position)
            position += 1
        ids.append(5)
        for axis in axes:
            axis.append(position)
        prefix = len(ids)
        for i, token in enumerate(suffix):
            ids.append(token)
            for axis in axes:
                axis.append(position + 1 + i)
        return dict(ids=np.asarray(ids), axes=np.asarray(axes), prefix=prefix, images=images)
    first = case([(image_grid, image_grid)], [6, 7, 8])
    sibling = dict(first, ids=np.concatenate([first['ids'][:first['prefix']], [9, 10]]),
                   axes=first['axes'][:, :-1].copy())
    rotary = dict(first, axes=first['axes'].copy())
    rotary['axes'][1, 3] += 1
    return [first, sibling, case([(image_grid, image_grid)], [6, 7, 8], zero=True),
            case([], [6, 7] * 32), case([(image_grid, image_grid)] * 2 if image_grid >= 64 else [(2, 4), (4, 2)], [8, 9]), rotary,
            dict(first, ids=first['ids'][:first['prefix']], axes=first['axes'][:, :first['prefix']])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--image-grid', type=int, default=4)
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    dtype = getattr(torch, args.dtype)
    if args.model:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=dtype,
            attn_implementation='sdpa').to(args.device).eval()
    else:
        text = Qwen3_5TextConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=128,
            max_position_embeddings=256, linear_num_key_heads=1, linear_num_value_heads=1,
            linear_key_head_dim=128, linear_value_head_dim=128,
            layer_types=['linear_attention', 'full_attention'] * 2,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                'partial_rotary_factor': 1., 'mrope_interleaved': True, 'mrope_section': [4, 2, 2]})
        vision = Qwen3_5VisionConfig(depth=2, hidden_size=16, intermediate_size=24, num_heads=2,
            patch_size=2, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
            num_position_embeddings=16)
        cfg = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
            image_token_id=101, vision_start_token_id=100, vision_end_token_id=102)
        model = Qwen3_5ForConditionalGeneration(cfg).to(dtype=dtype, device=args.device).eval()
    cases = cases_for(model.config, args.image_grid)
    if args.text_only:
        cases = [case for case in cases if not case['images']]
    native = Native(args.library, model, args.device, args.dtype)
    original_mask = implementation.create_causal_mask
    block_ids = None
    def mask(*a, **kw):
        return original_mask(*a, **kw, block_sequence_ids=block_ids)
    implementation.create_causal_mask = mask
    report = []
    try:
        with torch.inference_mode():
            for case in cases:
                ids = torch.tensor(case['ids'][None], device=args.device)
                block_ids = (torch.arange(ids.shape[1], device=args.device) >= case['prefix']).long()[None]
                media = {}
                if case['images']:
                    media = dict(pixel_values=torch.tensor(np.concatenate([i['pixels'] for i in case['images']]), device=args.device),
                                 image_grid_thw=torch.tensor([i['grid'] for i in case['images']], device=args.device))
                output = model(input_ids=ids, position_ids=torch.tensor(case['axes'][:, None], device=args.device),
                    attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True, logits_to_keep=1, **media)
                hidden = output.hidden_states[-1][0, -1].float()
                case['reference'] = (model.get_output_embeddings().weight[[11, 12, 13]].float() @ hidden).cpu().numpy()
            for ordered in [cases, list(reversed(cases))]:
                actual = native.prepare(ordered)()
                for row, case in zip(actual, ordered):
                    single = native.prepare([case])()[0]
                    tolerance = .001 if args.dtype == 'float32' else .1
                    # Only relative option logits determine OneJev scores. BF16
                    # full-sequence, split-prefix and chunked GDN execution can
                    # differ by a common offset; record that raw drift, but gate
                    # centered logits and the actual probabilities independently.
                    def centered(values):
                        return values - values.mean()
                    compare = centered if args.dtype != 'float32' else lambda values: values
                    np.testing.assert_allclose(compare(row), compare(case['reference']),
                        rtol=.01 if args.dtype != 'float32' else .001, atol=tolerance)
                    np.testing.assert_allclose(compare(row), compare(single), rtol=.01, atol=tolerance)
                    def softmax(values):
                        probabilities = np.exp(values - np.max(values))
                        return probabilities / probabilities.sum()
                    expected_p, actual_p = softmax(case['reference']), softmax(row)
                    probability_tolerance = .001 if args.dtype == 'float32' else .02
                    np.testing.assert_allclose(actual_p, expected_p, rtol=0, atol=probability_tolerance)
                    ordered_p = np.sort(expected_p)
                    if ordered_p[-1] - ordered_p[-2] > 2 * probability_tolerance:
                        assert np.argmax(row) == np.argmax(case['reference'])
                    report.append(dict(tokens=len(case['ids']), images=len(case['images']),
                        max_logit_error=float(np.max(np.abs(row-case['reference']))),
                        max_centered_logit_error=float(np.max(np.abs(centered(row)-centered(case['reference'])))),
                        max_probability_error=float(np.max(np.abs(actual_p-expected_p)))))
            # One warmed packed call versus one independently executed call per
            # question: covers shared visual prefills, not hardware qualification.
            repeated = [cases[0]] * 8
            packed = native.prepare(repeated)
            serial = [native.prepare([case]) for case in repeated]
            for _ in range(3):
                packed()
                for run in serial:
                    run()
            def measure(run):
                samples = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    run()
                    samples.append((time.perf_counter()-start)*1000)
                return float(np.median(samples))
            timings = dict(packed_eight_ms=measure(packed), serial_eight_ms=measure(lambda: [run() for run in serial]))
            profile = None
            if args.profile and cases[0]['images']:
                distinct = []
                for index in range(8):
                    case = dict(cases[0], ids=cases[0]['ids'].copy())
                    case['ids'][0] = 14 + index
                    distinct.append(case)
                run = native.prepare(distinct)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if args.device.startswith('cuda'):
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True) as capture:
                    run()
                shape = list(cases[0]['images'][0]['pixels'].shape)
                events = capture.events()
                projections = sum(event.name == 'aten::linear' and event.input_shapes[0] == shape for event in events if event.input_shapes)
                copies = sum(event.name == 'aten::_to_copy' and event.input_shapes[0] == shape for event in events if event.input_shapes)
                assert projections == 1, f'Vision projection ran {projections} times for eight shared images'
                if args.device.startswith('cuda'):
                    assert copies == 2, f'Expected one F32 upload plus one BF16 conversion, got {copies}'
                profile = dict(distinct_text_prefixes=8, vision_patch_projections=projections, pixel_copies=copies,
                    chunked_gdn=any('scan_chunks' in event.name for event in events))
            if cases[0]['images'] and cases[0]['images'][0]['pixels'].nbytes > 1024 * 1024:
                # One shared upload fits; eight independently owned pixel buffers
                # exceed this same limit even if their contents are identical.
                limit = 2 * cases[0]['images'][0]['pixels'].nbytes + 1024 * 1024
                native.check(native.lib.tei_option(native.handle, b'_input_upload_max_bytes', str(limit).encode()))
                packed()
                distinct_storage = [dict(case, images=[dict(image, pixels=image['pixels'].copy()) for image in case['images']]) for case in repeated]
                try:
                    native.prepare(distinct_storage)()
                except RuntimeError as error:
                    assert 'upload' in str(error).lower(), error
                else:
                    raise AssertionError('Distinct pixel uploads escaped the input budget')
                finally:
                    native.check(native.lib.tei_option(native.handle, b'_input_upload_max_bytes', str(512 << 20).encode()))
            result = dict(model=str(args.model or 'tiny-random-hybrid-vision'), device=args.device,
                dtype=args.dtype, cases=report, timings=timings, profile=profile)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result, indent=2), flush=True)
    finally:
        implementation.create_causal_mask = original_mask
        native.close()


if __name__ == '__main__':
    main()
