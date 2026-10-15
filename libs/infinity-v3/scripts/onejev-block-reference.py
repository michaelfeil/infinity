#!/usr/bin/env python3
"""Generate independent Transformers block-attention decision references (5.19.0)."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(19)
    if args.model:
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, attn_implementation='sdpa').to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        system = 'Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.'
        cases = []
        for state, question in [('Paris is the capital of France.', 'Is Paris in France?'),
                                ('Paris is the capital of France.', 'Is Paris in Germany?'),
                                ('Berlin is the capital of Germany.', 'Is Paris in France?'),
                                ('Paris is the capital of France. café 東京', 'Is the text multilingual?')]:
            suffix = f'Question: {question}\n\nOptions:\nA. yes: the statement is true / the answer is yes\nB. no: the statement is false / the answer is no\n\nAnswer with one letter: A, B.'
            prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': f'<state>\n{state}\n</state>\n\n{suffix}'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            boundary = prompt.rindex(suffix)
            prefix = sum(end <= boundary for _, end in enc['offset_mapping'])
            cases.append(dict(input_ids=enc['input_ids'], prefix_length=prefix, token_ids=[tokenizer.encode(s, add_special_tokens=False)[0] for s in ['A', 'B']], state=state, question=question))
        model_path = args.model.resolve()
    else:
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=96,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4, max_position_embeddings=1024,
            layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 0.5, 'mrope_section': [1, 1, 2]})
        config._attn_implementation = 'sdpa'
        model = Qwen3_5ForCausalLM(config).to(dtype=dtype, device=args.device).eval()
        model_path = args.output / 'model'
        model.save_pretrained(model_path)
        (model_path / 'readout.json').write_text(json.dumps({'attention': 'noncausal', 'readout': 'slot_lm'}))
        cases = [dict(input_ids=ids, prefix_length=p, token_ids=[11, 12, 13]) for ids, p in [
            ([3, 4, 5, 6, 7, 8, 9], 4), ([3, 4, 5, 6, 11, 12], 4),
            ([3, 4, 15, 6, 7, 8, 9], 4), ([3, 4, 5, 6, 7, 8, 9], 5),
            ([3, 4, 5], 1), ([3, 4, 5, 6], 3)]]
    # Use Transformers' own block overlay on a full, uncached sequence.
    original_mask = implementation.create_causal_mask
    block_ids = None
    def mask(*a, **kw):
        return original_mask(*a, **kw, block_sequence_ids=block_ids)
    implementation.create_causal_mask = mask
    try:
        with torch.inference_mode():
            for case in cases:
                ids = torch.tensor([case['input_ids']], device=args.device)
                block_ids = (torch.arange(ids.shape[1], device=args.device) >= case['prefix_length']).long()[None]
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    output_hidden_states=True, logits_to_keep=1)
                h = output.hidden_states[-1][0, -1].float()
                case['logits'] = (model.get_output_embeddings().weight[case['token_ids']].float() @ h).cpu().tolist()
                prefix = case['prefix_length']
                block_ids = torch.zeros((1, prefix), dtype=torch.long, device=args.device)
                cached = model(input_ids=ids[:, :prefix], attention_mask=torch.ones_like(ids[:, :prefix]), use_cache=True, logits_to_keep=1).past_key_values
                block_ids = (torch.arange(ids.shape[1], device=args.device) >= prefix).long()[None]
                suffix = model(input_ids=ids[:, prefix:], attention_mask=torch.ones_like(ids), past_key_values=cached,
                    use_cache=True, output_hidden_states=True, logits_to_keep=1)
                h = suffix.hidden_states[-1][0, -1].float()
                case['cached_logits'] = (model.get_output_embeddings().weight[case['token_ids']].float() @ h).cpu().tolist()
                print(len(case['input_ids']), case['prefix_length'], case['logits'], case['cached_logits'], flush=True)
    finally:
        implementation.create_causal_mask = original_mask
    (args.output / 'reference.json').write_text(json.dumps(dict(model=str(model_path), cases=cases), indent=2)+'\n')


if __name__ == '__main__':
    main()
