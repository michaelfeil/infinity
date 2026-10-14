"""Generate small independent Transformers fixtures for packed DeBERTa CUDA tests.

python integration_tests/deberta_reference.py /tmp/deberta-fixtures
INFINITY_DEBERTA_FIXTURES=/tmp/deberta-fixtures cargo test -p infinity_v3-backend-candle \
  --features experimental-deberta --test test_deberta -- --ignored --nocapture
No model downloads are required. Tested with transformers 5.17.0 / torch 2.13.
"""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    torch.manual_seed(71)
    lengths = [1, 7, 31, 65, 129]
    ids = [torch.randint(1, 257, (n,), device='cuda') for n in lengths]
    types = [torch.arange(n, device='cuda') % 2 for n in lengths]
    variants = [
        dict(share_att_key=True, pos_att_type=['c2p', 'p2c'], position_buckets=32,
             norm_rel_ebd='layer_norm', position_biased_input=False),
        dict(share_att_key=False, pos_att_type=['c2p', 'p2c'], position_buckets=32,
             norm_rel_ebd='none', position_biased_input=True, embedding_size=96,
             conv_kernel_size=3, conv_groups=2, conv_act='gelu'),
        dict(share_att_key=False, pos_att_type=['c2p'], position_buckets=-1,
             norm_rel_ebd='none', position_biased_input=False),
        dict(share_att_key=False, pos_att_type=['p2c'], position_buckets=32,
             norm_rel_ebd='layer_norm', position_biased_input=False),
        dict(relative_attention=False, pos_att_type=[], position_biased_input=True),
    ]
    variants += [dict(variants[0], task="sequence"), dict(variants[0], task="token"),
                 dict(relative_attention=False, pos_att_type=["c2p", "p2c"],
                      position_biased_input=True)]
    for i, extra in enumerate(variants):
        extra = dict(extra)
        task = extra.pop("task", None)
        cls = {None:DebertaV2Model,"sequence":DebertaV2ForSequenceClassification,"token":DebertaV2ForTokenClassification}[task]
        cfg = dict(vocab_size=257, hidden_size=128, intermediate_size=192,
                   num_hidden_layers=2, num_attention_heads=2,
                   relative_attention=True, max_position_embeddings=256,
                   max_relative_positions=128, type_vocab_size=2,
                   hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        cfg.update(extra)
        config = DebertaV2Config(**cfg)
        model = cls(config).cuda().eval()
        folder = args.output / str(i)
        folder.mkdir(parents=True, exist_ok=True)
        config.architectures = [cls.__name__]
        config.save_pretrained(folder)
        save_file({k:v.contiguous().cpu() for k,v in model.state_dict().items()}, folder/'model.safetensors')
        (folder/'inputs.json').write_text(json.dumps(dict(lengths=lengths,
            input_ids=torch.cat(ids).tolist(), token_type_ids=torch.cat(types).tolist())))
        for name, dtype in [('float16',torch.float16),('bfloat16',torch.bfloat16)]:
            # Recreate from FP32 weights for each dtype (never half -> bf16).
            from safetensors.torch import load_file
            model = cls(config)
            model.load_state_dict(load_file(folder/'model.safetensors'))
            model = model.to(device='cuda',dtype=dtype).eval()
            with torch.inference_mode():
                backbone=model if task is None else model.deberta
                outputs=[backbone(input_ids=x[None],token_type_ids=y[None]).last_hidden_state[0].float() for x,y in zip(ids,types)]
                tensors={'hidden':torch.cat(outputs).cpu()}
                if task:
                    logits=[model(input_ids=x[None],token_type_ids=y[None]).logits[0].float() for x,y in zip(ids,types)]
                    tensors['logits']=(torch.stack(logits) if task=='sequence' else torch.cat(logits)).cpu()
            save_file(tensors, folder/f'{name}.safetensors')
        print(folder,flush=True)

if __name__=='__main__': main()
