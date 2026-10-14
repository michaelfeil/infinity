"""Generate independent Transformers references; Python is only a test dependency."""
import json
import pathlib
import sys

import torch
from safetensors.torch import save_file
from transformers import (DebertaV2Config, DebertaV2Model, MPNetConfig, MPNetModel,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification)

root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(731)
torch.set_num_threads(1)
ids = [[2, 5, 7], [3, 8, 11, 9, 4, 12, 14, 17, 19, 21, 23, 25, 27]]
configs = [("mpnet", MPNetConfig(vocab_size=64, hidden_size=32,
    intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
    max_position_embeddings=32, hidden_dropout_prob=0., attention_probs_dropout_prob=0.))]
for shared in [False, True]:
    for conv in [0, 3]:
        configs.append((f"deberta-shared{shared}-conv{conv}", DebertaV2Config(
            vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=32, type_vocab_size=2,
            relative_attention=True, position_buckets=8, max_relative_positions=16,
            pos_att_type=["c2p", "p2c"], share_att_key=shared,
            position_biased_input=False, norm_rel_ebd="layer_norm",
            conv_kernel_size=conv, conv_groups=1,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)))
for kind in ["sequence", "token"]:
    config = DebertaV2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        type_vocab_size=2, relative_attention=True, position_buckets=8,
        max_relative_positions=16, pos_att_type=["c2p", "p2c"],
        position_biased_input=False, num_labels=3,
        hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    configs.append((f"deberta-{kind}", config))
for name, config in configs:
    factory = (MPNetModel if name == "mpnet" else
        DebertaV2ForSequenceClassification if name.endswith("sequence") else
        DebertaV2ForTokenClassification if name.endswith("token") else DebertaV2Model)
    model = factory(config).eval()
    path = root / name
    path.mkdir(exist_ok=True)
    model.save_pretrained(path)
    outputs, logits = [], []
    with torch.inference_mode():
        for row in ids:
            inputs = {"input_ids": torch.tensor([row])}
            if name != "mpnet":
                inputs["token_type_ids"] = torch.ones(1, len(row), dtype=torch.long)
            base = getattr(model, "deberta", model)
            outputs.append(base(**inputs).last_hidden_state.squeeze(0))
            if model is not base:
                value = model(**inputs).logits
                logits.append(value.squeeze(0) if name.endswith("token") else value)
    tensors = {"hidden": torch.cat(outputs).contiguous()}
    if logits:
        tensors["logits"] = torch.cat(logits).contiguous()
    save_file(tensors, path / "reference.safetensors")
    (path / "inputs.json").write_text(json.dumps({"lengths": list(map(len, ids)), "ids": sum(ids, [])}))
    print(path)
