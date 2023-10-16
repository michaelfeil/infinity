# from typing import List, Dict
# from infinity_emb.inference.primitives import NpEmbeddingType
# from infinity_emb.transformer.abstract import BaseTransformer
# import numpy as np
# import copy

# class FlagEmbeddingFake:
#     def __init__(self, *args, **kwargs) -> None:
#         pass

# try:
#     from fastembed.embedding import FlagEmbedding, normalize
# except:
#     FlagEmbedding = FlagEmbeddingFake

# class FastEmbed(FlagEmbedding, BaseTransformer):
#     def __init__(self, *args, **kwargs):
#         FlagEmbedding.__init__(self)(*args, **kwargs)
#         if FlagEmbedding == FlagEmbeddingFake:
#             raise ImportError("fastembed is not installed.")
#         self._infinity_tokenizer = copy.deepcopy(self.tokenizer)

#     def encode_pre(self, sentences: List[str]) -> Dict[str, np.ndarray[int]]:
#         encoded = self.tokenizer.encode_batch(sentences)
#         input_ids = np.array([e.ids for e in encoded])
#         attention_mask = np.array([e.attention_mask for e in encoded])

#         onnx_input = {
#             "input_ids": np.array(input_ids, dtype=np.int64),
#             "attention_mask": np.array(attention_mask, dtype=np.int64),
#         }

#         if not self.exclude_token_type_ids:
#             onnx_input["token_type_ids"] = np.array(
#                 [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
#             )
#         return onnx_input

#     def encode_core(self, features: Dict[str, np.ndarray[int]]) -> np.ndarray:
#         model_output = self.model.run(None, features)
#         last_hidden_state = model_output[0][:, 0]
#         return last_hidden_state

#     def encode_post(self, embedding: np.ndarray) -> NpEmbeddingType:
#         return normalize(embedding).astype(np.float32)

#     def tokenize_lengths(self, sentences: List[str]) -> List[int]:
#         # tks = self._infinity_tokenizer.encode_batch(
#         #     sentences,
#         # )
#         # return [len(t.tokens) for t in tks]
#         return [len(s) for s in sentences]
