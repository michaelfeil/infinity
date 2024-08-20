# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from infinity_emb.fastapi_schemas.pymodels import OpenAIEmbeddingResult, ReRankResult

# LEGACY, TODO: remove them
to_rerank_response = ReRankResult.to_rerank_response
list_embeddings_to_response = OpenAIEmbeddingResult.to_embeddings_response
