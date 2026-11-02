// SPDX-License-Identifier: MIT
// Canonical31 export order; this file is the native mode ABI.
// Fields: name, Q heads, KV heads, width, causal, local.
// Modes 0..19 are published and must not change. Modes 20..30 are append-only.
// Mode 26 is Qwen3.5-MoE Q16/KV2/d256 causal global. Mode 27/28 are the
// official Qwen3 dense H5120 Q40/KV8/d128 causal/noncausal globals.
// Mode 29 is Mistral noncausal W4096 left=4096, right=-1 (unbounded future).
// Mode 30 is Qwen3.5-MoE group8/d256 noncausal global, enabled by the Native
// use_bidirectional_attention / noncausal_full_attention selector.
FA4_PROFILE(bert, 16, 16, 64, 0, 0)
FA4_PROFILE(modern_local, 12, 12, 64, 0, 1)
FA4_PROFILE(qwen, 32, 8, 128, 1, 0)
FA4_PROFILE(voyage, 16, 8, 128, 0, 0)
FA4_PROFILE(gqa6, 12, 2, 128, 0, 0)
FA4_PROFILE(gqa8, 32, 4, 128, 0, 0)
FA4_PROFILE(qwen35, 16, 4, 256, 0, 0)
FA4_PROFILE(mha32, 12, 12, 32, 0, 0)
FA4_PROFILE(media72, 16, 16, 72, 0, 0)
FA4_PROFILE(mha128, 6, 6, 128, 0, 0)
FA4_PROFILE(local128, 6, 6, 128, 0, 1)
FA4_PROFILE(causal_gqa2, 16, 8, 128, 1, 0)
FA4_PROFILE(gqa3, 24, 8, 128, 0, 0)
FA4_PROFILE(gqa4, 32, 8, 128, 0, 0)
FA4_PROFILE(gqa7, 28, 4, 128, 0, 0)
FA4_PROFILE(causal_local128, 32, 8, 128, 1, 1)
FA4_PROFILE(wide_gqa6, 24, 4, 256, 0, 0)
FA4_PROFILE(causal_wide6, 24, 4, 256, 1, 0)
FA4_PROFILE(causal_wide4, 16, 4, 256, 1, 0)
FA4_PROFILE(causal_gqa8_d64, 32, 4, 64, 1, 0)
FA4_PROFILE(mistral_local4096, 32, 8, 128, 1, 1)
FA4_PROFILE(gemma3_local512, 4, 1, 256, 1, 1)
FA4_PROFILE(causal_gqa8, 32, 4, 128, 1, 0)
FA4_PROFILE(embedding_gemma2_local256, 4, 2, 256, 0, 1)
FA4_PROFILE(gemma4_local256_causal, 8, 1, 256, 1, 1)
FA4_PROFILE(gemma4_local256_noncausal, 8, 1, 256, 0, 1)
FA4_PROFILE(qwen35_moe_gqa8, 16, 2, 256, 1, 0)
FA4_PROFILE(qwen3_q40kv8_d128_causal, 40, 8, 128, 1, 0)
FA4_PROFILE(qwen3_q40kv8_d128_noncausal, 40, 8, 128, 0, 0)
FA4_PROFILE(mistral_local4096_noncausal, 32, 8, 128, 0, 1)
FA4_PROFILE(qwen35_moe_gqa8_noncausal, 16, 2, 256, 0, 0)
