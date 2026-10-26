/// \file
/// Packed Qwen3.5 execution across full attention, gated-delta, and MLP/MoE blocks.

#include "batching/prefix_cache.h"
#include "helpers/attention.h"
#include "helpers/fp8_linear.h"
#include "kernels/fp8_fusions.h"
#include "helpers/vendor_error.h"
#include "helpers/workspace_admission.h"
// Native adaptation of Candle Qwen3.5. Upstream notices retained in LICENSE/NOTICE.
#include "models/qwen35_models.h"
#include "env_config.h"
#include "helpers/dense_mlp.h"
#include "models/qwen3_vl_models.h"
#include "models/decoder_models.h"
#include "kernels/qwen35_kernels.h"
#include "kernels/qwen35_reference.h"
#include "kernels/torch_flash_fma.h"
#include "kernels/moe_kernels.h"
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include "batching/radix.h"
#include <array>
#include <map>
#include <memory>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#endif

namespace infinity {
namespace {
// Adapted from Candle qwen3_vl/vision.rs; upstream Apache-2.0 notices retained.
class Qwen35 final : public Model {
  /// Packed attention/history boundaries shared by cached Qwen3.5 block execution.
  struct BlockMetadata {
    /// Device int32 boundaries for assembled K/V and convolution-history rows. Reuse across layers
    /// avoids repeated uploads when the request's block layout is unchanged.
    at::Tensor kv_cumulative, history_cumulative;
    std::vector<int32_t> history_offsets;
  };

  /// Per-block attention and recurrent buffers paired with their packed boundaries.
  struct BlockCache {
    BlockMetadata& metadata;
    /// key/value retain representative-prefix attention rows; state is FP32
    /// [prefixes,value_heads,128,128]. history retains raw QKV rows
    /// preceding each suffix convolution.
    at::Tensor key, value, state, history;
    /// Host boundaries index the representative prefix rows and their separately packed convolution
    /// histories; suffixes select a prefix through block_prefix_sources.
    std::vector<int32_t> offsets, history_offsets;
  };

  /// Select the explicit block-bidirectional decision mode, whose representative prefixes use a
  /// different execution path from causal radix sharing.
  bool block_attention() const {
    return cfg.string("_decision_attention_mode") == "block_bidirectional";
  }

  Options cfg;
  Weights weights;
  c10::Device device;
  at::ScalarType dtype;
  std::string prefix;
  int64_t hidden, heads, kv_heads, dim, layer_count, rotary;
  double epsilon;
  bool delta_prefill = false, delta_suffix_prefill = false;
  at::Tensor cos_cache, sin_cache;

  /// Dense, routed, and shared-expert projections for one Qwen3.5 feed-forward block.
  struct Mlp {
    /// Dense weights use gate_up/down. MoE layers add router plus [E,2I,H]/[E,H,I] expert tensors
    /// and a sigmoid-gated shared expert; these layouts are distinct from grouped-GEMM transposes.
    at::Tensor gate_up, down, router, experts, expert_down, shared_gate_up, shared_down,
        shared_gate;
    DenseMlp dense_mlp, shared_mlp;
  };

  using QuantizedRows = std::pair<at::Tensor, at::Tensor>;
  std::vector<Mlp> mlps;
  std::map<std::string, std::unique_ptr<Fp8Linear>> fp8_projections;
  std::unique_ptr<Qwen3VL> visual;
  at::Tensor rotary_frequencies, rotary_axes;

  std::string key(const std::string& name) const {
    return cfg.values.count("text_config." + name) ? "text_config." + name : name;
  }

  int64_t integer(const std::string& name, int64_t fallback = 0) const {
    return cfg.integer(key(name), fallback);
  }

  double real(const std::string& name, double fallback = 0) const {
    return cfg.real(key(name), fallback);
  }

  std::string string(const std::string& name, const std::string& fallback = "") const {
    return cfg.string(key(name), fallback);
  }

  // Choose the same accumulation tree as the original logical-row GEMM,
  // using descriptors only. Execute on actual compact rows, with no padding.
#ifdef TEI_TORCH_CUDA_KERNELS
  /// Retained projection plan and workspace ownership for logical-row GEMMs.
  struct RadixLinearPlan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight{}, input{}, output{};
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes{};
    // Captured GEMMs retain this allocation even when a later policy grows it.
    std::shared_ptr<WorkspaceMemoryPermit> memory_permit;
    at::Tensor workspace;
    bool valid = false;

    ~RadixLinearPlan() {
      if (output)
        cublasLtMatrixLayoutDestroy(output);
      if (input)
        cublasLtMatrixLayoutDestroy(input);
      if (weight)
        cublasLtMatrixLayoutDestroy(weight);
      if (operation)
        cublasLtMatmulDescDestroy(operation);
    }
  };

  // Includes device, real logical/compact dimensions, operand strides/types,
  // and the ATen workspace policy. Plans are shared by identical layer shapes.
  using RadixLinearKey = std::array<int64_t, 15>;
  mutable std::map<RadixLinearKey, std::unique_ptr<RadixLinearPlan>> radix_linear_plans;
  mutable std::shared_ptr<WorkspaceMemoryPermit> radix_linear_memory_permit;
  mutable at::Tensor radix_linear_workspace;
#endif
  /// Project compact rows using qualified logical-shape GEMM plans where available. Cached
  /// descriptors preserve numerical planning; unsupported profiles retain the
  /// existing logical-row fallback.
  at::Tensor linear_radix_projection(const at::Tensor& x,
                                     const std::string& name,
                                     const PackedInput& packed) const {
    if (auto it = fp8_projections.find(name); it != fp8_projections.end())
      return it->second->forward(x);
#ifdef TEI_TORCH_CUDA_KERNELS
    const auto& matrix = w(name + ".weight");
    // This profile qualifies logical-shape planning for the MoE output projection. Other matrices
    // deliberately unfold to their real logical rows in the fallback, preserving the
    // original projection recipe.
    if (integer("num_experts", 0) > 0 && device.is_cuda() && dtype == at::kBFloat16 &&
        hidden == 2048 && at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt &&
        x.dim() == 2 && x.size(1) == 4096 && x.size(0) > 0 && x.is_contiguous() &&
        matrix.is_contiguous() && x.scalar_type() == at::kBFloat16 &&
        matrix.scalar_type() == at::kBFloat16 && matrix.dim() == 2 && matrix.size(0) == 2048 &&
        matrix.size(1) == 4096) {
      c10::cuda::CUDAGuard guard(x.device());
      auto property = at::cuda::getDeviceProperties(device.index());
      if (property->major >= 8 && reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 256 == 0 &&
          reinterpret_cast<uintptr_t>(matrix.const_data_ptr()) % 256 == 0) {
        const auto m = x.size(0), full_m = prefix_logical_input(packed).ids.numel(), k = x.size(1),
                   n = matrix.size(0);
        const auto limit = at::cuda::getCUDABlasLtWorkspaceSize();
        const RadixLinearKey key{
            { device.index(),
              full_m,
              m,
              k,
              n,
              x.stride(0),
              x.stride(1),
              matrix.stride(0),
              matrix.stride(1),
              n,
              1,
              static_cast<int64_t>(x.scalar_type()),
              static_cast<int64_t>(matrix.scalar_type()),
              static_cast<int64_t>(x.scalar_type()),
              static_cast<int64_t>(limit) }};
        auto found = radix_linear_plans.find(key);
        if (found == radix_linear_plans.end() && radix_linear_plans.size() < 64) {
          auto plan = std::make_unique<RadixLinearPlan>();

          /// Temporary cuBLASLt layouts and preference used during projection-plan selection.
          struct Planning {
            cublasLtMatrixLayout_t input{}, output{};
            cublasLtMatmulPreference_t preference{};

            ~Planning() {
              if (preference)
                cublasLtMatmulPreferenceDestroy(preference);
              if (output)
                cublasLtMatrixLayoutDestroy(output);
              if (input)
                cublasLtMatrixLayoutDestroy(input);
            }
          } logical;

          auto transpose = CUBLAS_OP_T, no_transpose = CUBLAS_OP_N;
          const auto data_type = CUDA_R_16BF;
          bool valid = cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                       CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatmulDescSetAttribute(plan->operation,
                                                   CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &transpose,
                                                   sizeof(transpose)) == CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatmulDescSetAttribute(plan->operation,
                                                   CUBLASLT_MATMUL_DESC_TRANSB,
                                                   &no_transpose,
                                                   sizeof(no_transpose)) == CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatrixLayoutCreate(&plan->weight, data_type, k, n, k) ==
                    CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatrixLayoutCreate(&plan->input, data_type, k, m, k) ==
                    CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatrixLayoutCreate(&plan->output, data_type, n, m, n) ==
                    CUBLAS_STATUS_SUCCESS;
          // Logical layouts never allocate or compute duplicated rows.
          if (valid)
            valid = cublasLtMatrixLayoutCreate(&logical.input, data_type, k, full_m, k) ==
                    CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatrixLayoutCreate(&logical.output, data_type, n, full_m, n) ==
                    CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatmulPreferenceCreate(&logical.preference) == CUBLAS_STATUS_SUCCESS;
          if (valid)
            valid = cublasLtMatmulPreferenceSetAttribute(logical.preference,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &limit,
                                                         sizeof(limit)) == CUBLAS_STATUS_SUCCESS;
          cublasLtMatmulHeuristicResult_t original{}, check{};
          int count = 0;
          auto handle = at::cuda::getCurrentCUDABlasLtHandle();
          if (valid)
            valid = cublasLtMatmulAlgoGetHeuristic(handle,
                                                   plan->operation,
                                                   plan->weight,
                                                   logical.input,
                                                   logical.output,
                                                   logical.output,
                                                   logical.preference,
                                                   1,
                                                   &original,
                                                   &count) == CUBLAS_STATUS_SUCCESS &&
                    count == 1 && original.state == CUBLAS_STATUS_SUCCESS;
          if (valid) {
            plan->algorithm = original.algo;
            valid = cublasLtMatmulAlgoCheck(handle,
                                            plan->operation,
                                            plan->weight,
                                            plan->input,
                                            plan->output,
                                            plan->output,
                                            &plan->algorithm,
                                            &check) == CUBLAS_STATUS_SUCCESS &&
                    check.state == CUBLAS_STATUS_SUCCESS && check.workspaceSize <= limit;
            plan->workspace_bytes = check.workspaceSize;
          }
          plan->valid = valid;
          found = radix_linear_plans.emplace(key, std::move(plan)).first;
        }
        if (found != radix_linear_plans.end() && found->second->valid) {
          auto& plan = *found->second;
          bool admitted = true;
          if (plan.workspace_bytes > 0 && !plan.workspace.defined()) {
            if (!radix_linear_workspace.defined() ||
                static_cast<size_t>(radix_linear_workspace.numel()) < plan.workspace_bytes) {
              auto permit = reserve_workspace_memory(device.index(), limit);
              if (!permit)
                admitted = false;
              else {
                auto workspace =
                    at::empty({static_cast<int64_t>(limit)}, x.options().dtype(at::kByte));
                radix_linear_workspace = std::move(workspace);
                radix_linear_memory_permit = std::move(permit);
              }
            }
            if (admitted) {
              plan.workspace = radix_linear_workspace;
              plan.memory_permit = radix_linear_memory_permit;
            }
          }
          // Supported descriptors survive transient credit denial. Retain the
          // exact workspace generation with each plan for captured addresses.
          if (admitted) {
            auto output = at::empty({m, n}, x.options());
            const float alpha = 1.f, beta = 0.f;
            auto status =
                cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(),
                               plan.operation,
                               &alpha,
                               matrix.const_data_ptr(),
                               plan.weight,
                               x.const_data_ptr(),
                               plan.input,
                               &beta,
                               output.const_data_ptr(),
                               plan.output,
                               output.mutable_data_ptr(),
                               plan.output,
                               &plan.algorithm,
                               plan.workspace_bytes ? plan.workspace.mutable_data_ptr() : nullptr,
                               plan.workspace_bytes,
                               c10::cuda::getCurrentCUDAStream(device.index()).stream());
            if (status == CUBLAS_STATUS_ALLOC_FAILED)
              throw VendorAllocationError{"Qwen3.5 packed projection cublasLtMatmul",
                                          static_cast<int32_t>(status)};
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                        "Qwen3.5 packed attention output projection failed: ",
                        static_cast<int>(status));
            return output;
          }
        }
      }
    }
#endif
    // Unsupported shapes retain original full-row arithmetic. These are the
    // request's real logical rows, never rounded-up or padded rows.
    return radix_fold(linear(radix_unfold(x, packed), name), packed);
  }

  const at::Tensor& w(const std::string& name) const {
    return weight(weights, prefix + name);
  }

  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    return at::linear(x, w(name + ".weight"));
  }

  /// Prefer a prepared FP8 projection when registered; otherwise preserve the full logical
  /// projection recipe during suffix execution.
  at::Tensor
  prefix_projection(const at::Tensor& x, const std::string& name, const PackedInput& input) const {
    if (auto it = fp8_projections.find(name); it != fp8_projections.end())
      return it->second->forward(x);
    return prefix_linear(x, w(name + ".weight"), {}, input);
  }

  /// Interpret the final dimension as [gate | up] and apply the reference SiLU/product stages,
  /// returning half as many features.
  at::Tensor gated(const at::Tensor& gu) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if (device.is_cuda() && dtype == at::kBFloat16)
      return gemma_gated_cuda(gu, false);
#endif
    auto gate = gu.narrow(-1, 0, gu.size(-1) / 2);
    return (gate / (at::ones({}, gate.options()) + at::exp(-gate))) *
           gu.narrow(-1, gu.size(-1) / 2, gu.size(-1) / 2);
  }

  /// Run dense or routed feed-forward layers with the checkpoint's
  /// selected activation and prefix recipe.
  at::Tensor feed_forward(const at::Tensor& x,
                          int64_t layer,
                          const PackedInput& input,
                          const QuantizedRows* quantized = nullptr) const {
    const auto& m = mlps[layer];
    MlpContext context;
    context.gemm = [&](const MlpProjection& projection, const at::Tensor& rows) {
      return prefix_linear(rows, projection.weight(), projection.gemm_bias(), input);
    };
    if (!m.router.defined())
      return quantized ? m.dense_mlp.forward(quantized->first, context, quantized->second)
                       : m.dense_mlp.forward(x, context);
    auto top = integer("num_experts_per_tok", 8);
    auto renormalize = cfg.boolean(key("norm_topk_prob"), true);
    const bool native = top == 8 && device.is_cuda() && dtype == at::kBFloat16 &&
                        (m.experts.size(0) == 128 || m.experts.size(0) == 256) &&
                        x.size(1) % 8 == 0 && m.expert_down.size(2) % 8 == 0;
    auto output = native ? routed_moe_cuda(x,
                                           prefix_linear(x, m.router, {}, input).to(at::kFloat),
                                           m.experts,
                                           m.expert_down,
                                           renormalize)
                         : at::Tensor();
    if (!output.defined())
      output = decoder_moe_forward(x, m.router, m.experts, m.expert_down, top, renormalize);
    // The shared expert is evaluated for every token in addition to selected routed experts. Its
    // learned sigmoid gate scales the shared branch before the two branches are added.
    auto shared = m.shared_mlp.forward(x, context);
    auto logits = prefix_linear(x, m.shared_gate, {}, input);
    auto gate = device.is_cuda() && dtype == at::kBFloat16
                    ? at::reciprocal(at::ones_like(logits) + at::exp(-logits))
                    : at::sigmoid(logits);
    return output + shared * gate;
  }

  /// Apply Qwen3.5 RMSNorm with an offset affine weight: scale is 1 + checkpoint weight. Statistics
  /// and affine scaling stay FP32 until the final cast back to model dtype.
  at::Tensor norm(const at::Tensor& x, const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    // Keep FP32 statistics and affine scaling in one CUDA kernel.
    if (device.is_cuda() && dtype == at::kBFloat16 && x.size(-1) <= 8192)
      return qwen35_reference_norm_cuda(x, w(name + ".weight"), epsilon);
#endif
    auto f = x.to(at::kFloat);
    return (f * at::reciprocal(at::sqrt(f.square().mean(-1, true) + epsilon)) *
            (w(name + ".weight").to(at::kFloat) + 1.))
        .to(x.scalar_type());
  }

  std::pair<at::Tensor, at::Tensor> residual_norm(const at::Tensor& x,
                                                  const at::Tensor& addition,
                                                  const std::string& name,
                                                  QuantizedRows* quantized = nullptr) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if (device.is_cuda() && dtype == at::kBFloat16 && x.size(-1) <= 8192)
      return qwen35_reference_residual_norm_cuda(
          x, addition, w(name + ".weight"), epsilon, quantized);
#endif
    auto updated = x + addition;
    return {updated, norm(updated, name)};
  }

  /// Rotate only the configured leading rotary features and copy the remaining head features
  /// unchanged. Multimodal axes are selected in logical or compact row order to
  /// match the input projection.
  at::Tensor rotate(const at::Tensor& x, const PackedInput& input) const {
    auto r = x.narrow(-1, 0, rotary);
#ifdef TEI_TORCH_CUDA_KERNELS
    if (device.is_cuda() && dtype == at::kBFloat16 && !input.multimodal_positions.defined()) {
      auto rotated = gemma_rotary_cuda(r, cos_cache, sin_cache, radix_positions(input));
      return rotary == dim ? rotated : at::cat({rotated, x.narrow(-1, rotary, dim - rotary)}, -1);
    }
#endif
    auto a = r.narrow(-1, 0, rotary / 2), b = r.narrow(-1, rotary / 2, rotary / 2);
    at::Tensor cos, sin;
    if (input.multimodal_positions.defined()) {
      TORCH_CHECK(visual, "Qwen3.5 multimodal positions require vision configuration");
      auto positions = radix_enabled(input)
                           ? input.multimodal_positions.index_select(1, input.fold_gather)
                           : input.multimodal_positions;
      auto angles = positions.index_select(0, rotary_axes).transpose(0, 1).to(at::kFloat) *
                    rotary_frequencies;
      cos = angles.cos().to(dtype).unsqueeze(1);
      sin = angles.sin().to(dtype).unsqueeze(1);
    } else {
      cos = cos_cache.index_select(0, radix_positions(input)).unsqueeze(1);
      sin = sin_cache.index_select(0, radix_positions(input)).unsqueeze(1);
    }
    return at::cat({at::cat({a * cos - b * sin, b * cos + a * sin}, -1),
                    x.narrow(-1, rotary, dim - rotary)},
                   -1)
        .contiguous();
  }

  using AttentionProjections = std::array<at::Tensor, 3>;

  /// Produce attention projections from real compact/suffix rows while
  /// retaining their logical GEMM recipe.
  AttentionProjections project_attention(const at::Tensor& x,
                                         const std::string& p,
                                         const PackedInput& input,
                                         bool linear,
                                         const QuantizedRows* prepared = nullptr) const {
    QuantizedRows quantized = prepared ? *prepared : QuantizedRows{};
    auto project = [&](const std::string& name) {
      if (auto it = fp8_projections.find(name); it != fp8_projections.end()) {
        if (!quantized.first.defined())
          quantized = fp8_quantize_rows(x.contiguous());
        return it->second->forward_quantized(quantized.first, quantized.second);
      }
      return prefix_projection(x, name, input);
    };
    if (fp8_projections.count(p + "qkv_proj")) {
      auto rows = project(p + "qkv_proj");
      // Norm accepts row strides. Own V separately so prefix caches cannot
      // retain the larger merged Q/K/gate allocation through its storage.
      const auto query = heads * 2 * dim, key_value = kv_heads * dim;
      return {rows.narrow(1, 0, query),
              rows.narrow(1, query, key_value),
              rows.narrow(1, query + key_value, key_value).clone(at::MemoryFormat::Contiguous)};
    }
    // The three outputs represent different layouts by layer type: GDN uses [QKV, Z, A|B], whereas
    // full attention uses [query|gate, K, V].
    if (linear && fp8_projections.count(p + "in_proj_qkvz")) {
      auto rows = project(p + "in_proj_qkvz");
      const auto qkv_width =
          (2 * integer("linear_num_key_heads") + integer("linear_num_value_heads")) * 128;
      // Both slices retain the GEMM storage; GDN reads their original row strides.
      return {rows.narrow(1, 0, qkv_width),
              rows.narrow(1, qkv_width, rows.size(1) - qkv_width),
              project(p + "in_proj_ab").contiguous()};
    }
    if (linear)
      return {project(p + "in_proj_qkv").contiguous(),
              project(p + "in_proj_z").contiguous(),
              project(p + "in_proj_ab").contiguous()};
    return {project(p + "q_proj"), project(p + "k_proj"), project(p + "v_proj")};
  }

  /// Run a full-attention layer, combining retained block/prefix K/V with
  /// the current query rows when supplied.
  at::Tensor full_attention(const at::Tensor& x,
                            const std::string& p,
                            const PackedInput& input,
                            BlockCache* cache = nullptr,
                            bool prefill = false,
                            int64_t cache_layer = -1,
                            const AttentionProjections* projected = nullptr) const {
    auto projections = projected ? *projected : project_attention(x, p, input, false);
    auto tokens = x.size(0);
    // Each query head packs [query | output gate], while K/V use kv_heads * dim features. Normalize
    // and rotate only query/key channels; the gate is applied after attention.
    auto qg = projections[0].view({tokens, heads, 2 * dim});
    auto q = rotate(norm(qg.narrow(-1, 0, dim), p + "q_norm"), input);
    auto k = rotate(norm(projections[1].view({tokens, kv_heads, dim}), p + "k_norm"), input);
    auto v = projections[2].view({tokens, kv_heads, dim});
    auto unfolded = radix_unfold_qkv(q, k, v, input);
    q = unfolded[0];
    k = unfolded[1];
    v = unfolded[2];
    auto mode =
        cfg.string("_decision_attention_mode", cfg.string("decision_attention_mode", "causal"));
    auto causal =
        mode != "noncausal_full_attention" && !cfg.boolean("use_bidirectional_attention", false);
    const bool reference =
        integer("num_experts", 0) > 0 && device.is_cuda() && dtype == at::kBFloat16;
    at::Tensor attention;
    if (cache) {
      std::vector<int32_t> kv_offsets{0};
      int64_t max_kv = input.max_sequence;
      if (prefill) {
        cache->key = k;
        cache->value = v;
      } else {
        std::vector<at::Tensor> keys, values;
        for (int64_t row = 0; row < input.batch; ++row) {
          const auto source = input.block_prefix_sources[row];
          const auto start = cache->offsets[source], length = cache->offsets[source + 1] - start;
          const auto suffix_start = input.offsets[row],
                     suffix_length = input.offsets[row + 1] - suffix_start;
          keys.insert(
              keys.end(),
              {cache->key.narrow(0, start, length), k.narrow(0, suffix_start, suffix_length)});
          values.insert(
              values.end(),
              {cache->value.narrow(0, start, length), v.narrow(0, suffix_start, suffix_length)});
          kv_offsets.push_back(kv_offsets.back() + length + suffix_length);
          max_kv = std::max(max_kv, int64_t(length + suffix_length));
        }
        k = at::cat(keys, 0);
        v = at::cat(values, 0);
      }
      if (device.is_cuda()) {
        auto& cached = cache->metadata.kv_cumulative;
        if (!prefill && !cached.defined())
          cached = at::tensor(kv_offsets, input.cumulative.options());
        auto cuk = prefill ? input.cumulative : cached;
        attention = packed_cuda_attention(q,
                                          k,
                                          v,
                                          input.cumulative,
                                          cuk,
                                          input.max_sequence,
                                          max_kv,
                                          1. / std::sqrt(double(dim)));
      } else {
        std::vector<at::Tensor> rows;
        const auto* offsets = prefill ? input.offsets : kv_offsets.data();
        for (int64_t row = 0; row < input.batch; ++row) {
          auto qq = q.narrow(0, input.offsets[row], input.offsets[row + 1] - input.offsets[row])
                        .transpose(0, 1)
                        .unsqueeze(0);
          auto kk =
              at::repeat_interleave(
                  k.narrow(0, offsets[row], offsets[row + 1] - offsets[row]), heads / kv_heads, 1)
                  .transpose(0, 1)
                  .unsqueeze(0);
          auto vv =
              at::repeat_interleave(
                  v.narrow(0, offsets[row], offsets[row + 1] - offsets[row]), heads / kv_heads, 1)
                  .transpose(0, 1)
                  .unsqueeze(0);
          rows.push_back(
              at::scaled_dot_product_attention(qq, kk, vv).squeeze(0).transpose(0, 1).contiguous());
        }
        attention = at::cat(rows, 0);
      }
    } else if (input.prefix)
      attention = prefix_attention(
          q, k, v, input, cache_layer, 1. / std::sqrt(static_cast<double>(dim)), -1, -1, reference);
    else
      attention = packed_attention(q,
                                   k,
                                   v,
                                   input,
                                   1. / std::sqrt(double(dim)),
                                   causal,
                                   -1,
                                   -1,
                                   std::nullopt,
                                   std::nullopt,
                                   reference);
    auto a = radix_fold(attention, input).view({tokens, heads, dim});
    auto logits = qg.narrow(-1, dim, dim);
    auto gate =
        reference ? at::reciprocal(at::ones_like(logits) + at::exp(-logits)) : at::sigmoid(logits);
    auto gated = (a * gate).reshape({tokens, heads * dim});
    // Block mode projects the combined prefix and suffix in block_forward.
    if (cache)
      return gated;
    return radix_enabled(input) ? linear_radix_projection(gated, p + "o_proj", input)
                                : prefix_projection(gated, p + "o_proj", input);
  }

  /// Evaluate sequence-local convolution and gated-delta updates
  /// on CPU with explicit recurrent state.
  at::Tensor delta_cpu(const at::Tensor& qkv,
                       const at::Tensor& z,
                       const at::Tensor& ab,
                       const std::string& p,
                       const PackedInput& input,
                       BlockCache* cache = nullptr,
                       bool prefill = false,
                       int64_t cache_layer = -1) const {
    // Exact-length reference recurrence. No sequence padding and no persistent state.
    auto kh = integer("linear_num_key_heads"), vh = integer("linear_num_value_heads");
    auto channels = (2 * kh + vh) * 128;
    auto conv = w(p + "conv1d.weight").view({channels, -1});
    auto kernel = conv.size(-1);
    auto alog = w(p + "A_log").to(at::kFloat), dt = w(p + "dt_bias").to(at::kFloat);
    std::vector<at::Tensor> outputs;
    std::vector<at::Tensor> states;
    for (int64_t row = 0; row < input.batch; ++row) {
      auto start = input.offsets[row], length = input.offsets[row + 1] - start;
      auto prior = input.prefix ? input.prefix->previous(row, cache_layer) : nullptr;
      TORCH_CHECK(!input.prefix || !input.prefix->hits[row] ||
                      (prior && prior->recurrent.defined()),
                  "Missing recurrent prefix checkpoint");
      const auto source = cache && !prefill ? input.block_prefix_sources[row] : row;
      auto history = prior && prior->convolution.defined() ? prior->convolution
                     : cache && !prefill
                         ? cache->history.narrow(0,
                                                 cache->history_offsets[source],
                                                 cache->history_offsets[source + 1] -
                                                     cache->history_offsets[source])
                         : at::Tensor();
      const auto previous = history.defined() ? history.size(0) : 0;
      auto raw = history.defined() ? at::cat({history, qkv.narrow(0, start, length)}, 0)
                                   : qkv.narrow(0, start, length);
      std::vector<at::Tensor> convolved;
      for (int64_t t = 0; t < length; ++t) {
        auto sum = at::zeros({channels}, qkv.options().dtype(at::kFloat));
        for (int64_t j = 0; j < kernel; ++j) {
          auto source = previous + t + j - kernel + 1;
          if (source >= 0)
            sum = sum + raw.select(0, source).to(at::kFloat) * conv.select(1, j).to(at::kFloat);
        }
        convolved.push_back(at::silu(sum.to(dtype)).to(dtype));
      }
      auto values = at::stack(convolved, 0);
      // Depthwise-convolved rows contain [Q, K, V] with fixed feature width 128 per head. Q/K are
      // normalized in FP32, then repeated to match the larger value-head count.
      auto q = values.narrow(-1, 0, kh * 128).view({length, kh, 128}).to(at::kFloat);
      auto k = values.narrow(-1, kh * 128, kh * 128).view({length, kh, 128}).to(at::kFloat);
      auto v = values.narrow(-1, 2 * kh * 128, vh * 128).view({length, vh, 128}).to(at::kFloat);
      q = q * at::rsqrt(q.square().sum(-1, true) + 1e-6) / std::sqrt(128.);
      k = k * at::rsqrt(k.square().sum(-1, true) + 1e-6);
      q = at::repeat_interleave(q, vh / kh, 1);
      k = at::repeat_interleave(k, vh / kh, 1);
      auto alpha = ab.narrow(0, start, length).narrow(-1, 0, vh).to(at::kFloat) + dt;
      auto decay = at::exp(-at::exp(alog) * at::softplus(alpha));
      auto beta = at::sigmoid(ab.narrow(0, start, length).narrow(-1, vh, vh).to(at::kFloat))
                      .to(dtype)
                      .to(at::kFloat);
      // The FP32 state is [value_heads, key_dim, value_dim]. Clone retained state before mutation
      // so separate suffixes never overwrite their representative prefix checkpoint.
      auto state = prior && prior->recurrent.defined() ? prior->recurrent.clone()
                   : cache && !prefill                 ? cache->state.select(0, source).clone()
                                                       : at::zeros({vh, 128, 128}, q.options());
      for (int64_t t = 0; t < length; ++t) {
        auto kt = k.select(0, t), qt = q.select(0, t);
        // Decay old memory first, compute the value prediction at K, then apply the beta-weighted
        // rank-one correction before reading the updated state with Q.
        state = state * decay.select(0, t).view({vh, 1, 1});
        auto memory = (state * kt.unsqueeze(-1)).sum(1);
        auto delta = (v.select(0, t) - memory) * beta.select(0, t).unsqueeze(-1);
        state = state + kt.unsqueeze(-1) * delta.unsqueeze(1);
        if (input.prefix && input.prefix->writers[row] &&
            input.prefix->hits[row] + t + 1 ==
                int64_t(input.prefix->writers[row]->identity.tokens.size())) {
          auto history = qkv.narrow(0, start, t + 1);
          if (prior && prior->convolution.defined())
            history = at::cat({prior->convolution, history}, 0);
          history = history.narrow(0,
                                   std::max<int64_t>(0, history.size(0) - (kernel - 1)),
                                   std::min<int64_t>(history.size(0), kernel - 1));
          input.prefix->checkpoint(row, cache_layer, state, history);
        }
        auto out = (state * qt.unsqueeze(-1)).sum(1).to(dtype);
        auto f = out.to(at::kFloat);
        out = (f * at::rsqrt(f.square().mean(-1, true) + epsilon)).to(dtype);
        out = (out * w(p + "norm.weight")).to(dtype);
        outputs.push_back(
            (out.to(at::kFloat) * at::silu(z.select(0, start + t).view({vh, 128}).to(at::kFloat)))
                .to(dtype)
                .reshape({vh * 128}));
      }
      if (cache && prefill)
        states.push_back(state);
    }
    if (cache && prefill)
      cache->state = at::stack(states, 0);
    return at::stack(outputs, 0);
  }

  /// Run linear attention with recurrent and convolution history, publishing
  /// state at requested cache boundaries.
  at::Tensor delta_attention(const at::Tensor& x,
                             const std::string& p,
                             const PackedInput& input,
                             BlockCache* cache = nullptr,
                             bool prefill = false,
                             int64_t cache_layer = -1,
                             const AttentionProjections* projected = nullptr,
                             const QuantizedRows* quantized_output = nullptr) const {
    auto projections = projected ? *projected : project_attention(x, p, input, true);
    auto qkv = projections[0], z = projections[1], ab = projections[2];
    qkv = radix_unfold(qkv, input);
    z = radix_unfold(z, input);
    ab = radix_unfold(ab, input);
    at::Tensor output;
    if (device.is_cuda()) {
#ifdef TEI_TORCH_CUDA_KERNELS
      auto conv = w(p + "conv1d.weight").view({qkv.size(1), -1}).contiguous();
      const auto& alog = w(p + "A_log");
      const auto& dt = w(p + "dt_bias");
      if (integer("num_experts", 0) > 0)
        output = qwen35_reference_delta_cuda(qkv.contiguous(),
                                             z.contiguous(),
                                             ab,
                                             conv,
                                             alog,
                                             dt,
                                             w(p + "norm.weight"),
                                             input,
                                             integer("linear_num_key_heads"),
                                             integer("linear_num_value_heads"),
                                             epsilon,
                                             cache_layer);
      else {
        // Wide chunk scans amortize their setup for multiple
        // long prefills. Keep short/single sequences on the faster recurrence.
        const auto* properties = at::cuda::getDeviceProperties(device.index());
        const bool chunks = properties->major >= 8 &&
                            properties->sharedMemPerBlockOptin >= 163840 && input.batch >= 4 &&
                            qkv.size(0) / input.batch >= 1024 &&
                            integer("linear_num_value_heads") >= 32;
        output =
            qwen35_delta_cuda(qkv,
                              z,
                              ab,
                              conv,
                              alog,
                              dt,
                              w(p + "norm.weight"),
                              input,
                              integer("linear_num_key_heads"),
                              integer("linear_num_value_heads"),
                              epsilon,
                              cache && !prefill ? cache->state : at::Tensor(),
                              cache && prefill ? &cache->state : nullptr,
                              cache && !prefill ? cache->history : at::Tensor(),
                              cache && !prefill ? cache->metadata.history_cumulative : at::Tensor(),
                              chunks,
                              cache_layer,
                              delta_prefill && !cache,
                              quantized_output,
                              delta_suffix_prefill);
      }
#else
      TORCH_CHECK(false, "Qwen3.5 requires the native DeltaNet CUDA kernel build");
#endif
    } else
      output = delta_cpu(qkv, z, ab, p, input, cache, prefill, cache_layer);
    if (cache && prefill) {
      std::vector<at::Tensor> histories;
      cache->history_offsets = {0};
      for (int64_t row = 0; row < input.batch; ++row) {
        auto n = std::min(int64_t(input.offsets[row + 1] - input.offsets[row]),
                          w(p + "conv1d.weight").size(-1) - 1);
        // Retain raw projected QKV history, not activated convolution outputs. The next suffix must
        // recompute its boundary convolution from at most kernel-1 preceding rows.
        histories.push_back(qkv.narrow(0, input.offsets[row + 1] - n, n));
        cache->history_offsets.push_back(cache->history_offsets.back() + n);
      }
      cache->history = at::cat(histories, 0);
      auto& metadata = cache->metadata;
      if (!metadata.history_cumulative.defined() ||
          metadata.history_offsets != cache->history_offsets) {
        metadata.history_cumulative =
            at::tensor(cache->history_offsets, input.cumulative.options());
        metadata.history_offsets = cache->history_offsets;
      }
    }
    if (quantized_output)
      return {};
    auto compact = radix_fold(output, input);
    if (cache)
      return compact;
    return radix_enabled(input) ? linear_radix_projection(compact, p + "out_proj", input)
                                : prefix_projection(compact, p + "out_proj", input);
  }

  /// Embed selected token rows, batch distinct image-feature misses, and replace packed image
  /// spans. Fold image-injected rows only after token placement is complete.
  at::Tensor embeddings(const PackedInput& input, RequestImageFeatures& shared_features) const {
    auto states = at::embedding(w("embed_tokens.weight"),
                                input.images.empty() ? radix_ids(input) : input.ids);
    TORCH_CHECK(input.images.empty() || visual, "Qwen3.5 vision tower unavailable");
    auto batch =
        shared_features.get_batch(input.images, [&](const std::vector<ImageInput>& images) {
          return visual->image_features(images);
        });
    for (size_t i = 0; i < input.images.size(); ++i) {
      const auto& image = input.images[i];
      auto features = batch[i].to(dtype);
      TORCH_CHECK(features.size(0) == image.token_count && features.size(1) == hidden,
                  "Qwen3.5 visual token geometry mismatch");
      states.narrow(0, image.token_start, image.token_count).copy_(features);
    }
    if (!input.images.empty())
      states = radix_fold(states, input);
    return states;
  }

  /// Execute distinct bidirectional prefixes once and route each suffix to its representative
  /// prefix state. Host metadata separates real prefix/suffix rows; output
  /// reconstruction restores every logical sequence.
  at::Tensor block_forward(const PackedInput& input) const {
    TORCH_CHECK(input.audios.empty() && !radix_enabled(input) &&
                    input.block_prefix_lengths.size() == size_t(input.batch) &&
                    input.block_prefix_sources.size() == size_t(input.batch),
                "Missing OneJev block boundaries");
    // Structural packing is host-side; projections, MLPs, attention and GDN run
    // once per layer over actual rows. No activation/sequence padding is added.
    std::vector<int64_t> representatives, source_index(input.batch, -1), suffix_rows;
    std::vector<int32_t> prefix_offsets{0}, suffix_offsets{0};
    std::vector<int64_t> suffix_sources;
    std::vector<at::Tensor> prefix_ids, prefix_types, prefix_positions, suffix_ids, suffix_types,
        suffix_positions, prefix_axes, suffix_axes;
    int64_t max_prefix = 0, max_suffix = 0;
    for (int64_t row = 0; row < input.batch; ++row) {
      const int64_t prefix = input.block_prefix_lengths[row],
                    length = input.offsets[row + 1] - input.offsets[row];
      const auto source = input.block_prefix_sources[row];
      TORCH_CHECK(prefix > 0 && prefix <= length && source >= 0 && source <= row &&
                      input.block_prefix_sources[source] == source &&
                      input.block_prefix_lengths[source] == prefix,
                  "Invalid shared OneJev prefix");
      if (source == row) {
        source_index[row] = representatives.size();
        representatives.push_back(row);
        prefix_ids.push_back(input.ids.narrow(0, input.offsets[row], prefix));
        prefix_types.push_back(input.types.narrow(0, input.offsets[row], prefix));
        prefix_positions.push_back(input.positions.narrow(0, input.offsets[row], prefix));
        if (input.multimodal_positions.defined())
          prefix_axes.push_back(input.multimodal_positions.narrow(1, input.offsets[row], prefix));
        prefix_offsets.push_back(prefix_offsets.back() + prefix);
        max_prefix = std::max(max_prefix, prefix);
      }
      if (length > prefix) {
        suffix_rows.push_back(row);
        suffix_sources.push_back(source_index[source]);
        suffix_ids.push_back(input.ids.narrow(0, input.offsets[row] + prefix, length - prefix));
        suffix_types.push_back(input.types.narrow(0, input.offsets[row] + prefix, length - prefix));
        suffix_positions.push_back(
            input.positions.narrow(0, input.offsets[row] + prefix, length - prefix));
        if (input.multimodal_positions.defined())
          suffix_axes.push_back(
              input.multimodal_positions.narrow(1, input.offsets[row] + prefix, length - prefix));
        suffix_offsets.push_back(suffix_offsets.back() + length - prefix);
        max_suffix = std::max(max_suffix, length - prefix);
      }
    }
    PackedInput prefixes{at::cat(prefix_ids, 0),
                         at::cat(prefix_types, 0),
                         at::cat(prefix_positions, 0),
                         input.block_layout->prefix_cumulative,
                         prefix_offsets.data(),
                         int64_t(representatives.size()),
                         max_prefix};
    if (!prefix_axes.empty())
      prefixes.multimodal_positions = at::cat(prefix_axes, 1);
    auto attach_images =
        [&](PackedInput& part, int64_t row, int64_t begin, int64_t length, int64_t destination) {
          const auto start = input.offsets[row] + begin, end = start + length;
          for (const auto& image : input.images)
            if (image.sequence_start == input.offsets[row] && image.token_start < end &&
                image.token_start + image.token_count > start) {
              TORCH_CHECK(image.token_start >= start &&
                              image.token_start + image.token_count <= end,
                          "OneJev image span crosses a block boundary");
              auto relocated = image;
              relocated.token_start = destination + image.token_start - start;
              relocated.sequence_start = destination;
              part.images.push_back(std::move(relocated));
            }
        };
    for (size_t i = 0; i < representatives.size(); ++i) {
      const auto row = representatives[i];
      attach_images(prefixes, row, 0, input.block_prefix_lengths[row], prefix_offsets[i]);
    }
    PackedInput suffixes{};
    if (!suffix_rows.empty()) {
      suffixes = PackedInput{at::cat(suffix_ids, 0),
                             at::cat(suffix_types, 0),
                             at::cat(suffix_positions, 0),
                             input.block_layout->suffix_cumulative,
                             suffix_offsets.data(),
                             int64_t(suffix_rows.size()),
                             max_suffix};
      if (!suffix_axes.empty())
        suffixes.multimodal_positions = at::cat(suffix_axes, 1);
      for (size_t i = 0; i < suffix_rows.size(); ++i) {
        const auto row = suffix_rows[i], prefix = input.block_prefix_lengths[row];
        attach_images(suffixes,
                      row,
                      prefix,
                      input.offsets[row + 1] - input.offsets[row] - prefix,
                      suffix_offsets[i]);
      }
      // Each executed suffix maps to a representative prefix index, not its original batch row.
      // Host indices drive structural assembly; the device copy selects cached recurrent state.
      suffixes.block_prefix_sources = suffix_sources;
      suffixes.block_cache_sources = input.block_layout->cache_sources;
    }
    // Count only images in the executed representative prefixes and suffixes.
    // One cache spans both passes, releasing features after their final use.
    auto executed_images = prefixes.images;
    executed_images.insert(executed_images.end(), suffixes.images.begin(), suffixes.images.end());
    RequestImageFeatures shared_features(executed_images);
    auto hidden = embeddings(prefixes, shared_features);
    const auto prefix_tokens = hidden.size(0);
    const bool has_suffix = !suffix_rows.empty();
    if (has_suffix)
      hidden = at::cat({hidden, embeddings(suffixes, shared_features)}, 0);
    // These sequence boundaries are shared by all layers of this request.
    BlockMetadata metadata{input.block_layout->kv_cumulative,
                           input.block_layout->history_cumulative,
                           input.block_layout->history_offsets};
    auto normalized = norm(hidden, "layers.0.input_layernorm");
    QuantizedRows attention_rows;
    for (int64_t layer = 0; layer < layer_count; ++layer) {
      const auto p = "layers." + std::to_string(layer) + ".";
      BlockCache cache{metadata};
      cache.offsets = prefix_offsets;
      const bool linear = string("layer_types." + std::to_string(layer)) == "linear_attention";
      const auto ap = p + (linear ? "linear_attn." : "self_attn.");
      // Project the combined representative-prefix/suffix rows once per layer, then slice those
      // projections for separate attention passes. Row-local output projections
      // and MLPs can remain batched.
      auto projections =
          project_attention(normalized,
                            ap,
                            input,
                            linear,
                            attention_rows.first.defined() ? &attention_rows : nullptr);
      const auto output_name = ap + (linear ? "out_proj" : "o_proj");
      const auto output_projection = fp8_projections.find(output_name);
      QuantizedRows output_rows;
      if (linear && output_projection != fp8_projections.end()) {
        const auto width = integer("linear_num_value_heads") * 128;
        output_rows = {
            at::empty({hidden.size(0), width}, hidden.options().dtype(at::kFloat8_e4m3fn)),
            at::empty({hidden.size(0), 1}, hidden.options().dtype(at::kFloat))};
      }
      auto attention = [&](const PackedInput& part, bool prefill) {
        const auto start = prefill ? 0 : prefix_tokens,
                   count = prefill ? prefix_tokens : hidden.size(0) - prefix_tokens;
        AttentionProjections block;
        for (size_t i = 0; i < block.size(); ++i)
          block[i] = projections[i].narrow(0, start, count);
        auto x = normalized.narrow(0, start, count);
        QuantizedRows slice;
        if (output_rows.first.defined())
          slice = {output_rows.first.narrow(0, start, count),
                   output_rows.second.narrow(0, start, count)};
        return linear ? delta_attention(x,
                                        ap,
                                        part,
                                        &cache,
                                        prefill,
                                        -1,
                                        &block,
                                        slice.first.defined() ? &slice : nullptr)
                      : full_attention(x, ap, part, &cache, prefill, -1, &block);
      };
      auto prefix_attention = attention(prefixes, true);
      at::Tensor attended;
      if (output_rows.first.defined()) {
        if (has_suffix)
          attention(suffixes, false);
        attended =
            output_projection->second->forward_quantized(output_rows.first, output_rows.second);
      } else {
        attended = has_suffix ? at::cat({prefix_attention, attention(suffixes, false)}, 0)
                              : prefix_attention;
        attended = prefix_projection(attended, output_name, input);
      }
      QuantizedRows mlp_rows;
      auto post = residual_norm(hidden,
                                attended,
                                p + "post_attention_layernorm",
                                mlps[layer].dense_mlp.fp8() ? &mlp_rows : nullptr);
      auto mlp =
          feed_forward(post.second, layer, input, mlp_rows.first.defined() ? &mlp_rows : nullptr);
      auto next = residual_norm(
          post.first,
          mlp,
          layer + 1 == layer_count ? "norm"
                                   : "layers." + std::to_string(layer + 1) + ".input_layernorm",
          layer + 1 < layer_count && !fp8_projections.empty() ? &attention_rows : nullptr);
      hidden = next.first;
      normalized = next.second;
    }
    hidden = normalized;
    // Execution storage contains each representative prefix once. Reconstruction below repeats its
    // hidden rows for every logical request and appends that request's own suffix.
    auto prefix_hidden = hidden.narrow(0, 0, prefix_tokens);
    auto suffix_hidden =
        has_suffix ? hidden.narrow(0, prefix_tokens, hidden.size(0) - prefix_tokens) : at::Tensor();
    std::vector<at::Tensor> results;
    size_t suffix_row = 0;
    for (int64_t row = 0; row < input.batch; ++row) {
      const auto source = source_index[input.block_prefix_sources[row]];
      auto prefix = prefix_hidden.narrow(
          0, prefix_offsets[source], prefix_offsets[source + 1] - prefix_offsets[source]);
      if (suffix_row < suffix_rows.size() && suffix_rows[suffix_row] == row) {
        results.push_back(at::cat(
            {prefix,
             suffix_hidden.narrow(0,
                                  suffix_offsets[suffix_row],
                                  suffix_offsets[suffix_row + 1] - suffix_offsets[suffix_row])},
            0));
        ++suffix_row;
      } else
        results.push_back(prefix);
    }
    return at::cat(results, 0);
  }

public:
  Qwen35(const Options& config, Weights& tensors, c10::Device dev, at::ScalarType type)
      : cfg(config), weights(std::move(tensors)), device(dev), dtype(type) {
    hidden = integer("hidden_size");
    heads = integer("num_attention_heads");
    kv_heads = integer("num_key_value_heads");
    dim = integer("head_dim", hidden / (heads ? heads : 1));
    layer_count = integer("num_hidden_layers");
    epsilon = real("rms_norm_eps", 1e-6);
    rotary = static_cast<int64_t>(dim * real("rope_parameters.partial_rotary_factor", 1.));
    for (const auto& p : {"model.language_model.", "language_model.", "model.", ""})
      if (weights.count(std::string(p) + "embed_tokens.weight")) {
        prefix = p;
        break;
      }
    if (cfg.integer("vision_config.depth", 0) > 0)
      visual = std::make_unique<Qwen3VL>(cfg, weights, device, dtype, true);
  }

  int64_t output_width() const override {
    return hidden;
  }

  bool supports_images() const override {
    return bool(visual);
  }

  void ready() override {
    // Let the native provider select supported GPU families and GDN layouts.
    delta_prefill = device.is_cuda() && !cfg.boolean("_fp8_dynamic", false);
    const auto* suffix_policy = std::getenv(INFINITY_ENV("TORCH_GDN_SUFFIX_PREFILL"));
    delta_suffix_prefill = suffix_policy && std::string_view(suffix_policy) == "1";
    const auto fp8 = cfg.boolean("_fp8_dynamic", false);
    const auto scope = cfg.string("_fp8_scope", fp8 ? "all" : "mlp");
    TORCH_CHECK(scope == "mlp" || scope == "all", "FP8 scope must be mlp or all");
    TORCH_CHECK(scope != "all" || fp8, "All-linear FP8 requires dynamic FP8 enabled");
    if (fp8) {
      TORCH_CHECK(device.is_cuda() && dtype == at::kBFloat16,
                  "Dynamic FP8 Qwen3.5 requires CUDA BF16");
      TORCH_CHECK(integer("num_experts", 0) == 0 && (cfg.string("model_type") == "qwen3_5" ||
                                                     cfg.string("model_type") == "qwen3_5_text"),
                  "Dynamic FP8 Qwen3.5 supports dense models only");
      for (const auto& [name, value] : cfg.values)
        TORCH_CHECK(
            name != "quantization_config" && name.rfind("quantization_config.", 0) != 0 &&
                name != "text_config.quantization_config" &&
                name.rfind("text_config.quantization_config.", 0) != 0,
            "Dynamic FP8 requires an unquantized checkpoint without supplied quantization scales");
    }
    TORCH_CHECK(!block_attention() || integer("num_experts", 0) == 0,
                "OneJev block attention currently requires a dense Qwen3.5 model");
    TORCH_CHECK(hidden > 0 && heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && dim > 0 &&
                    rotary > 0 && rotary <= dim && rotary % 2 == 0 && layer_count > 0,
                "Invalid Qwen3.5 attention geometry");
    TORCH_CHECK(integer("linear_key_head_dim", 128) == 128 &&
                    integer("linear_value_head_dim", 128) == 128,
                "Qwen3.5 DeltaNet requires head dimension128");
    TORCH_CHECK(device.is_cpu() || device.is_cuda(), "Native Qwen3.5 supports CPU/CUDA");
    if (device.is_cuda())
      TORCH_CHECK(dtype == at::kBFloat16, "Native Qwen3.5 CUDA DeltaNet requires BF16");
    auto theta = real("rope_parameters.rope_theta", 10000.);
    TORCH_CHECK(theta > 0., "Qwen3.5 rotary theta must be positive");
    std::vector<float> inverse(rotary / 2);
    for (int64_t i = 0; i < rotary / 2; ++i)
      inverse[i] = 1.f / std::pow(float(theta), float(2 * i) / float(rotary));
    auto inv = at::from_blob(inverse.data(), {rotary / 2}, at::TensorOptions().dtype(at::kFloat))
                   .clone()
                   .to(device);
    rotary_frequencies = inv;
    auto angles =
        at::arange(integer("max_position_embeddings", 32768), inv.options()).unsqueeze(1) *
        inv.unsqueeze(0);
    cos_cache = angles.cos().to(dtype);
    sin_cache = angles.sin().to(dtype);
    w("embed_tokens.weight");
    w("norm.weight");
    if (visual) {
      TORCH_CHECK(cfg.boolean(key("rope_parameters.mrope_interleaved"), false),
                  "Qwen3.5 vision requires interleaved mRoPE");
      int64_t sections = 0;
      for (int i = 0; i < 3; ++i)
        sections += integer("rope_parameters.mrope_section." + std::to_string(i));
      TORCH_CHECK(sections == rotary / 2 &&
                      !cfg.values.count("vision_config.deepstack_visual_indexes.0"),
                  "Invalid Qwen3.5 vision mRoPE/deepstack configuration");
      auto s1 = integer("rope_parameters.mrope_section.1"),
           s2 = integer("rope_parameters.mrope_section.2");
      std::vector<int64_t> axes;
      for (int64_t j = 0; j < rotary / 2; ++j)
        axes.push_back(j % 3 == 1 && j < s1 * 3 ? 1 : j % 3 == 2 && j < s2 * 3 ? 2 : 0);
      rotary_axes = at::from_blob(axes.data(), {rotary / 2}, at::TensorOptions().dtype(at::kLong))
                        .clone()
                        .to(device);
      visual->ready();
    }
    for (int64_t i = 0; i < layer_count; ++i) {
      auto kind = string("layer_types." + std::to_string(i));
      TORCH_CHECK(kind == "full_attention" || kind == "linear_attention",
                  "Invalid Qwen3.5 attention layer ",
                  i);
      auto p = "layers." + std::to_string(i) + ".";
      w(p + "input_layernorm.weight");
      w(p + "post_attention_layernorm.weight");
      auto mp = p + "mlp.";
      Mlp m;
      std::vector<std::string> consumed;
      auto experts = integer("num_experts", 0);
      if (experts == 0) {
        m.gate_up = at::cat({w(mp + "gate_proj.weight"), w(mp + "up_proj.weight")}, 0);
        m.down = w(mp + "down_proj.weight");
        auto recipe = MlpActivationRecipe::numerical(
            [this](const at::Tensor& rows) { return gated(rows); }, true);
        recipe.fusion = MlpFusion::RoundedSwiGlu;
        m.dense_mlp = DenseMlp({m.gate_up}, {m.down}, std::move(recipe));
        consumed = {mp + "gate_proj.weight", mp + "up_proj.weight", mp + "down_proj.weight"};
        if (fp8) {
          m.dense_mlp.prepare(MlpPrecision::Fp8);
          m.gate_up = at::Tensor();
          m.down = at::Tensor();
        }
      } else {
        auto top = integer("num_experts_per_tok", 8);
        TORCH_CHECK(top > 0 && top <= experts, "Invalid Qwen3.5 routed expert count");
        m.router = w(mp + "gate.weight");
        consumed.push_back(mp + "gate.weight");
        if (weights.count(prefix + mp + "experts.gate_up_proj")) {
          m.experts = w(mp + "experts.gate_up_proj");
          m.expert_down = w(mp + "experts.down_proj");
          consumed.push_back(mp + "experts.gate_up_proj");
          consumed.push_back(mp + "experts.down_proj");
        } else {
          std::vector<at::Tensor> gu, down;
          for (int64_t e = 0; e < experts; ++e) {
            auto ep = mp + "experts." + std::to_string(e) + ".";
            gu.push_back(at::cat({w(ep + "gate_proj.weight"), w(ep + "up_proj.weight")}, 0));
            down.push_back(w(ep + "down_proj.weight"));
            for (const auto& name : {"gate_proj.weight", "up_proj.weight", "down_proj.weight"})
              consumed.push_back(ep + name);
          }
          m.experts = at::stack(gu);
          m.expert_down = at::stack(down);
        }
        m.shared_gate_up = at::cat(
            {w(mp + "shared_expert.gate_proj.weight"), w(mp + "shared_expert.up_proj.weight")}, 0);
        m.shared_down = w(mp + "shared_expert.down_proj.weight");
        m.shared_mlp = DenseMlp({m.shared_gate_up},
                                {m.shared_down},
                                MlpActivationRecipe::numerical(
                                    [this](const at::Tensor& rows) { return gated(rows); }, true));
        m.shared_gate = w(mp + "shared_expert_gate.weight");
        for (const auto& name : {"shared_expert.gate_proj.weight",
                                 "shared_expert.up_proj.weight",
                                 "shared_expert.down_proj.weight",
                                 "shared_expert_gate.weight"})
          consumed.push_back(mp + name);
      }
      mlps.push_back(std::move(m));
      // Every FFN tensor is now owned by its inference cache. The dense/shared
      // gate and up projections, and individually stacked experts, otherwise
      // retain a second full allocation after concatenation. Forward reads FFNs
      // exclusively through mlps; keep all attention/embedding/vision/head keys.
      for (const auto& name : consumed)
        weights.erase(prefix + name);
      if (fp8 && scope == "all") {
        // Keep the small DeltaNet decay/update gates, convolution and recurrent
        // state in their original precision, along with vision and output heads.
        if (kind == "full_attention") {
          const auto attention = p + "self_attn.";
          auto query = w(attention + "q_proj.weight"), key = w(attention + "k_proj.weight"),
               value = w(attention + "v_proj.weight");
          TORCH_CHECK(query.sizes() == at::IntArrayRef({heads * 2 * dim, hidden}) &&
                          key.sizes() == at::IntArrayRef({kv_heads * dim, hidden}) &&
                          value.sizes() == at::IntArrayRef({kv_heads * dim, hidden}),
                      "Invalid Qwen3.5 Q/K/V projection shapes");
          auto matrix = at::cat({query, key, value}, 0);
          fp8_projections.emplace(attention + "qkv_proj", std::make_unique<Fp8Linear>(matrix));
          for (const auto& name : {"q_proj.weight", "k_proj.weight", "v_proj.weight"})
            weights.erase(prefix + attention + name);
        }
        if (kind == "linear_attention") {
          const auto attention = p + "linear_attn.";
          auto matrix =
              at::cat({w(attention + "in_proj_qkv.weight"), w(attention + "in_proj_z.weight")}, 0);
          fp8_projections.emplace(attention + "in_proj_qkvz", std::make_unique<Fp8Linear>(matrix));
          weights.erase(prefix + attention + "in_proj_qkv.weight");
          weights.erase(prefix + attention + "in_proj_z.weight");
        }
        const std::vector<std::string> projections =
            kind == "linear_attention" ? std::vector<std::string>{"linear_attn.out_proj"}
                                       : std::vector<std::string>{"self_attn.o_proj"};
        for (const auto& projection : projections) {
          const auto name = p + projection;
          fp8_projections.emplace(name, std::make_unique<Fp8Linear>(w(name + ".weight")));
          weights.erase(prefix + name + ".weight");
        }
      }
      if (kind == "linear_attention") {
        auto kh = integer("linear_num_key_heads"), vh = integer("linear_num_value_heads");
        TORCH_CHECK(kh > 0 && vh > 0 && vh % kh == 0, "Invalid Qwen3.5 DeltaNet heads");
        const auto ap = p + "linear_attn.";
        weights[prefix + ap + "in_proj_ab.weight"] =
            at::cat({w(ap + "in_proj_a.weight"), w(ap + "in_proj_b.weight")}, 0);
        weights.erase(prefix + ap + "in_proj_a.weight");
        weights.erase(prefix + ap + "in_proj_b.weight");
        w(p + "linear_attn.conv1d.weight");
        // Both CUDA and CPU recurrence consume these fixed parameters in FP32.
        for (const auto& name : {ap + "A_log", ap + "dt_bias"})
          weights[prefix + name] = w(name).to(at::kFloat).contiguous();
        w(p + "linear_attn.norm.weight");
      }
    }
  }

  int64_t prefix_layers() const override {
    return layer_count;
  }

  /// Estimate output rows plus per-layer KV or FP32 recurrent state and bounded convolution
  /// history. Saturating arithmetic keeps impossible admissions from wrapping the byte budget.
  uint64_t prefix_payload_bytes(int64_t tokens) const override {
    const auto element_size = w("embed_tokens.weight").element_size();
    auto bytes = prefix_storage_bytes(tokens, hidden, element_size);
    for (int64_t layer = 0; layer < layer_count; ++layer) {
      if (string("layer_types." + std::to_string(layer)) == "linear_attention") {
        const auto kh = integer("linear_num_key_heads"), vh = integer("linear_num_value_heads");
        const auto kernel =
            w("layers." + std::to_string(layer) + ".linear_attn.conv1d.weight").size(-1);
        bytes = prefix_storage_sum(bytes, prefix_storage_bytes(vh, 128 * 128, sizeof(float)));
        bytes = prefix_storage_sum(bytes,
                                   prefix_storage_bytes(std::min<int64_t>(tokens, kernel - 1),
                                                        (2 * kh + vh) * 128,
                                                        element_size));
      } else
        bytes = prefix_storage_sum(bytes,
                                   prefix_storage_bytes(tokens, 2 * kv_heads * dim, element_size));
    }
    return bytes;
  }

  bool supports_prefix_cache(int operation) const override {
    return operation == 0 && supports_radix();
  }

  bool prefix_checkpoint_only() const override {
    return true;
  }

  bool supports_radix() const override {
    auto mode =
        cfg.string("_decision_attention_mode", cfg.string("decision_attention_mode", "causal"));
    return !block_attention() && mode != "noncausal_full_attention" &&
           !cfg.boolean("use_bidirectional_attention", false);
  }

  void prepare_input(PackedInput& input) const override {
    if (!block_attention() || input.block_layout)
      return;
    TORCH_CHECK(input.block_prefix_lengths.size() == size_t(input.batch) &&
                    input.block_prefix_sources.size() == size_t(input.batch),
                "Missing OneJev block boundaries");
    auto layout = std::make_shared<BlockLayout>();
    std::vector<int32_t> prefixes{0}, suffixes{0}, kv{0};
    std::vector<int64_t> indices(input.batch, -1), sources;
    layout->history_offsets = {0};
    const auto history = integer("linear_conv_kernel_dim", 4) - 1;
    for (int64_t row = 0; row < input.batch; ++row) {
      const auto prefix = input.block_prefix_lengths[row], source = input.block_prefix_sources[row];
      const int64_t length = input.offsets[row + 1] - input.offsets[row];
      TORCH_CHECK(prefix > 0 && prefix <= length && source >= 0 && source <= row &&
                      input.block_prefix_sources[source] == source &&
                      input.block_prefix_lengths[source] == prefix,
                  "Invalid shared OneJev prefix");
      if (source == row) {
        indices[row] = prefixes.size() - 1;
        prefixes.push_back(prefixes.back() + prefix);
        layout->history_offsets.push_back(layout->history_offsets.back() +
                                          std::min(prefix, history));
      }
      if (length > prefix) {
        suffixes.push_back(suffixes.back() + length - prefix);
        kv.push_back(kv.back() + length);
        sources.push_back(indices[source]);
      }
    }
    layout->prefix_cumulative = at::tensor(prefixes, input.cumulative.options());
    layout->suffix_cumulative = at::tensor(suffixes, input.cumulative.options());
    layout->kv_cumulative = at::tensor(kv, input.cumulative.options());
    layout->history_cumulative = at::tensor(layout->history_offsets, input.cumulative.options());
    layout->cache_sources = at::tensor(sources, input.ids.options());
    input.block_layout = std::move(layout);
  }

  /// Alternate full-attention and gated-delta layers according to checkpoint layer_types.
  /// Block-attention checkpoints use their dedicated path; causal
  /// compact/prefix paths restore logical rows.
  at::Tensor forward(const PackedInput& input) const override {
    if (block_attention()) {
      auto prepared = input;
      prepare_input(prepared);
      return block_forward(prepared);
    }
    TORCH_CHECK(!radix_enabled(input) || supports_radix(),
                "Noncausal Qwen3.5 attention does not support radix folding");
    TORCH_CHECK(!radix_enabled(input) || input.audios.empty(),
                "Qwen3.5 radix audio is not supported");
    RequestImageFeatures shared_features(input.images,
                                         cfg.string("_decision.kind") == "d3" ||
                                             cfg.string("_decision.kind") == "pplx");
    auto states = embeddings(input, shared_features);
    for (int64_t i = 0; i < layer_count; ++i) {
      auto p = "layers." + std::to_string(i) + ".";
      auto normalized = norm(states, p + "input_layernorm");
      auto output = string("layer_types." + std::to_string(i)) == "linear_attention"
                        ? delta_attention(normalized, p + "linear_attn.", input, nullptr, false, i)
                        : full_attention(normalized, p + "self_attn.", input, nullptr, false, i);
      states = states + output;
      normalized = norm(states, p + "post_attention_layernorm");
      states = states + feed_forward(normalized, i, input);
    }
    return prefix_restore(input, radix_unfold(norm(states, "norm"), input));
  }
};
}

std::unique_ptr<Model>
create_qwen35(const Options& cfg, Weights& weights, c10::Device device, at::ScalarType dtype) {
  auto family = cfg.string("model_type");
  if (family == "qwen3_5" || family == "qwen3_5_text" || family == "qwen3_5_moe" ||
      family == "qwen3_5_moe_text")
    return std::make_unique<Qwen35>(cfg, weights, device, dtype);
  return nullptr;
}
}
