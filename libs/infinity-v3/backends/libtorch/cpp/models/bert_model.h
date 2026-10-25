/// \file
/// Packed BERT/RoBERTa backbone and task-head implementation.
/// Includes encoder attention, ColBERT query-key selection, and native linear paths.

#pragma once
#include "bridge.h"
#include "models/model.h"
#include "helpers/options.h"
#include "helpers/fp8_linear.h"
#include "helpers/attention.h"
#include "helpers/vendor_error.h"
#include "helpers/workspace_admission.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/fast_kernels.h"
#include "kernels/encoder_norm.h"
#include "kernels/encoder_activation.h"
#include "kernels/infinity_fa4.h"
#include "kernels/torch_flash_fma.h"
#include "kernels/torch_flash_exact32.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <ATen/ops/_addmm_activation.h>
#include "helpers/dense_mlp.h"
#include <ATen/ops/_cudnn_attention_forward.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <array>
#include <map>

namespace {
/// Packed BERT/RoBERTa backbone with native attention and optional task readouts.
struct Bert : infinity::Model {
  TeiConfig config;
  c10::Device device;
  at::ScalarType dtype;
  std::unordered_map<std::string, at::Tensor> weights;
  std::string prefix;
  infinity::Options options;
  std::unordered_map<std::string, infinity::DenseMlp> mlps;
  std::optional<infinity::DenseMlp> classifier_mlp;
  bool roberta = false;
  bool cudnn_varlen = false;
  bool private_flash_embedding = false;
  bool native_bert_embedding = false;
  bool private_flash_roberta_embedding = false;
  bool private_flash_exact_classifier = false;
  bool exact_biased_norm = false;
#ifdef TEI_TORCH_CUDA_KERNELS
  bool cached_classifier_gemm = false;

  /// Owns cuBLASLt descriptors and algorithm metadata for a cached classifier GEMM.
  struct ClassifierPlan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight{}, input{}, output{};
    cublasLtMatmulAlgo_t algorithm{};
    size_t bytes{};
    bool valid = false;

    ~ClassifierPlan() {
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

  // All projections execute serially on the model's stream. Keep one stable
  // allocation for every plan, including addresses retained by CUDA graphs.
  mutable std::shared_ptr<infinity::WorkspaceMemoryPermit> classifier_workspace_permit;
  mutable at::Tensor classifier_workspace;
  mutable std::map<std::array<int64_t, 5>, std::unique_ptr<ClassifierPlan>> classifier_plans;
#endif
  Bert(TeiConfig cfg, c10::Device dev, at::ScalarType type)
      : config(cfg), device(dev), dtype(type) {}

  const at::Tensor& w(const std::string& name) const {
    auto entry = weights.find(prefix + name);
    TORCH_CHECK(entry != weights.end(), "Missing BERT weight: ", prefix, name);
    return entry->second;
  }

  /// Apply a classifier projection with optional GELU using qualified cached cuBLASLt plans. Keep
  /// the ordinary ATen path when layout/policy checks fail; retained workspace survives graph use.
  at::Tensor classifier_projection(const at::Tensor& x,
                                   const at::Tensor& matrix,
                                   const at::Tensor& bias,
                                   bool gelu = false) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    // This cache stores GEMM+bias/activation descriptors. Feature multiples, operand alignment, and
    // the scalar-output exception bound that epilogue path; an ineligible projection still
    // uses its original operator.
    if (cached_classifier_gemm && x.dim() == 2 && x.size(0) > 0 && x.is_contiguous() &&
        matrix.dim() == 2 && matrix.is_contiguous() && bias.defined() && bias.is_contiguous() &&
        (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
        matrix.scalar_type() == x.scalar_type() && bias.scalar_type() == x.scalar_type() &&
        matrix.size(1) == x.size(1) &&
        (matrix.size(0) % 16 == 0 || (!gelu && matrix.size(0) == 1)) && x.size(1) % 16 == 0 &&
        bias.dim() == 1 && bias.numel() == matrix.size(0) &&
        (!classifier_workspace.defined() ||
         at::cuda::getCUDABlasLtWorkspaceSize() <=
             static_cast<size_t>(classifier_workspace.numel())) &&
        reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 256 == 0 &&
        reinterpret_cast<uintptr_t>(matrix.const_data_ptr()) % 256 == 0 &&
        reinterpret_cast<uintptr_t>(bias.const_data_ptr()) % 256 == 0) {
      c10::cuda::CUDAGuard guard(x.device());
      const int64_t m = x.size(0), k = x.size(1), n = matrix.size(0);
      const auto limit = at::cuda::getCUDABlasLtWorkspaceSize();
      const std::array<int64_t, 5> key{{m, k, n, gelu, static_cast<int64_t>(limit)}};
      auto found = classifier_plans.find(key);
      if (found == classifier_plans.end() && classifier_plans.size() < 64) {
        auto plan = std::make_unique<ClassifierPlan>();
        auto transpose = CUBLAS_OP_T;
        auto epilogue = gelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
        auto pointer = bias.const_data_ptr();
        const auto storage = x.scalar_type() == at::kBFloat16 ? CUDA_R_16BF : CUDA_R_16F;
        bool valid = cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                     CUBLAS_STATUS_SUCCESS;
        if (valid)
          valid = cublasLtMatmulDescSetAttribute(plan->operation,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &transpose,
                                                 sizeof(transpose)) == CUBLAS_STATUS_SUCCESS;
        if (valid)
          valid = cublasLtMatmulDescSetAttribute(plan->operation,
                                                 CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                 &epilogue,
                                                 sizeof(epilogue)) == CUBLAS_STATUS_SUCCESS;
        if (valid)
          valid = cublasLtMatmulDescSetAttribute(plan->operation,
                                                 CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                 &pointer,
                                                 sizeof(pointer)) == CUBLAS_STATUS_SUCCESS;
        if (valid)
          valid =
              cublasLtMatrixLayoutCreate(&plan->weight, storage, k, n, k) == CUBLAS_STATUS_SUCCESS;
        if (valid)
          valid =
              cublasLtMatrixLayoutCreate(&plan->input, storage, k, m, k) == CUBLAS_STATUS_SUCCESS;
        if (valid)
          valid =
              cublasLtMatrixLayoutCreate(&plan->output, storage, n, m, n) == CUBLAS_STATUS_SUCCESS;
        cublasLtMatmulPreference_t preference{};
        if (valid)
          valid = cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS;
        if (valid)
          valid = cublasLtMatmulPreferenceSetAttribute(preference,
                                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &limit,
                                                       sizeof(limit)) == CUBLAS_STATUS_SUCCESS;
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        if (valid)
          valid = cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(),
                                                 plan->operation,
                                                 plan->weight,
                                                 plan->input,
                                                 plan->output,
                                                 plan->output,
                                                 preference,
                                                 1,
                                                 &result,
                                                 &count) == CUBLAS_STATUS_SUCCESS &&
                  count == 1 && result.state == CUBLAS_STATUS_SUCCESS;
        if (preference)
          cublasLtMatmulPreferenceDestroy(preference);
        if (valid) {
          plan->algorithm = result.algo;
          plan->bytes = result.workspaceSize;
        }
        plan->valid = valid;
        found = classifier_plans.emplace(key, std::move(plan)).first;
      }
      if (found != classifier_plans.end() && found->second->valid) {
        auto& plan = *found->second;
        auto pointer = bias.const_data_ptr();
        if (plan.bytes > 0 && !classifier_workspace.defined()) {
          auto permit = infinity::reserve_workspace_memory(device.index(), limit);
          // Keep supported descriptors retryable while another model owns the
          // credit. The stable tensor and permit outlive all captured uses.
          if (!permit)
            return gelu ? at::_addmm_activation(bias, x, matrix.t(), 1, 1, true)
                        : at::linear(x, matrix, bias);
          auto workspace = at::empty({static_cast<int64_t>(limit)}, x.options().dtype(at::kByte));
          classifier_workspace = std::move(workspace);
          classifier_workspace_permit = std::move(permit);
        }
        if (cublasLtMatmulDescSetAttribute(
                plan.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &pointer, sizeof(pointer)) ==
            CUBLAS_STATUS_SUCCESS) {
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
                             plan.bytes ? classifier_workspace.mutable_data_ptr() : nullptr,
                             plan.bytes,
                             at::cuda::getCurrentCUDAStream());
          if (status == CUBLAS_STATUS_ALLOC_FAILED)
            throw infinity::VendorAllocationError{"BERT classifier cublasLtMatmul",
                                                  static_cast<int32_t>(status)};
          if (status == CUBLAS_STATUS_SUCCESS)
            return output;
        }
        // Match ATen's retry policy and stop resubmitting this failed plan.
        plan.valid = false;
      }
    }
#endif
    return gelu ? at::_addmm_activation(bias, x, matrix.t(), 1, 1, true)
                : at::linear(x, matrix, bias);
  }

  /// Project model-dtype rows through the classifier projection recipe and loaded bias.
  /// Quantized feed-forward projections are owned by DenseMlp.
  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    return classifier_projection(x, w(name + ".weight"), w(name + ".bias"));
  }

  infinity::MlpContext mlp_context() const {
    infinity::MlpContext context;
    context.gemm = [this](const infinity::MlpProjection& projection, const at::Tensor& rows) {
      return classifier_projection(rows, projection.weight(), projection.gemm_bias());
    };
    context.activated_gemm = [this](const infinity::MlpProjection& projection,
                                    const at::Tensor& rows,
                                    infinity::MlpEpilogue epilogue) {
      if (epilogue == infinity::MlpEpilogue::Relu)
        return at::_addmm_activation(projection.bias(), rows, projection.weight().t(), 1, 1, false);
      return classifier_projection(rows,
                                   projection.weight(),
                                   projection.gemm_bias(),
                                   epilogue == infinity::MlpEpilogue::Gelu);
    };
    return context;
  }

  at::Tensor mlp(const at::Tensor& x, const std::string& p) const {
    return mlps.at(p).forward(x, mlp_context());
  }

  /// Apply affine LayerNorm over hidden features; the checkpoint supplies both scale and bias.
  at::Tensor norm(const at::Tensor& x, const std::string& name) const {
    return at::layer_norm(
        x, {config.hidden}, w(name + ".weight"), w(name + ".bias"), config.epsilon);
  }

  /// Normalize projected rows plus their residual. Qualified exact paths compute statistics from
  /// the unrounded sum; other CUDA and CPU paths retain their existing fused or composed recipe.
  at::Tensor
  residual_norm(const at::Tensor& x, const at::Tensor& residual, const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if (x.is_cuda() && (exact_biased_norm || config.hidden == 384 ||
                        (private_flash_embedding && options.boolean("_fp8_dynamic", false))))
      return infinity::encoder_layer_norm_only(
          x, w(name + ".weight"), config.epsilon, residual, w(name + ".bias"), false);
    if (x.is_cuda())
      return infinity::fused_add_layer_norm(
          x, residual, w(name + ".weight"), w(name + ".bias"), config.epsilon);
#endif
    return norm(x + residual, name);
  }

  /// Require the checkpoint tensor to match every expected dimension before
  /// preparing inference state.
  void check(const std::string& name, std::initializer_list<int64_t> shape) const {
    TORCH_CHECK(w(name).sizes() == at::IntArrayRef(shape), "Invalid shape for ", name);
  }

  /// Resolve checkpoint namespaces, validate parameter geometry, and prepare eligible native
  /// kernels. This load-time qualification keeps per-request dispatch tied
  /// to the supported numerical profile.
  void ready() override {
    TORCH_CHECK(
        device.is_cpu() || device.is_cuda(),
        "LibTorch 2.14.1 varlen Flash Attention requires CUDA. CPU supports an unpadded reference path; MPS/XPU are not implemented.");
    if (device.is_cuda()) {
      TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                  "CUDA varlen attention requires float16 or bfloat16");
#ifdef TEI_TORCH_CUDA_KERNELS
      const auto major = at::cuda::getDeviceProperties(device.index())->major;
      cudnn_varlen = options.boolean("_cudnn_varlen", false) && (major == 9 || major == 10) &&
                     at::detail::getCUDAHooks().versionRuntimeCuDNN() >= 91800;
#endif
      const auto head = config.hidden / config.heads;
      TORCH_CHECK(head % 8 == 0 && head <= 256,
                  "CUDA varlen attention requires a head dimension divisible by 8, at most 256");
    }
    roberta = options.string("model_type", "bert") != "bert";
    prefix = "";
    for (const auto* stem : {"bert.", "roberta.", "xlm-roberta.", "camembert.", ""}) {
      if (weights.count(std::string(stem) + "embeddings.word_embeddings.weight")) {
        prefix = stem;
        break;
      }
    }
#ifdef TEI_TORCH_CUDA_KERNELS
    // Standard small/large BERT CLS embeddings use the same packed and biased
    // normalization contracts as the existing base-width implementation.
    native_bert_embedding =
        device.is_cuda() && dtype == at::kHalf && !roberta &&
        ((config.hidden == 384 && config.heads == 12) ||
         (config.hidden == 1024 && config.heads == 16)) &&
        config.intermediate == 4 * config.hidden && classification_width() == 0 &&
        options.string("_embedding_pool") == "cls" && !options.boolean("_multi_vector", false) &&
        !options.boolean("_fp8_dynamic", false) &&
        (config.activation == 0 || config.activation == 1) &&
        at::cuda::getDeviceProperties(device.index())->major >= 8;
    // Qualified H100 FP16 BERT embedding geometry. Dynamic FP8 also needs
    // the original Flash rounding before quantization, including trained heads.
    // BERT classifier backbones share this geometry; keep SPLADE and RoBERTa selectors unchanged.
    private_flash_embedding =
        device.is_cuda() && dtype == at::kHalf && !roberta && config.hidden == 768 &&
        config.heads == 12 && (config.activation == 0 || config.activation == 1) &&
        at::cuda::getDeviceProperties(device.index())->major == 9 &&
        (options.string("_pooling") != "splade" || options.boolean("_fp8_dynamic", false));
    // RoBERTa classifier projections use the same biased/GELU GEMM contract;
    // retain its existing attention selector while reusing projection plans.
    // BF16 RoBERTa retains its storage type throughout the same epilogue.
    cached_classifier_gemm =
        (private_flash_embedding || roberta || (!roberta && dtype == at::kBFloat16)) &&
        device.is_cuda() && (dtype == at::kHalf || dtype == at::kBFloat16) &&
        (config.hidden == 768 || config.hidden == 384) && config.heads == 12 &&
        (config.activation == 0 || config.activation == 1) &&
        at::cuda::getDeviceProperties(device.index())->major == 9 && classification_width() > 0 &&
        !options.boolean("_fp8_dynamic", false) &&
        at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt;
    // Reuse the bounded biased-GEMM plans for standard CLS embeddings too.
    // ATen's default addmm path already selects cuBLASLt for these layouts.
    cached_classifier_gemm |=
        native_bert_embedding && !at::globalContext().deterministicAlgorithms() &&
        !at::globalContext().allowFP16AccumulationCuBLAS() &&
        at::globalContext().allowFP16ReductionCuBLAS() ==
            at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK &&
        !at::cuda::tunable::getTuningContext()->IsTunableOpEnabled() &&
        (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublas ||
         at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt);
#endif
#ifdef TEI_TORCH_CUDA_KERNELS
    // BERT and its aliases share Candle's biased LayerNorm primitive: residual
    // statistics use the unrounded FP32 sum. Kernel eligibility is independent
    // of model alias, pooling and head architecture; other profiles retain their selectors.
    exact_biased_norm =
        device.is_cuda() && (dtype == at::kHalf || dtype == at::kBFloat16) &&
        (native_bert_embedding || ((config.hidden == 768 || config.hidden == 384) &&
                                   at::cuda::getDeviceProperties(device.index())->major == 9));
    // These trained profiles use Candle's packed attention arithmetic, including
    // long ragged inputs. Dynamic FP8 retains its existing selector.
    private_flash_exact_classifier =
        device.is_cuda() && (dtype == at::kHalf || dtype == at::kBFloat16) &&
        classification_width() > 0 && config.heads == 12 &&
        ((roberta && config.hidden == 384) ||
         (!roberta && dtype == at::kBFloat16 && config.hidden == 768)) &&
        (config.activation == 0 || config.activation == 1) &&
        at::cuda::getDeviceProperties(device.index())->major == 9 &&
        !options.boolean("_fp8_dynamic", false);
    // Initial measured RoBERTa eager scope: H100 FP16 hidden768/head64 mean
    // embeddings. Keep dynamic FP8, trained heads and other pools unchanged.
    private_flash_roberta_embedding =
        device.is_cuda() && dtype == at::kHalf && options.string("model_type") == "roberta" &&
        config.hidden == 768 && config.heads == 12 &&
        (config.activation == 0 || config.activation == 1) &&
        at::cuda::getDeviceProperties(device.index())->major == 9 && classification_width() == 0 &&
        options.string("_embedding_pool") == "mean" && !options.boolean("_fp8_dynamic", false);
#endif
    check("embeddings.word_embeddings.weight", {config.vocab, config.hidden});
    check("embeddings.position_embeddings.weight", {config.positions, config.hidden});
    check("embeddings.token_type_embeddings.weight", {config.types, config.hidden});
    auto norm_check = [&](const std::string& p) {
      check(p + ".weight", {config.hidden});
      check(p + ".bias", {config.hidden});
    };
    auto linear_check = [&](const std::string& p, int64_t out, int64_t in) {
      check(p + ".weight", {out, in});
      check(p + ".bias", {out});
    };
    norm_check("embeddings.LayerNorm");
    for (int64_t i = 0; i < config.layers; ++i) {
      std::string p = "encoder.layer." + std::to_string(i) + ".";
      for (auto qkv : {"query", "key", "value"})
        linear_check(p + "attention.self." + qkv, config.hidden, config.hidden);
      weights[prefix + p + "attention.self.qkv.weight"] =
          at::cat({w(p + "attention.self.query.weight"),
                   w(p + "attention.self.key.weight"),
                   w(p + "attention.self.value.weight")},
                  0);
      weights[prefix + p + "attention.self.qkv.bias"] =
          at::cat({w(p + "attention.self.query.bias"),
                   w(p + "attention.self.key.bias"),
                   w(p + "attention.self.value.bias")},
                  0);
      linear_check(p + "attention.output.dense", config.hidden, config.hidden);
      norm_check(p + "attention.output.LayerNorm");
      linear_check(p + "intermediate.dense", config.intermediate, config.hidden);
      linear_check(p + "output.dense", config.hidden, config.intermediate);
      norm_check(p + "output.LayerNorm");
      const bool exact_gelu =
          options.boolean("_multi_vector", false) && options.string("hidden_act", "gelu") == "gelu";
      infinity::MlpActivationRecipe activation(
          exact_gelu               ? infinity::MlpActivation::Gelu
          : config.activation == 3 ? infinity::MlpActivation::Silu
          : config.activation == 2 ? infinity::MlpActivation::Relu
                                   : infinity::MlpActivation::GeluTanh);
      if (!exact_gelu && config.activation != 3) {
        activation.epilogue =
            config.activation == 2 ? infinity::MlpEpilogue::Relu : infinity::MlpEpilogue::Gelu;
      }
      if (!exact_gelu && config.activation != 2) {
        const auto kind = config.activation == 3 ? infinity::MlpActivation::RoundedSilu
                                                 : infinity::MlpActivation::RoundedGelu;
        activation.quantized_custom = [kind](const at::Tensor& rows) {
          return infinity::mlp_activation(rows, kind);
        };
        if (private_flash_embedding && config.activation != 3)
          activation.fusion = infinity::MlpFusion::RoundedBiasGelu;
      }
      auto& mlp = mlps[p];
      mlp =
          infinity::DenseMlp({w(p + "intermediate.dense.weight"), w(p + "intermediate.dense.bias")},
                             {w(p + "output.dense.weight"), w(p + "output.dense.bias")},
                             std::move(activation));
      if (options.boolean("_fp8_dynamic", false)) {
        mlp.prepare(infinity::MlpPrecision::Fp8);
        for (const auto& name : {p + "intermediate.dense", p + "output.dense"}) {
          weights.erase(prefix + name + ".weight");
        }
      }
    }
    if (weights.count("classifier.dense.weight"))
      classifier_mlp.emplace(
          infinity::MlpProjection{infinity::weight(weights, "classifier.dense.weight"),
                                  infinity::weight(weights, "classifier.dense.bias")},
          infinity::MlpProjection{infinity::weight(weights, "classifier.out_proj.weight"),
                                  infinity::weight(weights, "classifier.out_proj.bias")},
          infinity::MlpActivationRecipe{infinity::MlpActivation::Tanh});
  }

  int64_t classification_width() const override {
    auto name = weights.count("classifier.out_proj.weight") ? "classifier.out_proj.weight"
                                                            : "classifier.weight";
    auto it = weights.find(name);
    return it == weights.end() ? 0 : it->second.size(0);
  }

  /// Apply a task-head projection from the root checkpoint namespace,
  /// independently of the backbone prefix.
  at::Tensor root_linear(const at::Tensor& h, const std::string& name) const {
    return classifier_projection(
        h, infinity::weight(weights, name + ".weight"), infinity::weight(weights, name + ".bias"));
  }

  /// Apply a root task-head projection followed by its activation code: ReLU (2), SiLU
  /// (3), or tanh GELU otherwise.
  at::Tensor head_activation(const at::Tensor& h,
                             const at::Tensor& matrix,
                             const at::Tensor& bias,
                             int activation) const {
    if (h.is_cuda() && h.dim() == 2 && bias.defined() && bias.dim() == 1 &&
        (h.scalar_type() == at::kHalf || h.scalar_type() == at::kBFloat16) && activation != 3)
      return at::_addmm_activation(bias, h, matrix.t(), 1, 1, activation != 2);
    auto projected = at::linear(h, matrix, bias);
    return activation == 3   ? at::silu(projected)
           : activation == 2 ? at::relu(projected)
                             : at::gelu(projected, "tanh");
  }

  /// Classify every token or each sequence's first packed token using the checkpoint's task head.
  /// Optional BERT pooler and RoBERTa dense/tanh stages are part
  /// of sequence readout, not embeddings.
  at::Tensor predict(const infinity::PackedInput& input, bool tokenwise) const override {
    auto h = forward(input);
    if (!tokenwise) {
      auto indices = input.cumulative.narrow(0, 0, input.batch).to(at::kLong);
      h = h.index_select(0, indices);
    }
    if (classifier_mlp)
      return classifier_mlp->forward(h, mlp_context());
    if (!tokenwise) {
      // Candle's optional classifier pooler namespace is independent of the backbone.
      const auto pooler = weights.find("bert.pooler.dense.weight");
      if (pooler != weights.end() && pooler->second.dim() == 2 &&
          pooler->second.size(0) == config.hidden && pooler->second.size(1) == config.hidden) {
        const auto bias = weights.find("bert.pooler.dense.bias");
        TORCH_CHECK(bias != weights.end() && bias->second.dim() == 1 &&
                        bias->second.numel() == config.hidden,
                    "Invalid BERT classifier pooler bias");
        h = at::tanh(root_linear(h, "bert.pooler.dense"));
      }
    }
    return root_linear(h, "classifier");
  }

  /// Return nonnegative log1p vocabulary scores for SPLADE before sequence-level max pooling. Use
  /// the checkpoint decoder matrix when present, otherwise the tied word embedding matrix.
  at::Tensor token_scores(const at::Tensor& h) const override {
    at::Tensor transformed, matrix, bias;
    if (roberta) {
      transformed = head_activation(h,
                                    infinity::weight(weights, "lm_head.dense.weight"),
                                    infinity::weight(weights, "lm_head.dense.bias"),
                                    1);
      transformed = at::layer_norm(transformed,
                                   {config.hidden},
                                   infinity::weight(weights, "lm_head.layer_norm.weight"),
                                   infinity::weight(weights, "lm_head.layer_norm.bias"),
                                   config.epsilon);
      matrix = weights.count("lm_head.decoder.weight") ? weights.at("lm_head.decoder.weight")
                                                       : w("embeddings.word_embeddings.weight");
      bias = infinity::weight(weights, "lm_head.bias");
    } else {
      transformed =
          head_activation(h,
                          infinity::weight(weights, "cls.predictions.transform.dense.weight"),
                          infinity::weight(weights, "cls.predictions.transform.dense.bias"),
                          config.activation);
      transformed =
          at::layer_norm(transformed,
                         {config.hidden},
                         infinity::weight(weights, "cls.predictions.transform.LayerNorm.weight"),
                         infinity::weight(weights, "cls.predictions.transform.LayerNorm.bias"),
                         config.epsilon);
      matrix = weights.count("cls.predictions.decoder.weight")
                   ? weights.at("cls.predictions.decoder.weight")
                   : w("embeddings.word_embeddings.weight");
      bias = infinity::weight(weights, "cls.predictions.bias");
    }
    return at::log1p(head_activation(transformed, matrix, bias, 2));
  }

  int64_t output_width() const override {
    return config.hidden;
  }

  /// Return [packed_tokens, hidden] states with sequence-local bidirectional attention. Selected
  /// native embedding, norm, and activation paths preserve their qualified rounding stages.
  at::Tensor forward(const infinity::PackedInput& input) const override {
    const auto& ids = input.ids;
    const auto& types = input.types;
    const auto& positions = input.positions;
    const auto& cumulative = input.cumulative;
    const auto* ends = input.offsets;
    const auto batch = input.batch, max_sequence = input.max_sequence;
    at::Tensor h;
#ifdef TEI_TORCH_CUDA_KERNELS
    if (private_flash_roberta_embedding && !exact_biased_norm) {
      h = infinity::fused_embedding_layer_norm(ids,
                                               types,
                                               positions,
                                               w("embeddings.word_embeddings.weight"),
                                               w("embeddings.token_type_embeddings.weight"),
                                               w("embeddings.position_embeddings.weight"),
                                               w("embeddings.LayerNorm.weight"),
                                               w("embeddings.LayerNorm.bias"),
                                               config.epsilon);
    } else
#endif
    {
      auto word = at::embedding(w("embeddings.word_embeddings.weight"), ids);
      auto token_type = at::embedding(w("embeddings.token_type_embeddings.weight"), types);
      auto position = at::embedding(w("embeddings.position_embeddings.weight"), positions);
#ifdef TEI_TORCH_CUDA_KERNELS
      if (word.is_cuda() && exact_biased_norm && config.hidden == 768 &&
          word.scalar_type() == at::kHalf)
        h = infinity::encoder_embedding_layer_norm(word,
                                                   token_type,
                                                   position,
                                                   w("embeddings.LayerNorm.weight"),
                                                   config.epsilon,
                                                   w("embeddings.LayerNorm.bias"));
      else if (word.is_cuda() && (exact_biased_norm || (private_flash_embedding &&
                                                        options.boolean("_fp8_dynamic", false))))
        h = infinity::encoder_layer_norm_only(word + token_type,
                                              w("embeddings.LayerNorm.weight"),
                                              config.epsilon,
                                              position,
                                              w("embeddings.LayerNorm.bias"),
                                              false);
      else if (word.is_cuda() && config.hidden == 384)
        h = infinity::encoder_layer_norm_only(word + token_type + position,
                                              w("embeddings.LayerNorm.weight"),
                                              config.epsilon,
                                              at::Tensor(),
                                              w("embeddings.LayerNorm.bias"),
                                              false);
      else if (word.is_cuda())
        h = infinity::fused_add_layer_norm(word,
                                           token_type,
                                           w("embeddings.LayerNorm.weight"),
                                           w("embeddings.LayerNorm.bias"),
                                           config.epsilon,
                                           position);
      else
#endif
        h = norm(word + token_type + position, "embeddings.LayerNorm");
    }
    const int64_t tokens = ids.size(0), head = config.hidden / config.heads;
    // ColBERT query expansion keeps MASK tokens as output queries, but excludes
    // them from attention keys. Pack real keys separately; never pad activations.
    std::vector<int64_t> selected_keys;
    std::vector<int32_t> key_offsets{0};
    at::Tensor key_indices, key_cumulative;
    int64_t max_keys = 0;
    const auto query_marker = options.integer("_colbert_query_marker", -1);
    const auto mask_token = options.integer("_colbert_mask_token", -1);
    if (options.boolean("_multi_vector", false) && query_marker >= 0 && mask_token >= 0 &&
        input.host_ids.defined()) {
      const auto* host = input.host_ids.const_data_ptr<int64_t>();
      for (int64_t row = 0; row < input.batch; ++row) {
        const auto begin = input.offsets[row], end = input.offsets[row + 1];
        const bool query = end - begin > 1 && host[begin + 1] == query_marker;
        for (int64_t token = begin; token < end; ++token)
          if (!query || host[token] != mask_token)
            selected_keys.push_back(token);
        max_keys = std::max(max_keys, int64_t(selected_keys.size()) - key_offsets.back());
        key_offsets.push_back(selected_keys.size());
      }
      if (selected_keys.size() != static_cast<size_t>(tokens)) {
        key_indices = at::tensor(selected_keys, at::TensorOptions().dtype(at::kLong)).to(device);
        key_cumulative = at::tensor(key_offsets, at::TensorOptions().dtype(at::kInt)).to(device);
      }
    }
    for (int64_t i = 0; i < config.layers; ++i) {
      std::string p = "encoder.layer." + std::to_string(i) + ".";
      // Q/K/V are disjoint views of one projection: [tokens, 3, heads, head_dim]. All query rows
      // remain present even when ColBERT uses a shorter key sequence.
      auto qkv = linear(h, p + "attention.self.qkv").view({tokens, 3, config.heads, head});
      auto q = qkv.select(1, 0), k = qkv.select(1, 1), v = qkv.select(1, 2);
      at::Tensor attended;
      if (key_indices.defined()) {
        // Only K/V follow selected_keys. Query boundaries still describe the complete sequence, so
        // expansion MASK rows continue to produce output embeddings.
        auto keys = k.index_select(0, key_indices).contiguous(),
             values = v.index_select(0, key_indices).contiguous();
        if (device.is_cuda()) {
          attended = std::get<0>(at::_flash_attention_forward(q.contiguous(),
                                                              keys,
                                                              values,
                                                              cumulative,
                                                              key_cumulative,
                                                              max_sequence,
                                                              max_keys,
                                                              0.,
                                                              false,
                                                              false,
                                                              std::nullopt,
                                                              -1,
                                                              -1));
        } else {
          std::vector<at::Tensor> sequences;
          for (int64_t row = 0; row < input.batch; ++row) {
            auto queries =
                q.narrow(0, input.offsets[row], input.offsets[row + 1] - input.offsets[row])
                    .transpose(0, 1)
                    .unsqueeze(0);
            auto slice = [&](const at::Tensor& x) {
              return x.narrow(0, key_offsets[row], key_offsets[row + 1] - key_offsets[row])
                  .transpose(0, 1)
                  .unsqueeze(0);
            };
            sequences.push_back(at::scaled_dot_product_attention(
                                    queries, slice(keys), slice(values), std::nullopt, 0., false)
                                    .squeeze(0)
                                    .transpose(0, 1));
          }
          attended = at::cat(sequences, 0).reshape({tokens, config.hidden});
        }
      } else if (device.is_cuda()) {
        // The same ATen operator used by torch.nn.attention.varlen.varlen_attn's
        // Flash Attention path. Q/K/V stay [total_tokens, heads, head_dim].
        // Cumulative INT32 offsets are on-device. No padded batch or mask exists.
        // cuDNN helps long sequences; FlashAttention is faster on the measured
        // ragged batches with maximum length 256, despite a similar token total.
#ifdef TEI_TORCH_CUDA_KERNELS
        // Native Torch packed Flash avoids cuDNN setup overhead for moderate
        // token totals. FP8 needs the fused Torch arithmetic at every total
        // to preserve quantization inputs; normal FP16 larger totals retain
        // the measured cuDNN advantage.
        if (!cudnn_varlen)
          attended = infinity::infinity_fa4(q,
                                            k,
                                            v,
                                            cumulative,
                                            cumulative,
                                            max_sequence,
                                            1.0 / std::sqrt(static_cast<double>(head)));
        if (attended.defined()) {
          // InfinityFA4 consumed the original packed views on this stream.
        } else if ((private_flash_exact_classifier || native_bert_embedding) && head == 32) {
          attended = infinity::torch_flash_exact32(
              q, k, v, cumulative, max_sequence, 1.0 / std::sqrt(static_cast<double>(head)));
        } else if ((private_flash_exact_classifier ||
                    ((private_flash_embedding || private_flash_roberta_embedding) &&
                     (tokens <= 4096 ||
                      (private_flash_embedding && options.boolean("_fp8_dynamic", false))))) &&
                   infinity::torch_flash_fma_supported(q, k, v)) {
          attended = infinity::torch_flash_fma(
              q, k, v, cumulative, max_sequence, 1.0 / std::sqrt(static_cast<double>(head)), false);
        } else
#endif
            if (cudnn_varlen && max_sequence >= 512) {
          attended = std::get<0>(at::_cudnn_attention_forward(q,
                                                              k,
                                                              v,
                                                              std::nullopt,
                                                              cumulative,
                                                              cumulative,
                                                              max_sequence,
                                                              max_sequence,
                                                              false,
                                                              0.0,
                                                              false,
                                                              false));
        } else {
          attended = std::get<0>(at::_flash_attention_forward(q,
                                                              k,
                                                              v,
                                                              cumulative,
                                                              cumulative,
                                                              max_sequence,
                                                              max_sequence,
                                                              0.0,
                                                              false,
                                                              false,
                                                              std::nullopt,
                                                              -1,
                                                              -1));
        }
      } else {
        // CPU-only reference path: each actual sequence uses SDPA at its exact length.
        // The GPU path never falls back to this loop or to dense padded attention.
        std::vector<at::Tensor> sequences;
        for (int64_t row = 0; row < batch; ++row) {
          auto slice = [&](const at::Tensor& t) {
            return t.narrow(0, ends[row], ends[row + 1] - ends[row]).transpose(0, 1).unsqueeze(0);
          };
          sequences.push_back(at::scaled_dot_product_attention(
                                  slice(q), slice(k), slice(v), std::nullopt, 0.0, false)
                                  .squeeze(0)
                                  .transpose(0, 1));
        }
        attended = at::cat(sequences, 0);
      }
      // Collapse heads only after attention. Each post-norm block consumes its own projected output
      // and the preceding hidden residual before the next MLP begins.
      attended = attended.contiguous().view({tokens, config.hidden});
      h = residual_norm(
          linear(attended, p + "attention.output.dense"), h, p + "attention.output.LayerNorm");
      h = residual_norm(mlp(h, p), h, p + "output.LayerNorm");
    }
    return h;
  }
};
}
