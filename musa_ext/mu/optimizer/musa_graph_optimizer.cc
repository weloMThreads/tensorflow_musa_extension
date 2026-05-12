/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "graph/fusion/fusion_pattern_manager.h"
#include "graph/fusion/gelu_fusion.h"
#include "graph/fusion/layernorm_fusion.h"
#include "mu/optimizer/graph_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

// Import musa namespace for graph utils
using namespace ::tensorflow::grappler::musa;

namespace {

// Device type for MUSA
constexpr char kMusaDeviceType[] = "MUSA";
constexpr char kDisabledFusionPatternsParam[] = "disabled_fusion_patterns";

std::string NormalizeFusionPatternName(const std::string& value) {
  size_t begin = 0;
  size_t end = value.size();

  while (begin < end &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }

  std::string normalized = value.substr(begin, end - begin);
  std::transform(
      normalized.begin(), normalized.end(), normalized.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return normalized;
}

bool EnvFlagEnabledLocal(const char* env_name) {
  const char* env_val = std::getenv(env_name);
  if (env_val == nullptr) return false;
  const std::string value(env_val);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "YES" || value == "on" ||
         value == "ON";
}

// Tri-state configuration for optimizers
enum class TriState { kDefault = 0, kOff = 1, kOn = 2 };

// Optimizer configurations - controls interaction with TensorFlow built-in
// optimizers Based on TF Modular Graph C API TP_OptimizerConfigs
// CRITICAL FIX: Restore to kDefault to match working 9ded154 configuration
struct MusaOptimizerConfigs {
  // Keep as Default for stability (was kOn, causing OOM and illegal memory
  // access)
  TriState arithmetic_optimization = TriState::kDefault;
  TriState constant_folding = TriState::kDefault;
  TriState remapping = TriState::kDefault;
  TriState shape_optimization = TriState::kDefault;

  // Restore to kDefault to avoid unexpected memory layout changes
  TriState implementation_selector = TriState::kDefault;
  TriState function_optimization = TriState::kDefault;
  TriState common_subgraph_elimination = TriState::kDefault;
  TriState memory_optimization = TriState::kDefault;

  // Inference-specific optimizations - use Default for safety
  TriState debug_stripper = TriState::kDefault;
  TriState pin_to_host_optimization = TriState::kDefault;

  // Keep disabled (handled internally by MUSA)
  TriState auto_mixed_precision = TriState::kOff;
  TriState layout_optimizer = TriState::kOff;

  // Keep as Default or enable as needed
  TriState disable_model_pruning = TriState::kDefault;
  TriState loop_optimization = TriState::kDefault;
  TriState dependency_optimization = TriState::kDefault;
  TriState auto_parallel = TriState::kDefault;
  TriState scoped_allocator_optimization = TriState::kDefault;
  TriState optimizer_remove_ios_node = TriState::kDefault;
};

// MUSA AMP Configuration
class MusaAmpConfig {
 public:
  std::unordered_set<string> fp16_compute_ops = {"MatMul",
                                                 "BatchMatMul",
                                                 "BatchMatMulV2",
                                                 "Conv2D",
                                                 "Conv2DBackpropInput",
                                                 "Conv2DBackpropFilter",
                                                 "DepthwiseConv2dNative",
                                                 "Conv3D",
                                                 "FusedBatchNorm",
                                                 "FusedBatchNormV2",
                                                 "FusedBatchNormV3"};

  std::unordered_set<string> fp32_keep_ops = {
      "Softmax",
      "LogSoftmax",
      "SoftmaxCrossEntropyWithLogits",
      "SparseSoftmaxCrossEntropyWithLogits",
      "SigmoidCrossEntropyWithLogits",
      "Mean",
      "Sum",
      "Prod",
      "L2Loss",
      "Norm",
      "Exp",
      "Log",
      "Sqrt",
      "Rsqrt",
      "Reciprocal",
      "Square"};

  std::unordered_set<string> conditional_ops = {
      "Add", "AddV2", "Sub", "Mul", "Div", "BiasAdd", "BiasAddGrad"};

  std::unordered_set<string> activation_ops = {
      "Relu", "Relu6", "Elu", "Selu", "LeakyRelu", "Sigmoid", "Tanh"};

  bool aggressive_mode = false;
  DataType target_dtype = DT_HALF;
};

// Graph utilities
class MusaGraphUtils {
 public:
  static NodeDef* CreateConstNode(GraphDef* graph, const string& name,
                                  const std::vector<int32>& values,
                                  const string& device) {
    NodeDef* node = graph->add_node();
    node->set_name(name);
    node->set_op("Const");
    node->set_device(device);

    auto* attr = node->mutable_attr();
    (*attr)["dtype"].set_type(DT_INT32);
    auto* tensor = (*attr)["value"].mutable_tensor();
    tensor->set_dtype(DT_INT32);
    tensor->mutable_tensor_shape()->add_dim()->set_size(values.size());

    for (int32 v : values) {
      tensor->add_int_val(v);
    }
    return node;
  }

  static NodeDef* InsertTranspose(GraphDef* graph, const string& base_name,
                                  const string& input_name,
                                  const std::vector<int32>& perm,
                                  DataType dtype, const string& device) {
    string perm_node_name = base_name + "/perm";
    CreateConstNode(graph, perm_node_name, perm, device);

    NodeDef* node = graph->add_node();
    node->set_name(base_name);
    node->set_op("Transpose");
    node->set_device(device);
    node->add_input(input_name);
    node->add_input(perm_node_name);

    auto* attr = node->mutable_attr();
    (*attr)["T"].set_type(dtype);
    (*attr)["Tperm"].set_type(DT_INT32);

    return node;
  }

  static NodeDef* InsertCast(GraphDef* graph, const string& name,
                             const string& input_name, DataType src_dtype,
                             DataType dst_dtype, const string& device) {
    NodeDef* node = graph->add_node();
    node->set_name(name);
    node->set_op("Cast");
    node->set_device(device);
    node->add_input(input_name);

    auto* attr = node->mutable_attr();
    (*attr)["SrcT"].set_type(src_dtype);
    (*attr)["DstT"].set_type(dst_dtype);
    (*attr)["Truncate"].set_b(false);

    return node;
  }

  static void RedirectEdges(GraphDef* graph, const string& old_node_name,
                            const string& new_node_name) {
    for (int i = 0; i < graph->node_size(); ++i) {
      NodeDef* node = graph->mutable_node(i);
      if (node->name() == new_node_name) continue;

      for (int j = 0; j < node->input_size(); ++j) {
        if (node->input(j) == old_node_name) {
          node->set_input(j, new_node_name);
        }
      }
    }
  }

  static void RewriteLayoutAttributes(NodeDef* node) {
    auto* attr = node->mutable_attr();
    std::vector<string> layout_attrs = {"strides", "dilations"};

    for (const string& attr_name : layout_attrs) {
      if (attr->count(attr_name)) {
        auto* list = (*attr)[attr_name].mutable_list();
        if (list->i_size() == 4) {
          int64_t h = list->i(1);
          int64_t w = list->i(2);
          list->set_i(1, 1);
          list->set_i(2, h);
          list->set_i(3, w);
        }
      }
    }
  }

  static bool IsMusaNCHWSupported(const NodeDef& node) {
    if (node.device().find(kMusaDeviceType) == std::string::npos) return false;
    return kLayoutSensitiveOps(node) || kLayoutAgnosticOps(node);
  }

  static bool kLayoutSensitiveOps(const NodeDef& node) {
    static const std::unordered_set<string> sensitive_ops = {
        "Conv2D",  "DepthwiseConv2dNative", "MaxPool",
        "AvgPool", "FusedBatchNorm",        "FusedBatchNormV3"};
    return sensitive_ops.count(node.op()) > 0;
  }

  static bool kLayoutAgnosticOps(const NodeDef& node) {
    static const std::unordered_set<string> agnostic_ops = {
        "Relu", "Sigmoid", "Tanh", "BiasAdd", "Add", "Sub", "Mul", "Identity"};
    return agnostic_ops.count(node.op()) > 0;
  }
};

// Check if graph contains MUSA device nodes
bool GraphHasMusaNodes(const GraphDef& graph) {
  for (const auto& node : graph.node()) {
    if (node.device().find(kMusaDeviceType) != std::string::npos) {
      return true;
    }
  }
  return graph.node_size() > 0;
}

bool IsFusionResidualConst(const NodeDef& node) {
  if (node.op() != "Const") {
    return false;
  }
  return node.name().find("/Gelu/") != string::npos ||
         node.name().find("/LayerNorm/") != string::npos;
}

bool IsFullyIsolatedNode(const NodeDef& node) { return node.input_size() == 0; }

int RemoveIsolatedNodes(GraphDef* graph) {
  // Fusion may leave behind folded scalar constants that no longer feed any
  // live node. Also drop nodes that are completely disconnected from the
  // executable graph. Prune them here so the dumped graph reflects the final
  // shape of the executable graph more closely.
  int removed_count = 0;

  while (true) {
    std::unordered_set<string> referenced_nodes;
    referenced_nodes.reserve(graph->node_size());
    for (const auto& node : graph->node()) {
      for (int i = 0; i < node.input_size(); ++i) {
        referenced_nodes.insert(
            ::tensorflow::grappler::musa_fusion::FusionGraphUtils::
                GetProducerNodeName(node.input(i)));
      }
    }

    std::vector<int> isolated_node_indices;
    for (int i = 0; i < graph->node_size(); ++i) {
      const auto& node = graph->node(i);
      const bool has_consumers =
          referenced_nodes.find(node.name()) != referenced_nodes.end();
      if ((IsFusionResidualConst(node) && !has_consumers) ||
          (IsFullyIsolatedNode(node) && !has_consumers)) {
        isolated_node_indices.push_back(i);
      }
    }

    if (isolated_node_indices.empty()) {
      return removed_count;
    }

    std::sort(isolated_node_indices.begin(), isolated_node_indices.end(),
              std::greater<int>());
    for (int node_idx : isolated_node_indices) {
      ::tensorflow::grappler::musa_fusion::FusionGraphUtils::RemoveNode(
          graph, node_idx);
      removed_count++;
    }
  }
}

string NodeNameFromInputLocal(const string& input) {
  if (input.empty()) return "";
  const string data_input = input[0] == '^' ? input.substr(1) : input;
  const size_t colon_pos = data_input.find(':');
  if (colon_pos == std::string::npos) return data_input;
  return data_input.substr(0, colon_pos);
}

bool GetConstIntVector(const NodeDef* node, std::vector<int64_t>* values) {
  if (node == nullptr || node->op() != "Const") return false;
  const auto& attr = node->attr();
  const auto dtype_it = attr.find("dtype");
  const auto value_it = attr.find("value");
  if (dtype_it == attr.end() || value_it == attr.end()) return false;

  const DataType dtype = dtype_it->second.type();
  if (dtype != DT_INT32 && dtype != DT_INT64) return false;

  const TensorProto& tensor = value_it->second.tensor();
  int64_t element_count = 1;
  if (tensor.tensor_shape().unknown_rank()) return false;
  for (const auto& dim : tensor.tensor_shape().dim()) {
    if (dim.size() < 0) return false;
    element_count *= dim.size();
  }

  values->clear();
  if (!tensor.tensor_content().empty()) {
    const string& content = tensor.tensor_content();
    const size_t elem_size = dtype == DT_INT32 ? sizeof(int32) : sizeof(int64_t);
    if (content.size() % elem_size != 0) return false;
    const size_t count = content.size() / elem_size;
    values->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (dtype == DT_INT32) {
        int32 value;
        std::memcpy(&value, content.data() + i * elem_size, elem_size);
        values->push_back(value);
      } else {
        int64_t value;
        std::memcpy(&value, content.data() + i * elem_size, elem_size);
        values->push_back(value);
      }
    }
    return element_count == 0 || static_cast<int64_t>(values->size()) == element_count;
  }

  if (dtype == DT_INT32) {
    if (tensor.int_val_size() == 0) return element_count == 0;
    if (tensor.int_val_size() == 1 && element_count > 1) {
      values->assign(element_count, tensor.int_val(0));
    } else {
      values->reserve(tensor.int_val_size());
      for (int i = 0; i < tensor.int_val_size(); ++i) {
        values->push_back(tensor.int_val(i));
      }
    }
  } else {
    if (tensor.int64_val_size() == 0) return element_count == 0;
    if (tensor.int64_val_size() == 1 && element_count > 1) {
      values->assign(element_count, tensor.int64_val(0));
    } else {
      values->reserve(tensor.int64_val_size());
      for (int i = 0; i < tensor.int64_val_size(); ++i) {
        values->push_back(tensor.int64_val(i));
      }
    }
  }

  return element_count == 0 || static_cast<int64_t>(values->size()) == element_count;
}

bool IsFloatOrBFloat16Type(DataType type) {
  return type == DT_FLOAT || type == DT_BFLOAT16;
}

struct SliceGradSegment {
  const NodeDef* node = nullptr;
  string dy_input;
  std::vector<int64_t> output_shape;
  int64_t start = -1;
  int64_t end = -1;
};

struct DynamicSliceGradSegment {
  const NodeDef* node = nullptr;
  string dy_input;
  std::vector<int64_t> output_shape;
  string split_input;
  bool is_prefix = false;
};

bool TryParseConstAxis1SliceGrad(
    const NodeDef& node,
    const std::unordered_map<string, const NodeDef*>& node_map,
    SliceGradSegment* segment) {
  if (node.op() != "StridedSliceGrad" || node.input_size() < 5) return false;
  const auto t_it = node.attr().find("T");
  if (t_it == node.attr().end() ||
      !IsFloatOrBFloat16Type(t_it->second.type())) {
    return false;
  }

  auto get_input_node = [&](int idx) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(node.input(idx)));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::vector<int64_t> shape;
  std::vector<int64_t> begin;
  std::vector<int64_t> end;
  std::vector<int64_t> strides;
  if (!GetConstIntVector(get_input_node(0), &shape) ||
      !GetConstIntVector(get_input_node(1), &begin) ||
      !GetConstIntVector(get_input_node(2), &end) ||
      !GetConstIntVector(get_input_node(3), &strides)) {
    return false;
  }

  if (shape.size() < 2 || begin.size() < 2 || end.size() < 2 ||
      strides.size() < 2) {
    return false;
  }
  if (shape[1] <= 0 || begin[0] != 0 || end[0] != 0 ||
      strides[0] != 1 || strides[1] != 1) {
    return false;
  }

  const int64_t axis_dim = shape[1];
  int64_t start = -1;
  int64_t finish = -1;
  if (begin[1] == 0 && end[1] > 0 && end[1] < axis_dim) {
    start = 0;
    finish = end[1];
  } else if (begin[1] > 0 && begin[1] < axis_dim &&
             (end[1] == 0 || end[1] == axis_dim)) {
    start = begin[1];
    finish = axis_dim;
  } else {
    return false;
  }

  segment->node = &node;
  segment->dy_input = node.input(4);
  segment->output_shape = std::move(shape);
  segment->start = start;
  segment->end = finish;
  return true;
}

bool IsConstAxis1ZeroVector(const NodeDef* node) {
  std::vector<int64_t> values;
  return GetConstIntVector(node, &values) && values.size() >= 2 &&
         values[0] == 0 && values[1] == 0;
}

bool TryGetPackAxis1Boundary(
    const NodeDef* node,
    const std::unordered_map<string, const NodeDef*>& node_map,
    string* split_input) {
  if (node == nullptr || node->op() != "Pack" || node->input_size() != 2) {
    return false;
  }

  const auto first_it = node_map.find(NodeNameFromInputLocal(node->input(0)));
  if (first_it == node_map.end()) return false;
  std::vector<int64_t> first_values;
  if (!GetConstIntVector(first_it->second, &first_values) ||
      first_values.empty() || first_values[0] != 0) {
    return false;
  }

  *split_input = NodeNameFromInputLocal(node->input(1));
  return !split_input->empty();
}

bool TryParseDynamicAxis1SliceGrad(
    const NodeDef& node,
    const std::unordered_map<string, const NodeDef*>& node_map,
    DynamicSliceGradSegment* segment) {
  if (node.op() != "StridedSliceGrad" || node.input_size() < 5) return false;
  const auto t_it = node.attr().find("T");
  if (t_it == node.attr().end() ||
      !IsFloatOrBFloat16Type(t_it->second.type())) {
    return false;
  }

  auto get_input_node = [&](int idx) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(node.input(idx)));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  if (!GetConstIntVector(get_input_node(0), &shape) ||
      !GetConstIntVector(get_input_node(3), &strides)) {
    return false;
  }
  if (shape.size() < 2 || shape[1] <= 0 || strides.size() < 2 ||
      strides[0] != 1 || strides[1] != 1) {
    return false;
  }

  const NodeDef* begin_node = get_input_node(1);
  const NodeDef* end_node = get_input_node(2);
  string split_input;
  bool is_prefix = false;
  if (IsConstAxis1ZeroVector(begin_node) &&
      TryGetPackAxis1Boundary(end_node, node_map, &split_input)) {
    is_prefix = true;
  } else if (TryGetPackAxis1Boundary(begin_node, node_map, &split_input) &&
             IsConstAxis1ZeroVector(end_node)) {
    is_prefix = false;
  } else {
    return false;
  }

  segment->node = &node;
  segment->dy_input = node.input(4);
  segment->output_shape = std::move(shape);
  segment->split_input = std::move(split_input);
  segment->is_prefix = is_prefix;
  return true;
}


bool TryParseConstAxis1SuffixSliceGrad(
    const NodeDef& node,
    const std::unordered_map<string, const NodeDef*>& node_map,
    SliceGradSegment* segment, int64_t* inner_dim) {
  if (node.op() != "StridedSliceGrad" || node.input_size() < 5) return false;
  const auto t_it = node.attr().find("T");
  if (t_it == node.attr().end() ||
      !IsFloatOrBFloat16Type(t_it->second.type())) {
    return false;
  }

  auto get_input_node = [&](int idx) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(node.input(idx)));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::vector<int64_t> shape;
  std::vector<int64_t> begin;
  std::vector<int64_t> end;
  std::vector<int64_t> strides;
  if (!GetConstIntVector(get_input_node(0), &shape) ||
      !GetConstIntVector(get_input_node(1), &begin) ||
      !GetConstIntVector(get_input_node(2), &end) ||
      !GetConstIntVector(get_input_node(3), &strides)) {
    return false;
  }
  if (shape.size() < 2 || begin.size() < 2 || end.size() < 2 ||
      strides.size() < 2 || shape[1] <= 0 || begin[0] != 0 ||
      end[0] != 0 || strides[0] != 1 || strides[1] != 1) {
    return false;
  }

  const int64_t axis_dim = shape[1];
  const int64_t start = begin[1] < 0 ? axis_dim + begin[1] : begin[1];
  if (start <= 0 || start >= axis_dim ||
      !(end[1] == 0 || end[1] == axis_dim)) {
    return false;
  }

  *inner_dim = 1;
  for (size_t i = 2; i < shape.size(); ++i) {
    *inner_dim *= shape[i];
  }
  segment->node = &node;
  segment->dy_input = node.input(4);
  segment->output_shape = std::move(shape);
  segment->start = start;
  segment->end = axis_dim;
  return *inner_dim > 0;
}

bool IsConstLastAxis(const NodeDef* node) {
  std::vector<int64_t> value;
  if (!GetConstIntVector(node, &value) || value.size() != 1) {
    return false;
  }
  return value[0] == -1 || value[0] == 2;
}

void RedirectNodeOutputs(GraphDef* graph, const string& old_node_name,
                         const string& new_node_name) {
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if (node->name() == new_node_name) continue;
    for (int j = 0; j < node->input_size(); ++j) {
      const string input = node->input(j);
      if (input == old_node_name || input == old_node_name + ":0") {
        node->set_input(j, new_node_name);
      } else if (input == "^" + old_node_name) {
        node->set_input(j, "^" + new_node_name);
      }
    }
  }
}

void RemoveNodesByName(GraphDef* graph,
                       const std::unordered_set<string>& remove_names) {
  if (remove_names.empty()) return;
  GraphDef kept;
  *kept.mutable_versions() = graph->versions();
  *kept.mutable_library() = graph->library();
  for (const auto& node : graph->node()) {
    if (remove_names.find(node.name()) != remove_names.end()) {
      continue;
    }
    *kept.add_node() = node;
  }
  graph->Swap(&kept);
}


bool StringEndsWithLocal(const string& value, const string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool HasFloatTypeAttr(const NodeDef& node) {
  const auto it = node.attr().find("T");
  return it != node.attr().end() && IsFloatOrBFloat16Type(it->second.type());
}

DataType GetNodeTypeAttr(const NodeDef& node) {
  const auto it = node.attr().find("T");
  return it == node.attr().end() ? DT_INVALID : it->second.type();
}

float BFloat16BitsToFloat(uint16_t bits) {
  uint32_t raw = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

bool GetConstScalarFloat(const NodeDef* node, float* value) {
  if (node == nullptr || node->op() != "Const") return false;
  const auto dtype_it = node->attr().find("dtype");
  const auto value_it = node->attr().find("value");
  if (dtype_it == node->attr().end() || value_it == node->attr().end()) {
    return false;
  }
  const DataType dtype = dtype_it->second.type();
  const TensorProto& tensor = value_it->second.tensor();
  if (dtype == DT_BFLOAT16) {
    if (!tensor.tensor_content().empty()) {
      if (tensor.tensor_content().size() < sizeof(uint16_t)) return false;
      uint16_t bits = 0;
      std::memcpy(&bits, tensor.tensor_content().data(), sizeof(bits));
      *value = BFloat16BitsToFloat(bits);
      return true;
    }
    if (tensor.half_val_size() <= 0) return false;
    *value = BFloat16BitsToFloat(static_cast<uint16_t>(tensor.half_val(0)));
    return true;
  }
  if (dtype != DT_FLOAT) return false;
  if (!tensor.tensor_content().empty()) {
    if (tensor.tensor_content().size() < sizeof(float)) return false;
    std::memcpy(value, tensor.tensor_content().data(), sizeof(float));
    return true;
  }
  if (tensor.float_val_size() <= 0) return false;
  *value = tensor.float_val(0);
  return true;
}

bool IsFloatZeroConst(const NodeDef* node) {
  float value = 0.0f;
  return GetConstScalarFloat(node, &value) && value == 0.0f;
}

bool IsOnlyConsumer(const std::unordered_map<string, std::vector<string>>& consumers,
                    const string& producer, const string& consumer) {
  const auto it = consumers.find(producer);
  return it != consumers.end() && it->second.size() == 1 &&
         it->second[0] == consumer;
}

void RedirectDataConsumersToInput(GraphDef* graph, const string& old_node_name,
                                  const string& new_input,
                                  const string& skip_node_name) {
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if (node->name() == skip_node_name) continue;
    for (int j = 0; j < node->input_size(); ++j) {
      const string input = node->input(j);
      if (!input.empty() && input[0] == '^') continue;
      if (NodeNameFromInputLocal(input) == old_node_name) {
        node->set_input(j, new_input);
      }
    }
  }
}

int OptimizeForwardRmsNorm(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] == '^') continue;
      consumers[NodeNameFromInputLocal(input)].push_back(node.name());
    }
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    NodeDef* out = graph->mutable_node(i);
    if (out->op() != "Mul" || out->input_size() != 2 ||
        !HasFloatTypeAttr(*out) ||
        out->name().find("rms_layer_norm") == string::npos ||
        out->name().find("gradient_tape/") != string::npos ||
        (!StringEndsWithLocal(out->name(), "/mul") &&
         !StringEndsWithLocal(out->name(), "/mul_1"))) {
      continue;
    }

    const NodeDef* in0 = find_node(out->input(0));
    const NodeDef* in1 = find_node(out->input(1));
    if (in0 == nullptr || in1 == nullptr) continue;

    const NodeDef* norm = nullptr;
    string gamma_input;
    if ((in0->op() == "Mul" || in0->op() == "RealDiv" ||
         in0->op() == "Div") && HasFloatTypeAttr(*in0)) {
      norm = in0;
      gamma_input = out->input(1);
    } else if ((in1->op() == "Mul" || in1->op() == "RealDiv" ||
                in1->op() == "Div") && HasFloatTypeAttr(*in1)) {
      norm = in1;
      gamma_input = out->input(0);
    } else {
      continue;
    }
    if (remove_names.find(norm->name()) != remove_names.end() ||
        norm->input_size() != 2) {
      continue;
    }

    const NodeDef* scale_node = find_node(gamma_input);
    if (scale_node == nullptr) {
      continue;
    }
    if (scale_node->op() == "Cast") {
      const auto src_it = scale_node->attr().find("SrcT");
      const auto dst_it = scale_node->attr().find("DstT");
      const NodeDef* cast_input =
          scale_node->input_size() > 0 ? find_node(scale_node->input(0)) : nullptr;
      if (src_it == scale_node->attr().end() ||
          dst_it == scale_node->attr().end() ||
          src_it->second.type() != DT_FLOAT ||
          dst_it->second.type() != GetNodeTypeAttr(*out) ||
          cast_input == nullptr || cast_input->op() != "ReadVariableOp") {
        continue;
      }
    } else if (scale_node->op() != "ReadVariableOp") {
      continue;
    }

    const NodeDef* inv_or_denom = nullptr;
    string x_input;
    bool denominator_is_sqrt = false;
    if (norm->op() == "Mul") {
      const NodeDef* norm_in0 = find_node(norm->input(0));
      const NodeDef* norm_in1 = find_node(norm->input(1));
      if (norm_in0 != nullptr && norm_in0->op() == "Rsqrt") {
        inv_or_denom = norm_in0;
        x_input = norm->input(1);
      } else if (norm_in1 != nullptr && norm_in1->op() == "Rsqrt") {
        inv_or_denom = norm_in1;
        x_input = norm->input(0);
      } else {
        continue;
      }
    } else {
      const NodeDef* norm_in0 = find_node(norm->input(0));
      const NodeDef* norm_in1 = find_node(norm->input(1));
      if (norm_in1 != nullptr && norm_in1->op() == "Sqrt") {
        inv_or_denom = norm_in1;
        x_input = norm->input(0);
        denominator_is_sqrt = true;
      } else if (norm_in0 != nullptr && norm_in0->op() == "Sqrt") {
        inv_or_denom = norm_in0;
        x_input = norm->input(1);
        denominator_is_sqrt = true;
      } else {
        continue;
      }
    }
    if (inv_or_denom == nullptr || inv_or_denom->input_size() != 1) continue;

    const NodeDef* add = find_node(inv_or_denom->input(0));
    if (add == nullptr || add->op() != "AddV2" || add->input_size() != 2 ||
        !HasFloatTypeAttr(*add)) {
      continue;
    }

    const NodeDef* add_in0 = find_node(add->input(0));
    const NodeDef* add_in1 = find_node(add->input(1));
    const NodeDef* mean = nullptr;
    const NodeDef* eps = nullptr;
    if (add_in0 != nullptr && add_in0->op() == "Mean" &&
        add_in1 != nullptr && add_in1->op() == "Const") {
      mean = add_in0;
      eps = add_in1;
    } else if (add_in1 != nullptr && add_in1->op() == "Mean" &&
               add_in0 != nullptr && add_in0->op() == "Const") {
      mean = add_in1;
      eps = add_in0;
    } else {
      continue;
    }
    if (mean->input_size() != 2 || !HasFloatTypeAttr(*mean)) continue;

    const auto keep_it = mean->attr().find("keep_dims");
    if (keep_it == mean->attr().end() || !keep_it->second.b()) continue;

    const NodeDef* square = find_node(mean->input(0));
    const NodeDef* reduce = find_node(mean->input(1));
    if (square == nullptr || reduce == nullptr || square->op() != "Square" ||
        square->input_size() != 1 || !HasFloatTypeAttr(*square)) {
      continue;
    }
    if (NodeNameFromInputLocal(square->input(0)) !=
        NodeNameFromInputLocal(x_input)) {
      continue;
    }

    std::vector<int64_t> reduction;
    if (!GetConstIntVector(reduce, &reduction) || reduction.size() != 1 ||
        reduction[0] != -1) {
      continue;
    }

    float epsilon = 0.0f;
    if (!GetConstScalarFloat(eps, &epsilon)) continue;

    if (!IsOnlyConsumer(consumers, square->name(), mean->name()) ||
        !IsOnlyConsumer(consumers, mean->name(), add->name()) ||
        !IsOnlyConsumer(consumers, add->name(), inv_or_denom->name())) {
      continue;
    }

    RedirectDataConsumersToInput(graph, norm->name(), out->name() + ":1",
                                 out->name());
    if (!denominator_is_sqrt) {
      RedirectDataConsumersToInput(graph, inv_or_denom->name(),
                                   out->name() + ":2", out->name());
    }

    const DataType dtype = GetNodeTypeAttr(*out);
    out->set_op("MusaRmsNorm");
    out->clear_input();
    out->add_input(x_input);
    out->add_input(gamma_input);
    out->mutable_attr()->clear();
    (*out->mutable_attr())["T"].set_type(dtype);
    (*out->mutable_attr())["epsilon"].set_f(epsilon);

    remove_names.insert(norm->name());
    if (!denominator_is_sqrt ||
        IsOnlyConsumer(consumers, inv_or_denom->name(), norm->name())) {
      remove_names.insert(inv_or_denom->name());
      remove_names.insert(add->name());
      remove_names.insert(mean->name());
      remove_names.insert(square->name());
      if (IsOnlyConsumer(consumers, eps->name(), add->name())) {
        remove_names.insert(eps->name());
      }
      if (IsOnlyConsumer(consumers, reduce->name(), mean->name())) {
        remove_names.insert(reduce->name());
      }
    }
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " forward RMSNorm node(s)";
  }
  return rewrites;
}


bool ReplaceTwoInputsWithFused(NodeDef* node, const string& first,
                               const string& second,
                               const string& fused_name) {
  bool saw_first = false;
  bool saw_second = false;
  bool inserted = false;
  std::vector<string> new_inputs;
  new_inputs.reserve(node->input_size());
  for (int i = 0; i < node->input_size(); ++i) {
    const string clean = NodeNameFromInputLocal(node->input(i));
    if (clean == first || clean == second) {
      saw_first = saw_first || clean == first;
      saw_second = saw_second || clean == second;
      if (!inserted) {
        new_inputs.push_back(fused_name);
        inserted = true;
      }
      continue;
    }
    new_inputs.push_back(node->input(i));
  }
  if (!saw_first || !saw_second) return false;
  node->clear_input();
  for (const auto& input : new_inputs) node->add_input(input);
  return true;
}

int OptimizeRmsNormGradDxWarp128(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] == '^') continue;
      consumers[NodeNameFromInputLocal(input)].push_back(node.name());
    }
  }

  auto find_node = [&](const string& name) -> const NodeDef* {
    const auto it = node_map.find(name);
    return it == node_map.end() ? nullptr : it->second;
  };

  auto is_read_variable = [&](const string& input) -> bool {
    const NodeDef* node = find_node(NodeNameFromInputLocal(input));
    if (node == nullptr) return false;
    if (node->op() == "ReadVariableOp") return true;
    if (node->op() == "Cast" && node->input_size() == 1) {
      const NodeDef* cast_input = find_node(NodeNameFromInputLocal(node->input(0)));
      return cast_input != nullptr && cast_input->op() == "ReadVariableOp";
    }
    return false;
  };

  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& direct = graph->node(i);
    static const string kDirectSuffix = "/mul/Mul";
    if (direct.op() != "Mul" || direct.input_size() != 2 ||
        !HasFloatTypeAttr(direct) ||
        direct.name().find("gradient_tape/") == string::npos ||
        direct.name().find("rms_layer_norm") == string::npos ||
        !StringEndsWithLocal(direct.name(), kDirectSuffix)) {
      continue;
    }

    const string prefix =
        direct.name().substr(0, direct.name().size() - kDirectSuffix.size());
    const string correction_name = prefix + "/Mul_1";
    const string dy_gamma_name = prefix + "/mul_1/Mul";
    const string x_times_two_name = prefix + "/Mul";
    const string fused_name = prefix + "/musa_rmsnorm_grad_dx";
    if (node_map.find(fused_name) != node_map.end()) continue;

    const NodeDef* correction = find_node(correction_name);
    const NodeDef* dy_gamma = find_node(dy_gamma_name);
    const NodeDef* x_times_two = find_node(x_times_two_name);
    if (correction == nullptr || dy_gamma == nullptr ||
        x_times_two == nullptr || correction->op() != "Mul" ||
        dy_gamma->op() != "Mul" || x_times_two->op() != "Mul" ||
        correction->input_size() != 2 || dy_gamma->input_size() != 2 ||
        x_times_two->input_size() != 2 || !HasFloatTypeAttr(*correction) ||
        !HasFloatTypeAttr(*dy_gamma) || !HasFloatTypeAttr(*x_times_two)) {
      continue;
    }

    const auto direct_consumer_it = consumers.find(direct.name());
    const auto correction_consumer_it = consumers.find(correction_name);
    if (direct_consumer_it == consumers.end() ||
        correction_consumer_it == consumers.end() ||
        direct_consumer_it->second.size() != 1 ||
        correction_consumer_it->second.size() != 1 ||
        direct_consumer_it->second[0] != correction_consumer_it->second[0]) {
      continue;
    }

    NodeDef* consumer = nullptr;
    for (int node_idx = 0; node_idx < graph->node_size(); ++node_idx) {
      if (graph->node(node_idx).name() == direct_consumer_it->second[0]) {
        consumer = graph->mutable_node(node_idx);
        break;
      }
    }
    if (consumer == nullptr ||
        (consumer->op() != "AddN" &&
         consumer->op() != "MusaAddNWithSliceGrad")) {
      continue;
    }

    string inv_input;
    string dy_gamma_input;
    for (int input_idx = 0; input_idx < 2; ++input_idx) {
      const string clean = NodeNameFromInputLocal(direct.input(input_idx));
      if (clean == dy_gamma_name) {
        dy_gamma_input = direct.input(input_idx);
      } else if (direct.input(input_idx).find(":2") != string::npos) {
        inv_input = direct.input(input_idx);
      }
    }
    if (inv_input.empty() || dy_gamma_input.empty()) continue;

    string gamma_input;
    string dy_input;
    if (is_read_variable(dy_gamma->input(0))) {
      gamma_input = dy_gamma->input(0);
      dy_input = dy_gamma->input(1);
    } else if (is_read_variable(dy_gamma->input(1))) {
      gamma_input = dy_gamma->input(1);
      dy_input = dy_gamma->input(0);
    } else {
      continue;
    }

    string x_input;
    for (int input_idx = 0; input_idx < 2; ++input_idx) {
      const NodeDef* producer =
          find_node(NodeNameFromInputLocal(x_times_two->input(input_idx)));
      if (producer != nullptr && producer->op() != "Const") {
        x_input = x_times_two->input(input_idx);
      }
    }
    if (x_input.empty()) continue;

    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaRmsNormGradDx");
    fused->set_device(direct.device());
    fused->add_input(x_input);
    fused->add_input(inv_input);
    fused->add_input(gamma_input);
    fused->add_input(dy_input);
    (*fused->mutable_attr())["T"].set_type(GetNodeTypeAttr(direct));

    if (!ReplaceTwoInputsWithFused(consumer, direct.name(), correction_name,
                                   fused_name)) {
      graph->mutable_node(graph->node_size() - 1)->set_name(
          fused_name + "/unused");
      continue;
    }
    if (consumer->op() == "AddN") {
      (*consumer->mutable_attr())["N"].set_i(consumer->input_size());
    } else {
      const int old_n = consumer->attr().at("N").i();
      (*consumer->mutable_attr())["N"].set_i(std::max(1, old_n - 1));
    }
    rewrites++;
  }

  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " RMSNorm grad dx node(s) with warp128 kernel";
  }
  return rewrites;
}



bool IsReadVariableFromResource(
    const NodeDef* node, const string& resource_input) {
  return node != nullptr && node->op() == "ReadVariableOp" &&
         node->input_size() >= 1 &&
         NodeNameFromInputLocal(node->input(0)) ==
             NodeNameFromInputLocal(resource_input);
}

bool ExtractTf215SgdZeroSlotMul(
    const NodeDef* mul, const string& slot_resource,
    const std::unordered_map<string, const NodeDef*>& node_map,
    std::vector<string>* remove_names) {
  if (mul == nullptr || mul->op() != "Mul" || mul->input_size() != 2 ||
      !HasFloatTypeAttr(*mul)) {
    return false;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  for (int read_idx = 0; read_idx < 2; ++read_idx) {
    const int zero_idx = 1 - read_idx;
    const NodeDef* read = find_node(mul->input(read_idx));
    const NodeDef* zero = find_node(mul->input(zero_idx));
    if (IsReadVariableFromResource(read, slot_resource) &&
        IsFloatZeroConst(zero)) {
      remove_names->push_back(mul->name());
      remove_names->push_back(read->name());
      return true;
    }
  }
  return false;
}

bool ExtractTf215SgdNegGradLrMul(
    const NodeDef* mul,
    const std::unordered_map<string, const NodeDef*>& node_map,
    string* grad_input, string* lr_input,
    std::vector<string>* remove_names) {
  if (mul == nullptr || mul->op() != "Mul" || mul->input_size() != 2 ||
      !HasFloatTypeAttr(*mul)) {
    return false;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  for (int neg_idx = 0; neg_idx < 2; ++neg_idx) {
    const int lr_idx = 1 - neg_idx;
    const NodeDef* neg = find_node(mul->input(neg_idx));
    const NodeDef* lr = find_node(mul->input(lr_idx));
    if (neg != nullptr && neg->op() == "Neg" && neg->input_size() == 1 &&
        HasFloatTypeAttr(*neg) && lr != nullptr &&
        lr->op() == "ReadVariableOp") {
      *grad_input = neg->input(0);
      *lr_input = mul->input(lr_idx);
      remove_names->push_back(mul->name());
      remove_names->push_back(neg->name());
      return true;
    }
  }
  return false;
}

int OptimizeTf215SgdAssignAdd(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& assign_add = graph->node(i);
    if (assign_add.op() != "AssignAddVariableOp" ||
        assign_add.input_size() < 2 ||
        assign_add.name().find("SGD/") != 0) {
      continue;
    }
    const auto dtype_it = assign_add.attr().find("dtype");
    if (dtype_it == assign_add.attr().end() ||
        !IsFloatOrBFloat16Type(dtype_it->second.type())) {
      continue;
    }

    const NodeDef* slot_read = find_node(assign_add.input(1));
    if (slot_read == nullptr || slot_read->op() != "ReadVariableOp" ||
        slot_read->input_size() < 2) {
      continue;
    }

    const string slot_resource = slot_read->input(0);
    const NodeDef* slot_assign = nullptr;
    for (int input_idx = 1; input_idx < slot_read->input_size(); ++input_idx) {
      const string& ctrl = slot_read->input(input_idx);
      if (ctrl.empty() || ctrl[0] != '^') continue;
      const NodeDef* ctrl_node = find_node(ctrl);
      if (ctrl_node != nullptr && ctrl_node->op() == "AssignVariableOp" &&
          ctrl_node->input_size() >= 2 &&
          NodeNameFromInputLocal(ctrl_node->input(0)) ==
              NodeNameFromInputLocal(slot_resource)) {
        slot_assign = ctrl_node;
        break;
      }
    }
    if (slot_assign == nullptr) continue;

    const NodeDef* add = find_node(slot_assign->input(1));
    if (add == nullptr || add->op() != "AddV2" || add->input_size() != 2 ||
        !HasFloatTypeAttr(*add)) {
      continue;
    }

    string grad_input;
    string lr_input;
    std::vector<string> local_remove;
    bool matched = false;
    for (int update_idx = 0; update_idx < 2; ++update_idx) {
      const int zero_idx = 1 - update_idx;
      std::vector<string> update_remove;
      string candidate_grad;
      string candidate_lr;
      if (ExtractTf215SgdNegGradLrMul(
              find_node(add->input(update_idx)), node_map, &candidate_grad,
              &candidate_lr, &update_remove) &&
          ExtractTf215SgdZeroSlotMul(find_node(add->input(zero_idx)),
                                     slot_resource, node_map,
                                     &update_remove)) {
        grad_input = candidate_grad;
        lr_input = candidate_lr;
        local_remove = std::move(update_remove);
        matched = true;
        break;
      }
    }
    if (!matched) continue;

    const string fused_name =
        assign_add.name() + "/musa_resource_apply_gradient_descent";
    if (node_map.find(fused_name) != node_map.end()) continue;

    const DataType dtype = dtype_it->second.type();
    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("ResourceApplyGradientDescent");
    fused->set_device(assign_add.device());
    fused->add_input(assign_add.input(0));
    fused->add_input(lr_input);
    fused->add_input(grad_input);
    for (const auto& input : assign_add.input()) {
      if (!input.empty() && input[0] == '^') {
        fused->add_input(input);
      }
    }
    (*fused->mutable_attr())["T"].set_type(dtype);
    (*fused->mutable_attr())["use_locking"].set_b(true);

    RedirectNodeOutputs(graph, assign_add.name(), fused_name);
    RedirectNodeOutputs(graph, slot_assign->name(), fused_name);
    RedirectNodeOutputs(graph, slot_read->name(), fused_name);
    RedirectNodeOutputs(graph, add->name(), fused_name);
    for (const string& name : local_remove) {
      RedirectNodeOutputs(graph, name, fused_name);
    }
    remove_names.insert(assign_add.name());
    remove_names.insert(slot_assign->name());
    remove_names.insert(slot_read->name());
    remove_names.insert(add->name());
    for (const string& name : local_remove) {
      remove_names.insert(name);
    }
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " TF2.15 SGD update node(s) with ResourceApplyGradientDescent";
  }
  return rewrites;
}

bool IsBFloat16ToFloatCast(const NodeDef* node) {
  if (node == nullptr || node->op() != "Cast" || node->input_size() != 1) {
    return false;
  }
  const auto src_it = node->attr().find("SrcT");
  const auto dst_it = node->attr().find("DstT");
  return src_it != node->attr().end() && dst_it != node->attr().end() &&
         src_it->second.type() == DT_BFLOAT16 &&
         dst_it->second.type() == DT_FLOAT;
}

int DataConsumerCount(const GraphDef& graph, const string& producer_name) {
  int count = 0;
  for (const auto& node : graph.node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] == '^') continue;
      if (NodeNameFromInputLocal(input) == producer_name) {
        ++count;
      }
    }
  }
  return count;
}

int OptimizeBFloat16GradientDescentCast(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& apply = graph->node(i);
    if (apply.op() != "ResourceApplyGradientDescent" ||
        apply.input_size() < 3) {
      continue;
    }
    const auto dtype_it = apply.attr().find("T");
    if (dtype_it == apply.attr().end() ||
        dtype_it->second.type() != DT_FLOAT) {
      continue;
    }

    const NodeDef* grad_cast = find_node(apply.input(2));
    if (!IsBFloat16ToFloatCast(grad_cast)) {
      continue;
    }

    const string fused_name = apply.name() + "/musa_mixed_bf16";
    if (node_map.find(fused_name) != node_map.end()) {
      continue;
    }

    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaResourceApplyGradientDescentMixed");
    fused->set_device(apply.device());
    fused->add_input(apply.input(0));
    fused->add_input(apply.input(1));
    fused->add_input(grad_cast->input(0));
    auto add_input_if_missing = [&](const string& input) {
      for (const auto& existing : fused->input()) {
        if (existing == input) return;
      }
      fused->add_input(input);
    };
    for (const auto& input : grad_cast->input()) {
      if (!input.empty() && input[0] == '^') {
        fused->add_input(input);
      }
    }
    for (const auto& input : apply.input()) {
      if (!input.empty() && input[0] == '^') {
        add_input_if_missing(input);
      }
    }

    (*fused->mutable_attr())["T"].set_type(DT_FLOAT);
    (*fused->mutable_attr())["Tgrad"].set_type(DT_BFLOAT16);
    const auto locking_it = apply.attr().find("use_locking");
    (*fused->mutable_attr())["use_locking"].set_b(
        locking_it != apply.attr().end() && locking_it->second.b());

    RedirectNodeOutputs(graph, apply.name(), fused_name);
    remove_names.insert(apply.name());
    if (DataConsumerCount(*graph, grad_cast->name()) == 1) {
      RedirectNodeOutputs(graph, grad_cast->name(), fused_name);
      remove_names.insert(grad_cast->name());
    }
    ++rewrites;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " bfloat16-gradient SGD update node(s) with mixed update";
  }
  return rewrites;
}

int OptimizeBFloat16SparseSgdScatter(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& scatter = graph->node(i);
    if (scatter.op() != "ResourceScatterAdd" || scatter.input_size() < 3) {
      continue;
    }
    const auto dtype_it = scatter.attr().find("dtype");
    const auto indices_it = scatter.attr().find("Tindices");
    if (dtype_it == scatter.attr().end() ||
        dtype_it->second.type() != DT_FLOAT ||
        indices_it == scatter.attr().end()) {
      continue;
    }

    const NodeDef* mul = find_node(scatter.input(2));
    if (mul == nullptr || mul->op() != "Mul" || mul->input_size() != 2 ||
        GetNodeTypeAttr(*mul) != DT_FLOAT) {
      continue;
    }

    const NodeDef* neg = nullptr;
    string alpha_input;
    for (int neg_idx = 0; neg_idx < 2; ++neg_idx) {
      const NodeDef* candidate_neg = find_node(mul->input(neg_idx));
      if (candidate_neg != nullptr && candidate_neg->op() == "Neg" &&
          candidate_neg->input_size() == 1 &&
          GetNodeTypeAttr(*candidate_neg) == DT_FLOAT) {
        neg = candidate_neg;
        alpha_input = mul->input(1 - neg_idx);
        break;
      }
    }
    if (neg == nullptr) continue;

    const NodeDef* reshape = find_node(neg->input(0));
    if (reshape == nullptr || reshape->op() != "Reshape" ||
        reshape->input_size() < 1 || GetNodeTypeAttr(*reshape) != DT_FLOAT) {
      continue;
    }

    const NodeDef* cast = find_node(reshape->input(0));
    if (!IsBFloat16ToFloatCast(cast)) {
      continue;
    }

    const string fused_name = scatter.name() + "/musa_mixed_bf16";
    if (node_map.find(fused_name) != node_map.end()) {
      continue;
    }

    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaResourceScatterSubBFloat16");
    fused->set_device(scatter.device());
    fused->add_input(scatter.input(0));
    fused->add_input(scatter.input(1));
    fused->add_input(cast->input(0));
    fused->add_input(alpha_input);
    for (const auto& input : scatter.input()) {
      if (!input.empty() && input[0] == '^') {
        fused->add_input(input);
      }
    }

    (*fused->mutable_attr())["T"].set_type(DT_FLOAT);
    (*fused->mutable_attr())["Tgrad"].set_type(DT_BFLOAT16);
    (*fused->mutable_attr())["Tindices"].set_type(indices_it->second.type());

    RedirectNodeOutputs(graph, scatter.name(), fused_name);
    remove_names.insert(scatter.name());
    if (DataConsumerCount(*graph, mul->name()) == 1) {
      remove_names.insert(mul->name());
    }
    if (DataConsumerCount(*graph, neg->name()) == 1) {
      remove_names.insert(neg->name());
    }
    if (DataConsumerCount(*graph, reshape->name()) == 1) {
      remove_names.insert(reshape->name());
    }
    if (DataConsumerCount(*graph, cast->name()) == 1) {
      remove_names.insert(cast->name());
    }
    ++rewrites;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " bfloat16 sparse SGD scatter update node(s)";
  }
  return rewrites;
}

bool AddInputIfMissing(NodeDef* node, const string& input) {
  for (const auto& existing : node->input()) {
    if (existing == input) return false;
  }
  node->add_input(input);
  return true;
}

bool ConsumersAreSubset(
    const std::unordered_map<string, std::vector<string>>& consumers,
    const string& producer, const std::unordered_set<string>& allowed) {
  const auto it = consumers.find(producer);
  if (it == consumers.end()) return true;
  for (const string& consumer : it->second) {
    if (allowed.find(consumer) == allowed.end()) return false;
  }
  return true;
}

NodeDef* MutableNodeByName(
    GraphDef* graph, const std::unordered_map<string, int>& node_index,
    const string& name) {
  const auto it = node_index.find(name);
  return it == node_index.end() ? nullptr : graph->mutable_node(it->second);
}

bool TryBypassTf215SparseSgdAggregation(
    GraphDef* graph, const NodeDef* scatter,
    const std::unordered_map<string, const NodeDef*>& node_map,
    const std::unordered_map<string, int>& node_index,
    const std::unordered_map<string, std::vector<string>>& consumers,
    string* indices_input, std::vector<string>* remove_names) {
  if (scatter == nullptr || scatter->input_size() < 3) return false;

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  const NodeDef* unique = find_node(scatter->input(1));
  const NodeDef* update_mul = find_node(scatter->input(2));
  if (unique == nullptr || unique->op() != "Unique" ||
      unique->input_size() < 1 || update_mul == nullptr ||
      update_mul->op() != "Mul" || update_mul->input_size() != 2 ||
      !HasFloatTypeAttr(*update_mul)) {
    return false;
  }

  const NodeDef* neg = nullptr;
  for (int input_idx = 0; input_idx < 2; ++input_idx) {
    const NodeDef* input = find_node(update_mul->input(input_idx));
    if (input != nullptr && input->op() == "Neg" && input->input_size() == 1 &&
        HasFloatTypeAttr(*input)) {
      neg = input;
      break;
    }
  }
  if (neg == nullptr) return false;

  const NodeDef* segment_sum = find_node(neg->input(0));
  if (segment_sum == nullptr || segment_sum->op() != "UnsortedSegmentSum" ||
      segment_sum->input_size() < 3 || !HasFloatTypeAttr(*segment_sum) ||
      NodeNameFromInputLocal(segment_sum->input(1)) != unique->name() ||
      segment_sum->input(1).find(":1") == string::npos) {
    return false;
  }

  const NodeDef* num_segments = find_node(segment_sum->input(2));
  if (num_segments == nullptr || num_segments->op() != "StridedSlice" ||
      num_segments->input_size() < 1) {
    return false;
  }
  const NodeDef* shape = find_node(num_segments->input(0));
  if (shape == nullptr || shape->op() != "Shape" || shape->input_size() < 1 ||
      NodeNameFromInputLocal(shape->input(0)) != unique->name()) {
    return false;
  }

  const std::unordered_set<string> unique_consumers = {
      scatter->name(), shape->name(), segment_sum->name()};
  if (!ConsumersAreSubset(consumers, unique->name(), unique_consumers) ||
      !ConsumersAreSubset(consumers, segment_sum->name(), {neg->name()}) ||
      !ConsumersAreSubset(consumers, shape->name(), {num_segments->name()}) ||
      !ConsumersAreSubset(consumers, num_segments->name(),
                          {segment_sum->name()})) {
    return false;
  }

  NodeDef* neg_mutable = MutableNodeByName(graph, node_index, neg->name());
  if (neg_mutable == nullptr) return false;
  neg_mutable->set_input(0, segment_sum->input(0));
  *indices_input = unique->input(0);
  remove_names->push_back(unique->name());
  remove_names->push_back(segment_sum->name());
  remove_names->push_back(shape->name());
  remove_names->push_back(num_segments->name());
  return true;
}

int OptimizeTf215SparseSgdAssignAdd(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  std::unordered_map<string, int> node_index;
  node_map.reserve(graph->node_size());
  node_index.reserve(graph->node_size());
  for (int i = 0; i < graph->node_size(); ++i) {
    const NodeDef& node = graph->node(i);
    node_map[node.name()] = &node;
    node_index[node.name()] = i;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  auto find_mutable_node = [&](const string& name) -> NodeDef* {
    const auto it = node_index.find(name);
    return it == node_index.end() ? nullptr : graph->mutable_node(it->second);
  };

  std::unordered_map<string, std::vector<string>> consumers;
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      consumers[NodeNameFromInputLocal(input)].push_back(node.name());
    }
  }

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& assign_add = graph->node(i);
    if (assign_add.op() != "AssignAddVariableOp" ||
        assign_add.input_size() < 2 ||
        assign_add.name().find("SGD/") != 0) {
      continue;
    }
    const auto dtype_it = assign_add.attr().find("dtype");
    if (dtype_it == assign_add.attr().end() ||
        !IsFloatOrBFloat16Type(dtype_it->second.type())) {
      continue;
    }

    const NodeDef* slot_read = find_node(assign_add.input(1));
    if (slot_read == nullptr || slot_read->op() != "ReadVariableOp" ||
        slot_read->input_size() < 2) {
      continue;
    }

    const string slot_resource = slot_read->input(0);
    const NodeDef* scatter = nullptr;
    for (int input_idx = 1; input_idx < slot_read->input_size(); ++input_idx) {
      const string& ctrl = slot_read->input(input_idx);
      if (ctrl.empty() || ctrl[0] != '^') continue;
      const NodeDef* ctrl_node = find_node(ctrl);
      if (ctrl_node != nullptr && ctrl_node->op() == "ResourceScatterAdd" &&
          ctrl_node->input_size() >= 3 &&
          NodeNameFromInputLocal(ctrl_node->input(0)) ==
              NodeNameFromInputLocal(slot_resource)) {
        const auto scatter_dtype_it = ctrl_node->attr().find("dtype");
        if (scatter_dtype_it != ctrl_node->attr().end() &&
            IsFloatOrBFloat16Type(scatter_dtype_it->second.type())) {
          scatter = ctrl_node;
          break;
        }
      }
    }
    if (scatter == nullptr) continue;

    const NodeDef* slot_assign = nullptr;
    for (int input_idx = 3; input_idx < scatter->input_size(); ++input_idx) {
      const string& ctrl = scatter->input(input_idx);
      if (ctrl.empty() || ctrl[0] != '^') continue;
      const NodeDef* ctrl_node = find_node(ctrl);
      if (ctrl_node != nullptr && ctrl_node->op() == "AssignVariableOp" &&
          ctrl_node->input_size() >= 2 &&
          NodeNameFromInputLocal(ctrl_node->input(0)) ==
              NodeNameFromInputLocal(slot_resource)) {
        slot_assign = ctrl_node;
        break;
      }
    }
    if (slot_assign == nullptr) continue;

    std::vector<string> local_remove;
    if (!ExtractTf215SgdZeroSlotMul(find_node(slot_assign->input(1)),
                                    slot_resource, node_map,
                                    &local_remove)) {
      continue;
    }
    if (NodeNameFromInputLocal(assign_add.input(0)) ==
        NodeNameFromInputLocal(slot_resource)) {
      continue;
    }

    NodeDef* scatter_mutable = find_mutable_node(scatter->name());
    if (scatter_mutable == nullptr) continue;

    const string scatter_name = scatter->name();
    string indices_input = scatter->input(1);
    const string updates_input = scatter->input(2);
    std::vector<string> aggregation_remove;
    if (TryBypassTf215SparseSgdAggregation(
            graph, scatter, node_map, node_index, consumers, &indices_input,
            &aggregation_remove)) {
      local_remove.insert(local_remove.end(), aggregation_remove.begin(),
                          aggregation_remove.end());
    }
    scatter_mutable->clear_input();
    scatter_mutable->add_input(assign_add.input(0));
    scatter_mutable->add_input(indices_input);
    scatter_mutable->add_input(updates_input);
    for (int input_idx = 3; input_idx < scatter->input_size(); ++input_idx) {
      const string& input = scatter->input(input_idx);
      if (NodeNameFromInputLocal(input) != slot_assign->name()) {
        AddInputIfMissing(scatter_mutable, input);
      }
    }
    for (int input_idx = 2; input_idx < assign_add.input_size(); ++input_idx) {
      const string& input = assign_add.input(input_idx);
      if (!input.empty() && input[0] == '^') {
        AddInputIfMissing(scatter_mutable, input);
      }
    }

    RedirectNodeOutputs(graph, assign_add.name(), scatter_name);
    RedirectNodeOutputs(graph, slot_assign->name(), scatter_name);
    RedirectNodeOutputs(graph, slot_read->name(), scatter_name);
    for (const string& name : local_remove) {
      RedirectNodeOutputs(graph, name, scatter_name);
    }
    remove_names.insert(assign_add.name());
    remove_names.insert(slot_assign->name());
    remove_names.insert(slot_read->name());
    for (const string& name : local_remove) {
      remove_names.insert(name);
    }
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " TF2.15 sparse SGD slot update node(s) with ResourceScatterAdd";
  }
  return rewrites;
}

int OptimizeRmsNormSqrtGradDxWarp128(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] == '^') continue;
      consumers[NodeNameFromInputLocal(input)].push_back(node.name());
    }
  }

  auto find_node = [&](const string& name) -> const NodeDef* {
    const auto it = node_map.find(name);
    return it == node_map.end() ? nullptr : it->second;
  };

  auto is_read_variable = [&](const string& input) -> bool {
    const NodeDef* node = find_node(NodeNameFromInputLocal(input));
    if (node == nullptr) return false;
    if (node->op() == "ReadVariableOp") return true;
    if (node->op() == "Cast" && node->input_size() == 1) {
      const NodeDef* cast_input = find_node(NodeNameFromInputLocal(node->input(0)));
      return cast_input != nullptr && cast_input->op() == "ReadVariableOp";
    }
    return false;
  };

  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& direct = graph->node(i);
    static const string kDirectSuffix = "/truediv/Reshape";
    static const string kRealDivDirectSuffix = "/truediv/RealDiv";
    const bool is_legacy_direct =
        direct.op() == "Reshape" &&
        StringEndsWithLocal(direct.name(), kDirectSuffix);
    const bool is_tf215_direct =
        direct.op() == "RealDiv" &&
        StringEndsWithLocal(direct.name(), kRealDivDirectSuffix);
    if (!HasFloatTypeAttr(direct) ||
        direct.name().find("gradient_tape/") == string::npos ||
        direct.name().find("rms_layer_norm") == string::npos ||
        (!is_legacy_direct && !is_tf215_direct)) {
      continue;
    }

    const string direct_suffix =
        is_legacy_direct ? kDirectSuffix : kRealDivDirectSuffix;
    const string prefix =
        direct.name().substr(0, direct.name().size() - direct_suffix.size());
    const string forward_prefix = prefix.substr(strlen("gradient_tape/"));
    const string forward_fused_name = forward_prefix + "/mul";
    const string correction_name = prefix + "/Mul_1";
    const string dy_gamma_name = prefix + "/mul/Mul";
    const string fused_name = prefix + "/musa_rmsnorm_grad_dx";
    if (node_map.find(fused_name) != node_map.end()) continue;

    const NodeDef* forward_fused = find_node(forward_fused_name);
    const NodeDef* correction = find_node(correction_name);
    const NodeDef* dy_gamma = find_node(dy_gamma_name);
    if (forward_fused == nullptr || correction == nullptr ||
        dy_gamma == nullptr || forward_fused->op() != "MusaRmsNorm" ||
        correction->op() != "Mul" || dy_gamma->op() != "Mul" ||
        forward_fused->input_size() != 2 || correction->input_size() != 2 ||
        dy_gamma->input_size() != 2 || !HasFloatTypeAttr(*correction) ||
        !HasFloatTypeAttr(*dy_gamma)) {
      continue;
    }

    const auto direct_consumer_it = consumers.find(direct.name());
    const auto correction_consumer_it = consumers.find(correction_name);
    if (direct_consumer_it == consumers.end() ||
        correction_consumer_it == consumers.end() ||
        direct_consumer_it->second.size() != 1 ||
        correction_consumer_it->second.size() != 1 ||
        direct_consumer_it->second[0] != correction_consumer_it->second[0]) {
      continue;
    }

    NodeDef* consumer = nullptr;
    for (int node_idx = 0; node_idx < graph->node_size(); ++node_idx) {
      if (graph->node(node_idx).name() == direct_consumer_it->second[0]) {
        consumer = graph->mutable_node(node_idx);
        break;
      }
    }
    if (consumer == nullptr ||
        (consumer->op() != "AddN" &&
         consumer->op() != "MusaAddNWithSliceGrad")) {
      continue;
    }

    string gamma_input;
    string dy_input;
    if (is_read_variable(dy_gamma->input(0))) {
      gamma_input = dy_gamma->input(0);
      dy_input = dy_gamma->input(1);
    } else if (is_read_variable(dy_gamma->input(1))) {
      gamma_input = dy_gamma->input(1);
      dy_input = dy_gamma->input(0);
    } else {
      continue;
    }

    const DataType dtype = GetNodeTypeAttr(direct);
    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaRmsNormGradDx");
    fused->set_device(direct.device());
    fused->add_input(forward_fused->input(0));
    fused->add_input(forward_fused_name + ":2");
    fused->add_input(gamma_input);
    fused->add_input(dy_input);
    (*fused->mutable_attr())["T"].set_type(dtype);

    if (!ReplaceTwoInputsWithFused(consumer, direct.name(), correction_name,
                                   fused_name)) {
      graph->mutable_node(graph->node_size() - 1)->set_name(
          fused_name + "/unused");
      continue;
    }
    if (consumer->op() == "AddN") {
      (*consumer->mutable_attr())["N"].set_i(consumer->input_size());
    } else {
      const int old_n = consumer->attr().at("N").i();
      (*consumer->mutable_attr())["N"].set_i(std::max(1, old_n - 1));
    }
    rewrites++;
  }

  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " Sqrt RMSNorm grad dx node(s) with warp128 kernel";
  }
  return rewrites;
}

bool ExtractGeluGradPrefix(const string& node_name, string* prefix) {
  const size_t gelu_pos = node_name.find("/Gelu");
  if (gelu_pos == string::npos) return false;
  const size_t next_slash = node_name.find('/', gelu_pos + 1);
  if (next_slash == string::npos) return false;
  *prefix = node_name.substr(0, next_slash);
  return prefix->find("gradient_tape/") == 0;
}

string StripGradientTapePrefix(const string& name) {
  static const string kPrefix = "gradient_tape/";
  if (name.compare(0, kPrefix.size(), kPrefix) != 0) return "";
  return name.substr(kPrefix.size());
}

int OptimizeExactGeluGrad(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  auto find_node = [&](const string& name) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(name));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& addn = graph->node(i);
    if (addn.op() != "AddN" || addn.input_size() != 2 ||
        !HasFloatTypeAttr(addn)) {
      continue;
    }

    const NodeDef* lhs = find_node(addn.input(0));
    const NodeDef* rhs = find_node(addn.input(1));
    if (lhs == nullptr || rhs == nullptr) continue;

    string lhs_prefix;
    string rhs_prefix;
    if (!ExtractGeluGradPrefix(lhs->name(), &lhs_prefix) ||
        !ExtractGeluGradPrefix(rhs->name(), &rhs_prefix) ||
        lhs_prefix != rhs_prefix) {
      continue;
    }

    const string forward_prefix = StripGradientTapePrefix(lhs_prefix);
    if (forward_prefix.empty()) continue;

    const NodeDef* forward_gelu = find_node(forward_prefix + "/mul_1");
    const NodeDef* forward_add = find_node(forward_prefix + "/add");
    const NodeDef* grad_mul = find_node(lhs_prefix + "/mul_1/Mul");
    if (forward_gelu == nullptr || forward_add == nullptr ||
        grad_mul == nullptr || forward_gelu->op() != "MusaGelu" ||
        forward_gelu->input_size() < 1 || grad_mul->op() != "Mul" ||
        grad_mul->input_size() != 2 || !HasFloatTypeAttr(*grad_mul)) {
      continue;
    }

    string dy_input;
    for (int input_idx = 0; input_idx < grad_mul->input_size(); ++input_idx) {
      if (NodeNameFromInputLocal(grad_mul->input(input_idx)) !=
          forward_add->name()) {
        dy_input = grad_mul->input(input_idx);
      }
    }
    if (dy_input.empty()) continue;

    const string fused_name = addn.name() + "/musa_gelu_grad";
    if (node_map.find(fused_name) != node_map.end()) continue;

    const DataType dtype = GetNodeTypeAttr(addn);
    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaGeluGrad");
    fused->set_device(addn.device());
    fused->add_input(forward_gelu->input(0));
    fused->add_input(dy_input);
    (*fused->mutable_attr())["T"].set_type(dtype);

    RedirectNodeOutputs(graph, addn.name(), fused_name);
    remove_names.insert(addn.name());
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " exact GELU grad node(s)";
  }
  return rewrites;
}



bool IsFloatOp(const NodeDef& node) {
  const auto it = node.attr().find("T");
  return it != node.attr().end() && IsFloatOrBFloat16Type(it->second.type());
}

bool IsReductionIndexMinusOne(const NodeDef* node) {
  std::vector<int64_t> values;
  return GetConstIntVector(node, &values) && values.size() == 1 &&
         values[0] == -1;
}



void AddConstInt32ScalarNode(GraphDef* graph, const string& name, int value,
                             const string& device) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("Const");
  node->set_device(device);
  (*node->mutable_attr())["dtype"].set_type(DT_INT32);
  TensorProto* tensor = (*node->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  tensor->add_int_val(value);
}

NodeDef* AddTypedConcat3Node(GraphDef* graph, const string& name,
                             const string& in0, const string& in1,
                             const string& in2, const string& axis,
                             DataType dtype, const string& device) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("ConcatV2");
  node->set_device(device);
  node->add_input(in0);
  node->add_input(in1);
  node->add_input(in2);
  node->add_input(axis);
  (*node->mutable_attr())["N"].set_i(3);
  (*node->mutable_attr())["T"].set_type(dtype);
  (*node->mutable_attr())["Tidx"].set_type(DT_INT32);
  return node;
}

bool IsQkvDenseDwMatMul(const NodeDef* node) {
  if (node == nullptr || node->op() != "MatMul" || node->input_size() != 2 ||
      node->name().find("gradient_tape/") == string::npos ||
      node->name().find("pyramid_mixed_causal_attention") == string::npos ||
      node->name().find("/dense_") == string::npos ||
      !HasFloatTypeAttr(*node)) {
    return false;
  }
  const auto transpose_a_it = node->attr().find("transpose_a");
  const auto transpose_b_it = node->attr().find("transpose_b");
  return transpose_a_it != node->attr().end() &&
         transpose_a_it->second.b() &&
         (transpose_b_it == node->attr().end() ||
          !transpose_b_it->second.b());
}

int OptimizeQkvDenseDw(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::unordered_map<string, std::vector<const NodeDef*>> groups;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& node = graph->node(i);
    if (!IsQkvDenseDwMatMul(&node)) continue;
    const NodeDef* x_reshape = find_node(node.input(0));
    if (x_reshape == nullptr || x_reshape->op() != "Reshape" ||
        x_reshape->input_size() < 1) {
      continue;
    }
    groups[NodeNameFromInputLocal(x_reshape->input(0))].push_back(&node);
  }

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  for (auto& item : groups) {
    std::vector<const NodeDef*>& nodes = item.second;
    if (nodes.size() != 3) continue;
    std::sort(nodes.begin(), nodes.end(),
              [](const NodeDef* lhs, const NodeDef* rhs) {
                return lhs->name() < rhs->name();
              });

    const NodeDef* x0 = find_node(nodes[0]->input(0));
    const NodeDef* x1 = find_node(nodes[1]->input(0));
    const NodeDef* x2 = find_node(nodes[2]->input(0));
    const DataType dtype = GetNodeTypeAttr(*nodes[0]);
    if (x0 == nullptr || x1 == nullptr || x2 == nullptr ||
        GetNodeTypeAttr(*nodes[1]) != dtype ||
        GetNodeTypeAttr(*nodes[2]) != dtype ||
        NodeNameFromInputLocal(x0->input(0)) !=
            NodeNameFromInputLocal(x1->input(0)) ||
        NodeNameFromInputLocal(x0->input(0)) !=
            NodeNameFromInputLocal(x2->input(0))) {
      continue;
    }

    const string axis_name = nodes[0]->name() + "/musa_qkv_dense_dw_axis";
    const string dy_concat_name =
        nodes[0]->name() + "/musa_qkv_dense_dw_dy_concat";
    const string fused_name = nodes[0]->name() + "/musa_qkv_dense_dw_matmul";
    const string split_name = fused_name + "/split";
    if (node_map.find(axis_name) != node_map.end() ||
        node_map.find(dy_concat_name) != node_map.end() ||
        node_map.find(fused_name) != node_map.end() ||
        node_map.find(split_name) != node_map.end()) {
      continue;
    }

    AddConstInt32ScalarNode(graph, axis_name, 1, nodes[0]->device());
    AddTypedConcat3Node(graph, dy_concat_name, nodes[0]->input(1),
                        nodes[1]->input(1), nodes[2]->input(1), axis_name,
                        dtype, nodes[0]->device());

    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MatMul");
    fused->set_device(nodes[0]->device());
    fused->add_input(nodes[0]->input(0));
    fused->add_input(dy_concat_name);
    (*fused->mutable_attr())["T"].set_type(dtype);
    (*fused->mutable_attr())["transpose_a"].set_b(true);
    (*fused->mutable_attr())["transpose_b"].set_b(false);

    NodeDef* split = graph->add_node();
    split->set_name(split_name);
    split->set_op("Split");
    split->set_device(nodes[0]->device());
    split->add_input(axis_name);
    split->add_input(fused_name);
    (*split->mutable_attr())["num_split"].set_i(3);
    (*split->mutable_attr())["T"].set_type(dtype);

    for (int part = 0; part < 3; ++part) {
      RedirectDataConsumersToInput(graph, nodes[part]->name(),
                                   split_name + ":" + std::to_string(part),
                                   split_name);
      remove_names.insert(nodes[part]->name());
    }
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " QKV dense dW group(s) to concat MatMul+Split";
  }
  return rewrites;
}


int OptimizeCausalMaskedSoftmax(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  consumers.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] != '^') {
        consumers[NodeNameFromInputLocal(input)].push_back(node.name());
      }
    }
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  struct ScaledMatMulInfo {
    string logits_input;
    string scale_input;
    const NodeDef* scale_mul = nullptr;
  };

  auto get_scaled_matmul = [&](const NodeDef* node,
                               ScaledMatMulInfo* info) -> bool {
    if (node == nullptr || node->op() != "Mul" || node->input_size() != 2 ||
        !HasFloatTypeAttr(*node)) {
      return false;
    }
    const NodeDef* lhs = find_node(node->input(0));
    const NodeDef* rhs = find_node(node->input(1));
    if (lhs != nullptr && lhs->op() == "BatchMatMulV2") {
      info->logits_input = node->input(0);
      info->scale_input = node->input(1);
      info->scale_mul = node;
      return true;
    }
    if (rhs != nullptr && rhs->op() == "BatchMatMulV2") {
      info->logits_input = node->input(1);
      info->scale_input = node->input(0);
      info->scale_mul = node;
      return true;
    }
    return false;
  };

  auto is_mask_side = [&](const NodeDef* node) -> bool {
    if (node == nullptr) return false;
    if (node->op() == "Const") return true;
    // Original playground builds the causal mask dynamically as
    // Mul(StridedSlice(Cast(Greater(...))), negative_large_const).
    if (node->op() != "Mul" || node->input_size() != 2 ||
        !HasFloatTypeAttr(*node)) {
      return false;
    }
    const NodeDef* lhs = find_node(node->input(0));
    const NodeDef* rhs = find_node(node->input(1));
    return (lhs != nullptr && (lhs->op() == "StridedSlice" ||
                              lhs->op() == "Cast" || lhs->op() == "Const")) ||
           (rhs != nullptr && (rhs->op() == "StridedSlice" ||
                              rhs->op() == "Cast" || rhs->op() == "Const"));
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* softmax = graph->mutable_node(i);
    if (softmax->op() != "Softmax" || softmax->input_size() != 1 ||
        softmax->name().find("pyramid_mixed_causal_attention") ==
            string::npos) {
      continue;
    }

    const NodeDef* add = find_node(softmax->input(0));
    if (add == nullptr || add->input_size() != 2 ||
        (add->op() != "AddV2" && add->op() != "Add") ||
        !HasFloatTypeAttr(*add) ||
        !IsOnlyConsumer(consumers, add->name(), softmax->name())) {
      continue;
    }

    const NodeDef* lhs = find_node(add->input(0));
    const NodeDef* rhs = find_node(add->input(1));
    if (lhs == nullptr || rhs == nullptr) continue;

    ScaledMatMulInfo scaled;
    const NodeDef* mask_node = nullptr;
    if (get_scaled_matmul(lhs, &scaled) && is_mask_side(rhs)) {
      mask_node = rhs;
    } else if (get_scaled_matmul(rhs, &scaled) && is_mask_side(lhs)) {
      mask_node = lhs;
    } else {
      continue;
    }
    if (!IsOnlyConsumer(consumers, scaled.scale_mul->name(), add->name())) {
      continue;
    }

    const DataType dtype = GetNodeTypeAttr(*add);
    softmax->set_op("MusaCausalMaskedSoftmax");
    softmax->clear_input();
    softmax->add_input(scaled.logits_input);
    softmax->add_input(scaled.scale_input);
    softmax->mutable_attr()->clear();
    (*softmax->mutable_attr())["T"].set_type(dtype);

    remove_names.insert(add->name());
    remove_names.insert(scaled.scale_mul->name());
    if (mask_node != nullptr &&
        IsOnlyConsumer(consumers, mask_node->name(), add->name())) {
      remove_names.insert(mask_node->name());
    }
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " causal mask Softmax node(s)";
  }
  return rewrites;
}

int OptimizeCausalAttentionForward(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  consumers.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] != '^') {
        consumers[NodeNameFromInputLocal(input)].push_back(node.name());
      }
    }
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };
  auto bool_attr = [](const NodeDef& node, const char* name) -> bool {
    const auto it = node.attr().find(name);
    return it != node.attr().end() && it->second.b();
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    NodeDef* softmax = graph->mutable_node(i);
    if (softmax->op() != "MusaCausalMaskedSoftmax" ||
        softmax->input_size() != 2 ||
        softmax->name().find("pyramid_mixed_causal_attention") ==
            string::npos) {
      continue;
    }
    const NodeDef* qk_bmm = find_node(softmax->input(0));
    if (qk_bmm == nullptr || qk_bmm->op() != "BatchMatMulV2" ||
        qk_bmm->input_size() != 2 || !HasFloatTypeAttr(*qk_bmm) ||
        bool_attr(*qk_bmm, "adj_x") || !bool_attr(*qk_bmm, "adj_y") ||
        !IsOnlyConsumer(consumers, qk_bmm->name(), softmax->name())) {
      continue;
    }
    const DataType dtype = GetNodeTypeAttr(*qk_bmm);

    const NodeDef* value_bmm = nullptr;
    const auto softmax_consumers = consumers.find(softmax->name());
    if (softmax_consumers == consumers.end()) continue;
    for (const string& consumer_name : softmax_consumers->second) {
      const NodeDef* candidate = find_node(consumer_name);
      if (candidate == nullptr || candidate->op() != "BatchMatMulV2" ||
          candidate->input_size() != 2 || !HasFloatTypeAttr(*candidate) ||
          GetNodeTypeAttr(*candidate) != dtype ||
          candidate->name().find("gradient_tape/") != string::npos) {
        continue;
      }
      if (NodeNameFromInputLocal(candidate->input(0)) == softmax->name() &&
          !bool_attr(*candidate, "adj_x") && !bool_attr(*candidate, "adj_y")) {
        value_bmm = candidate;
        break;
      }
    }
    if (value_bmm == nullptr) continue;
    const NodeDef* value = find_node(value_bmm->input(1));
    if (value == nullptr || value->op() != "Transpose" ||
        !HasFloatTypeAttr(*value) || GetNodeTypeAttr(*value) != dtype) {
      continue;
    }

    const string query_input = qk_bmm->input(0);
    const string key_input = qk_bmm->input(1);
    const string value_input = value_bmm->input(1);
    const string scale_input = softmax->input(1);

    RedirectDataConsumersToInput(graph, value_bmm->name(), softmax->name() + ":1",
                                 softmax->name());

    softmax->set_op("MusaCausalAttentionForward");
    softmax->clear_input();
    softmax->add_input(query_input);
    softmax->add_input(key_input);
    softmax->add_input(value_input);
    softmax->add_input(scale_input);
    softmax->mutable_attr()->clear();
    (*softmax->mutable_attr())["T"].set_type(dtype);

    remove_names.insert(qk_bmm->name());
    remove_names.insert(value_bmm->name());
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " causal attention forward core group(s)";
  }
  return rewrites;
}

int OptimizeSoftmaxGradSmallLastDim(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  consumers.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] != '^') {
        consumers[NodeNameFromInputLocal(input)].push_back(node.name());
      }
    }
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& final_mul = graph->node(i);
    if (final_mul.op() != "Mul" || final_mul.input_size() != 2 ||
        !IsFloatOp(final_mul) ||
        final_mul.name().find("pyramid_mixed_causal_attention") ==
            string::npos) {
      continue;
    }

    const NodeDef* sub = nullptr;
    const NodeDef* softmax = nullptr;
    for (int input_idx = 0; input_idx < 2; ++input_idx) {
      const NodeDef* candidate = find_node(final_mul.input(input_idx));
      if (candidate == nullptr) continue;
      if (candidate->op() == "Sub") {
        sub = candidate;
      } else if (candidate->op() == "Softmax" ||
                 candidate->op() == "MusaCausalMaskedSoftmax" ||
                 candidate->op() == "MusaCausalAttentionForward") {
        softmax = candidate;
      }
    }
    if (sub == nullptr || softmax == nullptr || sub->input_size() != 2 ||
        !IsFloatOp(*sub) ||
        !IsOnlyConsumer(consumers, sub->name(), final_mul.name())) {
      continue;
    }

    const NodeDef* sum = find_node(sub->input(1));
    if (sum == nullptr || sum->op() != "Sum" || sum->input_size() != 2 ||
        !IsFloatOp(*sum) ||
        !IsOnlyConsumer(consumers, sum->name(), sub->name())) {
      continue;
    }
    const auto keep_dims_it = sum->attr().find("keep_dims");
    if (keep_dims_it == sum->attr().end() || !keep_dims_it->second.b()) {
      continue;
    }
    if (!IsReductionIndexMinusOne(find_node(sum->input(1)))) {
      continue;
    }

    const NodeDef* first_mul = find_node(sum->input(0));
    if (first_mul == nullptr || first_mul->op() != "Mul" ||
        first_mul->input_size() != 2 || !IsFloatOp(*first_mul) ||
        !IsOnlyConsumer(consumers, first_mul->name(), sum->name())) {
      continue;
    }

    const string dy_input = sub->input(0);
    const string softmax_name = softmax->name();
    bool first_mul_has_dy = false;
    bool first_mul_has_softmax = false;
    for (const auto& input : first_mul->input()) {
      const string clean = NodeNameFromInputLocal(input);
      if (clean == NodeNameFromInputLocal(dy_input)) first_mul_has_dy = true;
      if (clean == softmax_name) first_mul_has_softmax = true;
    }
    if (!first_mul_has_dy || !first_mul_has_softmax) continue;
    if (NodeNameFromInputLocal(final_mul.input(0)) != sub->name() &&
        NodeNameFromInputLocal(final_mul.input(1)) != sub->name()) {
      continue;
    }

    const string fused_name = final_mul.name() + "/musa_softmax_grad";
    if (node_map.find(fused_name) != node_map.end()) continue;
    const DataType dtype = GetNodeTypeAttr(final_mul);
    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaSoftmaxGrad");
    fused->set_device(final_mul.device());
    fused->add_input(softmax->name());
    fused->add_input(dy_input);
    (*fused->mutable_attr())["T"].set_type(dtype);

    RedirectNodeOutputs(graph, final_mul.name(), fused_name);
    remove_names.insert(final_mul.name());
    remove_names.insert(sub->name());
    remove_names.insert(sum->name());
    remove_names.insert(first_mul->name());
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " softmax grad node(s) with MusaSoftmaxGrad";
  }
  return rewrites;
}

int OptimizeCausalAttentionGrad(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  auto get_bool_attr = [](const NodeDef& node, const char* name) -> bool {
    const auto it = node.attr().find(name);
    return it != node.attr().end() && it->second.b();
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& softmax_grad = graph->node(i);
    if (softmax_grad.op() != "MusaSoftmaxGrad" ||
        softmax_grad.input_size() != 2 ||
        softmax_grad.name().find("pyramid_mixed_causal_attention") ==
            string::npos) {
      continue;
    }
    const DataType dtype = GetNodeTypeAttr(softmax_grad);
    if (!IsFloatOrBFloat16Type(dtype)) continue;

    const NodeDef* softmax = find_node(softmax_grad.input(0));
    const NodeDef* ds_bmm = find_node(softmax_grad.input(1));
    const bool softmax_is_causal =
        softmax != nullptr &&
        ((softmax->op() == "MusaCausalMaskedSoftmax" &&
          softmax->input_size() == 2) ||
         (softmax->op() == "MusaCausalAttentionForward" &&
          softmax->input_size() == 4));
    if (softmax_is_causal && GetNodeTypeAttr(*softmax) != dtype) continue;
    if (!softmax_is_causal || ds_bmm == nullptr || ds_bmm->op() != "BatchMatMulV2" ||
        ds_bmm->input_size() != 2 ||
        !HasFloatTypeAttr(*ds_bmm) || GetNodeTypeAttr(*ds_bmm) != dtype ||
        get_bool_attr(*ds_bmm, "adj_x") ||
        !get_bool_attr(*ds_bmm, "adj_y")) {
      continue;
    }
    const string dout_input = ds_bmm->input(0);
    const string value_input = ds_bmm->input(1);

    const NodeDef* scale_mul = nullptr;
    const NodeDef* dq_bmm = nullptr;
    const NodeDef* dk_bmm = nullptr;
    const NodeDef* dv_bmm = nullptr;
    string query_input;
    string key_input;
    for (const auto& candidate : graph->node()) {
      if (candidate.op() == "Mul" && candidate.input_size() == 2 &&
          !remove_names.count(candidate.name())) {
        const string in0 = NodeNameFromInputLocal(candidate.input(0));
        const string in1 = NodeNameFromInputLocal(candidate.input(1));
        if (in0 == softmax_grad.name() || in1 == softmax_grad.name()) {
          scale_mul = &candidate;
        }
      }
      if (candidate.op() != "BatchMatMulV2" || candidate.input_size() != 2 ||
          !HasFloatTypeAttr(candidate) || GetNodeTypeAttr(candidate) != dtype) {
        continue;
      }
      const string in0 = NodeNameFromInputLocal(candidate.input(0));
      const string in1 = NodeNameFromInputLocal(candidate.input(1));
      if (in0 == softmax->name() &&
          in1 == NodeNameFromInputLocal(dout_input) &&
          get_bool_attr(candidate, "adj_x") &&
          !get_bool_attr(candidate, "adj_y")) {
        dv_bmm = &candidate;
      }
    }
    if (scale_mul == nullptr || dv_bmm == nullptr) continue;

    for (const auto& candidate : graph->node()) {
      if (candidate.op() != "BatchMatMulV2" || candidate.input_size() != 2 ||
          !HasFloatTypeAttr(candidate) || GetNodeTypeAttr(candidate) != dtype) {
        continue;
      }
      const string in0 = NodeNameFromInputLocal(candidate.input(0));
      if (in0 != scale_mul->name()) continue;
      if (!get_bool_attr(candidate, "adj_x") &&
          !get_bool_attr(candidate, "adj_y")) {
        dq_bmm = &candidate;
        key_input = candidate.input(1);
      } else if (get_bool_attr(candidate, "adj_x") &&
                 !get_bool_attr(candidate, "adj_y")) {
        dk_bmm = &candidate;
        query_input = candidate.input(1);
      }
    }
    if (dq_bmm == nullptr || dk_bmm == nullptr || query_input.empty() ||
        key_input.empty()) {
      continue;
    }

    const string fused_name = softmax_grad.name() + "/musa_causal_attention_grad";
    if (node_map.find(fused_name) != node_map.end()) continue;
    const string scale_input =
        softmax->op() == "MusaCausalAttentionForward" ? softmax->input(3)
                                                       : softmax->input(1);

    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaCausalAttentionGrad");
    fused->set_device(softmax_grad.device());
    fused->add_input(softmax->name());
    fused->add_input(dout_input);
    fused->add_input(query_input);
    fused->add_input(key_input);
    fused->add_input(value_input);
    fused->add_input(scale_input);
    (*fused->mutable_attr())["T"].set_type(dtype);

    RedirectDataConsumersToInput(graph, dq_bmm->name(), fused_name, fused_name);
    RedirectDataConsumersToInput(graph, dk_bmm->name(), fused_name + ":1",
                                 fused_name);
    RedirectDataConsumersToInput(graph, dv_bmm->name(), fused_name + ":2",
                                 fused_name);

    remove_names.insert(ds_bmm->name());
    remove_names.insert(softmax_grad.name());
    remove_names.insert(scale_mul->name());
    remove_names.insert(dq_bmm->name());
    remove_names.insert(dk_bmm->name());
    remove_names.insert(dv_bmm->name());
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " causal attention grad group(s)";
  }
  return rewrites;
}


bool IsBoolToFloatCast(const NodeDef* node) {
  if (node == nullptr || node->op() != "Cast" || node->input_size() != 1) {
    return false;
  }
  const auto src_it = node->attr().find("SrcT");
  const auto dst_it = node->attr().find("DstT");
  return src_it != node->attr().end() && dst_it != node->attr().end() &&
         src_it->second.type() == DT_BOOL &&
         IsFloatOrBFloat16Type(dst_it->second.type());
}

bool TryGetMulConstScale(const NodeDef* mul,
                         const std::unordered_map<string, const NodeDef*>&
                             node_map,
                         string* data_input, float* scale) {
  if (mul == nullptr || mul->op() != "Mul" || mul->input_size() != 2 ||
      !HasFloatTypeAttr(*mul)) {
    return false;
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  const NodeDef* lhs = find_node(mul->input(0));
  const NodeDef* rhs = find_node(mul->input(1));
  float lhs_scale = 0.0f;
  float rhs_scale = 0.0f;
  if (GetConstScalarFloat(lhs, &lhs_scale)) {
    *data_input = mul->input(1);
    *scale = lhs_scale;
    return true;
  }
  if (GetConstScalarFloat(rhs, &rhs_scale)) {
    *data_input = mul->input(0);
    *scale = rhs_scale;
    return true;
  }
  return false;
}

int OptimizeDropoutScaledMaskMul(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  consumers.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] != '^') {
        consumers[NodeNameFromInputLocal(input)].push_back(node.name());
      }
    }
  }

  auto find_node = [&](const string& input) -> const NodeDef* {
    const auto it = node_map.find(NodeNameFromInputLocal(input));
    return it == node_map.end() ? nullptr : it->second;
  };

  auto create_fused = [&](const NodeDef& old_node, const string& data_input,
                          const string& mask_input, float scale,
                          std::unordered_set<string>* remove_names) -> bool {
    if (mask_input.empty()) return false;
    const DataType dtype = GetNodeTypeAttr(old_node);
    const string fused_name = old_node.name() + "/musa_scaled_masked_mul";
    if (node_map.find(fused_name) != node_map.end()) return false;

    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaScaledMaskedMul");
    fused->set_device(old_node.device());
    fused->add_input(data_input);
    fused->add_input(mask_input);
    (*fused->mutable_attr())["T"].set_type(dtype);
    (*fused->mutable_attr())["scale"].set_f(scale);

    RedirectNodeOutputs(graph, old_node.name(), fused_name);
    remove_names->insert(old_node.name());
    return true;
  };

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& node = graph->node(i);
    if (node.name().find("/dropout") == string::npos) {
      continue;
    }

    if (node.op() == "SelectV2" && node.input_size() == 3 &&
        HasFloatTypeAttr(node)) {
      const NodeDef* pred = find_node(node.input(0));
      const NodeDef* scaled_mul = find_node(node.input(1));
      const NodeDef* false_value = find_node(node.input(2));
      string data_input;
      float scale = 0.0f;
      if (pred != nullptr && pred->op() == "GreaterEqual" &&
          IsFloatZeroConst(false_value) &&
          TryGetMulConstScale(scaled_mul, node_map, &data_input, &scale)) {
        if (create_fused(node, data_input, node.input(0), scale,
                         &remove_names)) {
          if (IsOnlyConsumer(consumers, scaled_mul->name(), node.name())) {
            remove_names.insert(scaled_mul->name());
          }
          rewrites++;
        }
        continue;
      }
    }

    const NodeDef& final_mul = node;
    if (final_mul.op() != "Mul" || final_mul.input_size() != 2 ||
        !HasFloatTypeAttr(final_mul)) {
      continue;
    }

    const NodeDef* in0 = find_node(final_mul.input(0));
    const NodeDef* in1 = find_node(final_mul.input(1));
    if (in0 == nullptr || in1 == nullptr) continue;

    const NodeDef* select = nullptr;
    float select_scale = 0.0f;
    if (GetConstScalarFloat(in0, &select_scale) && in1->op() == "SelectV2") {
      select = in1;
    } else if (GetConstScalarFloat(in1, &select_scale) &&
               in0->op() == "SelectV2") {
      select = in0;
    }
    if (select != nullptr && select->input_size() == 3 &&
        HasFloatTypeAttr(*select)) {
      const NodeDef* pred = find_node(select->input(0));
      const NodeDef* false_value = find_node(select->input(2));
      if (pred != nullptr && pred->op() == "GreaterEqual" &&
          IsFloatZeroConst(false_value)) {
        if (create_fused(final_mul, select->input(1), select->input(0),
                         select_scale, &remove_names)) {
          if (IsOnlyConsumer(consumers, select->name(), final_mul.name())) {
            remove_names.insert(select->name());
          }
          rewrites++;
        }
        continue;
      }
    }

    const NodeDef* cast = nullptr;
    const NodeDef* scaled_mul = nullptr;
    if (IsBoolToFloatCast(in0) && in1->op() == "Mul") {
      cast = in0;
      scaled_mul = in1;
    } else if (IsBoolToFloatCast(in1) && in0->op() == "Mul") {
      cast = in1;
      scaled_mul = in0;
    }

    string data_input;
    float scale = 0.0f;
    if (cast != nullptr &&
        TryGetMulConstScale(scaled_mul, node_map, &data_input, &scale)) {
      if (create_fused(final_mul, data_input, cast->input(0), scale,
                       &remove_names)) {
        const auto scaled_consumer_it = consumers.find(scaled_mul->name());
        if (scaled_consumer_it == consumers.end() ||
            (scaled_consumer_it->second.size() == 1 &&
             scaled_consumer_it->second[0] == final_mul.name())) {
          remove_names.insert(scaled_mul->name());
        }
        rewrites++;
      }
      continue;
    }

    const NodeDef* scale_mul = nullptr;
    float final_scale = 0.0f;
    string mask_mul_input;
    if (GetConstScalarFloat(in0, &final_scale) && in1->op() == "Mul") {
      scale_mul = in1;
    } else if (GetConstScalarFloat(in1, &final_scale) && in0->op() == "Mul") {
      scale_mul = in0;
    }
    if (scale_mul == nullptr || scale_mul->input_size() != 2 ||
        !HasFloatTypeAttr(*scale_mul) ||
        !IsOnlyConsumer(consumers, scale_mul->name(), final_mul.name())) {
      continue;
    }

    const NodeDef* scale_in0 = find_node(scale_mul->input(0));
    const NodeDef* scale_in1 = find_node(scale_mul->input(1));
    const NodeDef* mask_cast = nullptr;
    if (IsBoolToFloatCast(scale_in0)) {
      mask_cast = scale_in0;
      mask_mul_input = scale_mul->input(1);
    } else if (IsBoolToFloatCast(scale_in1)) {
      mask_cast = scale_in1;
      mask_mul_input = scale_mul->input(0);
    }
    if (mask_cast == nullptr) continue;

    if (create_fused(final_mul, mask_mul_input, mask_cast->input(0),
                     final_scale, &remove_names)) {
      remove_names.insert(scale_mul->name());
      rewrites++;
    }
  }

  for (const auto& node : graph->node()) {
    if (!IsBoolToFloatCast(&node)) continue;
    const auto consumer_it = consumers.find(node.name());
    if (consumer_it == consumers.end() || consumer_it->second.empty()) {
      continue;
    }
    bool all_consumers_removed = true;
    for (const string& consumer : consumer_it->second) {
      if (remove_names.find(consumer) == remove_names.end()) {
        all_consumers_removed = false;
        break;
      }
    }
    if (all_consumers_removed) {
      remove_names.insert(node.name());
    }
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " dropout scaled-mask Mul node(s)";
  }
  return rewrites;
}

int OptimizeSliceGradAddNConcat(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] == '^') continue;
      consumers[NodeNameFromInputLocal(input)].push_back(node.name());
    }
  }

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& addn = graph->node(i);
    if (addn.op() != "AddN" || addn.input_size() != 2) continue;
    const auto t_it = addn.attr().find("T");
    if (t_it == addn.attr().end() ||
        !IsFloatOrBFloat16Type(t_it->second.type())) {
      continue;
    }

    const NodeDef* lhs =
        node_map.count(NodeNameFromInputLocal(addn.input(0)))
            ? node_map[NodeNameFromInputLocal(addn.input(0))]
            : nullptr;
    const NodeDef* rhs =
        node_map.count(NodeNameFromInputLocal(addn.input(1)))
            ? node_map[NodeNameFromInputLocal(addn.input(1))]
            : nullptr;
    SliceGradSegment a;
    SliceGradSegment b;
    if (lhs == nullptr || rhs == nullptr ||
        consumers[NodeNameFromInputLocal(addn.input(0))].size() != 1 ||
        consumers[NodeNameFromInputLocal(addn.input(1))].size() != 1 ||
        consumers[NodeNameFromInputLocal(addn.input(0))][0] != addn.name() ||
        consumers[NodeNameFromInputLocal(addn.input(1))][0] != addn.name()) {
      continue;
    }

    const NodeDef* first_segment = nullptr;
    const NodeDef* second_segment = nullptr;
    string first_dy;
    string second_dy;
    bool matched = false;

    if (TryParseConstAxis1SliceGrad(*lhs, node_map, &a) &&
        TryParseConstAxis1SliceGrad(*rhs, node_map, &b) &&
        a.output_shape == b.output_shape && a.end == b.start &&
        a.start == 0 && b.end == a.output_shape[1]) {
      first_segment = a.node;
      second_segment = b.node;
      first_dy = a.dy_input;
      second_dy = b.dy_input;
      matched = true;
    }

    if (!matched) {
      DynamicSliceGradSegment da;
      DynamicSliceGradSegment db;
      if (TryParseDynamicAxis1SliceGrad(*lhs, node_map, &da) &&
          TryParseDynamicAxis1SliceGrad(*rhs, node_map, &db) &&
          da.output_shape == db.output_shape &&
          da.split_input == db.split_input &&
          da.is_prefix != db.is_prefix) {
        const DynamicSliceGradSegment& prefix = da.is_prefix ? da : db;
        const DynamicSliceGradSegment& suffix = da.is_prefix ? db : da;
        first_segment = prefix.node;
        second_segment = suffix.node;
        first_dy = prefix.dy_input;
        second_dy = suffix.dy_input;
        matched = true;
      }
    }

    if (!matched) {
      continue;
    }

    const string concat_name = addn.name() + "/musa_slice_grad_concat";
    const string axis_name = concat_name + "/axis";

    NodeDef* axis = graph->add_node();
    axis->set_name(axis_name);
    axis->set_op("Const");
    auto* axis_attr = axis->mutable_attr();
    (*axis_attr)["dtype"].set_type(DT_INT32);
    TensorProto* axis_tensor = (*axis_attr)["value"].mutable_tensor();
    axis_tensor->set_dtype(DT_INT32);
    axis_tensor->add_int_val(1);

    NodeDef* concat = graph->add_node();
    concat->set_name(concat_name);
    concat->set_op("ConcatV2");
    concat->set_device(addn.device());
    concat->add_input(first_dy);
    concat->add_input(second_dy);
    concat->add_input(axis_name);
    auto* concat_attr = concat->mutable_attr();
    (*concat_attr)["N"].set_i(2);
    (*concat_attr)["T"].set_type(t_it->second.type());
    (*concat_attr)["Tidx"].set_type(DT_INT32);

    RedirectNodeOutputs(graph, addn.name(), concat_name);
    remove_names.insert(addn.name());
    remove_names.insert(first_segment->name());
    remove_names.insert(second_segment->name());
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " StridedSliceGrad+AddN pair(s) to ConcatV2";
  }
  return rewrites;
}


int OptimizeAddNWithSuffixSliceGrad(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] == '^') continue;
      consumers[NodeNameFromInputLocal(input)].push_back(node.name());
    }
  }

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& addn = graph->node(i);
    if (addn.op() != "AddN" || addn.input_size() < 2 ||
        addn.input_size() > 9) {
      continue;
    }
    const auto t_it = addn.attr().find("T");
    if (t_it == addn.attr().end() ||
        !IsFloatOrBFloat16Type(t_it->second.type())) {
      continue;
    }

    int slice_input_idx = -1;
    SliceGradSegment segment;
    int64_t inner_dim = 1;
    for (int input_idx = 0; input_idx < addn.input_size(); ++input_idx) {
      const string producer_name = NodeNameFromInputLocal(addn.input(input_idx));
      const auto node_it = node_map.find(producer_name);
      if (node_it == node_map.end()) continue;
      SliceGradSegment candidate;
      int64_t candidate_inner_dim = 1;
      if (!TryParseConstAxis1SuffixSliceGrad(*node_it->second, node_map,
                                             &candidate,
                                             &candidate_inner_dim)) {
        continue;
      }
      if (consumers[producer_name].size() != 1 ||
          consumers[producer_name][0] != addn.name()) {
        continue;
      }
      if (slice_input_idx != -1) {
        slice_input_idx = -1;
        break;
      }
      slice_input_idx = input_idx;
      segment = std::move(candidate);
      inner_dim = candidate_inner_dim;
    }
    if (slice_input_idx == -1) continue;

    const int num_base_inputs = addn.input_size() - 1;
    const string fused_name = addn.name() + "/musa_addn_slice_grad";
    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaAddNWithSliceGrad");
    fused->set_device(addn.device());
    for (int input_idx = 0; input_idx < addn.input_size(); ++input_idx) {
      if (input_idx == slice_input_idx) continue;
      fused->add_input(addn.input(input_idx));
    }
    fused->add_input(segment.dy_input);

    auto* attr = fused->mutable_attr();
    (*attr)["N"].set_i(num_base_inputs);
    (*attr)["T"].set_type(t_it->second.type());
    (*attr)["axis_dim"].set_i(segment.output_shape[1]);
    (*attr)["slice_start"].set_i(segment.start);
    (*attr)["inner_dim"].set_i(inner_dim);

    RedirectNodeOutputs(graph, addn.name(), fused_name);
    remove_names.insert(addn.name());
    remove_names.insert(segment.node->name());
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " AddN+suffix StridedSliceGrad node(s)";
  }
  return rewrites;
}

int OptimizeConcatWithSuffixSliceGrad(GraphDef* graph) {
  std::unordered_map<string, const NodeDef*> node_map;
  node_map.reserve(graph->node_size());
  for (const auto& node : graph->node()) {
    node_map[node.name()] = &node;
  }

  std::unordered_map<string, std::vector<string>> consumers;
  for (const auto& node : graph->node()) {
    for (const auto& input : node.input()) {
      if (!input.empty() && input[0] == '^') continue;
      consumers[NodeNameFromInputLocal(input)].push_back(node.name());
    }
  }

  std::unordered_set<string> remove_names;
  int rewrites = 0;
  const int original_node_size = graph->node_size();
  for (int i = 0; i < original_node_size; ++i) {
    const NodeDef& concat = graph->node(i);
    if (concat.op() != "ConcatV2" || concat.input_size() != 4) continue;
    const auto n_it = concat.attr().find("N");
    const auto t_it = concat.attr().find("T");
    if (n_it == concat.attr().end() || n_it->second.i() != 3 ||
        t_it == concat.attr().end() ||
        !IsFloatOrBFloat16Type(t_it->second.type())) {
      continue;
    }

    const string axis_name = NodeNameFromInputLocal(concat.input(3));
    const auto axis_it = node_map.find(axis_name);
    if (axis_it == node_map.end() || !IsConstLastAxis(axis_it->second)) {
      continue;
    }

    const string slice_name = NodeNameFromInputLocal(concat.input(0));
    const auto slice_it = node_map.find(slice_name);
    if (slice_it == node_map.end()) continue;
    if (consumers[slice_name].size() != 1 ||
        consumers[slice_name][0] != concat.name()) {
      continue;
    }

    SliceGradSegment segment;
    int64_t inner_dim = 1;
    if (!TryParseConstAxis1SuffixSliceGrad(*slice_it->second, node_map,
                                           &segment, &inner_dim)) {
      continue;
    }
    if (segment.output_shape.size() != 3 ||
        inner_dim != segment.output_shape[2]) {
      continue;
    }

    const string fused_name = concat.name() + "/musa_concat_slice_grad";
    NodeDef* fused = graph->add_node();
    fused->set_name(fused_name);
    fused->set_op("MusaConcatWithSliceGrad");
    fused->set_device(concat.device());
    fused->add_input(segment.dy_input);
    fused->add_input(concat.input(1));
    fused->add_input(concat.input(2));

    auto* attr = fused->mutable_attr();
    (*attr)["T"].set_type(t_it->second.type());
    (*attr)["axis_dim"].set_i(segment.output_shape[1]);
    (*attr)["slice_start"].set_i(segment.start);
    (*attr)["inner_dim"].set_i(inner_dim);

    RedirectNodeOutputs(graph, concat.name(), fused_name);
    remove_names.insert(concat.name());
    remove_names.insert(segment.node->name());
    rewrites++;
  }

  RemoveNodesByName(graph, remove_names);
  if (rewrites > 0) {
    LOG(INFO) << "MusaGraphOptimizer: Rewrote " << rewrites
              << " ConcatV2+suffix StridedSliceGrad node(s)";
  }
  return rewrites;
}

}  // namespace

// Unified MUSA Graph Optimizer
// Combines Layout optimization and AMP (Automatic Mixed Precision)
// Based on Modular TensorFlow Graph C API design principles
class MusaGraphOptimizer : public CustomGraphOptimizer {
 public:
  MusaGraphOptimizer() : device_type_(kMusaDeviceType) {}
  ~MusaGraphOptimizer() override {}

  std::string name() const override { return "musa_graph_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    disabled_fusion_patterns_.clear();
    disable_all_fusion_patterns_ = false;

    // Environment variable control for AMP (performance quick win)
    const char* amp_env = std::getenv("MUSA_AUTO_MIXED_PRECISION");
    if (amp_env && std::string(amp_env) == "1") {
      configs_.auto_mixed_precision = TriState::kOn;
      VLOG(1)
          << "MusaGraphOptimizer: AMP enabled via MUSA_AUTO_MIXED_PRECISION=1";
    }

    // Environment variable for AMP mode (FP16 or BF16)
    const char* amp_mode_env = std::getenv("MUSA_AMP_MODE");
    if (amp_mode_env) {
      std::string mode(amp_mode_env);
      if (mode == "BF16" || mode == "BFLOAT16") {
        amp_config_.target_dtype = DT_BFLOAT16;
        VLOG(1) << "MusaGraphOptimizer: AMP mode set to BF16";
      } else if (mode == "FP16") {
        amp_config_.target_dtype = DT_HALF;
        VLOG(1) << "MusaGraphOptimizer: AMP mode set to FP16";
      }
    }

    // Environment variable to disable all Grappler optimizations
    const char* disable_grappler_env = std::getenv("MUSA_DISABLE_GRAPPLER");
    if (disable_grappler_env && std::string(disable_grappler_env) == "1") {
      configs_.constant_folding = TriState::kOff;
      configs_.remapping = TriState::kOff;
      configs_.arithmetic_optimization = TriState::kOff;
      configs_.shape_optimization = TriState::kOff;
      VLOG(1) << "MusaGraphOptimizer: All Grappler optimizations disabled via "
                 "MUSA_DISABLE_GRAPPLER=1";
    }

    if (config) {
      for (const auto& param : config->parameter_map()) {
        if (param.first == "aggressive_mode") {
          amp_config_.aggressive_mode = param.second.b();
        } else if (param.first == "precision_mode") {
          string mode = param.second.s();
          if (mode == "BF16" || mode == "BFLOAT16") {
            amp_config_.target_dtype = DT_BFLOAT16;
          } else {
            amp_config_.target_dtype = DT_HALF;
          }
        } else if (param.first == "disable_layout_optimizer") {
          // Allow user to disable layout optimization
          if (param.second.b()) {
            configs_.layout_optimizer = TriState::kOff;
          }
        } else if (param.first == "disable_amp") {
          // Allow user to disable AMP
          if (param.second.b()) {
            configs_.auto_mixed_precision = TriState::kOff;
          }
        } else if (param.first == kDisabledFusionPatternsParam) {
          SetDisabledFusionPatterns(param.second.s());
        }
      }
    }
    return ::tensorflow::OkStatus();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
    *optimized_graph = item.graph;

    // Initialize dumper for this optimization run
    GraphDefDumper dumper("musa_optimizer");
    dumper.DumpInitial(*optimized_graph);

    // Skip optimization if graph doesn't contain MUSA nodes
    if (!GraphHasMusaNodes(*optimized_graph)) {
      VLOG(2)
          << "MusaGraphOptimizer: No MUSA nodes found, skipping optimization";
      dumper.DumpFinal(*optimized_graph);
      return ::tensorflow::OkStatus();
    }

    VLOG(1) << "MusaGraphOptimizer: Optimizing graph with "
            << optimized_graph->node_size() << " nodes";

    // Step 1: Layout optimization (NHWC -> NCHW)
    if (configs_.layout_optimizer != TriState::kOff) {
      dumper.DumpBeforePass(*optimized_graph, "layout");
      OptimizeLayout(optimized_graph);
      dumper.DumpAfterPass(*optimized_graph, "layout");
    }

    // Step 2: AMP optimization (FP32 -> FP16)
    if (configs_.auto_mixed_precision != TriState::kOff) {
      dumper.DumpBeforePass(*optimized_graph, "amp");
      OptimizeAMP(optimized_graph);
      dumper.DumpAfterPass(*optimized_graph, "amp");
    }

    // Step 3: Fusion optimization (LayerNorm, GELU, etc.)
    if (configs_.remapping != TriState::kOff) {
      dumper.DumpBeforePass(*optimized_graph, "fusion");
      TF_RETURN_IF_ERROR(OptimizeFusion(optimized_graph));
      dumper.DumpAfterPass(*optimized_graph, "fusion");
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_QKV_DENSE_DW_FUSION")) {
      const int qkv_dense_dw_rewrites =
          OptimizeQkvDenseDw(optimized_graph);
      if (qkv_dense_dw_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_qkv_dense_dw_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_FORWARD_RMSNORM_FUSION")) {
      const int forward_rmsnorm_rewrites =
          OptimizeForwardRmsNorm(optimized_graph);
      if (forward_rmsnorm_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph, "after_forward_rmsnorm_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_SLICE_GRAD_ADDN_CONCAT_FUSION")) {
      const int slice_grad_addn_rewrites =
          OptimizeSliceGradAddNConcat(optimized_graph);
      if (slice_grad_addn_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph, "after_slice_grad_addn_concat");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_ADDN_SUFFIX_SLICE_GRAD_FUSION")) {
      const int addn_slice_grad_rewrites =
          OptimizeAddNWithSuffixSliceGrad(optimized_graph);
      if (addn_slice_grad_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_addn_suffix_slice_grad_fusion");
      }
    }
    if (!EnvFlagEnabledLocal("MUSA_DISABLE_CONCAT_SUFFIX_SLICE_GRAD_FUSION")) {
      const int concat_slice_grad_rewrites =
          OptimizeConcatWithSuffixSliceGrad(optimized_graph);
      if (concat_slice_grad_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_concat_suffix_slice_grad_fusion");
      }
    }
    if (!EnvFlagEnabledLocal("MUSA_DISABLE_GELU_GRAD_FUSION")) {
      const int gelu_grad_rewrites = OptimizeExactGeluGrad(optimized_graph);
      if (gelu_grad_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_exact_gelu_grad_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_DROPOUT_SCALED_MASK_FUSION")) {
      const int dropout_scaled_mask_rewrites =
          OptimizeDropoutScaledMaskMul(optimized_graph);
      if (dropout_scaled_mask_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_dropout_scaled_mask_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_CAUSAL_MASK_SOFTMAX_FUSION")) {
      const int causal_mask_softmax_rewrites =
          OptimizeCausalMaskedSoftmax(optimized_graph);
      if (causal_mask_softmax_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_causal_mask_softmax_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_CAUSAL_ATTENTION_FORWARD_FUSION")) {
      const int causal_attention_forward_rewrites =
          OptimizeCausalAttentionForward(optimized_graph);
      if (causal_attention_forward_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_causal_attention_forward_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_SOFTMAX_GRAD_FUSION")) {
      const int softmax_grad_rewrites =
          OptimizeSoftmaxGradSmallLastDim(optimized_graph);
      if (softmax_grad_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_softmax_grad_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_CAUSAL_ATTENTION_GRAD_FUSION")) {
      const int causal_attention_grad_rewrites =
          OptimizeCausalAttentionGrad(optimized_graph);
      if (causal_attention_grad_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_causal_attention_grad_fusion");
      }
    }

    if (!EnvFlagEnabledLocal("MUSA_DISABLE_RMSNORM_GRAD_DX_FUSION")) {
      const int rms_norm_grad_dx_rewrites =
          OptimizeRmsNormGradDxWarp128(optimized_graph);
      if (rms_norm_grad_dx_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_rmsnorm_grad_dx_warp128_fusion");
      }
    }
    if (!EnvFlagEnabledLocal("MUSA_DISABLE_RMSNORM_SQRT_GRAD_DX_FUSION")) {
      const int rms_norm_sqrt_grad_dx_rewrites =
          OptimizeRmsNormSqrtGradDxWarp128(optimized_graph);
      if (rms_norm_sqrt_grad_dx_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_rmsnorm_sqrt_grad_dx_warp128_fusion");
      }
    }
    if (!EnvFlagEnabledLocal("MUSA_DISABLE_TF215_SGD_ASSIGN_ADD_FUSION")) {
      const int tf215_sgd_rewrites = OptimizeTf215SgdAssignAdd(optimized_graph);
      if (tf215_sgd_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_tf215_sgd_assign_add_fusion");
      }
    }
    if (!EnvFlagEnabledLocal("MUSA_DISABLE_TF215_MIXED_BF16_SGD_APPLY_FUSION")) {
      const int mixed_bf16_sgd_rewrites =
          OptimizeBFloat16GradientDescentCast(optimized_graph);
      if (mixed_bf16_sgd_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_tf215_mixed_bf16_sgd_apply_fusion");
      }
    }
    if (!EnvFlagEnabledLocal("MUSA_DISABLE_TF215_SPARSE_SGD_ASSIGN_ADD_FUSION")) {
      const int tf215_sparse_sgd_rewrites =
          OptimizeTf215SparseSgdAssignAdd(optimized_graph);
      if (tf215_sparse_sgd_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_tf215_sparse_sgd_assign_add_fusion");
      }
    }
    if (!EnvFlagEnabledLocal("MUSA_DISABLE_TF215_MIXED_BF16_SPARSE_SGD_FUSION")) {
      const int mixed_bf16_sparse_sgd_rewrites =
          OptimizeBFloat16SparseSgdScatter(optimized_graph);
      if (mixed_bf16_sparse_sgd_rewrites > 0) {
        dumper.DumpAtStage(*optimized_graph,
                           "after_tf215_mixed_bf16_sparse_sgd_fusion");
      }
    }
    if (configs_.optimizer_remove_ios_node != TriState::kOff) {
      const int removed_isolated_nodes = RemoveIsolatedNodes(optimized_graph);
      if (removed_isolated_nodes > 0) {
        VLOG(1) << "MusaGraphOptimizer: Removed " << removed_isolated_nodes
                << " isolated node(s) after optimization";
      }
    }

    VLOG(1) << "MusaGraphOptimizer: Optimization complete, graph now has "
            << optimized_graph->node_size() << " nodes";

    // Dump final graph
    dumper.DumpFinal(*optimized_graph);

    // Debug: print all node names and check for consumers of fused node
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "MusaGraphOptimizer: Nodes in optimized graph:";
      for (const auto& node : optimized_graph->node()) {
        VLOG(2) << "  - " << node.name() << " (" << node.op() << ")";
      }
    }
    return ::tensorflow::OkStatus();
  }

  // Feedback method removed - not available in TF 2.6.1 CustomGraphOptimizer
  // interface void Feedback(Cluster* cluster, const GrapplerItem& item,
  //               const GraphDef& optimized_graph, double result) override {}

  // Get optimizer configurations - used for coordination with other optimizers
  const MusaOptimizerConfigs& GetConfigs() const { return configs_; }

  // Fusion optimization - applies registered fusion patterns
  Status OptimizeFusion(GraphDef* graph) {
    using namespace ::tensorflow::grappler::musa_fusion;

    VLOG(1) << "MusaGraphOptimizer: Starting fusion optimization";

    auto& pattern_manager = FusionPatternManager::GetInstance();
    auto patterns = pattern_manager.GetSortedPatterns();

    if (patterns.empty()) {
      VLOG(1) << "MusaGraphOptimizer: No fusion patterns registered";
      return ::tensorflow::OkStatus();
    }

    VLOG(1) << "MusaGraphOptimizer: Applying " << patterns.size()
            << " fusion patterns";

    int fusion_applied_count = 0;
    int fusion_fallback_count = 0;

    std::map<int, std::vector<const FusionPattern*>, std::greater<int>>
        priority_groups;
    for (const auto* pattern : patterns) {
      if (!pattern->IsEnabled()) {
        continue;
      }
      if (IsFusionPatternDisabled(pattern->GetName())) {
        VLOG(1) << "MusaGraphOptimizer: Fusion pattern '" << pattern->GetName()
                << "' disabled by " << kDisabledFusionPatternsParam;
        continue;
      }
      priority_groups[pattern->GetPriority()].push_back(pattern);
    }

    if (priority_groups.empty()) {
      VLOG(1) << "MusaGraphOptimizer: No enabled fusion patterns after "
              << kDisabledFusionPatternsParam << " filtering";
      return ::tensorflow::OkStatus();
    }

    auto run_scan =
        [&](const std::vector<const FusionPattern*>& active_patterns,
            bool reverse) -> bool {
      bool pass_modified = false;

      while (true) {
        bool applied_in_sweep = false;
        const int node_count = graph->node_size();

        for (int offset = 0; offset < node_count; ++offset) {
          const int node_idx = reverse ? (node_count - 1 - offset) : offset;

          for (const auto* pattern : active_patterns) {
            auto match_result = pattern->Match(*graph, node_idx);
            if (!match_result.matched) {
              continue;
            }

            if (!pattern->IsKernelAvailable()) {
              VLOG(1) << "MusaGraphOptimizer: Pattern '" << pattern->GetName()
                      << "' matched at node " << node_idx
                      << " but kernel not available - using fallback";

              Status status = pattern->Apply(graph, match_result);
              if (!status.ok()) {
                LOG(WARNING) << "MusaGraphOptimizer: Fallback for pattern '"
                             << pattern->GetName() << "' failed: " << status;
              }
              fusion_fallback_count++;
              continue;
            }

            VLOG(1) << "MusaGraphOptimizer: Applying pattern '"
                    << pattern->GetName() << "' at node " << node_idx;

            Status status = pattern->Apply(graph, match_result);
            if (status.ok()) {
              pass_modified = true;
              fusion_applied_count++;
              applied_in_sweep = true;
              VLOG(1) << "MusaGraphOptimizer: Pattern '" << pattern->GetName()
                      << "' applied successfully";
              break;
            } else {
              LOG(WARNING) << "MusaGraphOptimizer: Pattern '"
                           << pattern->GetName()
                           << "' apply failed: " << status;
            }
          }

          if (applied_in_sweep) {
            break;
          }
        }

        if (!applied_in_sweep) {
          return pass_modified;
        }
      }
    };

    bool graph_modified = true;
    int iteration = 0;
    const int kMaxIterations = 50;  // Prevent infinite loops

    while (graph_modified && iteration < kMaxIterations) {
      graph_modified = false;
      iteration++;

      for (const auto& priority_group : priority_groups) {
        const int priority = priority_group.first;
        const auto& active_patterns = priority_group.second;

        bool priority_modified = true;
        int priority_iteration = 0;
        while (priority_modified && priority_iteration < kMaxIterations) {
          priority_modified = false;
          priority_iteration++;

          if (run_scan(active_patterns, false)) {
            priority_modified = true;
            graph_modified = true;
          }
          if (run_scan(active_patterns, true)) {
            priority_modified = true;
            graph_modified = true;
          }
        }

        if (priority_modified && priority_iteration >= kMaxIterations) {
          LOG(WARNING) << "MusaGraphOptimizer: Priority " << priority
                       << " group hit iteration limit (" << kMaxIterations
                       << ") before reaching a fixed point";
        } else {
          VLOG(2) << "MusaGraphOptimizer: Priority " << priority
                  << " group reached fixed point";
        }
      }
    }

    if (graph_modified && iteration >= kMaxIterations) {
      LOG(WARNING) << "MusaGraphOptimizer: Fusion optimization hit iteration "
                   << "limit (" << kMaxIterations
                   << ") before reaching a fixed point. Remaining fusible "
                   << "subgraphs may require a higher cap or a matcher "
                   << "investigation.";
    }

    VLOG(1) << "MusaGraphOptimizer: Fusion optimization complete. "
            << "Applied: " << fusion_applied_count
            << ", Fallbacks: " << fusion_fallback_count;

    return ::tensorflow::OkStatus();
  }

 private:
  void SetDisabledFusionPatterns(const string& disabled_patterns) {
    disabled_fusion_patterns_.clear();
    disable_all_fusion_patterns_ = false;

    size_t token_begin = 0;
    while (token_begin <= disabled_patterns.size()) {
      const size_t token_end = disabled_patterns.find(',', token_begin);
      const string token = NormalizeFusionPatternName(
          disabled_patterns.substr(token_begin, token_end - token_begin));

      if (!token.empty()) {
        if (token == "all") {
          disable_all_fusion_patterns_ = true;
          disabled_fusion_patterns_.clear();
          VLOG(1) << "MusaGraphOptimizer: All fusion patterns disabled by "
                  << kDisabledFusionPatternsParam;
          return;
        }
        disabled_fusion_patterns_.insert(token);
      }

      if (token_end == string::npos) {
        break;
      }
      token_begin = token_end + 1;
    }

    if (!disabled_fusion_patterns_.empty()) {
      VLOG(1) << "MusaGraphOptimizer: Disabled "
              << disabled_fusion_patterns_.size() << " fusion pattern(s) by "
              << kDisabledFusionPatternsParam;
    }
  }

  bool IsFusionPatternDisabled(const string& pattern_name) const {
    return disable_all_fusion_patterns_ ||
           disabled_fusion_patterns_.count(
               NormalizeFusionPatternName(pattern_name)) > 0;
  }

  MusaAmpConfig amp_config_;
  MusaOptimizerConfigs configs_;
  string device_type_;
  bool disable_all_fusion_patterns_ = false;
  std::unordered_set<string> disabled_fusion_patterns_;

  // Layout Optimization
  void OptimizeLayout(GraphDef* graph) {
    bool changed = true;
    int iteration = 0;
    const int kMaxIterations = 5;

    while (changed && iteration < kMaxIterations) {
      changed = false;
      iteration++;

      for (int i = 0; i < graph->node_size(); ++i) {
        NodeDef* node = graph->mutable_node(i);

        if (!MusaGraphUtils::IsMusaNCHWSupported(*node)) {
          continue;
        }

        auto* attr = node->mutable_attr();
        bool is_already_nchw = (attr->count("data_format") &&
                                (*attr)["data_format"].s() == "NCHW");
        if (is_already_nchw) continue;

        bool has_nchw_upstream = false;
        if (node->input_size() > 0) {
          if (node->input(0).find("/post_transpose_nhwc") !=
              std::string::npos) {
            has_nchw_upstream = true;
          }
        }

        bool should_transform = false;
        if (MusaGraphUtils::kLayoutSensitiveOps(*node)) {
          should_transform = true;
        } else if (MusaGraphUtils::kLayoutAgnosticOps(*node) &&
                   has_nchw_upstream) {
          should_transform = true;
        }

        if (should_transform) {
          std::string op_name = node->name();
          DataType dtype = (*attr)["T"].type();
          std::string device = node->device();

          if (has_nchw_upstream) {
            std::string real_src = node->input(0).substr(
                0, node->input(0).find("/post_transpose_nhwc"));
            node->set_input(0, real_src);
          } else {
            std::string pre_name = op_name + "/pre_transpose_nchw";
            MusaGraphUtils::InsertTranspose(graph, pre_name, node->input(0),
                                            {0, 3, 1, 2}, dtype, device);
            node->set_input(0, pre_name);
          }

          (*attr)["data_format"].set_s("NCHW");
          if (MusaGraphUtils::kLayoutSensitiveOps(*node)) {
            MusaGraphUtils::RewriteLayoutAttributes(node);
          }

          std::string post_name = op_name + "/post_transpose_nhwc";
          MusaGraphUtils::InsertTranspose(graph, post_name, op_name,
                                          {0, 2, 3, 1}, dtype, device);
          MusaGraphUtils::RedirectEdges(graph, op_name, post_name);

          changed = true;
        }
      }
    }
  }

  // AMP Optimization
  void OptimizeAMP(GraphDef* graph) {
    std::unordered_map<string, bool> should_convert;
    AnalyzeGraphForAMP(*graph, should_convert);

    int original_node_size = graph->node_size();
    for (int i = 0; i < original_node_size; ++i) {
      NodeDef* node = graph->mutable_node(i);

      if (node->device().find(kMusaDeviceType) == std::string::npos) {
        continue;
      }

      if (!should_convert[node->name()]) {
        continue;
      }

      DataType dtype = GetNodeDataType(node);
      if (dtype != DT_FLOAT) {
        continue;
      }

      ConvertNodeToLowPrecision(graph, node);
    }
  }

  void AnalyzeGraphForAMP(const GraphDef& graph,
                          std::unordered_map<string, bool>& should_convert) {
    std::unordered_map<string, const NodeDef*> node_map;
    for (const auto& node : graph.node()) {
      node_map[node.name()] = &node;
    }

    for (const auto& node : graph.node()) {
      bool convert = false;

      if (amp_config_.fp16_compute_ops.count(node.op())) {
        convert = true;
      }

      if (amp_config_.fp32_keep_ops.count(node.op())) {
        convert = false;
      }

      if (amp_config_.activation_ops.count(node.op())) {
        if (node.input_size() > 0) {
          string input_name = GetNodeNameFromInput(node.input(0));
          if (node_map.count(input_name)) {
            const NodeDef* input_node = node_map.at(input_name);
            if (amp_config_.fp16_compute_ops.count(input_node->op())) {
              convert = true;
            }
          }
        }
      }

      if (amp_config_.conditional_ops.count(node.op())) {
        if (amp_config_.aggressive_mode) {
          convert = true;
        } else {
          int low_prec_inputs = 0;
          for (const auto& input : node.input()) {
            if (input[0] == '^') continue;
            string input_name = GetNodeNameFromInput(input);
            if (node_map.count(input_name)) {
              const NodeDef* input_node = node_map.at(input_name);
              if (amp_config_.fp16_compute_ops.count(input_node->op())) {
                low_prec_inputs++;
              }
            }
          }
          if (low_prec_inputs >= 1) {
            convert = true;
          }
        }
      }

      should_convert[node.name()] = convert;
    }
  }

  string GetNodeNameFromInput(const string& input) {
    if (input.empty()) return "";
    if (input[0] == '^') return input.substr(1);

    size_t colon_pos = input.find(':');
    if (colon_pos != std::string::npos) {
      return input.substr(0, colon_pos);
    }
    return input;
  }

  DataType GetNodeDataType(const NodeDef* node) {
    if (node->attr().count("T")) {
      return node->attr().at("T").type();
    } else if (node->attr().count("dtype")) {
      return node->attr().at("dtype").type();
    }
    return DT_INVALID;
  }

  bool ConvertNodeToLowPrecision(GraphDef* graph, NodeDef* node) {
    string op_name = node->name();
    string device = node->device();
    DataType target_t = amp_config_.target_dtype;

    if (node->mutable_attr()->count("T")) {
      (*node->mutable_attr())["T"].set_type(target_t);
    } else if (node->mutable_attr()->count("dtype")) {
      (*node->mutable_attr())["dtype"].set_type(target_t);
    }

    std::vector<string> new_inputs;
    for (int idx = 0; idx < node->input_size(); ++idx) {
      string input_name = node->input(idx);

      if (input_name.empty() || input_name[0] == '^') {
        new_inputs.push_back(input_name);
        continue;
      }

      if (input_name.find("/CastF2Lower") != std::string::npos) {
        new_inputs.push_back(input_name);
        continue;
      }

      string cast_in_name =
          op_name + "/Input_" + std::to_string(idx) + "/CastF2Lower";

      MusaGraphUtils::InsertCast(graph, cast_in_name, input_name, DT_FLOAT,
                                 target_t, device);
      new_inputs.push_back(cast_in_name);
    }

    node->clear_input();
    for (const auto& input : new_inputs) {
      node->add_input(input);
    }

    string cast_out_name = op_name + "/Output/CastLower2F";
    MusaGraphUtils::InsertCast(graph, cast_out_name, op_name, target_t,
                               DT_FLOAT, device);

    for (int j = 0; j < graph->node_size(); ++j) {
      NodeDef* consumer = graph->mutable_node(j);
      if (consumer->name() == cast_out_name) continue;

      for (int k = 0; k < consumer->input_size(); ++k) {
        string inp = consumer->input(k);

        if (inp == op_name) {
          consumer->set_input(k, cast_out_name);
        } else if (inp.find(op_name + ":") == 0) {
          string suffix = inp.substr(op_name.length());
          consumer->set_input(k, cast_out_name + suffix);
        } else if (inp == "^" + op_name) {
          consumer->set_input(k, "^" + cast_out_name);
        }
      }
    }

    return true;
  }
};

REGISTER_GRAPH_OPTIMIZER_AS(MusaGraphOptimizer, "musa_graph_optimizer");

class MusaAutoGraphOptimizationPass : public ::tensorflow::GraphOptimizationPass {
 public:
  Status Run(const ::tensorflow::GraphOptimizationPassOptions& options) override {
    if (options.graph == nullptr || options.graph->get() == nullptr) {
      return ::tensorflow::OkStatus();
    }

    GraphDef graph_def;
    (*options.graph)->ToGraphDef(&graph_def);
    if (graph_def.node_size() == 0) {
      return ::tensorflow::OkStatus();
    }

    GrapplerItem item;
    item.id = "musa_auto_graph_pass";
    item.graph = graph_def;

    MusaGraphOptimizer optimizer;
    TF_RETURN_IF_ERROR(optimizer.Init(nullptr));

    GraphDef optimized_graph_def;
    TF_RETURN_IF_ERROR(optimizer.Optimize(nullptr, item, &optimized_graph_def));

    std::unique_ptr<::tensorflow::Graph> new_graph(
        new ::tensorflow::Graph((*options.graph)->op_registry()));
    ::tensorflow::GraphConstructorOptions constructor_opts;
    constructor_opts.allow_internal_ops = true;
    constructor_opts.add_default_attributes = true;
    TF_RETURN_IF_ERROR(::tensorflow::ConvertGraphDefToGraph(
        constructor_opts, std::move(optimized_graph_def), new_graph.get()));
    options.graph->swap(new_graph);
    LOG(INFO) << "MusaAutoGraphOptimizationPass applied to graph with "
              << graph_def.node_size() << " node(s)";
    return ::tensorflow::OkStatus();
  }
};

REGISTER_OPTIMIZATION(::tensorflow::OptimizationPassRegistry::PRE_PLACEMENT, 60,
                      MusaAutoGraphOptimizationPass);

}  // namespace grappler
}  // namespace tensorflow

extern "C" {
// This function will be called when the plugin is loaded
// Note: Full C API (TF_InitGraphPlugin) is available in TensorFlow 2.5+
// For TensorFlow 2.4.4, we use C++ API with REGISTER_GRAPH_OPTIMIZER_AS
void __attribute__((constructor)) ForceMusaGraphOptimizerLoad() {
  // Optimizer is automatically registered via REGISTER_GRAPH_OPTIMIZER_AS
  VLOG(1) << "MUSA Graph Optimizer plugin loaded (v2.4.4 C++ API mode)";
}
}
