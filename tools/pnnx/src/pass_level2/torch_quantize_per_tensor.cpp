// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pass_level2.h"

namespace pnnx {

class torch_quantize_per_tensor : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_scale 0 1 input_scale
pnnx.Input              input_zero_point 0 1 input_zero_point
aten::FloatImplicit     op_0        1 1 input_scale scale
aten::IntImplicit       op_1        1 1 input_zero_point zero_point
prim::Constant          op_2        0 1 dtype value=%dtype
aten::quantize_per_tensor op_3      4 1 input scale zero_point dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.quantize_per_tensor";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_quantize_per_tensor, 40)

} // namespace pnnx
