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

#ifndef PNNX_PASS_LEVEL1_H
#define PNNX_PASS_LEVEL1_H

#include "ir.h"

namespace pnnx {

void pass_level1(const torch::jit::Module& mod, const std::shared_ptr<torch::jit::Graph>& g, const std::vector<std::string>& module_operators, Graph& pg);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL1_H
