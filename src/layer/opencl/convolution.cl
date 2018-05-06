// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

__kernel void convolution(__global const float* ptr, const int w, const int h, const int channels,
                          __global float* outptr, const int outw, const int outh, const int outc,
                          __global const float* kptr,
                          __global const float* bias,
                          const int kernel_w, const int kernel_h,
                          const int conv_c, const int conv_n,
                          const int stride_w, const int stride_h,
                          const int pad_left, const int pad_right, const int pad_top, const int pad_bot)
{
    const int id = get_global_id(0);
}
