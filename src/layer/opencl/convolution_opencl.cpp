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

#include "convolution_opencl.h"

#include "gpu.h"
#include "net.h"

#include <stdio.h>

namespace ncnn {

Convolution_opencl::Convolution_opencl()
{
    one_blob_only = true;
    support_inplace = false;
    support_opencl = false;

    convolution = get_gpu_kernel("convolution");
}

int Convolution_opencl::finalize()
{
    fprintf(stderr, "hello convolution opencl finalize\n");

//     support_opencl = true;

    return 0;
}

int Convolution_opencl::forward(Queue& queue, const Mat& bottom_blob, Mat& top_blob) const
{
    // convolv with NxN kernel
    // value = value + bias

    fprintf(stderr, "hello convolution opencl\n");

    return 0;
}

} // namespace ncnn
