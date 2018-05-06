// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "relu_opencl.h"

#include "gpu.h"
#include "net.h"

#include <stdio.h>

namespace ncnn {

ReLU_opencl::ReLU_opencl()
{
    one_blob_only = true;
    support_inplace = true;
    support_opencl = false;

    relu = get_gpu_kernel("relu");
}

int ReLU_opencl::finalize()
{
//     fprintf(stderr, "hello relu opencl finalize\n");

    if (slope == 0.f)
        support_opencl = true;

    return 0;
}

int ReLU_opencl::forward_inplace(Queue& queue, Mat& bottom_top_blob) const
{
//     fprintf(stderr, "hello relu opencl\n");

    cl_mem cldata = bottom_top_blob.cldata;
    int size = bottom_top_blob.total();

    cl_int error = 0;

    error |= clSetKernelArg(relu, 0, sizeof(cl_mem), &cldata);
    error |= clSetKernelArg(relu, 1, sizeof(int), &size);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clSetKernelArg failed %d\n", error);
        return -123;
    }

    const size_t local_ws = 32;
    const size_t global_ws = size;
    error = clEnqueueNDRangeKernel(queue, relu, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clEnqueueNDRangeKernel failed %d\n", error);
        return -123;
    }

    return 0;
}

} // namespace ncnn
