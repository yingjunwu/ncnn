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

#include "gpu.h"

#include <stdio.h>

// opencl
#include <CL/cl.h>

namespace ncnn {

static cl_device_id g_device;
static cl_context g_context;

int init_gpu_device()
{
    cl_int error;

    cl_platform_id platform;
    error = clGetPlatformIDs(1, &platform, NULL);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clGetPlatformIDs failed\n");
        return -1;
    }

    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clGetDeviceIDs failed\n");
        return -1;
    }

    g_context = clCreateContext(0, 1, &g_device, NULL, NULL, &error);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clCreateContext failed\n");
        return -1;
    }

    return 0;
}

cl_device_id get_gpu_device()
{
    return g_device;
}

cl_context get_gpu_context()
{
    return g_context;
}

} // namespace ncnn
