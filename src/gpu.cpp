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

#include <map>

// opencl
#include <CL/cl.h>

namespace ncnn {

Queue::Queue()
{
    cl_int error;
    clqueue = clCreateCommandQueue(get_gpu_context(), get_gpu_device(), 0, &error);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clCreateCommandQueue failed %d\n", error);
    }
}

Queue::~Queue()
{
    clReleaseCommandQueue(clqueue);
}

static cl_device_id g_device;
static cl_context g_context;

int init_gpu_device()
{
    cl_int error;

    cl_platform_id platform;
    error = clGetPlatformIDs(1, &platform, NULL);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clGetPlatformIDs failed %d\n", error);
        return -1;
    }

    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clGetDeviceIDs failed %d\n", error);
        return -1;
    }

    g_context = clCreateContext(0, 1, &g_device, NULL, NULL, &error);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clCreateContext failed %d\n", error);
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

static cl_program g_program;

int compile_gpu_kernel()
{
    cl_int error;

    const char bigsource[] =
"__kernel void relu(__global float* ptr, const int size)"
"{                                                      "
"    const int i = get_global_id(0);                    "
"                                                       "
"    if (i < size)                                      "
"        ptr[i] = max(ptr[i], 0.f);                     "
"}                                                      ";

    const char* source = bigsource;
    size_t size = sizeof(bigsource);

    g_program = clCreateProgramWithSource(g_context, 1, &source, &size, &error);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clCreateProgramWithSource failed %d\n", error);
        return -1;
    }

    error = clBuildProgram(g_program, 1, &g_device, NULL, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clBuildProgram failed %d\n", error);

        size_t log_size;
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char* build_log = new char[log_size+1];
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);

        build_log[log_size] = '\0';

        fprintf(stderr, "%s\n", build_log);

        delete[] build_log;

        return -1;
    }

    return 0;
}

cl_kernel get_gpu_kernel(const char* name)
{
    cl_int error;

    cl_kernel kernel = clCreateKernel(g_program, name, &error);
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clCreateKernel failed %d\n", error);
        return 0;
    }

    // TODO cache kernel
    return kernel;
}

} // namespace ncnn
