/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel.h"

// 默认写的.cu是fp32的，TensorRT在fp16运行模式下，运行到不支持fp16的插件op时，会自动切换到fp32模式，等插件op运行完再切换回来。
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void pCustomKernel(const int n, const float negativeSlope, const float* input, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = input[i] > 0 ? input[i] : input[i] * negativeSlope;
    }
}

pluginStatus_t CustomGPU(cudaStream_t stream, const int n, const float negativeSlope, const void* input, void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;
    pCustomKernel<BS><<<GS, BS, 0, stream>>>(n, negativeSlope,
                                           (const float*) input,
                                           (float*) output);
    return STATUS_SUCCESS;
}

pluginStatus_t CustomInference(
    cudaStream_t stream, const int n, const float negativeSlope, const void* input, void* output)
{
    return CustomGPU(stream, n, negativeSlope, (const float*) input, (float*) output);
}
