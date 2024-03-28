// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: points(b, c, n)(1 3 20000)
// idx: (b, npoints, nsample)(1 1024 64)
// output: out(b, c, npoints, nsample)(1 3 1024 64)
// block size: (1024, 3)
// 已知点在原始点云中的idx，将对应的点填入out
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

void group_points_kernel_wrapper(int b, int c, int n, int npoints,
                                 int nsample,  // 1 3 20000 1024 64
                                 const float *points, const int *idx,
                                 float *out) {  // 1024*64
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // block 尺寸为1024，3 (num_seed, 3)
  group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
                                         int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}
