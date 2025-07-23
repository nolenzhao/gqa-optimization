#ifndef KERNELS_H
#define KERNELS_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

__global__ void gqa_naive(float* queries, float* key_mat, int group_size, int seq_len, int hidden_dim);

#endif