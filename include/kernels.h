#ifndef KERNELS_H
#define KERNELS_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "types.h"

__global__ void gqa_packed(float16_t* queries, float16_t* key_mat, int group_size, int seq_len, int hidden_dim, int lda, int ldb);

__device__ AFragT load_queries_16x16_col_major(float16_t const* input, int ld);

__device__ BFragT load_keys_16x16_row_major(float16_t const* input, int ld);

#endif