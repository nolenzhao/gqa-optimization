#ifndef KERNELS_H
#define KERNELS_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "types.h"

__global__ void gqa_packed(float16_t const* queries, float16_t const* key_mat, float32_t* attention_output, int group_size, int seq_len, int hidden_dim, int lda, int ldb, int ldd);

__device__ AFragT load_queries_4x4_col_major(float16_t const* input, int ld, int wave_id);

__device__ BFragT load_keys_4x4_row_major(float16_t const* input, int ld, int wave_id);

__device__ void load_data(float16_t* dst, float16_t const* src, int lda, int ldb, bool data_col_major);

__device__ void store_attention_pattern_4x4_col_major(float32_t* output, AccumFragT accum, int ld);

__device__ void load_queries(float16_t* dst, float16_t const* src, int ld);

__device__ void load_keys_quad(float16_t* dst, float16_t const* src, int ld);

__device__ int col_major(const std::pair<int, int>& coord, int ld);
__device__ int row_major(const std::pair<int, int>& coord, int ld);
#endif