#include "../include/helpers.h"
#include "../include/kernels.h"
#include <vector>
#include <random>


constexpr int GROUP_SIZE = 8;
constexpr int SEQ_LEN = 10;
constexpr int HIDDEN_DIM = 128;

int main(){

    size_t q_bytes = sizeof(float) * GROUP_SIZE * HIDDEN_DIM;
    size_t k_bytes = sizeof(float) * SEQ_LEN * HIDDEN_DIM;
    float* queries = (float*)malloc(q_bytes);
    // Let's assume it's been stored pre-transposed :)
    float* key_mat = (float*)malloc(k_bytes);

    fillMatrix(queries, q_bytes / sizeof(float));
    fillMatrix(key_mat, k_bytes / sizeof(float));

    float* d_queries, *d_key_mat;
    hipMalloc(&d_queries, q_bytes);
    hipMalloc(&d_key_mat, k_bytes);
    hipMemcpy(d_queries, queries, q_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_key_mat, key_mat, k_bytes, hipMemcpyHostToDevice);

    // at launch lets assume d_queries is stored in column-major
    // assume d_key_mat is stored in row-major (but its transposed)
    gqa_naive<<<1, 1024>>>(d_queries, d_key_mat, GROUP_SIZE, SEQ_LEN, HIDDEN_DIM);
    return 0;
}