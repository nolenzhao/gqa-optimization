#include "../include/helpers.h"
#include "../include/kernels.h"
#include "../include/types.h"
#include <vector>
#include <random>



int main(){


    std::vector<float16_t> keys(SEQ_LEN * HIDDEN_DIM);
    std::vector<float16_t> queries(GROUP_SIZE * HIDDEN_DIM);

    float16_t* d_queries, *d_keys;

    hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
    hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
    hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);


    dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N));
    dim3 blockDim = dim3(64, 1);

    gqa_packed<<<gridDim, blockDim>>>(
        d_queries, 
        d_keys, 
        GROUP_SIZE, 
        SEQ_LEN, 
        HIDDEN_DIM, 
        GROUP_SIZE, 
        SEQ_LEN);

    
    hipFree(d_queries);
    hipFree(d_keys);
    return 0;
}