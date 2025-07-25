#include "../include/helpers.h"
#include "../include/kernels.h"
#include "../include/types.h"
#include <vector>
#include <random>



int main(){


    std::vector<float16_t> keys(SEQ_LEN * HIDDEN_DIM);
    std::vector<float16_t> queries(GROUP_SIZE * HIDDEN_DIM);
    std::vector<float32_t> attention_output(GROUP_SIZE * SEQ_LEN);


    fillMatrix(keys.data(), keys.size(), false, static_cast<float16_t>(1.0));
    fillMatrix(queries.data(), queries.size(), false, static_cast<float16_t>(2.0));

    float16_t* d_queries, *d_keys;
    float32_t* d_attention_output;

    hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
    hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
    hipMalloc(&d_attention_output, sizeof(float32_t) * GROUP_SIZE * SEQ_LEN);
    hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

    

    dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N));
    dim3 blockDim = dim3(64, 1);

    // Leading dimension passed into a load/store should be format going to/from HBM
    gqa_packed<<<gridDim, blockDim>>>(
        d_queries, 
        d_keys, 
        d_attention_output,
        GROUP_SIZE, 
        SEQ_LEN, 
        HIDDEN_DIM, 
        GROUP_SIZE,  // column since we load from col-major data
        SEQ_LEN,  // row since we load from row-major data
        GROUP_SIZE // col since we store as col-major (even though registers are in row-major)
    );

    hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);

    print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, "attention_output");

    hipFree(d_queries);
    hipFree(d_keys);

    return 0;
}