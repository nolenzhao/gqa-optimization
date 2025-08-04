#include "../include/helpers.h"
#include "../include/kernels.h"
#include "../include/types.h"
#include <vector>
#include <random>



int main(){


    // For simplicity pretend its transposed already

    // This is to be interpreted row-major
    std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
    // This is to be interpreted col-major
    std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
    std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

    fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, false, static_cast<float16_t>(2.0), true);
    fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false, static_cast<float16_t>(1.0), false);

    float16_t* d_queries, *d_keys;
    float32_t* d_attention_output;

    hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
    hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
    hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
    hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

    // float16_t* d_queries_naive;
    // hipMalloc(&d_queries_naive, sizeof(float16_t) * GROUP_SIZE * BLOCK_M * PADDED_HIDDEN_DIM);


    dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);

    // group_sizes are rarely greater than 8 -> thus we should try to group waves into workgroups 
    // in the y dimension since output matrix likely does not need to grow in x direction for a 16x16 mfma
    // for a 4x4mfma, assume we are stretching blocks across seq_len dimension for B matrix 

    // TODO: Optimize A later to load more than one block at a time (in the k direction)
    dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

    // When this is called we assume d_queries is in col-major and d_keys is in row-major
    gqa_packed<<<gridDim, blockDim>>>(
        d_queries, 
        d_keys, 
        d_attention_output,
        GROUP_SIZE, 
        SEQ_LEN, 
        HIDDEN_DIM, 
        PADDED_GROUP_SIZE,  // lda -> # rows since we load from col-major data
        PADDED_SEQ_LEN,  // ldb -> #cols since we load from row-major data
        PADDED_GROUP_SIZE // ldd -> #rows since we store as col-major (even though registers are in row-major)
    );


    hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);

    print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "attention_output", true);

    hipFree(d_queries);
    hipFree(d_keys);
    hipFree(d_attention_output);

    return 0;
}