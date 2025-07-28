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

    // TODO: figure out logic to remove hardcoded 2
    // Since each threadBlock now computes a 2x2 output, we only need half as many in both directions
    dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M * 2), ceilDiv(SEQ_LEN, BLOCK_N * 2));

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

    // hipStream_t streams[GROUP_SIZE];

    // for(int i = 0; i < GROUP_SIZE; i++)
    //     hipStreamCreate(&streams[i]);


    // for(int i = 0; i < GROUP_SIZE; i++){
    //     gqa_packed<<<64, 1, 0, streams[i]>>>(
    //         d_queries_naive + (i * sizeof(float16_t) * PADDED_HIDDEN_DIM * BLOCK_M),
    //         d_keys, 
    //         d_attention_output,
    //         1, //group_size == 1 since we are dealing with a single query vector
    //         SEQ_LEN, 
    //         HIDDEN_DIM,
    //         16, 
    //         PADDED_SEQ_LEN, 
    //         16
    //     );
    // }

    hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);

    print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "attention_output", true);

    hipFree(d_queries);
    hipFree(d_keys);
    hipFree(d_attention_output);

    return 0;
}