#include "../include/helpers.h"
#include "../include/kernels.h"
#include "../include/types.h"
#include <vector>
#include <random>
#include <roctracer/roctx.h>


int main(int argc, char* argv[]) {

    // Test 1: Naive implementation
    if (std::stoi(argv[1]))
    {
        using namespace Naive;
        std::cout << "\n=== Unoptimized ===\n";
        
        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("Naive", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);

        // print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "4x4lds", true);

        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }

    // Test 2: 16x16 MFMA with LDS
    if (std::stoi(argv[2]))
    {
        using namespace Mfma16x16;
        std::cout << "\n=== 16x16 LDS ===\n";
        
        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("16x16-LDS", 
        [&](){
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
        });


        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);

        // print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "4x4lds", true);

        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }

    //Mfma 16x16-Half LDS
    if (std::stoi(argv[3]))
    {
        using namespace Mfma16x16HalfLDS;
        std::cout << "\n=== 16x16 Half-LDS ===\n";
        
        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("16x16Half-LDS", 
        [&](){
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
        });


        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);

        // print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "4x4lds", true);

        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }


    if (std::stoi(argv[4]))
    {
        using namespace Mfma4x4;
        std::cout << "\n=== 4x4 LDS ===\n";

        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("4x4-LDS", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);

        // print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "4x4lds", true);

        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }
    
    // Test 3: 4x4 MFMA with Half-LDS
    if (std::stoi(argv[5]))
    {
        using namespace Mfma4x4HalfLDS;
        std::cout << "\n=== 4x4 Half-LDS ===\n";

        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("4x4Half-LDS", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);

        // print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "4x4lds", true);

        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }

    if (std::stoi(argv[6]))
    {
        using namespace Mfma4x4Occup;
        std::cout << "\n=== 4x4 LDS Occup ===\n";

        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M * BLOCK_B_X * WAVES_PER_BLOCK_X), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B_Y * WAVES_PER_BLOCK_Y));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("4x4-LDS Occup", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);


        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }

    // Test 8: 4x4 MFMA with LDS and Ping-Ponging
    if (std::stoi(argv[7]))
    {
        using namespace Mfma4x4PingPong;
        std::cout << "\n=== 4x4 LDS (with ping-ponging) ===\n";

        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false, true);

        // fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, false, );
        // fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, true);
        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("4x4-LDS pingpong", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);


        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }
    
    if (std::stoi(argv[8]))
    {
        using namespace Mfma4x4HalfLDSOccup;
        std::cout << "\n=== 4x4 Half-LDS Occup ===\n";

        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
        dim3 gridDim = dim3(ceilDiv(GROUP_SIZE, BLOCK_M * BLOCK_B_X * WAVES_PER_BLOCK_X), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B_Y * WAVES_PER_BLOCK_Y));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("4x4-Half LDS Occup", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);


        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }

    // NOTE: THE FOLLOWING ASSUMES HIDDEN_DIM >= 16 (for KSPLIT = 4)
    // due to loop split into 4 iterations, with BLOCK_K = 4
    // therefore K_CHUNK_SIZE == HIDDEN_DIM / KSPLIT = 4 -> cant have chunk_size < BLOCK_K
    // Thus take caution when changing hidden_dim to make sure its a multiple of 16
    // Since we want to be able to split the hidden_dim into (KSPLIT chunks) 
    // (each chunk has HIDDEN_DIM / KSPLIT) elems 
    // These are then divided in the BLOCK_K dimension (4). Thus we need HIDDEN_DIM % 16 == 0
    if(std::stoi(argv[9]))
    {
        using namespace Mfma4x4HalfLDSSplitK;
        std::cout << "\n=== 4x4 Half-LDS SplitK ===\n";

        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        // Important to memeset since we use atomicAdd in store
        hipMemset(d_attention_output, 0, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);

        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);

        // We assign some number of TB's per output chunk; 
        // The blockIdx.X no longer corresponds to its output. we need to do blockIdx.X / 4 to get correct output chunk now 
        // Within that output chunk, we have 4 TB's to split the K loop amongst
        dim3 gridDim = dim3(KSPLIT * ceilDiv(GROUP_SIZE, BLOCK_M), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B * WAVES_PER_BLOCK));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("4x4Half-LDSSplitK", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);

        // print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "4x4lds", true);

        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }
    //mfma4x4 half lds Occup splitk
    if(std::stoi(argv[10]))
    {
        using namespace Mfma4x4HalfLDSOccupSplitK;
        std::cout << "\n=== 4x4 Half-LDS Occup SplitK ===\n";

        // This is to be interpreted row-major
        std::vector<float16_t> keys(PADDED_HIDDEN_DIM * PADDED_SEQ_LEN);
        // This is to be interpreted col-major
        std::vector<float16_t> queries(PADDED_GROUP_SIZE * PADDED_HIDDEN_DIM);
        std::vector<float32_t> attention_output(PADDED_GROUP_SIZE * PADDED_SEQ_LEN, std::numeric_limits<float32_t>::signaling_NaN());

        fillMatrix(queries.data(), GROUP_SIZE, HIDDEN_DIM, PADDED_GROUP_SIZE, PADDED_HIDDEN_DIM, true);
        fillMatrix(keys.data(), HIDDEN_DIM, SEQ_LEN, PADDED_HIDDEN_DIM, PADDED_SEQ_LEN, false);

        float16_t* d_queries, *d_keys;
        float32_t* d_attention_output;

        hipMalloc(&d_queries, sizeof(float16_t) * queries.size());
        hipMalloc(&d_keys, sizeof(float16_t) * keys.size());
        hipMalloc(&d_attention_output, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        // Important to memeset since we use atomicAdd in store
        hipMemset(d_attention_output, 0, sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);

        hipMemcpy(d_queries, queries.data(), queries.size() * sizeof(float16_t), hipMemcpyHostToDevice);
        hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(float16_t), hipMemcpyHostToDevice);

        dim3 blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);

        // We assign some number of TB's per output chunk; 
        // The blockIdx.X no longer corresponds to its output. we need to do blockIdx.X / 4 to get correct output chunk now 
        // Within that output chunk, we have 4 TB's to split the K loop amongst
        dim3 gridDim = dim3(KSPLIT * ceilDiv(GROUP_SIZE, BLOCK_M * BLOCK_B_X * WAVES_PER_BLOCK_X), ceilDiv(SEQ_LEN, BLOCK_N * BLOCK_B_Y * WAVES_PER_BLOCK_Y));

        float32_t* cpu_output = (float32_t*)malloc(sizeof(float32_t) * PADDED_GROUP_SIZE * PADDED_SEQ_LEN);
        cpu_gemm(queries.data(), keys.data(), cpu_output, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, PADDED_HIDDEN_DIM);

        time_kernel("4x4Half-LDSSplitK", 
        [&](){
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
        });

        hipMemcpy(attention_output.data(), d_attention_output, sizeof(float32_t) * attention_output.size(), hipMemcpyDeviceToHost);
        validate_valid_region(cpu_output, attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN);

        // print_matrix(attention_output.data(), GROUP_SIZE, SEQ_LEN, PADDED_GROUP_SIZE, PADDED_SEQ_LEN, "4x4lds", true);

        free(cpu_output);
        hipFree(d_queries);
        hipFree(d_keys);
        hipFree(d_attention_output);
    }
    return 0;
}
