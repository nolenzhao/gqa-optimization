#include "../include/kernels.h"
#include "../include/helpers.h"
#include "../include/types.h"



//Remember: a single wave computes the ful accum for a blkM x blkN mat
// it itself steps through the entire k loop
__global__ void gqa_packed(
    float16_t const* query, 
    float16_t const* keys, 
    float32_t* attention_output,
    int group_size, 
    int seq_len, 
    int hidden_dim, 
    int lda, 
    int ldb, 
    int ldd) {

    // for a 1x4 output matrix, need to load one rows of A
    // we have 4 waves in this t_block and they should be growing in y direction
    
    //load One block of A every iteration
    __shared__ float16_t shared_a[BLOCK_M * BLOCK_K];
    // load 4 blocks of B every iteration to perform mfma with single A 
    // Even though these values (for group_size < 16 and 16x16 mfma) are never used again 
    // We should use LDS to reduce register pressure
    
    // Just init to 4 16x16 
    // It would be nice to dynamically allocate LDS depending on the seq_len size, but that would 
    // require variable LDS size for diff threadBlocks
    __shared__ float16_t shared_b[BLOCK_K * BLOCK_N * WAVES_PER_BLOCK]; 

    auto fragA = AFragT{};
    auto fragB = BFragT{};
    auto fragAcc = AccumFragT{};
    fill_frag(fragAcc, 0.0f);

    // find which wave we are in (there are 4 now)
    int wave_id = threadIdx.x / WAVE_SIZE;
    // partition into row/col; we wil alwyas have one row and (4?) cols
    int wave_row = wave_id / (WAVES_PER_BLOCK * WAVE_SIZE);
    int wave_col = wave_id % (WAVES_PER_BLOCK);

    // Find which global wave this is. For each block in the output matrix, there is a corresponding wave
    auto waveGridX = blockIdx.x + wave_row;
    auto waveGridY = (blockIdx.y * WAVES_PER_BLOCK) + wave_col;

    // This gets the absolute row/col of upperleft C block coord that this threadBlock computes
    auto cRow = waveGridX * BLOCK_M;
    auto cCol = waveGridY * BLOCK_N;


    if(cRow < group_size && cCol < seq_len){
        // step through the K loop
        for(int i = 0; i < hidden_dim; i+= BLOCK_K){

            // assuming 128-bit optimal loads per thread we only need one wave to load 
            // the 16x16 A matrix -> we shuould try and load double 
            // Do we really need to load B into LDS? only really need to reuse A right?
            // when we call load_queries we are already pointing at the correct upper left
            if (wave_id == 0){
                load_queries(shared_a, query + (i * lda + cRow), lda);
            }
            __syncthreads();
            // Just have each wave load it's own matrix -> each thread loads 8 bytes
            // when we call this we are pointing at the correct row/col
            load_keys_quad(shared_b + (wave_id * BLOCK_K * BLOCK_N), keys + (i * ldb + cCol), ldb);

            __syncthreads();
            // Need to point to starting corner of two rows we want to load

            // Have each thread load its fragment (4 fp16's in each frag) 
            // shared_a has been loaded as row-major so the load to registers is correctly col-major
            fragA = load_queries_16x16_col_major(shared_a , BLOCK_K, wave_id);
            // B is in row-major order
            // keys gives us the start of matrix, ccol indexes into the row 
            // i * ldb calculates (block_k (col dimension)* size of row)
            // i.e. do a num rows * sizeof(rows) offset  

            // <<!---You're trying to have every thread use the same pointer to shared_b---!>
            // Instead you should properly point to the correct segment and then use local_t_id to idx
            // so use the info you have in this func (wave_id, etc. to point to correct offset into s_mem)
            // the same cna be said for loading fragA, just point to the correct segment using parity offset
            fragB = load_keys_16x16_row_major(shared_b + (wave_id * BLOCK_K * BLOCK_N), BLOCK_K, wave_id);

            // Acumulate the ouput 16x16 blocks
            // fragAcc holds 4 f32_t (row major order)

            fragAcc = __builtin_amdgcn_mfma_f32_16x16x16f16(fragA, fragB, fragAcc, 0, 0, 0);


        }
        store_attention_pattern_16x16_col_major(attention_output + (cCol * ldd + cRow), fragAcc, ldd);
    }

}

// Assume data is in row-major; we need to load as col-major
__device__ AFragT load_queries_16x16_col_major(float16_t const* input, int ld, int wave_id){

    static constexpr uint32_t VW = vectorSize(AFragT{});
    static constexpr uint32_t Dim = BLOCK_M;

    // Every wave needs to load A from LDS
    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 4 elements at maximum.
    // We need to know where they start, and where the next elements are.
    int local_t_id = threadIdx.x - wave_id * WAVE_SIZE;

    // Since we're storing this column-neighbor from column-major data 
    // we can only fit 4 columns in a SIMD so we calculate the offset between 
    // columns as {0, 4, 8, 12}  
    // then we index into the row [0..15]
    // then use kOffset to offset to neighboring columns (column major)

    auto startCoord2D = std::make_pair(local_t_id % Dim,         // Row
                                       (local_t_id / Dim) * VW); // Col
    // This is (row, col) so a step is a step to a diff row
    auto stepCoord2D = std::make_pair(0u, 1u);

    // startOffset calculated as the startOffset of each 4 column blob
    // and then we idx into those columns to specify the row
    auto startOffset = row_major(startCoord2D, ld);
    // == 1 since to step a column over (k direction for A) we need to offset ld (1)
    auto kOffset = row_major(stepCoord2D, ld);

    // load with 4 non-contiguous offsets
    auto fragA = AFragT
    {
        input[startOffset], 
        input[startOffset + kOffset],
        input[startOffset + 2 * kOffset],
        input[startOffset + 3 * kOffset],
    };

    return fragA;
}


// Assume data is in col-major need to load registers in row-major
__device__ BFragT load_keys_16x16_row_major(float16_t const* input, int ld, int wave_id)
{
    static constexpr uint32_t VW = vectorSize(BFragT{});
    static constexpr uint32_t Dim = BLOCK_N;

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 4 elements.
    // We need to know where they start, and where the next elements are.
    int local_t_id = threadIdx.x % WAVE_SIZE;

    // 256 threads -> 0-63 t_id
    // What we're really doing here is making the startCoord corresponds to the 
    // "row" of SIMD within that row which row we have
    // we're calculating the start offset of which row, {0 | 4 | 8 | 12}
    // of the 16x16 matrix we start at 
    auto startCoord2D = std::make_pair((local_t_id / Dim) * VW, // Row
                                        (local_t_id % Dim));   // Col
    // row, column step here
    auto stepCoord2D = std::make_pair(1u, 0u);

    // Flatten to 1D row_major offsets.
    // If we have row-major data, to step down a row, need to multiply by ld, 
    // then access into that row with coord.second

    // This gets the start idx at each block of 4 rows (since our VW is 4)
    // we can only store 4 rows in each SIMD
    auto startOffset = col_major(startCoord2D, ld);
    // kOffset is 16 since to go down a row, need to go 16 offset in row-major
    auto kOffset = col_major(stepCoord2D, ld);

    // If you notice carefully, kOffset != 1.
    // This means the following is vector is loaded with 4 non-contiguous offsets,
    // which the compiler will separate into 4 different global_load_short instructions.
    auto fragB = BFragT
    {
        input[startOffset],               // v[0] = Reg 0 [0:15]
        input[startOffset + kOffset],     // v[1] = Reg 0 [16:31]
        input[startOffset + 2 * kOffset], // v[2] = Reg 1 [0:15]
        input[startOffset + 3 * kOffset], // v[3] = Reg 1 [16:31]
    };

    return fragB;
}

__device__ void store_attention_pattern_16x16_col_major(float32_t* output, AccumFragT accum, int ld){

    static constexpr uint32_t Dim = BLOCK_N;
    static constexpr uint32_t VW = vectorSize(AccumFragT{});

    int local_t_id = threadIdx.x % WAVE_SIZE;
    // these are stored in registers  in row major, so they need to be indexed as such
    auto startCoord2D = std::make_pair((local_t_id / Dim) * VW, local_t_id % Dim);
    auto stepCoord2D = std::make_pair(1u, 0u);
    // accum is composed of 4 registers
    // the matrix is stored in row-major order

    // Takes the row-major view from the register and almost transposes it to column major indexing
    // use threadIdx.x = 14 and see how it actually indexes properly from a column persepctive
    // coord.first goes down into the column (across rows), and coord.second accesses across columns
    // imagine the registers on top of the matrix, this is how mem is laid out, thread 0 holds 4 elems 
    // of the first column (since the SIMD's hold row-major form) 
    // similarly thread 1 holds 4 elems of the second column. thus we can store these contiugously
    // coord.first = {0, 4, 8, 12} thus we are going into each column, and extracting, going across 
    // columns using coord.second (i.e. thread1 -> return 0 + 1*16 = 16) -> this is correct in col major

    auto startOffset = col_major(startCoord2D, ld);
    // when transposes the col values are contiguous where the row values are not
    auto kOffset = col_major(stepCoord2D, ld);

    output[startOffset] = accum[0]; 
    output[startOffset + kOffset] = accum[1];
    output[startOffset + 2 * kOffset] = accum[2];
    output[startOffset + 3 * kOffset] = accum[3];
}


// Load from HBM to LDS
__device__ void load_data(float16_t* dst, float16_t const* src, int lda, int ldb, bool data_col_major){
    // If the input data is in col_major, we should store as row-major and vice-versa
    // try to have each thread load  128bits
    // we need to load 512 f16_t values (2 * 16 * 16)
    // thus need to load 1024bytes -> need 256 threads to load 4 bytes each
    // 256threads/64 (threads_per_wave) = 4 waves 
    // imagine wave_id is arranged row-major for 4 quadrants of 2 row input
    // have the first 64 threads 
    // ___________________
    // | wave_0 | wave_1 |
    // | wave_2 | wave_3 |
    // Alternatively use one wave to load all 1024 bytes (64 * 16bytes) = 1024 bytes
    // have each threadIdx load half a column 

    // Given that the group_size is usually about 8 we should really be storing A matrix in LDS as a single row with multiple columns loaded
    // This way the b matrix, can access the single a row in LDS?


    // If the HBM data is col major, we store as row-major
    if (data_col_major){
        static constexpr uint32_t Dim = BLOCK_M;

        // TODO: figure out the math for dynamic # waves per block
        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * 8, // Row
                                        threadIdx.x % Dim); // Column

        auto stepCoord2D = std::make_pair(1u, 0u);


        // The src pointer points to the entire A or B matrix
        auto startOffsetSrc = col_major(startCoord2D, lda);
        // the dst pointer points to a small chunk, thus we use hte leading 
        // dimension of the chunk (in row-major leading dim is block_k)
        auto startOffsetDst = row_major(startCoord2D, BLOCK_K);
        auto kOffset = col_major(stepCoord2D, lda);

        // We want dst to be filled row-major but src is column-major
        dst[startOffsetDst] = src[startOffsetSrc];
        dst[startOffsetDst + BLOCK_K] = src[startOffsetSrc + kOffset];
        dst[startOffsetDst + 2 * BLOCK_K] = src[startOffsetSrc + 2 * kOffset];
        dst[startOffsetDst + 3 * BLOCK_K] = src[startOffsetSrc + 3 * kOffset];
        dst[startOffsetDst + 4 * BLOCK_K] = src[startOffsetSrc + 4 * kOffset];
        dst[startOffsetDst + 5 * BLOCK_K] = src[startOffsetSrc + 5 * kOffset];
        dst[startOffsetDst + 6 * BLOCK_K] = src[startOffsetSrc + 6 * kOffset];
        dst[startOffsetDst + 7 * BLOCK_K] = src[startOffsetSrc + 7 * kOffset];
    }
}


__device__ void load_queries(float16_t* dst, float16_t const* src, int ld){

    // We only need to load 1 16x16 matrices here
    static constexpr uint32_t Dim = BLOCK_K;

    // Every 32 elems, we move down to the next startOffset
    // Assume we are stored in col-major, each thread loads 4 f16_t -> 16 bytes 
    int local_t_id = threadIdx.x % WAVE_SIZE;

    auto startCoord2D = std::make_pair((local_t_id / Dim) * 4,  // row
                                        local_t_id % Dim); // col
    // We want to step down since we're loading a col per thread
    auto stepCoord2D = std::make_pair(1u, 0u);


    auto startOffsetSrc = col_major(startCoord2D, ld);
    auto startOffsetDst = row_major(startCoord2D, Dim);
    // kOffset == 1
    auto kOffset = col_major(stepCoord2D, ld);


    // Verbose
    // Want dst (LDS) to store this information row-major but src is col-major
    dst[startOffsetDst] = src[startOffsetSrc];
    dst[startOffsetDst + Dim] = src[startOffsetSrc + kOffset];
    dst[startOffsetDst + 2 * Dim] = src[startOffsetSrc + 2 * kOffset];
    dst[startOffsetDst + 3 * Dim] = src[startOffsetSrc + 3 * kOffset];
}

// Expect keys to be row-major in HBM -> load as col-major in LDS 
// So that we can do a coallesced load from LDS to VGPR's avoiding bank conflict
__device__ void load_keys_quad(float16_t* dst, float16_t const* src, int ld){

    static constexpr uint32_t Dim = BLOCK_K;

    int local_t_id = threadIdx.x % WAVE_SIZE;
   
    // need to remember: the pointer to src is arleady at the correct upper left
    auto startCoord2D = std::make_pair((local_t_id %  Dim), // row
                                        (local_t_id / Dim) * 4); // col
    auto stepCoord2D = std::make_pair(0u, 1u);

    auto startOffsetSrc = row_major(startCoord2D, ld);
    auto startOffsetDst = col_major(startCoord2D, Dim);
    // kOffset == 1
    auto kOffset = row_major(stepCoord2D, ld);

    dst[startOffsetDst] = src[startOffsetSrc];
    dst[startOffsetDst + Dim] = src[startOffsetSrc + kOffset];
    dst[startOffsetDst + 2 * Dim] = src[startOffsetSrc + 2 * kOffset];
    dst[startOffsetDst + 3 * Dim] = src[startOffsetSrc + 3 * kOffset];
}

__device__ int col_major(const std::pair<int, int>& coord, int ld){
    return coord.first + coord.second * ld;
}

__device__ int row_major(const std::pair<int, int>& coord, int ld){
    return coord.first * ld + coord.second;
}