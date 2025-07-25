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
    int ldd){

    auto fragA = AFragT{};
    auto fragB = BFragT{};
    auto fragAcc = AccumFragT{};

    fill_frag(fragAcc, 0.0f);

    // Find which wave this is. For each block in the output matrix, there is a corresponding wave
    auto waveGridX = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    auto waveGridY = (blockIdx.y * blockDim.y + threadIdx.y);

// This gets the absolute row/col of upperleft C block coord that this threadBlock computes
    auto cRow = waveGridX * BLOCK_M;
    auto cCol = waveGridY * BLOCK_N;

    if(cRow < group_size && cCol < seq_len){
        // step through the K loop
        for(int i = 0; i < hidden_dim; i+= BLOCK_K){
            // Have each thread load its fragment (4 fp16's in each frag) 
            // to load A (assuming its col major) we need to give start + row offset + columns * lda
            // the row offset controls how deep in a column we are
            // i is a multiple of block_k (gives column number) and lda gives column length (number of rows)
            fragA = load_queries_16x16_col_major(query + (cRow + i * lda), lda);

            // B is in row-major order
            // keys gives us the start of matrix, ccol indexes into the row 
            // i * ldb calculates (block_k (col dimension)* size of row)
            // i.e. do a num rows * sizeof(rows) offset  
            fragB = load_keys_16x16_row_major(keys + (i * ldb * cCol), ldb);

            // Acumulate the ouput 16x16 blocks
            // fragAcc holds 4 f32_t (row major order)
            fragAcc = __builtin_amdgcn_mfma_f32_16x16x16f16(fragA, fragB, fragAcc, 0, 0, 0);
        }
        store_attention_pattern_16x16_col_major(attention_output, fragAcc, ldd);
    }

}

// Assuem data is in col-major; we need to load as col-major as well
__device__ AFragT load_queries_16x16_col_major(float16_t const* input, int ld){

    static constexpr uint32_t VW = vectorSize(AFragT{});
    static constexpr uint32_t Dim = BLOCK_M;


    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 4 elements at maximum.
    // We need to know where they start, and where the next elements are.

    // Since we're storing this column-neighbor from column-major data 
    // we can only fit 4 columns in a SIMD so we calculate the offset between 
    // columns as {0, 4, 8, 12}  
    // then we index into the row [0..15]
    // then use kOffset to offset to neighboring columns (column major)
    auto startCoord2D = std::make_pair(threadIdx.x % Dim,         // Row
                                       (threadIdx.x / Dim) * VW); // Col
    // This is (row, col) so a step is a step to a diff column
    auto stepCoord2D = std::make_pair(0u, 1u);

    
    // col_major is calcualted as idx into the row then use the col * leading dim to step to other columns
    auto col_major = [](auto const& coord, auto ld) {return coord.first + coord.second * ld; };

    // startOffset calculated as the startOffset of each 4 column blob
    // and then we idx into those columns to specify the row
    auto startOffset = col_major(startCoord2D, ld);
    // == 16 since to step a column over (k direction for A) we need to offset ld (16)
    auto kOffset = col_major(stepCoord2D, ld);

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


// Assume data is in row-major need to load data in row-major too
__device__ BFragT load_keys_16x16_row_major(float16_t const* input, int ld)
{

    static constexpr uint32_t VW = vectorSize(BFragT{});
    static constexpr uint32_t Dim = BLOCK_N;

    // To start the loading process, let's visualize in 2D coords.
    // Each thread will load 4 elements.
    // We need to know where they start, and where the next elements are.

    // What we're really doing here is making the startCoord corresponds to the 
    // "row" of SIMD within that row which row we have
    // we're calculating the start offset of which row, {0 | 4 | 8 | 12}
    // of the 16x16 matrix we start at 
    // {and col offset [0..15] to specify within that row}
    // Then use kOffset to specify offset to rows {1, 2, 3, 5, etc.}


    // <!----SERIOUS out of bounds idxing error-----------!>
    /*
    If the mfma does not match perfectly to the input matrix, then we will be accessing 
    memory out of bounds or into another thread's fragment. 
    imagine threadIdx.x = 15 for a B matrix of (rowxcol) (16x10)
    The row will be calcualted correctly as 0
    The column will be calcualted as 15 -> this is obviously not a valid column
    when startOffset is calculated it will calc 0 * 16 + 15 = 15
    this means that we are going to access into t_id 4's high bits of VGPR[0] input access

    N.B. use `lane <lane_idx>` to switch lanes; `info lanes` to view lanes for the curr thread
    
    */
    auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, // Row
                                        threadIdx.x % Dim);      // Col
    // row, column step here
    auto stepCoord2D = std::make_pair(1u, 0u);

    // Flatten to 1D row_major offsets.
    // If we have row-major data, to step down a row, need to multiply by ld, 
    // then access into that row with coord.second
    auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

    // This gets the start idx at each block of 4 rows (since our VW is 4)
    // we can only store 4 rows in each SIMD
    auto startOffset = row_major(startCoord2D, ld);
    // kOffset is 16 since to go down a row, need to go 16 offset in row-major
    auto kOffset = row_major(stepCoord2D, ld);

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

    // these are stored in registers  in row major, so they need to be indexed as such
    auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, threadIdx.x % Dim);
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
    auto col_major = [](auto const& coord, auto ld){return coord.first + coord.second * ld;};

    auto startOffset = col_major(startCoord2D, ld);
    // when transposes the col values are contiguous where the row values are not
    auto kOffset = col_major(stepCoord2D, ld);

    output[startOffset] = accum[0]; 
    output[startOffset + kOffset] = accum[1];
    output[startOffset + 2 * kOffset] = accum[2];
    output[startOffset + 3 * kOffset] = accum[3];
}