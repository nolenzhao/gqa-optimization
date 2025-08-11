#ifndef CONSTANTS_H
#define CONSTANTS_H


inline constexpr int GROUP_SIZE = 18;
inline constexpr int SEQ_LEN = 34;
inline constexpr int HIDDEN_DIM = 10;
inline constexpr int WAVE_SIZE = 64;

// waves_per_block should be specified as 4 in Y direction and 1 in x direction

namespace Naive {

    inline constexpr int WAVES_PER_BLOCK = 1;
    inline constexpr int THREADS_PER_BLOCK = 64;
    inline constexpr int T_BLOCK_X = WAVES_PER_BLOCK * WAVE_SIZE;
    inline constexpr int T_BLOCK_Y = 1;
    inline constexpr int BLOCK_M = 16;
    inline constexpr int BLOCK_N = 16;
    inline constexpr int BLOCK_K = 16;
    // block_b should be specifies as 16 in y direction and 1 in x direction
    inline constexpr int BLOCK_B = 1;
    // Represents the number of Block B's per threadblock
    inline constexpr int BLOCK_B_PER_BLOCK = BLOCK_B * WAVES_PER_BLOCK;
    // Calculate padded dimensions 
    inline constexpr int PADDED_GROUP_SIZE = ((GROUP_SIZE + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    inline constexpr int PADDED_HIDDEN_DIM = ((HIDDEN_DIM + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
    inline constexpr int PADDED_SEQ_LEN = ((SEQ_LEN + (BLOCK_N * BLOCK_B_PER_BLOCK) - 1) / (BLOCK_N * BLOCK_B_PER_BLOCK)) * (BLOCK_N * BLOCK_B_PER_BLOCK);

}

namespace Mfma4x4{
    // 4x4x4f16_16B mfma
    inline constexpr int WAVES_PER_BLOCK = 4;
    inline constexpr int THREADS_PER_BLOCK = 4;
    inline constexpr int T_BLOCK_X = WAVES_PER_BLOCK * WAVE_SIZE;
    inline constexpr int T_BLOCK_Y = 1;
    inline constexpr int BLOCK_M = 4;
    inline constexpr int BLOCK_N = 4;
    inline constexpr int BLOCK_K = 4;
    // block_b should be specifies as 16 in y direction and 1 in x direction
    inline constexpr int BLOCK_B = 16;
    inline constexpr int BLOCK_B_PER_BLOCK = BLOCK_B * WAVES_PER_BLOCK;
    // Calculate padded dimensions 
    inline constexpr int PADDED_GROUP_SIZE = ((GROUP_SIZE + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    inline constexpr int PADDED_HIDDEN_DIM = ((HIDDEN_DIM + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
    inline constexpr int PADDED_SEQ_LEN = ((SEQ_LEN + (BLOCK_N * BLOCK_B_PER_BLOCK) - 1) / (BLOCK_N * BLOCK_B_PER_BLOCK)) * (BLOCK_N * BLOCK_B_PER_BLOCK);
}

namespace Mfma4x4PingPong{
    // 4x4x4f16_16B mfma
    inline constexpr int WAVES_PER_BLOCK = 4;
    inline constexpr int THREADS_PER_BLOCK = 4;
    inline constexpr int T_BLOCK_X = WAVES_PER_BLOCK * WAVE_SIZE;
    inline constexpr int T_BLOCK_Y = 1;
    inline constexpr int BLOCK_M = 4;
    inline constexpr int BLOCK_N = 4;
    inline constexpr int BLOCK_K = 4;
    // block_b should be specifies as 16 in y direction and 1 in x direction
    inline constexpr int BLOCK_B = 16;
    inline constexpr int BLOCK_B_PER_BLOCK = BLOCK_B * WAVES_PER_BLOCK;
    // Calculate padded dimensions 
    inline constexpr int PADDED_GROUP_SIZE = ((GROUP_SIZE + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    inline constexpr int PADDED_HIDDEN_DIM = ((HIDDEN_DIM + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
    inline constexpr int PADDED_SEQ_LEN = ((SEQ_LEN + (BLOCK_N * BLOCK_B_PER_BLOCK) - 1) / (BLOCK_N * BLOCK_B_PER_BLOCK)) * (BLOCK_N * BLOCK_B_PER_BLOCK);
}

namespace Mfma16x16{

    inline constexpr int WAVES_PER_BLOCK = 4;
    inline constexpr int THREADS_PER_BLOCK = 4;
    inline constexpr int T_BLOCK_X = WAVES_PER_BLOCK * WAVE_SIZE;
    inline constexpr int T_BLOCK_Y = 1;
    inline constexpr int BLOCK_M = 16;
    inline constexpr int BLOCK_N = 16;
    inline constexpr int BLOCK_K = 16;
    // block_b should be specifies as 16 in y direction and 1 in x direction
    inline constexpr int BLOCK_B = 1;
    // Represents the number of Block B's per threadblock
    inline constexpr int BLOCK_B_PER_BLOCK = BLOCK_B * WAVES_PER_BLOCK;
    // Calculate padded dimensions 
    inline constexpr int PADDED_GROUP_SIZE = ((GROUP_SIZE + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    inline constexpr int PADDED_HIDDEN_DIM = ((HIDDEN_DIM + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
    inline constexpr int PADDED_SEQ_LEN = ((SEQ_LEN + (BLOCK_N * BLOCK_B_PER_BLOCK) - 1) / (BLOCK_N * BLOCK_B_PER_BLOCK)) * (BLOCK_N * BLOCK_B_PER_BLOCK);

}


#endif