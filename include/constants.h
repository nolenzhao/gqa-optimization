#ifndef CONSTANTS_H
#define CONSTANTS_H

inline constexpr int GROUP_SIZE = 8;
inline constexpr int SEQ_LEN = 99;
inline constexpr int HIDDEN_DIM = 128;
inline constexpr int BLOCK_M = 16;
inline constexpr int BLOCK_N = 16;
inline constexpr int BLOCK_K = 16;
inline constexpr int WAVE_SIZE = 64;
inline constexpr int WAVES_PER_BLOCK = 4;
inline constexpr int T_BLOCK_X = WAVES_PER_BLOCK * WAVE_SIZE;
inline constexpr int T_BLOCK_Y = 1;

// Calculate padded dimensions 
inline constexpr int PADDED_GROUP_SIZE = ((GROUP_SIZE + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
inline constexpr int PADDED_HIDDEN_DIM = ((HIDDEN_DIM + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
inline constexpr int PADDED_SEQ_LEN = ((SEQ_LEN + (BLOCK_N * WAVES_PER_BLOCK) - 1) / (BLOCK_N * WAVES_PER_BLOCK)) * (BLOCK_N * WAVES_PER_BLOCK);

#endif