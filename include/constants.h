#ifndef CONSTANTS_H
#define CONSTANTS_H

inline constexpr int GROUP_SIZE = 4;
inline constexpr int SEQ_LEN = 10;
inline constexpr int HIDDEN_DIM = 16;
inline constexpr int BLOCK_M = 16;
inline constexpr int BLOCK_N = 16;
inline constexpr int BLOCK_K = 16;
inline constexpr int WAVE_SIZE = 64;

// Calculate padded dimensions once
constexpr int PADDED_GROUP_SIZE = ((GROUP_SIZE + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
constexpr int PADDED_HIDDEN_DIM = ((HIDDEN_DIM + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
constexpr int PADDED_SEQ_LEN = ((SEQ_LEN + BLOCK_N - 1) / BLOCK_N) * BLOCK_N;

#endif