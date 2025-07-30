#ifndef CONSTANTS_H
#define CONSTANTS_H

inline constexpr int GROUP_SIZE = 4;
inline constexpr int SEQ_LEN = 4;
inline constexpr int HIDDEN_DIM = 6;
constexpr int MAX_DIM = GROUP_SIZE > SEQ_LEN ? GROUP_SIZE : SEQ_LEN;
inline constexpr int BLOCK_M = MAX_DIM >= 5 ? 16 : 4;
inline constexpr int BLOCK_N = MAX_DIM >= 5 ? 16 : 4;
inline constexpr int BLOCK_K = MAX_DIM >= 5 ? 16 : 4;
inline constexpr int WAVE_SIZE = 64;

// Calculate padded dimensions once
inline constexpr int PADDED_GROUP_SIZE = ((GROUP_SIZE + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
inline constexpr int PADDED_HIDDEN_DIM = ((HIDDEN_DIM + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
inline constexpr int PADDED_SEQ_LEN = ((SEQ_LEN + BLOCK_N - 1) / BLOCK_N) * BLOCK_N;

#endif