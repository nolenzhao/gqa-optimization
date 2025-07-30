#ifndef TYPES_H
#define TYPES_H
#include "constants.h"
#include <stdint.h>

template<typename T, uint32_t Rank>
using VecT = T __attribute__((ext_vector_type(Rank)));
// Types used in this exercise
using float16_t = _Float16;
using float32_t = float;
using AFragT = VecT<float16_t, std::max(BLOCK_M * BLOCK_K / WAVE_SIZE, 4)>;
using BFragT = VecT<float16_t, std::max(BLOCK_M * BLOCK_K / WAVE_SIZE, 4)>;
using AccumFragT = VecT<float32_t, std::max(BLOCK_M * BLOCK_K / WAVE_SIZE, 4)>;
using CFragT = AccumFragT;



#endif