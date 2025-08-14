#ifndef TYPES_H
#define TYPES_H
#include "constants.h"
#include <stdint.h>

template<typename T, uint32_t Rank>
using VecT = T __attribute__((ext_vector_type(Rank)));
// Types used in this exercise
using float16_t = _Float16;
using float32_t = float;

namespace Naive {

    using AFragT = VecT<float16_t, BLOCK_M * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using BFragT = VecT<float16_t, BLOCK_N * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using AccumFragT = VecT<float32_t, BLOCK_M * BLOCK_N * BLOCK_B / WAVE_SIZE>;
    using CFragT = AccumFragT;

}

namespace Mfma4x4 {
    using AFragT = VecT<float16_t, BLOCK_M * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using BFragT = VecT<float16_t, BLOCK_N * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using AccumFragT = VecT<float32_t, BLOCK_M * BLOCK_N * BLOCK_B / WAVE_SIZE>;
    using CFragT = AccumFragT;
}


namespace Mfma4x4PingPong {
    using AFragT = VecT<float16_t, BLOCK_M * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using BFragT = VecT<float16_t, BLOCK_N * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using AccumFragT = VecT<float32_t, BLOCK_M * BLOCK_N * BLOCK_B / WAVE_SIZE>;
    using CFragT = AccumFragT;
}


namespace Mfma16x16PingPong {
    using AFragT = VecT<float16_t, BLOCK_M * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using BFragT = VecT<float16_t, BLOCK_N * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using AccumFragT = VecT<float32_t, BLOCK_M * BLOCK_N * BLOCK_B / WAVE_SIZE>;
    using CFragT = AccumFragT;
}


namespace Mfma16x16 {
    using AFragT = VecT<float16_t, BLOCK_M * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using BFragT = VecT<float16_t, BLOCK_N * BLOCK_K * BLOCK_B / WAVE_SIZE>;
    using AccumFragT = VecT<float32_t, BLOCK_M * BLOCK_N * BLOCK_B / WAVE_SIZE>;
    using CFragT = AccumFragT;
}

#endif