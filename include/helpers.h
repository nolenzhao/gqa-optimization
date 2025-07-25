#ifndef HELPERS_H
#define HELPERS_H

#include "types.h"
#include <random>


std::mt19937& getGenerator();

float getRandomFloat();

void fillMatrix(float* mat, size_t size);


// Helper for vec size
template<typename T, uint32_t Rank>
inline constexpr int32_t vectorSize(VecT<T, Rank>const& v)
{
    return Rank;
}

// Vector fill
// Assign a value to each vector register.
template<typename T, uint32_t Rank>
__device__ inline void fill_frag(VecT<T, Rank>& frag, T value)
{
    for(int i = 0; i < Rank; i++)
    {
        frag[i] = value;
    }
}

int ceilDiv(int num, int denom);

#endif