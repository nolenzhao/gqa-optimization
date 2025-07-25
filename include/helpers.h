#ifndef HELPERS_H
#define HELPERS_H

#include "types.h"
#include <iostream>
#include <iomanip>
#include <random>


std::mt19937& getGenerator();

float getRandomFloat();



// Helper for vec size
template<typename T, uint32_t Rank>
inline constexpr int32_t vectorSize(VecT<T, Rank>const& v)
{
    return Rank;
}


template <typename T>
inline void fillMatrix(T* mat, size_t size, bool rand = true, T val =0.0){
    for(int i = 0; i < size; i++){
        if (val == 0)
            mat[i] = getRandomFloat();
        else
            mat[i] = val;
    }
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

template <typename T>
__host__ inline void print_matrix(T* matrix, int rows, int cols, const std::string& name = "",bool col_major=true){
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    std::cout << std::fixed << std::setprecision(4);
    
    for(int i = 0; i < rows; i++) {
        std::cout << "[";
        for(int j = 0; j < cols; j++) {
            int idx;
            if(col_major) {
                idx = i + j * rows;  // column-major indexing
            } else {
                idx = i * cols + j;  // row-major indexing
            }
            
            std::cout << std::setw(8) << static_cast<float>(matrix[idx]);
            if(j < cols - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}


#endif