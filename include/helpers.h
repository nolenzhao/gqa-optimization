#ifndef HELPERS_H
#define HELPERS_H

#include "types.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>


std::mt19937& getGenerator();

float getRandomFloat();



// Helper for vec size
template<typename T, uint32_t Rank>
inline constexpr int32_t vectorSize(VecT<T, Rank>const& v)
{
    return Rank;
}


template <typename T>
inline void fillMatrix(T* mat, int valid_m, int valid_n, int padded_m, int padded_n, bool colMajor = true, bool rand = true, T val =0.0){
    if (rand) {
        // Random filling using the existing random generator
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // Adjust range as needed
        
        if (colMajor){
            for(int i = 0; i < valid_n; i++){
                for(int j = 0; j < valid_m; j++){
                    mat[i * padded_m + j] = static_cast<T>(dist(getGenerator()));
                }
            }
        }
        else {
            for(int i = 0; i < valid_m; i++){
                for(int j = 0; j < valid_n; j++){
                    mat[i * padded_n + j] = static_cast<T>(dist(getGenerator()));
                }
            }
        }
    } else {
        // Predetermined value filling
        if (colMajor){
            for(int i = 0; i < valid_n; i++){
                for(int j = 0; j < valid_m; j++){
                    mat[i * padded_m + j] = val;
                }
            }
        }
        else {
            for(int i = 0; i < valid_m; i++){
                for(int j = 0; j < valid_n; j++){
                    mat[i * padded_n + j] = val;
                }
            }
        }
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

constexpr int ceilDiv(int num, int denom) {
    return (num + denom - 1) / denom;
}

template <typename T>
__host__ inline void print_matrix(T* matrix, int valid_m, int valid_n, int padded_m, int padded_n, const std::string& name = "", bool col_major = true) {
    std::cout << "\n" << name << " (valid: " << valid_m << "x" << valid_n 
              << ", padded: " << padded_m << "x" << padded_n << "):\n";
    std::cout << std::fixed << std::setprecision(4);
    
    for(int i = 0; i < valid_m; i++) {
        std::cout << "[";
        for(int j = 0; j < valid_n; j++) {
            int idx;
            if(col_major) {
                idx = i + j * padded_m;  // Use padded_m for leading dimension
            } else {
                idx = i * padded_n + j;  // Use padded_n for leading dimension
            }
            
            std::cout << std::setw(8) << static_cast<float>(matrix[idx]);
            if(j < valid_n - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

template <typename T>
inline void pad_matrix(T* src, T* dst, int orig_M, int orig_K, int padded_M, int padded_K) {
    // Zero out the entire padded matrix first
    memset(dst, 0, padded_M * padded_K * sizeof(T));
    
    // Copy original data
    for (int i = 0; i < orig_M; i++) {
        memcpy(&dst[i * padded_K], &src[i * orig_K], orig_K * sizeof(T));
        // Padding columns are already zeroed from memset
    }
    // Padding rows are already zeroed from memset
}


void cpu_gemm(float16_t const* A, float16_t const* B, float32_t* C, int M, int N, int K);

void validate_valid_region(float32_t* C1, float32_t* C2, int valid_rows, int valid_cols, int padded_rows, int padded_cols);


template<typename KernelFunc>
inline float time_kernel(const std::string& name, KernelFunc&& kernel_call) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    kernel_call();
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << name << " execution time: " << milliseconds << " ms\n";
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return milliseconds;
}


                        
#endif