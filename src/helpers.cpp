#include "../include/helpers.h"


// Global random engine and distribution
std::mt19937& getGenerator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

float getRandomFloat() {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(getGenerator());
}

// corresponds to m, n , k
// A is in col-major, B is in row-major, C is in col-major (matching GPU kernel expectations)
void cpu_gemm(float16_t const* A, float16_t const* B, float32_t* C, int M, int N, int K){
    // Zero initialize C
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    // Perform the matrix multiplication: C = A * B
    // A is column-major: A[m + k * M]
    // B is row-major: B[k * N + n]  
    // C is column-major: C[m + n * M]
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                C[m + n * M] += A[m + k * M] * B[k * N + n];
            }
        }
    }
}

// Validate only the valid (non-padded) region
// This should be the primary validation function to use
void validate_valid_region(float32_t* C1, float32_t* C2, int valid_rows, int valid_cols, 
                          int padded_rows, int padded_cols) {
    const float TOLERANCE = 1e-2f;
    bool mismatch_found = false;
    int mismatch_count = 0;
    const int MAX_MISMATCHES_TO_SHOW = 10;
    
    std::cout << "Comparing valid region: " << valid_rows << "x" << valid_cols 
              << " within padded " << padded_rows << "x" << padded_cols << " matrices\n";
    
    for(int col = 0; col < valid_cols; col++) {
        for(int row = 0; row < valid_rows; row++) {
            int idx = row + col * padded_rows;  // Column-major indexing with padded leading dimension
            float diff = fabs(C1[idx] - C2[idx]);
            if (diff > TOLERANCE) {
                if (!mismatch_found) {
                    std::cout << "Mismatch found in valid region:\n";
                    mismatch_found = true;
                }
                if (mismatch_count < MAX_MISMATCHES_TO_SHOW) {
                    std::cout << "  At (" << row << "," << col << ") [idx=" << idx 
                              << "]: CPU=" << C1[idx] << ", GPU=" << C2[idx] 
                              << ", diff=" << diff << "\n";
                }
                mismatch_count++;
            }
        }
    }
    
    if (mismatch_found) {
        std::cout << "Total mismatches in valid region: " << mismatch_count 
                  << " out of " << (valid_rows * valid_cols) << " elements\n";
    } else {
        std::cout << "Valid regions match within tolerance (" << TOLERANCE << ")\n";
    }
}
