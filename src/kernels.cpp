#include "../include/kernels.h"


// let's assume that query in col format
__global__ void gqa_naive(float* query, float* key_max, int group_size, int seq_len, int hidden_dim){

    // again we assume query is in col-major with GROUP_SIZE number of vectors
    for(int i = 0; i < GROUP_SIZE; i++){

        int start_coord = query[i * hidden_dim];
    }



    
}

