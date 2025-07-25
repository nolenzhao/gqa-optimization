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


void fillMatrix(float* mat, size_t size){
    for(int i = 0; i < size; i++){
        mat[i] = getRandomFloat();
    }
}

int ceilDiv(int num, int denom){
    return (num + denom -1) / denom;
}