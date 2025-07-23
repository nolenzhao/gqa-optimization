#ifndef HELPERS_H
#define HELPERS_H


#include <random>


std::mt19937& getGenerator();

float getRandomFloat();

void fillMatrix(float* mat, size_t size);

#endif