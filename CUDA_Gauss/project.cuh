#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define COLUMN_LENGTH 8 // y
#define ROW_LENGTH 8 // x

#define NR_OF_BLOCKS 1
#define THREADS_PER_BLOCK 4

#define ELEMENTS_PER_THREAD (int)(COLUMN_LENGTH/THREADS_PER_BLOCK)

int* FillMatrixRandom(float** matrix, float* vector);
void FillMatrixDefault(float** matrix, float* vector);
void SortCPU(float** matrix, float* vector, float* answer);
void PrintMatrix(float** matrix, float* vector, float* answer);
void PrintMatrix(std::string stuff, float* matrix, float* vector, float* answer);
void CheckAnswer(float* answer, int* key);

void ForwardSubstitution(float** matrix, float* vector);
void BackSubstitution(float** matrix, float* vector, float* answer);

void InitCUDA(float** m, float* v, float* a);
void cuErrorCheck(cudaError_t cs);

__global__ void DeviceGaussForwardLower(float* d_m, float* d_v);
__global__ void DeviceGaussForwardUpper(float* d_m, float* d_v, float* d_a);