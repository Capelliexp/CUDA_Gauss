#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define COLUMN_LENGTH 4 // y
#define ROW_LENGTH 4	// x

void FillMAtrixRandom(float** matrix, float* vector);
void SortCPU(float** matrix, float* vector, float* answer);
void PrintMatrix(float** matrix, float* vector, float* answer);

void ForwardSubstitution(float** matrix, float* vector);
void BackSubstitution(float** matrix, float* vector, float* answer);

void InitCUDA(float** m, float* v, float* a);
void cuErrorCheck(cudaError_t cs);

__global__ void DeviceGaussForward(float** d_m/*, float* d_v*/);
__device__ void FloatArrayCpy(float* dst, float* src, int length);