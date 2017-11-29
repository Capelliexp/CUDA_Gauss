#pragma once
#include <stdio.h>
#include <string>
#include <iostream>

#define COLUMN_LENGTH 4 // y
#define ROW_LENGTH 4	// x

void FillMatrixStandard(float** matrix, float* vector);
void FillMAtrixRandom(float** matrix, float* vector);
void SortCPU(float** matrix, float* vector, float* answer);
void PrintMatrix(float** matrix, float* vector, float* answer);

void ForwardSubstitution(float** matrix, float* vector);
void BackSubstitution(float** matrix, float* vector, float* answer);