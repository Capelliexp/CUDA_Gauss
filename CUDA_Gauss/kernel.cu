
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CPU_Functions.cu"

#include <stdio.h>

#define COLUMN_LENGTH 4 // y
#define ROW_LENGTH 4	// x

int main(int argc, char *argv[]){
	float** matrix = new float*[COLUMN_LENGTH];
	for (int i = 0; i < 4; i++)
		matrix[i] = new float[ROW_LENGTH];
	float* vector = new float[COLUMN_LENGTH];

	FillMatrixStandard(matrix, vector);

	SortCPU(matrix, vector);

	return 0;
}

void FillMatrixStandard(float** matrix, float* vector) {
	matrix[0][0] = 2.f;
	matrix[0][1] = 4.f;
	matrix[0][2] = 6.f;
	matrix[0][3] = 8.f;
					
	matrix[1][0] = 1.f;
	matrix[1][1] = 2.f;
	matrix[1][2] = 3.f;
	matrix[1][3] = 4.f;
					
	matrix[2][0] = 0.f;
	matrix[2][1] = 3.f;
	matrix[2][2] = 6.f;
	matrix[2][3] = 9.f;
					
	matrix[3][0] = 0.f;
	matrix[3][1] = 4.f;
	matrix[3][2] = 8.f;
	matrix[3][3] = 12.f;

	vector[0] = 0.f;
	vector[1] = 1.f;
	vector[2] = 2.f;
	vector[3] = 3.f;
}

void FillMAtrixRandom(float** matrix, float* vector) {}

void SortCPU(float** matrix, float* vector) {
	OrderMatrixBySize(matrix);
	ForwardSubstitution(matrix, vector);
	BackSubstitution(matrix, vector);
}