//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "CPU_Functions.cu"

#include "project.cuh"

int main(int argc, char *argv[]){
	float** matrix = new float*[COLUMN_LENGTH];
	for (int i = 0; i < 4; i++)
		matrix[i] = new float[ROW_LENGTH];
	float* vector = new float[COLUMN_LENGTH];

	FillMatrixStandard(matrix, vector);

	SortCPU(matrix, vector);

	getchar();
	return 0;
}

void FillMatrixStandard(float** matrix, float* vector) {
	matrix[0][0] = 1.f;
	matrix[0][1] = 3.f;
	matrix[0][2] = 1.f;
	matrix[0][3] = 3.f;
					
	matrix[1][0] = 4.f;
	matrix[1][1] = -2.f;
	matrix[1][2] = -3.f;
	matrix[1][3] = 1.f;
					
	matrix[2][0] = 2.f;
	matrix[2][1] = 1.f;
	matrix[2][2] = -1.f;
	matrix[2][3] = -1.f;
					
	matrix[3][0] = 1.f;
	matrix[3][1] = 2.f;
	matrix[3][2] = -1.f;
	matrix[3][3] = -2.f;

	vector[0] = 14.f;
	vector[1] = 20.f;
	vector[2] = 9.f;
	vector[3] = 3.f;
}

void FillMAtrixRandom(float** matrix, float* vector) {}

void SortCPU(float** matrix, float* vector) {
	ForwardSubstitution(matrix, vector);
	BackSubstitution(matrix, vector);
}