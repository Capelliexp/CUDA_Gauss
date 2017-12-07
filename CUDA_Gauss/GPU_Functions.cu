#include "project.cuh"

__global__
void DeviceGaussForward(float** d_m, float* d_v) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	++id;
	int i = 0;

	__shared__ float pivotLHS[ROW_LENGTH];
	__shared__ float pivotRHS;
	float LHS[ROW_LENGTH];
	float RHS = d_v[id];

	memcpy(LHS, d_m[id], sizeof(float)*ROW_LENGTH);

	while (id > i && id < COLUMN_LENGTH) {
		memcpy(pivotLHS, d_m[i], sizeof(float)*ROW_LENGTH);
		pivotRHS = d_v[i];
		float factor = (LHS[i] / pivotLHS[i]) * (-1);

		for (int j = 0; j < ROW_LENGTH; j++) {
			LHS[j] += (factor * pivotLHS[j]);
		}
		RHS += (factor * pivotRHS);

		++i;
		__syncthreads();
	}
}

void InitCUDA(float** m, float* v, float* a) {
	float** d_m = nullptr;
	float* d_v = nullptr;
	float* d_a = nullptr;

	cuErrorCheck(cudaSetDevice(0));

	//Allocate memory
	cuErrorCheck(cudaMalloc((void**)&d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float)));
	cuErrorCheck(cudaMalloc((void**)&d_v, COLUMN_LENGTH * sizeof(float)));
	cuErrorCheck(cudaMalloc((void**)&d_a, COLUMN_LENGTH * sizeof(float)));

	cuErrorCheck(cudaMemcpy(d_m, m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	cuErrorCheck(cudaMemcpy(d_v, v, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	cuErrorCheck(cudaMemcpy(d_a, a, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));



	cudaFree(d_m);
	cudaFree(d_v);
	cudaFree(d_a);
}

void cuErrorCheck(cudaError_t cs)
{
	if (cs != cudaSuccess)
		fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(cs));
}