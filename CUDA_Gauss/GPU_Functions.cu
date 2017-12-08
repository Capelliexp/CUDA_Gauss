#include "project.cuh"

__global__
void DeviceGaussForward(float* d_m, float* d_v) {
	int id = threadIdx.x + blockIdx.x * blockDim.x + 1;

	if (id < COLUMN_LENGTH) {
		int i = 0;
		float factor;

		__shared__ float pivotRHS;
		__shared__ float pivotLHS[ROW_LENGTH];
		
		float RHS = d_v[id];
		float LHS[ROW_LENGTH];

		for (int j = 0; j < ROW_LENGTH; ++j) {
			int pos = (id * ROW_LENGTH) + j;
			LHS[j] = d_m[pos];
		}

		while (id > i) {
			__syncthreads();
			pivotRHS = d_v[i];
			for (int j = 0; j < ROW_LENGTH; ++j) pivotLHS[j] = d_m[i*ROW_LENGTH + j];

			factor = (LHS[i] / pivotLHS[i]) * (-1);

			for (int j = i; j < ROW_LENGTH; j++) LHS[j] += (factor * pivotLHS[j]);
			RHS += (factor * pivotRHS);

			++i;
		}

		for (int j = 0; j < ROW_LENGTH; ++j) d_m[ROW_LENGTH*id + j] = LHS[j]; 
		d_v[id] = RHS;
	}
}

void InitCUDA(float** m, float* v, float* a) {
	float* d_m = nullptr;
	float* d_v = nullptr;
	float* d_a = nullptr;


	float* cuda_m = new float[ROW_LENGTH*COLUMN_LENGTH];
	for (int i = 0; i < COLUMN_LENGTH; i++){
		for (int j = 0; j < ROW_LENGTH; j++) {
			cuda_m[i * ROW_LENGTH + j] = m[i][j];
		}
	}

	cuErrorCheck(cudaSetDevice(0));

	//Allocate memory
	cuErrorCheck(cudaMalloc((void**)&d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float)));
	cuErrorCheck(cudaMalloc((void**)&d_v, COLUMN_LENGTH * sizeof(float)));
	cuErrorCheck(cudaMalloc((void**)&d_a, COLUMN_LENGTH * sizeof(float)));

	cuErrorCheck(cudaMemcpy(d_m, cuda_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	cuErrorCheck(cudaMemcpy(d_v, v, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	cuErrorCheck(cudaMemcpy(d_a, a, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

	DeviceGaussForward<<<1, 4>>>(d_m, d_v);
	cuErrorCheck(cudaGetLastError());

	cuErrorCheck(cudaMemcpy(cuda_m, d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	cuErrorCheck(cudaMemcpy(v, d_v, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

	/*for (int i = 0; i < ROW_LENGTH * COLUMN_LENGTH; i++) {
		if (i%ROW_LENGTH == 0) std::cout << std::endl;
		std::cout << cuda_m[i] << " ";
	}
	std::cout << std::endl;*/

	for (int i = 0; i < COLUMN_LENGTH; i++) {
		for (int j = 0; j < ROW_LENGTH; j++)
			std::cout << cuda_m[(i*ROW_LENGTH) + j] << " ";
		std::cout << std::endl;
	}

	cudaFree(d_m);
	cudaFree(d_v);
	cudaFree(d_a);
}

void cuErrorCheck(cudaError_t cs)
{
	if (cs != cudaSuccess)
		fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(cs));
}