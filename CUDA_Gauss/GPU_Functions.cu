#include "project.cuh"

__global__
void DeviceGaussForward(float* d_m/*, float* d_v*/) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//++id;	//1-4
	int i = 0;	

	if (id < COLUMN_LENGTH) {
		printf("%f ", d_m[id]);
		__syncthreads();
		printf("\n");
		/*__shared__ float pivotLHS[ROW_LENGTH];
		__shared__ float pivotRHS;
		float LHS[ROW_LENGTH];
		float RHS = d_v[id];

		FloatArrayCpy(LHS, d_m[id], ROW_LENGTH);

		while (id > i) {
			FloatArrayCpy(pivotLHS, d_m[i], ROW_LENGTH);
			pivotRHS = d_v[i];
			float factor = (LHS[i] / pivotLHS[i]) * (-1);

			for (int j = 0; j < ROW_LENGTH; j++) {
				LHS[j] += (factor * pivotLHS[j]);
			}
			RHS += (factor * pivotRHS);

			++i;
			__syncthreads();
		}

		FloatArrayCpy(d_m[id], LHS, ROW_LENGTH);
		d_v[id] = RHS;*/
	}
}

__device__
void FloatArrayCpy(float* dst, float* src, int length) {
	for (int i = 0; i < length; ++i)
		dst[i] = src[i];
}

void InitCUDA(float** m, float* v, float* a) {
	float* d_m = nullptr;
	/*float* d_v = nullptr;
	float* d_a = nullptr;*/

	cuErrorCheck(cudaSetDevice(0));

	//Allocate memory
	//cuErrorCheck(cudaMalloc((void**)&d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float)));
	/*cuErrorCheck(cudaMalloc((void**)&d_v, COLUMN_LENGTH * sizeof(float)));
	cuErrorCheck(cudaMalloc((void**)&d_a, COLUMN_LENGTH * sizeof(float)));*/

	size_t pitch;
	cudaMallocPitch(&d_m, &pitch, sizeof(float)*ROW_LENGTH, COLUMN_LENGTH);

	cuErrorCheck(cudaMemcpy2D(d_m, pitch, m, sizeof(float)*COLUMN_LENGTH, sizeof(float)*ROW_LENGTH, COLUMN_LENGTH, cudaMemcpyHostToDevice));
	/*cuErrorCheck(cudaMemcpy(d_v, v, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	cuErrorCheck(cudaMemcpy(d_a, a, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));*/

	DeviceGaussForward<<<1, 4>>>(d_m/*, d_v*/);
	//cuErrorCheck(cudaGetLastError());

	//cuErrorCheck(cudaMemcpy(m, d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

	//cuErrorCheck(cudaMemcpy(v, d_v, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));



	cudaFree(d_m);
	/*cudaFree(d_v);
	cudaFree(d_a);*/
}

void cuErrorCheck(cudaError_t cs)
{
	if (cs != cudaSuccess)
		fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(cs));
}