#include "project.cuh"

__global__
void DeviceGaussForwardLower(float* d_m, float* d_v) {
	int id = threadIdx.x + blockIdx.x * blockDim.x + 1;

	if (id < COLUMN_LENGTH) {
		int i = 0;
		double factor;

		__shared__ double pivotRHS;
		__shared__ double pivotLHS[ROW_LENGTH];

		double RHS = d_v[id];
		double LHS[ROW_LENGTH];

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

			if (id < COLUMN_LENGTH) {
				for (int j = 0; j < ROW_LENGTH; ++j) {
					d_m[ROW_LENGTH*id + j] = LHS[j];
				}
				d_v[id] = RHS;
			}
		}
	}
}

__global__
void DeviceGaussForwardUpper(float* d_m, float* d_v, float* d_a) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < COLUMN_LENGTH) {
		int i = COLUMN_LENGTH - 1;
		double factor;

		__shared__ double pivotRHS;
		__shared__ double pivotLHS[ROW_LENGTH];

		double RHS = d_v[id];
		double LHS[ROW_LENGTH];

		for (int j = 0; j < ROW_LENGTH; ++j) {
			int pos = (id * ROW_LENGTH) + j;
			double a = d_m[pos];
			LHS[j] = a;
			//printf("id %d: LHS[%d] = %f \n",id, j, LHS[j]);
		}

		int iterator = 0;

		if (id == COLUMN_LENGTH-1) {
			d_a[id] = RHS / LHS[ROW_LENGTH-1];
		}

		while (id < i) {
			__syncthreads();
			pivotRHS = d_v[i];

			for (int j = 0; j < ROW_LENGTH; ++j) {
				pivotLHS[j] = d_m[i*ROW_LENGTH + j];
			}

			double a = LHS[i];
			double b = pivotLHS[i];
			double c = (a / b);
			factor = c * (-1);
			//factor = (LHS[i] / pivotLHS[i]) * (-1);

			//if (id == 0) printf("factor: %f \n", factor);
			for (int j = 0; j < ROW_LENGTH; j++) {
				LHS[j] += (factor * pivotLHS[j]);
				//if (id == 0) printf("i %d: LHS[%d] = %f \n",i, j, LHS[j]);
			}
			//printf("\n");

			RHS += (factor * pivotRHS);

			--i;
			++iterator;

			if (!(id < i)) {
				for (int j = 0; j < ROW_LENGTH; ++j) {
					d_m[ROW_LENGTH*id + j] = LHS[j];
				}
				d_v[id] = RHS;
				d_a[id] = RHS / LHS[i];
			}
			
		}
	}
}

void InitCUDA(float** m, float* v, float* a) {
	float* d_m = nullptr;
	float* d_v = nullptr;
	float* d_a = nullptr;

	float duration = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

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

	//Copy memory
	cuErrorCheck(cudaMemcpy(d_m, cuda_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	cuErrorCheck(cudaMemcpy(d_v, v, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
	cuErrorCheck(cudaMemcpy(d_a, a, COLUMN_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

	//cudaEventRecord(start);
	DeviceGaussForwardLower<<<1, 5>>>(d_m, d_v);	//upper
	cuErrorCheck(cudaGetLastError());
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&duration, start, stop);
	//std::cout << "CUDA 1 time: " << duration / 1000 << " sec" << std::endl;

	//Copy memory
	cuErrorCheck(cudaMemcpy(cuda_m, d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	cuErrorCheck(cudaMemcpy(v, d_v, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

	PrintMatrix(cuda_m, v, a);

	//cudaEventRecord(start);
	DeviceGaussForwardUpper<<<1, 5>>>(d_m, d_v, d_a);	//lower
	cuErrorCheck(cudaGetLastError());
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&duration, start, stop);
	//std::cout << "CUDA 2 time: " << duration / 1000 << " sec" << std::endl;

	//Copy memory
	cuErrorCheck(cudaMemcpy(cuda_m, d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	cuErrorCheck(cudaMemcpy(v, d_v, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	cuErrorCheck(cudaMemcpy(a, d_a, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

	PrintMatrix(cuda_m, v, a);

	//Free memory
	cudaFree(d_m);
	cudaFree(d_v);
	cudaFree(d_a);
}

void cuErrorCheck(cudaError_t cs)
{
	if (cs != cudaSuccess)
		fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(cs));
}