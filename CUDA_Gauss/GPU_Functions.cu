#include "project.cuh"

__global__
void DeviceGaussForwardLower(float* d_m, float* d_v) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if ((id *= ELEMENTS_PER_THREAD)< COLUMN_LENGTH) {
		++id;
		int i = 0;
		double factor;

		__shared__ double pivotRHS;
		__shared__ double pivotLHS[ROW_LENGTH];

		double RHS[ELEMENTS_PER_THREAD];
		double LHS[ELEMENTS_PER_THREAD][ROW_LENGTH];

		for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
			float value = d_v[id + j];
			RHS[j] = value;
		}

		for (int k = 0; k < ELEMENTS_PER_THREAD; ++k) {
			for (int j = 0; j < ROW_LENGTH; ++j) {
				int pos = ((id+k) * ROW_LENGTH) + j;	//fel?
				float value = d_m[pos];
				LHS[k][j] = value;
			}
		}

		while ((id+ELEMENTS_PER_THREAD) > i) {
			__syncthreads();

			pivotRHS = d_v[i];

			for (int j = 0; j < ROW_LENGTH; ++j) {
				float value = d_m[i*ROW_LENGTH + j];
				pivotLHS[j] = value;
			}

			for (int g = 0; g < ELEMENTS_PER_THREAD; ++g) {
				if ((id + g) > i) {
					float a = LHS[g][i];
					float b = pivotLHS[i];
					float c = (a / b);
					factor = c * (-1);
					//factor = (LHS[g][i] / pivotLHS[i]) * (-1);

					for (int j = i; j < ROW_LENGTH; ++j) {
						float value = LHS[g][j] + (factor * pivotLHS[j]);
						LHS[g][j] = value;
					}
					float value = RHS[g] + (factor * pivotRHS);
					RHS[g] = value;
				}

				for (int j = 0; j < ROW_LENGTH; ++j) {
					d_m[ROW_LENGTH*(id + g) + j] = (abs(LHS[g][j]) < 0.001 ? 0 : LHS[g][j]);
				}
				d_v[id + g] = (abs(RHS[g]) < 0.001 ? 0 : RHS[g]);

				/*if (!((id + g) > (i-1))) {	//FLYTTA IFSATSEN ÖVERS I WHILE OCH TESTA DÅ. KÖR SYNC EFTER IF
					
				}*/
			}
			++i;
		}
	}
}

__global__
void DeviceGaussForwardUpper(float* d_m, float* d_v, float* d_a) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if ((id *= ELEMENTS_PER_THREAD) < COLUMN_LENGTH) {
		int i = COLUMN_LENGTH - 1;
		double factor;

		__shared__ double pivotRHS;
		__shared__ double pivotLHS[ROW_LENGTH];

		double RHS[ELEMENTS_PER_THREAD];
		double LHS[ELEMENTS_PER_THREAD][ROW_LENGTH];

		for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
			RHS[j] = d_v[id+j];
		}
		
		for (int k = 0; k < ELEMENTS_PER_THREAD; ++k) {
			for (int j = 0; j < ROW_LENGTH; ++j) {
				int pos = (k*ROW_LENGTH)+(id * ROW_LENGTH) + j;
				double a = d_m[pos];
				LHS[k][j] = a;
				//printf("id %d: LHS[%d] = %f \n",id, j, LHS[j]);
			}
		}

		while (id < i) {
			__syncthreads();

			pivotRHS = d_v[i];

			for (int j = 0; j < ROW_LENGTH; ++j) {
				pivotLHS[j] = d_m[i*ROW_LENGTH + j];
			}

			for (int g = 0; g < ELEMENTS_PER_THREAD; ++g) {
				if (id+g < i) {
					double a = LHS[g][i];	//obs! antar att vi jobbar med square matrix
					double b = pivotLHS[i];
					double c = (a / b);
					factor = c * (-1);
					//factor = (LHS[i] / pivotLHS[i]) * (-1);

					//if (id == 0) printf("factor: %f \n", factor);
					for (int j = 0; j < ROW_LENGTH; j++) {
						LHS[g][j] += (factor * pivotLHS[j]);
						//if (id == 0) printf("i %d: LHS[%d] = %f \n",i, j, LHS[j]);
					}
					//printf("\n");

					float piv = pivotRHS;
					float pivAndFac = piv*factor;
					float RightHandSideOfG = RHS[g];
					float value = RightHandSideOfG + pivAndFac;
					RHS[g] = value;
				}

				for (int j = 0; j < ROW_LENGTH; ++j) {
					int pos = ROW_LENGTH*(id + g) + j;
					float value = LHS[g][j];
					d_m[pos] = value;	//fel?
				}
				float value1 = RHS[g];
				d_v[id + g] = value1;

				//float denominator = LHS[g][i];
				//float value2 = RHS[g] / denominator;	//antagligen rätt...
				//d_a[id + g] = value2;

				//float denominator = d_m[(g*ROW_LENGTH) + (id*ROW_LENGTH) + i];
				//float value2 = d_v[id + g] / denominator;	//antagligen rätt...
				//d_a[id + g] = value2;

				d_a[id + g] = d_v[id + g] / d_m[(id + g)*ROW_LENGTH + (g == 0 ? i-1 : i)];
				/*if (!(id+g < i)) {
					
				}*/
			}
			--i;
		}

		if (id == COLUMN_LENGTH - 1) {
			d_a[id] = RHS[0] / LHS[0][ROW_LENGTH - 1];	//fel? Nej, rätt?
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
	DeviceGaussForwardLower<<<NR_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_m, d_v);	//upper
	cuErrorCheck(cudaGetLastError());
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&duration, start, stop);
	//std::cout << "CUDA 1 time: " << duration / 1000 << " sec" << std::endl;

	//Copy memory
	cuErrorCheck(cudaMemcpy(cuda_m, d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	cuErrorCheck(cudaMemcpy(v, d_v, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

	//PrintMatrix("After lower", cuda_m, v, a);

	//cudaEventRecord(start);
	DeviceGaussForwardUpper<<<NR_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_m, d_v, d_a);	//lower
	cuErrorCheck(cudaGetLastError());
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&duration, start, stop);
	//std::cout << "CUDA 2 time: " << duration / 1000 << " sec" << std::endl;

	//Copy memory
	cuErrorCheck(cudaMemcpy(cuda_m, d_m, ROW_LENGTH * COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	cuErrorCheck(cudaMemcpy(v, d_v, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	cuErrorCheck(cudaMemcpy(a, d_a, COLUMN_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

	//PrintMatrix("After upper", cuda_m, v, a);

	std::cout << "GPU solution:" << std::endl;
	for (int i = 0; i < COLUMN_LENGTH-1; i++) {
		std::cout << a[i] << ", ";
	}
	std::cout << a[COLUMN_LENGTH - 1] << std::endl << std::endl;

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