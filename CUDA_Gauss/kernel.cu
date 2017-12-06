//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "CPU_Functions.cu"

#include "project.cuh"

int main(int argc, char *argv[]){
	srand((unsigned int)time(NULL));

	float** matrix = new float*[COLUMN_LENGTH];
	for (int i = 0; i < COLUMN_LENGTH; i++)
		matrix[i] = new float[ROW_LENGTH];
	float* vector = new float[COLUMN_LENGTH];
	float* answer = new float[ROW_LENGTH];

	//FillMatrixStandard(matrix, vector);
	while (true) {
		FillMAtrixRandom(matrix, vector);

		SortCPU(matrix, vector, answer);
		PrintMatrix(matrix, vector, answer);

		getchar();
		printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	}
	
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

	/*matrix[4][0] = 1.f;
	matrix[4][1] = 2.f;
	matrix[4][2] = -1.f;
	matrix[4][3] = -2.f;*/

	vector[0] = 14.f;
	vector[1] = 20.f;
	vector[2] = 9.f;
	vector[3] = 3.f;
	vector[4] = 3.f;
}

void FillMAtrixRandom(float** matrix, float* vector) {
	std::cout << "MATRIX:" << std::endl;
	for (int i = 0; i < COLUMN_LENGTH; i++) {
		for (int j = 0; j < ROW_LENGTH; j++) {
			matrix[i][j] = (int)rand()%(COLUMN_LENGTH*ROW_LENGTH) + 1;
			if ((int)rand()%3 == 0)
				matrix[i][j] *= -1;
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	std::cout << "ANSWERS:" << std::endl;
	int* answers = new int[COLUMN_LENGTH];
	for (int i = 0; i < COLUMN_LENGTH; i++) {
		answers[i] = (int)rand()%(COLUMN_LENGTH*ROW_LENGTH) + 1;
		if ((int)rand() % 3 == 0)
			answers[i] *= -1;
		std::cout << answers[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "VECTOR:" << std::endl;
	for (int i = 0; i < ROW_LENGTH; i++) {
		vector[i] = 0;
		for (int j = 0; j < COLUMN_LENGTH; j++)
			vector[i] += matrix[i][j] * answers[j];
		std::cout << vector[i] << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	free(answers);
}

void SortCPU(float** matrix, float* vector, float* answer) {
	ForwardSubstitution(matrix, vector);
	BackSubstitution(matrix, vector, answer);
}

void PrintMatrix(float** matrix, float* vector, float* answer) {
	for (int i = 0; i < COLUMN_LENGTH; i++) {
		std::cout << "| ";
		for (int j = 0; j < ROW_LENGTH; j++) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << " | " << vector[i] << std::endl;
	}

	std::cout << "Answers: ";
	for (int i = 0; i < ROW_LENGTH; i++) {
		std::cout << answer[i] << " ";
	}


	std::cout << std::endl;
}