#include "project.cuh"

int main(int argc, char *argv[]){
	srand((unsigned int)time(NULL));

	float** matrixCPU = new float*[COLUMN_LENGTH];
	float** matrixCUDA = new float*[COLUMN_LENGTH];
	for (int i = 0; i < COLUMN_LENGTH; i++) {
		matrixCPU[i] = new float[ROW_LENGTH];
		matrixCUDA[i] = new float[ROW_LENGTH];
	}		

	float* vectorCPU = new float[COLUMN_LENGTH];
	float* vectorCUDA = new float[COLUMN_LENGTH];

	float* answerCPU = new float[COLUMN_LENGTH];
	float* answerCUDA = new float[COLUMN_LENGTH];

	int* key = new int[COLUMN_LENGTH];

	key = FillMatrixRandom(matrixCPU, vectorCPU);

	for (int i = 0; i < COLUMN_LENGTH; i++) {
		for (int j = 0; j < ROW_LENGTH; j++) {
			matrixCUDA[i][j] = matrixCPU[i][j];
		}
	}

	for (int i = 0; i < COLUMN_LENGTH; i++)
		vectorCUDA[i] = vectorCPU[i];

	//HOST
	SortCPU(matrixCPU, vectorCPU, answerCPU);
	CheckAnswer("CPU", answerCPU, key);

	printf("\n\n");
	
	//Device
	InitCUDA(matrixCUDA, vectorCUDA, answerCUDA);
	CheckAnswer("CUDA", answerCUDA, key);

	getchar();

	free(matrixCPU);
	free(matrixCUDA);
	free(vectorCPU);
	free(vectorCUDA);
	free(answerCPU);
	free(answerCUDA);
	free(key);
	return 0;
}

int* FillMatrixRandom(float** matrix, float* vector) {
	for (int i = 0; i < COLUMN_LENGTH; i++) {	//generate matrix[][]
		for (int j = 0; j < ROW_LENGTH; j++) {
			//matrix[i][j] = (int)rand() % (COLUMN_LENGTH*ROW_LENGTH) + 1;
			matrix[i][j] = (int)rand()%(4) + 1;
			if ((int)rand()%3 == 0)
				matrix[i][j] *= -1;
		}
	}

	std::cout << "SOLUTION:" << std::endl;
	int* answers = new int[COLUMN_LENGTH];
	for (int i = 0; i < COLUMN_LENGTH; i++) {	//generate solution[] (x,y,z... values)
		//answers[i] = (int)rand() % (COLUMN_LENGTH*ROW_LENGTH) + 1;
		answers[i] = (int)rand()%(4) + 1;
		if ((int)rand() % 3 == 0)
			answers[i] *= -1;
		std::cout << answers[i] << " ";
	}
	std::cout << std::endl << std::endl;

	for (int i = 0; i < ROW_LENGTH; i++) {	//calc matrix[][] * solution[]
		vector[i] = 0;
		for (int j = 0; j < COLUMN_LENGTH; j++)
			vector[i] += matrix[i][j] * answers[j];
	}

	return answers;
}

void FillMatrixDefault(float ** matrix, float * vector){
	matrix[0][0] = 2;
	matrix[0][1] = 1;
	matrix[0][2] = -1;
	matrix[0][3] = 2;

	matrix[1][0] = 4;
	matrix[1][1] = 5;
	matrix[1][2] = -3;
	matrix[1][3] = 6;

	matrix[2][0] = -2;
	matrix[2][1] = 5;
	matrix[2][2] = -2;
	matrix[2][3] = 6;

	matrix[3][0] = 4;
	matrix[3][1] = 11;
	matrix[3][2] = -4;
	matrix[3][3] = 8;

	//---

	vector[0] = 5;
	vector[1] = 9;
	vector[2] = 4;
	vector[3] = 2;
}

void SortCPU(float** matrix, float* vector, float* answer) {
	std::clock_t start;

	start = std::clock();	//timer start
	ForwardSubstitution(matrix, vector);
	BackSubstitution(matrix, vector, answer);
	long double duration = (std::clock() - start) / (long double)CLOCKS_PER_SEC;	//timer stop

	std::cout << "CPU sorting time: " << duration << " sec" << std::endl;
}

void PrintMatrix(float** matrix, float* vector, float* answer) {
	std::cout << "Equation: " << std::endl;
	for (int i = 0; i < COLUMN_LENGTH; i++) {
		std::cout << "| ";
		for (int j = 0; j < ROW_LENGTH; j++) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << " | " << vector[i] << std::endl;
	}

	/*std::cout << std::endl << "Answers: ";
	for (int i = 0; i < ROW_LENGTH; i++) {
		std::cout << (answer[i] == -431602080 ? "NaN" : std::to_string(answer[i])) << " ";
	}*/

	std::cout << std::endl;
}

void PrintMatrix(std::string stuff, float* matrix, float* vector, float* answer) {
	std::cout << stuff << ":" << std::endl;
	for (int i = 0; i < COLUMN_LENGTH; i++) {
		std::cout << "| ";
		for (int j = 0; j < ROW_LENGTH; j++) {
			std::cout << matrix[i*ROW_LENGTH + j] << " ";
		}
		std::cout << " | " << vector[i] << std::endl;
	}

	std::cout << std::endl << stuff << " answers: ";
	for (int i = 0; i < ROW_LENGTH; i++) {
		std::cout << (answer[i] == -431602080 ? "NaN" : std::to_string(answer[i])) << " ";
	}

	std::cout << std::endl << std::endl;
}

void CheckAnswer(std::string type, float* answer, int* key) {
	std::cout << type << ": ";

	bool correct = true;
	for (int i = 0; i < COLUMN_LENGTH; i++) {
		if (abs(answer[i] - key[i]) > 0.05) {
			correct = false;
			//std::cout << i << ": " << key[i] << " - " << answer[i] << std::endl;
		}
	}
	if(correct == true) std::cout << "CORRECT!" << std::endl;
	else std::cout << "incorrect" << std::endl;
}