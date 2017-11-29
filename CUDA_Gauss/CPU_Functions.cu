#pragma once
#include "project.cuh"

void ForwardSubstitution(float** m, float* v) {
	for (int i = 0; i < ROW_LENGTH; i++) {
		std::cout << "---" << std::endl;
		for (int j = 0; j < COLUMN_LENGTH - 1; j++) {
			if ((i + j + 1) < COLUMN_LENGTH) {
				float factor = (m[i + j + 1][i] / m[i][i]) * (-1);
				std::cout << "factor: " << factor << std::endl;
				for (int k = 0; k < ROW_LENGTH; k++) {
					m[i + j + 1][k] += (factor * m[i][k]);
				}
				v[i + j + 1] += (factor * v[i]);
			}
		}
	}
}

void BackSubstitution(float** matrix, float* vector) {
	float* answers = new float[ROW_LENGTH];


}