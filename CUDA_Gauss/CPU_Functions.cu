#pragma once
#include "project.cuh"

void ForwardSubstitution(float** m, float* v) {
	for (int i = 0; i < ROW_LENGTH; i++) {
		for (int j = 0; j < COLUMN_LENGTH - 1; j++) {
			if ((i + j + 1) < COLUMN_LENGTH) {
				float factor = (m[i + j + 1][i] / m[i][i]) * (-1);
				for (int k = 0; k < ROW_LENGTH; k++) {
					m[i + j + 1][k] += (factor * m[i][k]);
				}
				v[i + j + 1] += (factor * v[i]);
			}
		}
	}
}

void BackSubstitution(float** m, float* v, float* a) {
	for (int i = ROW_LENGTH - 1; i >= 0; i--) {
		for (int j = ROW_LENGTH - 1; j > i; j--) {
			v[i] -= a[j]*m[i][j];
		}
		a[i] = round(v[i] / m[i][i]);
	}
}