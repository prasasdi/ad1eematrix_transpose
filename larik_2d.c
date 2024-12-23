#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> // AVX2 intrinsics

// Fungsi untuk mencetak matriks NxN
void print_matrix(int** matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Fungsi untuk mentransposisi matriks NxN menggunakan intrinsics AVX2
void transpose_nxn_avx2(int** matrix, int N) {
    // Loop untuk iterasi baris-blok utama dari matriks
    for (int i = 0; i < N; i += 8) {
        // Loop untuk iterasi kolom-blok utama dari matriks
        for (int j = i; j < N; j += 8) {
            // Loop untuk memuat baris-baris 8 elemen dari blok matriks
            for (int k = 0; k < 8; k++) {
                // Pastikan indeks dalam batas matriks
                if (i + k < N && j < N) {
                    // Load baris ke dalam register SIMD
                    __m256i row = _mm256_loadu_si256((__m256i*)&matrix[i + k][j]);
                    // Simpan baris kembali ke matriks
                    _mm256_storeu_si256((__m256i*)&matrix[i + k][j], row);
                }
            }

            // Loop untuk menukar elemen-elemen dalam blok 8x8
            for (int k = 0; k < 8; k++) {
                for (int l = k + 1; l < 8; l++) {
                    // Pastikan indeks dalam batas matriks
                    if (i + k < N && j + l < N && j + k < N && i + l < N) {
                        // Tukar elemen-elemen untuk transpose
                        int temp = matrix[i + k][j + l];
                        matrix[i + k][j + l] = matrix[j + k][i + l];
                        matrix[j + k][i + l] = temp;
                    }
                }
            }
        }
    }

    // Tangani baris dan kolom sisa jika N bukan kelipatan 8
    for (int i = (N / 8) * 8; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            // Tukar elemen-elemen di bagian sisa matriks
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

int main(void) {
    int N = 3; // Ukuran matriks NxN
    int** matrix = malloc(sizeof(int*) * N);

    // Alokasikan memori untuk setiap baris
    for (int i = 0; i < N; i++) {
        matrix[i] = malloc(sizeof(int) * N);
    }

    // Mengisi matriks dengan nilai
    int counter = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = counter++;
        }
    }

    // Mencetak matriks asli
    printf("Matriks asli:\n");
    print_matrix(matrix, N);

    // Melakukan transposisi matriks
    transpose_nxn_avx2(matrix, N);

    // Mencetak matriks setelah ditransposisi
    printf("Matriks setelah ditransposisi:\n");
    print_matrix(matrix, N);

    // Membebaskan memori yang telah dialokasikan
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);

    return 0;
}
