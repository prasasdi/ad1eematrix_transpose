#include <stdio.h>
#include <immintrin.h> // AVX2 intrinsics

// Fungsi untuk mencetak matriks NxN
void print_matrix(int* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Fungsi untuk mentransposisi matriks NxN menggunakan intrinsics AVX2
void transpose_nxn_avx2(int* matrix, int N) {
    // Loop untuk iterasi baris-blok utama dari matriks
    for (int i = 0; i < N; i += 8) {
        // Loop untuk iterasi kolom-blok utama dari matriks
        for (int j = i; j < N; j += 8) {
            // Loop untuk memuat baris-baris 8 elemen dari blok matriks
            for (int k = 0; k < 8; k++) {
                // Pastikan indeks dalam batas matriks
                if (i + k < N && j < N) {
                    // Load baris ke dalam register SIMD
                    __m256i row = _mm256_loadu_si256((__m256i*)&matrix[(i + k) * N + j]);
                    // Simpan baris kembali ke matriks
                    _mm256_storeu_si256((__m256i*)&matrix[(i + k) * N + j], row);
                }
            }

            // Loop untuk menukar elemen-elemen dalam blok 8x8
            for (int k = 0; k < 8; k++) {
                for (int l = k + 1; l < 8; l++) {
                    // Pastikan indeks dalam batas matriks
                    if (i + k < N && j + l < N && j + k < N && i + l < N) {
                        // Tukar elemen-elemen untuk transpose
                        int temp = matrix[(i + k) * N + (j + l)];
                        matrix[(i + k) * N + (j + l)] = matrix[(j + k) * N + (i + l)];
                        matrix[(j + k) * N + (i + l)] = temp;
                    }
                }
            }
        }
    }

    // Tangani baris dan kolom sisa jika N bukan kelipatan 8
    for (int i = (N / 8) * 8; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            // Tukar elemen-elemen di bagian sisa matriks
            int temp = matrix[i * N + j];
            matrix[i * N + j] = matrix[j * N + i];
            matrix[j * N + i] = temp;
        }
    }
}

int main(void) {
    int N = 5; // Ubah ini ke ukuran matriks NxN yang diinginkan
    int matrix[25] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    printf("Matriks asli:\n");
    print_matrix(matrix, N);

    transpose_nxn_avx2(matrix, N);

    printf("Matriks setelah ditransposisi:\n");
    print_matrix(matrix, N);

    return 0;
}
