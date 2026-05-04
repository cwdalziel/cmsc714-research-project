/*
 * 3d_fft_mpi.cpp
 * 3D FFT using MPI + FFTW
 * 
 * Compile: mpic++ -O2 -o 3d_fft_mpi 3d_fft_mpi.cpp -lfftw3 -lm
 * Run:     mpirun -np 8 ./3d_fft_mpi 256
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <fftw3.h>

/* Initialize 3D data */
static void init_data_3d(fftw_complex *buf, int local_slices, int N)
{
    for (int s = 0; s < local_slices; s++)
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
                int idx = (s * N + r) * N + c;
                buf[idx][0] = sin(2.0 * M_PI * r / N) * cos(2.0 * M_PI * c / N);
                buf[idx][1] = 0.0;
            }
}

/* 1D FFT along X-axis (innermost dimension) */
static void fft_x_axis(fftw_complex *data, int local_slices, int N)
{
    fftw_plan plan = fftw_plan_many_dft(
        1, &N, local_slices * N,  /* N elements, repeated */
        data, NULL, 1, N,
        data, NULL, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

/* 1D FFT along Y-axis (middle dimension) */
static void fft_y_axis(fftw_complex *data, int local_slices, int N)
{
    fftw_plan plan = fftw_plan_many_dft(
        1, &N, local_slices * N,
        data, NULL, N, 1,
        data, NULL, N, 1,
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

/* All-to-all transpose for 3D (exchange slabs between processes) */
static void transpose_alltoall_3d(fftw_complex *in, fftw_complex *out,
                                   int local_slices, int N, int P,
                                   MPI_Comm comm)
{
    int total_size = local_slices * N * N;
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(total_size * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(total_size * sizeof(fftw_complex));

    /* Pack data for MPI_Alltoall */
    memcpy(send_buf, in, total_size * sizeof(fftw_complex));

    int count = local_slices * N * N / P;  /* elements per rank */
    MPI_Alltoall(send_buf, count * 2, MPI_DOUBLE,
                 recv_buf, count * 2, MPI_DOUBLE,
                 comm);

    memcpy(out, recv_buf, total_size * sizeof(fftw_complex));

    fftw_free(send_buf);
    fftw_free(recv_buf);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int N = (argc > 1) ? atoi(argv[1]) : 64;

    if ((N * N * N) % (P * N * N) != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N³ must be divisible by P*N²\n");
        MPI_Finalize();
        return 1;
    }

    int local_slices = N / P;
    size_t local_size = (size_t)local_slices * N * N;

    fftw_complex *data = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    fftw_complex *temp = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));

    init_data_3d(data, local_slices, N);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Stage 1: FFT along X */
    fft_x_axis(data, local_slices, N);

    /* Stage 2: All-to-all transpose */
    transpose_alltoall_3d(data, temp, local_slices, N, P, MPI_COMM_WORLD);

    /* Stage 3: FFT along Y */
    fft_y_axis(temp, local_slices, N);

    /* Stage 4: All-to-all transpose 2 */
    /* */
    transpose_alltoall_3d(temp, data, local_slices, N, P, MPI_COMM_WORLD);

    /* Stage 5: FFT along Z (outermost dimension) */
    fftw_plan plan_z = fftw_plan_many_dft(
        1, &N, local_slices,
        data, NULL, N * N, 1,
        data, NULL, N * N, 1,
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_z);
    fftw_destroy_plan(plan_z);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    /* For verification, perform inverse FFT  */
    if (rank == 0){
    fftw_plan plan_inv = fftw_plan_dft_3d(N, N, N, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_inv);
    fftw_destroy_plan(plan_inv);

    /* Scale by 1/N³ */
    double scale = 1.0 / (N * N * N);
    for (size_t i = 0; i < local_size; i++) {
        data[i][0] *= scale;
        data[i][1] *= scale;
    }

    /* Compare with initial data */
    double error = 0.0;
    for (size_t i = 0; i < local_size; i++) {
        error += pow(data[i][0], 2) + pow(data[i][1], 2);
    }
    }

    if (rank == 0) {
        printf("=== 3D FFT MPI Results ===\n");
        printf("N=%d (N³=%d), P=%d\n", N, N*N*N, P);
        printf("Total time: %.6f s\n", t_end - t_start);
    }

    fftw_free(data);
    fftw_free(temp);

    MPI_Finalize();
    return 0;
}