/*
 * fft2d_mpi.c
 * 2D FFT using MPI + FFTW
 *
 * Algorithm:
 *   1. Distribute N rows across P processes (each gets N/P rows)
 *   2. Local 1D FFT on each row (FFTW)
 *   3. All-to-all transpose  <-- network topology bottleneck
 *   4. Local 1D FFT on each (transposed) row  = column FFTs
 *   5. All-to-all transpose back
 *
 * Compile (standard OpenMPI):
 *   mpicc -O2 -o fft2d_mpi fft2d_mpi.c -lfftw3 -lm
 *
 * Compile (SimGrid SMPI):
 *   smpicc -O2 -o fft2d_mpi fft2d_mpi.c -lfftw3 -lm
 *
 * Run (standard):
 *   mpirun -np 4 ./fft2d_mpi 1024
 *
 * Run (SimGrid):
 *   smpirun -np 4 -platform platform.xml -hostfile hostfile.txt ./fft2d_mpi 1024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <fftw3.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

/* Fill a local block with a simple test signal */
static void init_data(fftw_complex *buf, int rows, int N)
{
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < N; c++) {
            buf[r * N + c][0] = sin(2.0 * M_PI * r / N) * cos(2.0 * M_PI * c / N);
            buf[r * N + c][1] = 0.0;
        }
}

/* Compute local RMS magnitude (for sanity-check output) */
static double rms(fftw_complex *buf, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += buf[i][0] * buf[i][0] + buf[i][1] * buf[i][1];
    return sqrt(s / n);
}

/* ------------------------------------------------------------------ */
/* Transpose via MPI_Alltoall                                           */
/* ------------------------------------------------------------------ */
/*
 * Global matrix is N x N, distributed as (N/P) rows per process.
 * After transpose the global matrix is also N x N but columns become rows.
 *
 * Strategy:
 *   - Each process splits its (local_rows x N) block into P sub-blocks
 *     of size (local_rows x local_rows), one destined for each process.
 *   - MPI_Alltoall exchanges these blocks.
 *   - Each process then locally transposes each received (local_rows x local_rows)
 *     block so rows become columns.
 *
 * This is the classic slab-decomposition all-to-all transpose.
 */
static void transpose_alltoall(fftw_complex *in, fftw_complex *out,
                                int local_rows, int N, int P,
                                MPI_Comm comm)
{
    /* Pack: re-order so that data for rank r is contiguous */
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(local_rows * N * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(local_rows * N * sizeof(fftw_complex));

    int block = local_rows; /* = N/P */

    /* Pack send buffer: for each destination rank r, copy the sub-block
     * columns [r*block .. (r+1)*block) from every local row */
    for (int r = 0; r < P; r++)
        for (int row = 0; row < local_rows; row++)
            memcpy(&send_buf[r * local_rows * block + row * block],
                   &in[row * N + r * block],
                   block * sizeof(fftw_complex));

    int count = local_rows * block; /* elements per rank */
    MPI_Alltoall(send_buf, count * 2, MPI_DOUBLE,
                 recv_buf, count * 2, MPI_DOUBLE,
                 comm);

    /* Unpack & local transpose: received block from rank r is
     * (local_rows x block) = (block x block) for square case.
     * We want row i of the transposed matrix to be column i of original. */
    for (int r = 0; r < P; r++)
        for (int row = 0; row < block; row++)
            for (int col = 0; col < block; col++) {
                out[(r * block + col) * local_rows + row][0] =
                    recv_buf[r * local_rows * block + row * block + col][0];
                out[(r * block + col) * local_rows + row][1] =
                    recv_buf[r * local_rows * block + row * block + col][1];
            }

    fftw_free(send_buf);
    fftw_free(recv_buf);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /* Matrix size N x N.  N must be divisible by P. */
    int N = (argc > 1) ? atoi(argv[1]) : 16384;

    if (N % P != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N (%d) must be divisible by P (%d)\n", N, P);
        MPI_Finalize();
        return 1;
    }

    int local_rows = N / P;

    /* Allocate local slab: local_rows x N */
    fftw_complex *data     = (fftw_complex *)fftw_malloc(local_rows * N * sizeof(fftw_complex));
    fftw_complex *transposed = (fftw_complex *)fftw_malloc(local_rows * N * sizeof(fftw_complex));

    /* FFTW plans for 1D FFT along a row of length N */
    fftw_plan plan_row = fftw_plan_many_dft(
        1, &N, local_rows,
        data, NULL, 1, N,
        data, NULL, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_plan plan_col = fftw_plan_many_dft(
        1, &N, local_rows,
        transposed, NULL, 1, N,
        transposed, NULL, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE);

    /* Initialise with test signal */
    init_data(data, local_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ---- Step 1: Row-wise FFTs (local) ---- */
    double t_fft1_start = MPI_Wtime();
    fftw_execute(plan_row);
    double t_fft1_end = MPI_Wtime();

    /* ---- Step 2: Global transpose via Alltoall ---- */
    double t_trans1_start = MPI_Wtime();
    transpose_alltoall(data, transposed, local_rows, N, P, MPI_COMM_WORLD);
    double t_trans1_end = MPI_Wtime();

    /* ---- Step 3: Column-wise FFTs (now rows after transpose, local) ---- */
    double t_fft2_start = MPI_Wtime();
    fftw_execute(plan_col);
    double t_fft2_end = MPI_Wtime();

    /* ---- Step 4: Transpose back (optional — restores row-major order) ---- */
    double t_trans2_start = MPI_Wtime();
    transpose_alltoall(transposed, data, local_rows, N, P, MPI_COMM_WORLD);
    double t_trans2_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    /* ---- Timing report ---- */
    if (rank == 0) {
        double total       = t_end - t_start;
        double fft_time    = (t_fft1_end - t_fft1_start) + (t_fft2_end - t_fft2_start);
        double comm_time   = (t_trans1_end - t_trans1_start) + (t_trans2_end - t_trans2_start);

        printf("=== 2D FFT MPI Results ===\n");
        printf("N=%d, P=%d, local_rows=%d\n", N, P, local_rows);
        printf("Total time       : %.6f s\n", total);
        printf("FFT compute time : %.6f s  (%.1f%%)\n",
               fft_time, 100.0 * fft_time / total);
        printf("Alltoall comm    : %.6f s  (%.1f%%)\n",
               comm_time, 100.0 * comm_time / total);
        printf("RMS magnitude    : %.6f\n", rms(data, local_rows * N));
    }

    /* Cleanup */
    fftw_destroy_plan(plan_row);
    fftw_destroy_plan(plan_col);
    fftw_free(data);
    fftw_free(transposed);

    MPI_Finalize();
    return 0;
}