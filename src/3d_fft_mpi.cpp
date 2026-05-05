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
#include <iomanip>
#include <iostream>
/* Initialize 3D data on rank 0 */
/* x is the slowest varying dimension, then z then y, row major format*/
static void init_data_3d(fftw_complex *buf, int N)
{
    for (int x = 0; x < N; x++)
        for (int z = 0; z < N; z++)
            for (int y = 0; y < N; y++) {
                int idx = (x * N + z) * N + y;
                // buf[idx][0] = sin(2.0 * M_PI * z / N) * cos(2.0 * M_PI * y / N);
                // buf[idx][1] = 0.0;
                buf[idx][0] = idx;
                buf[idx][1] = 0.0;
            }
}

void print3dMatrix(fftw_complex *buf, int N, int local_slices)
{
    for (int x = 0; x < local_slices; x++) {
        std::cout << "Slice x=" << x << ":\n";
        for (int z = 0; z < N; z++) {
            std::cout << " [ ";
            for (int y = 0; y < N; y++) {
                int idx = (x * N + z) * N + y;
                std::cout << std::setw(4) << static_cast<float>(buf[idx][0]) << " ";    
            }
            std::cout << " ]" << std::endl;
        }
        std::cout << std::endl;
    }
}

/* 1D FFT along Y-axis on slabs */
static void fft_x_axis(fftw_complex *data, int local_slices, int N)
{
    int rank = 1;
    int howmany = local_slices * N;

    fftw_plan plan = fftw_plan_many_dft(
        1, &N, local_slices * N, 
        data, NULL, 1, N,
        data, NULL, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

/* 1D FFT along Y-axis (middle dimension) */
static void fft_1d_slices(fftw_complex *datain, fftw_complex *dataout, int local_slices, int N)
{
    fftw_plan plan = fftw_plan_many_dft(
        1, &N, local_slices * N,
        datain, NULL, 1, N,
        dataout, NULL, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

/* tanspose slices Y and Z dimensions locally   */
static void transpose_local_yz(fftw_complex *in, fftw_complex *out, int local_slices, int N)
{
    for (int x = 0; x < local_slices; x++)
        for (int z = 0; z < N; z++)
            for (int y = 0; y < N; y++) {
                int in_idx = (x * N + z) * N + y;
                int out_idx = (x * N + y) * N + z;
                out[out_idx][0] = in[in_idx][0];
                out[out_idx][1] = in[in_idx][1];
            }
}


/* All-to-all transpose for 3D - simple block-based redistribution
 * This is self-inverse: calling it twice restores original distribution
 */
static void transpose_alltoall_3d(fftw_complex *in, fftw_complex *out,
                                   int local_slices, int N, int P,
                                   MPI_Comm comm)
{
    int local_size = local_slices * N * N;
    int count_per_rank = local_size / P;
    
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));

    /* Simple block-based packing: send block i to rank i (self-inverse) */
    for (int dest_rank = 0; dest_rank < P; dest_rank++) {
        int block_start = dest_rank * count_per_rank;
        for (int i = 0; i < count_per_rank; i++) {
            send_buf[dest_rank * count_per_rank + i][0] = in[block_start + i][0];
            send_buf[dest_rank * count_per_rank + i][1] = in[block_start + i][1];
        }
    }

    MPI_Alltoall(send_buf, count_per_rank * 2, MPI_DOUBLE,
                 recv_buf, count_per_rank * 2, MPI_DOUBLE,
                 comm);

    memcpy(out, recv_buf, local_size * sizeof(fftw_complex));

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
    size_t local_size = sizeof(fftw_complex) * local_slices * N * N;

    fftw_complex *data = (fftw_complex *)fftw_malloc(local_size);
    fftw_complex *fft1out = (fftw_complex *)fftw_malloc(local_size);  //output of 1D fft along y for slices
    fftw_complex *fft2out = (fftw_complex *)fftw_malloc(local_size);  //output of 1D fft along x for transposed slices
    fftw_complex *data_t = (fftw_complex *)fftw_malloc(local_size);     //transposed slices in slices

    fftw_complex *cube = NULL;
    if(rank == 0){
        int total_size = N * N * N;
        cube = (fftw_complex *)fftw_malloc(total_size * sizeof(fftw_complex));
        #ifdef DEBUG
        printf("Initializing data on rank 0...\n");
        #endif
        init_data_3d(cube, N);
        // print3dMatrix(cube, N, N);  // Skip large print for now
        #ifdef DEBUG
        printf("Scattering data to all ranks...\n");
        #endif
    }
    MPI_Scatter(cube, local_size, MPI_BYTE, data, local_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    if(rank == 0){
        fftw_free(cube);
    }
    #ifdef DEBUG
    printf("Rank %d: After scatter, before FFTs\n", rank);
    fflush(stdout);
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    
    #ifdef DEBUG
    /* Compute input energy for verification via Parseval's theorem */
    double local_input_energy = 0.0;
    for (int i = 0; i < local_slices * N * N; i++) {
        double mag_sq = data[i][0] * data[i][0] + data[i][1] * data[i][1];
        local_input_energy += mag_sq;
    }
    
    if(rank == 1){
        printf("Data after scattering (local slice on rank 0):\n");
        print3dMatrix(data, N, local_slices);
    }
    // }

    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double fft_time = 0.0;
    double comm_time = 0.0;
    double intl_transpose_time = 0.0;
    // /* Stage 1: FFT along Y for slices */
    #ifdef DEBUG
    printf("Rank %d: Starting FFT along Y\n", rank);
    fflush(stdout);
    #endif
    double fft_start = MPI_Wtime();
    fft_1d_slices(data, fft1out, local_slices, N);
    fft_time  += MPI_Wtime() - fft_start;
    double tr_start_time = MPI_Wtime();
    transpose_local_yz(fft1out, data_t, local_slices, N);
    intl_transpose_time += MPI_Wtime() - tr_start_time;
    // printf("FFT along y done on rank %d, now doing FFT along x for transposed slices...\n", rank);
    fft_start = MPI_Wtime();
    fft_1d_slices(data_t, fft2out, local_slices, N);
    fft_time += MPI_Wtime() - fft_start;

    // printf("FFT along y and z done on rank %d, now doing all-to-all transpose...\n", rank);
    /* 2D fft for each slice in each rank is done, now do the all to all transpose.*/
    // if(rank == 0){
    //     print3dMatrix(data, N, local_slices);
    //     print3dMatrix(data_t, N, local_slices);
    // }
    tr_start_time = MPI_Wtime();
    transpose_local_yz(fft2out, data_t, local_slices, N);
    intl_transpose_time += MPI_Wtime() - tr_start_time;

    fftw_complex *recvd = (fftw_complex *)fftw_malloc(local_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_comm_start = MPI_Wtime();
    transpose_alltoall_3d(data_t, recvd, local_slices, N, P, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t_comm_start;
    #ifdef DEBUG
    printf("All-to-all transpose done on rank %d, now doing FFT along x for transposed slices...\n", rank);
    #endif
    // if(rank == 0){
    //     printf("Data after all-to-all transpose (local slice on rank 0):\n");
    //     print3dMatrix(recvd, N, local_slices);
    // }
    //Do the 1D fft along the x axis for the transposed data in slices.
    fftw_complex *fft3out = (fftw_complex *)fftw_malloc(local_size);
    fft_start = MPI_Wtime();
    fft_1d_slices(recvd,fft3out, local_slices, N);
    fft_time += MPI_Wtime() - fft_start;
    double t_end = MPI_Wtime();
    

    #ifdef DEBUG
    printf("FFT along x done on rank %d, now doing inverse all-to-all transpose to restore original distribution...\n", rank);
    /* Inverse all-to-all transpose - using self-inverse block-based redistribution */
    fftw_complex *data_restored = (fftw_complex *)fftw_malloc(local_size);
    t_comm_start = MPI_Wtime();
    transpose_alltoall_3d(fft3out, data_restored, local_slices, N, P, MPI_COMM_WORLD);
    
    
    /* Compute output energy for verification using Parseval's theorem */
    double local_output_energy = 0.0;
    for (int i = 0; i < local_slices * N * N; i++) {
        double mag_sq = data_restored[i][0] * data_restored[i][0] + data_restored[i][1] * data_restored[i][1];
        local_output_energy += mag_sq;
    }
    
    /* Reduce energies across all ranks */
    double global_input_energy, global_output_energy;
    MPI_Allreduce(&local_input_energy, &global_input_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_output_energy, &global_output_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double scale = 1.0 / (N * N * N);
        double parseval_ratio = (global_output_energy * scale) / global_input_energy;
        printf("Parseval verification: output_energy / (N³ * input_energy) = %.6f (should be ~1.0)\n", parseval_ratio);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    

    /* Gather all data to rank 0 for verification */
    fftw_complex *full_data = NULL;
    if (rank == 0)
        full_data = (fftw_complex *)fftw_malloc(N * N * N * sizeof(fftw_complex));

    int num_elements = local_slices * N * N;
    MPI_Gather(data_restored, num_elements, MPI_C_COMPLEX,
               full_data, num_elements, MPI_C_COMPLEX,
               0, MPI_COMM_WORLD);

    /* Calculate RMS magnitude of output */
    double local_rms_mag_sq = 0.0;
    for (int i = 0; i < local_slices * N * N; i++) {
        double mag_sq = data_restored[i][0] * data_restored[i][0] + data_restored[i][1] * data_restored[i][1];
        local_rms_mag_sq += mag_sq;
    }
    
    double global_rms_mag_sq = 0.0;
    MPI_Allreduce(&local_rms_mag_sq, &global_rms_mag_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double rms_magnitude = sqrt(global_rms_mag_sq / (N * N * N));

    fftw_free(data_restored);
    #endif

    if (rank == 0) {
        printf("FFT computation complete. Energy verification via Parseval's theorem shown above.\n");
        printf("=== 3D FFT MPI Results ===\n");
        printf("N=%d (N³=%d), P=%d\n", N, N*N*N, P);
        printf("Total time: %.6f s\n", t_end - t_start);
        printf("FFT compute time: %.6f s\n", fft_time);
        printf("Alltoall comm: %.6f s\n", comm_time);
        printf("Internal transpose time: %.6f s\n", intl_transpose_time);
        // printf("RMS magnitude: %.6e\n", rms_magnitude);
}

    
    fftw_free(data);
    fftw_free(data_t);

    MPI_Finalize();
    return 0;
}