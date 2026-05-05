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


/* All-to-all transpose for 3D (exchange slabs between processes) */
static void transpose_alltoall_3d(fftw_complex *in, fftw_complex *out,
                                   int local_slices, int N, int P,
                                   MPI_Comm comm)
{
    int local_size = local_slices * N * N;
    int count_per_rank = local_size / P;  /* elements per destination rank */
    
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));

    /* Pack send buffer: reorganize so data for rank r is contiguous */
    for (int dest_rank = 0; dest_rank < P; dest_rank++) {
        int send_offset = dest_rank * count_per_rank;
        
        /* Collect count_per_rank elements destined for this rank */
        int elem_count = 0;
        for (int src_idx = 0; src_idx < local_size && elem_count < count_per_rank; src_idx++) {
            /* Simple distribution: modulo mapping */
            if (src_idx % P == dest_rank) {
                send_buf[send_offset + elem_count][0] = in[src_idx][0];
                send_buf[send_offset + elem_count][1] = in[src_idx][1];
                elem_count++;
            }
        }
    }

    /* MPI_Alltoall exchange (count is in elements, multiply by 2 for complex) */
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

    fftw_complex *cube;
    if(rank == 0){
        int total_size = N * N * N;
        cube = (fftw_complex *)fftw_malloc(total_size * sizeof(fftw_complex));
        printf("Initializing data on rank 0...\n");
        init_data_3d(cube, N);
        print3dMatrix(cube, N,N);
        printf("Scattering data to all ranks...\n");
        
    }
    MPI_Scatter(cube, local_size, MPI_BYTE, data, local_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    if(rank == 0){
        fftw_free(cube);
    }
    
    if(rank == 1){
        // printf("Data after scattering (local slice on rank 0):\n");
        // print3dMatrix(data, N, local_slices);
    }
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // /* Stage 1: FFT along Y for slices */
    fft_1d_slices(data, fft1out, local_slices, N);
    transpose_local_yz(fft1out, data_t, local_slices, N);
    // printf("FFT along y done on rank %d, now doing FFT along x for transposed slices...\n", rank);
    fft_1d_slices(data_t, fft2out, local_slices, N);

    // printf("FFT along y and z done on rank %d, now doing all-to-all transpose...\n", rank);
    /* 2D fft for each slice in each rank is done, now do the all to all transpose.*/
    // if(rank == 0){
    //     print3dMatrix(data, N, local_slices);
    //     print3dMatrix(data_t, N, local_slices);
    // }
    transpose_local_yz(fft2out, data_t, local_slices, N);
    fftw_complex *recvd = (fftw_complex *)fftw_malloc(local_size);
    transpose_alltoall_3d(data_t, recvd, local_slices, N, P, MPI_COMM_WORLD);
    printf("All-to-all transpose done on rank %d, now doing FFT along x for transposed slices...\n", rank);
    // if(rank == 0){
    //     printf("Data after all-to-all transpose (local slice on rank 0):\n");
    //     print3dMatrix(recvd, N, local_slices);
    // }
    //Do the 1D fft along the x axis for the transposed data in slices.
    fftw_complex *fft3out = (fftw_complex *)fftw_malloc(local_size);
    fft_1d_slices(recvd,fft3out, local_slices, N); //Error

    printf("FFT along x done on rank %d, now doing inverse all-to-all transpose to restore original distribution...\n", rank);
    /* Inverse all-to-all transpose to restore original X distribution */
    fftw_complex *data_restored = (fftw_complex *)fftw_malloc(local_size);
    transpose_alltoall_3d(fft3out, data_restored, local_slices, N, P, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    /* Gather all data to rank 0 for verification */
    fftw_complex *full_data = NULL;
    if (rank == 0)
        full_data = (fftw_complex *)fftw_malloc(N * N * N * sizeof(fftw_complex));

    int num_elements = local_slices * N * N;
        MPI_Gather(data_restored, num_elements, MPI_C_COMPLEX,
           full_data, num_elements, MPI_C_COMPLEX,
           0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Performing inverse FFT for verification...\n");
        fftw_plan plan_inv = fftw_plan_dft_3d(N, N, N, full_data, full_data, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan_inv);
        fftw_destroy_plan(plan_inv);

        /* Scale by 1/N³ */
        double scale = 1.0 / (N * N * N);
        double error = 0.0;
        for (int i = 0; i < N * N * N; i++) {
            full_data[i][0] *= scale;
            error += pow(full_data[i][0] - i, 2) + pow(full_data[i][1], 2);
        }
        
        printf("Round-trip error: %.2e (should be < 1e-10)\n", sqrt(error / (N*N*N)));
        printf("=== 3D FFT MPI Results ===\n");
        printf("N=%d (N³=%d), P=%d\n", N, N*N*N, P);
        printf("Total time: %.6f s\n", t_end - t_start);
        fftw_free(full_data);
    }

    fftw_free(data_restored);
    fftw_free(data);
    fftw_free(data_t);

    MPI_Finalize();
    return 0;
}