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

/* Timing structure for all-to-all functions */
struct AlltoallTiming {
    double mpi_time;    /* Time spent in actual MPI calls */
    double memcpy_time; /* Time spent in memcpy operations */
};

  // Enable debug prints for small N
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
                                   MPI_Comm comm, AlltoallTiming *timing)
{
    int local_size = local_slices * N * N;
    int count_per_rank = local_size / P;
    
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));

    /* Simple block-based packing: send block i to rank i (self-inverse) */
    double memcpy_start = MPI_Wtime();
    for (int dest_rank = 0; dest_rank < P; dest_rank++) {
        int block_start = dest_rank * count_per_rank;
        for (int i = 0; i < count_per_rank; i++) {
            send_buf[dest_rank * count_per_rank + i][0] = in[block_start + i][0];
            send_buf[dest_rank * count_per_rank + i][1] = in[block_start + i][1];
        }
    }
    timing->memcpy_time += MPI_Wtime() - memcpy_start;

    double mpi_start = MPI_Wtime();
    MPI_Alltoall(send_buf, count_per_rank * 2, MPI_DOUBLE,
                 recv_buf, count_per_rank * 2, MPI_DOUBLE,
                 comm);
    timing->mpi_time += MPI_Wtime() - mpi_start;

    memcpy_start = MPI_Wtime();
    memcpy(out, recv_buf, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;

    fftw_free(send_buf);
    fftw_free(recv_buf);
}

/* Binary tree / Recursive Doubling All-to-All
 * 
 * Algorithm: Logarithmic stage communication using XOR-based partner selection
 * - Stage k: rank i exchanges with rank (i XOR 2^k)
 * - Creates a hypercube topology of communication
 * 
 * Behavior:
 * 1. O(log P) synchronization stages (vs O(P) for naive staged)
 * 2. Each stage, data volume INCREASES but number of stages DECREASES
 * 3. Stage 0: exchange 1/2 of local data with distance-1 partner
 * 4. Stage 1: exchange 2/3 of local data with distance-2 partner
 * 5. Stage k: exchange k/(k+1) of local data with distance-2^k partner
 * 6. Total messages: P * log(P) (same as naive), but latency is O(log P)
 * 7. More bandwidth-efficient pipelining possible
 * 
 * Example with P=8 ranks:
 *   Stage 0 (XOR 1): 0↔1, 2↔3, 4↔5, 6↔7
 *   Stage 1 (XOR 2): 0↔2, 1↔3, 4↔6, 5↔7  (now have data from groups)
 *   Stage 2 (XOR 4): 0↔4, 1↔5, 2↔6, 3↔7  (now have data from all)
 */
static void alltoall_recursive_doubling(fftw_complex *in, fftw_complex *out,
                                        int local_size, int count_per_rank, int P, int rank,
                                        MPI_Comm comm, AlltoallTiming *timing)
{
    fftw_complex *data = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    double memcpy_start = MPI_Wtime();
    memcpy(data, in, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    fftw_complex *temp_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    
    /* Recursive doubling: O(log P) stages */
    for (int stage = 0; (1 << stage) < P; stage++) {
        int partner = rank ^ (1 << stage);  /* XOR with 2^stage */
        
        if (partner < P) {
            /* Exchange entire current data buffer with partner */
            double mpi_start = MPI_Wtime();
            MPI_Sendrecv(data, local_size, MPI_C_COMPLEX, partner, stage,
                         temp_buf, local_size, MPI_C_COMPLEX, partner, stage,
                         comm, MPI_STATUS_IGNORE);
            timing->mpi_time += MPI_Wtime() - mpi_start;
            
            /* Merge received data - interleave based on XOR pattern */
            /* Rank with lower XOR bit keeps lower chunks, gets higher from partner */
            int bit_value = (rank >> stage) & 1;
            
            memcpy_start = MPI_Wtime();
            if (bit_value == 0) {
                /* Lower half: keep our lower half, replace upper half with partner's */
                memcpy(data + (local_size / 2), 
                       temp_buf + (local_size / 2), 
                       (local_size / 2) * sizeof(fftw_complex));
            } else {
                /* Upper half: replace lower half with partner's, keep our upper half */
                memcpy(data, 
                       temp_buf, 
                       (local_size / 2) * sizeof(fftw_complex));
            }
            timing->memcpy_time += MPI_Wtime() - memcpy_start;
        }
    }
    
    memcpy_start = MPI_Wtime();
    memcpy(out, data, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    fftw_free(data);
    fftw_free(temp_buf);
}

/* Naive all-to-all: Staged communication with no optimization
 * 
 * Algorithm: Completely sequential exchange with P stages
 * - Stage k (0 to P-1): Each rank sends block k to destination rank and receives block from source rank
 * - In stage k: rank i sends to rank (i+k)%P and receives from rank (i-k+P)%P
 * - Uses MPI_Sendrecv to avoid deadlock (no buffering of sends)
 * 
 * Behavior:
 * 1. P sequential synchronization barriers (one per stage)
 * 2. Each stage: exactly one message sent/received per rank
 * 3. No pipelining, no overlap between stages
 * 4. Bandwidth utilization: Very poor - only 2 edges active per rank per stage
 *    (1 out, 1 in) but across entire system only P messages active at a time
 * 5. Latency: O(P * link_latency) - must go through all P stages sequentially
 * 6. Total time: P × (latency + (data_size / bandwidth))
 * 
 * Why naive:
 * - No adaptive routing
 * - No pipelining or message batching
 * - No topology awareness
 * - All communication is strictly serialized by stages
 */
static void alltoall_naive(fftw_complex *in, fftw_complex *out,
                           int local_size, int count_per_rank, int P, int rank,
                           MPI_Comm comm, AlltoallTiming *timing)
{
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(count_per_rank * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(count_per_rank * sizeof(fftw_complex));
    
    /* Organize input data: in[block i] contains data destined for rank i */
    fftw_complex *all_data = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    double memcpy_start = MPI_Wtime();
    memcpy(all_data, in, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    /* Initialize output to hold received data */
    fftw_complex *all_output = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    memset(all_output, 0, local_size * sizeof(fftw_complex));
    
    /* P stages of communication - completely serialized */
    for (int stage = 0; stage < P; stage++) {
        int dest_rank = (rank + stage) % P;
        int src_rank = (rank - stage + P) % P;
        
        /* Prepare send data: block that should go to dest_rank */
        int send_block_idx = dest_rank;
        int send_offset = send_block_idx * count_per_rank;
        memcpy_start = MPI_Wtime();
        memcpy(send_buf, all_data + send_offset, count_per_rank * sizeof(fftw_complex));
        timing->memcpy_time += MPI_Wtime() - memcpy_start;
        
        /* Exchange data with specific pair */
        double mpi_start = MPI_Wtime();
        MPI_Sendrecv(send_buf, count_per_rank * 2, MPI_DOUBLE,
                     dest_rank, stage,
                     recv_buf, count_per_rank * 2, MPI_DOUBLE,
                     src_rank, stage,
                     comm, MPI_STATUS_IGNORE);
        timing->mpi_time += MPI_Wtime() - mpi_start;
        
        /* Store received data in correct output position */
        int recv_block_idx = src_rank;
        int recv_offset = recv_block_idx * count_per_rank;
        memcpy_start = MPI_Wtime();
        memcpy(all_output + recv_offset, recv_buf, count_per_rank * sizeof(fftw_complex));
        timing->memcpy_time += MPI_Wtime() - memcpy_start;
    }
    
    memcpy_start = MPI_Wtime();
    memcpy(out, all_output, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    fftw_free(send_buf);
    fftw_free(recv_buf);
    fftw_free(all_data);
    fftw_free(all_output);
}

static void alltoall_ring(fftw_complex *in, fftw_complex *out,
                          int local_size, int count_per_rank, int P, int rank,
                          MPI_Comm comm, AlltoallTiming *timing)
{
    /* The ring algorithm must compute the EXACT SAME permutation as the naive staged version
     * so that calling it twice (forward and inverse) restores original data.
     * 
     * Naive staged does:
     *   Stage k: send block (rank+k)%P to rank (rank+k)%P
     *            receive from rank (rank-k+P)%P into position (rank-k+P)%P
     * 
     * This exchanges data around the ring but visits all pairs.
     */
    
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(count_per_rank * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(count_per_rank * sizeof(fftw_complex));
    
    /* Copy input to temporary buffer for gathering data */
    fftw_complex *all_data = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    double memcpy_start = MPI_Wtime();
    memcpy(all_data, in, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    fftw_complex *all_output = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    memset(all_output, 0, local_size * sizeof(fftw_complex));
    
    /* Ring exchange: P stages, but optimized to use only ring links */
    for (int stage = 0; stage < P; stage++) {
        int dest_rank = (rank + stage) % P;
        int src_rank = (rank - stage + P) % P;
        
        /* Send the block designated for dest_rank */
        int send_block_idx = dest_rank;
        int send_offset = send_block_idx * count_per_rank;
        memcpy_start = MPI_Wtime();
        memcpy(send_buf, all_data + send_offset, count_per_rank * sizeof(fftw_complex));
        timing->memcpy_time += MPI_Wtime() - memcpy_start;
        
        /* Exchange with appropriate partner */
        double mpi_start = MPI_Wtime();
        MPI_Sendrecv(send_buf, count_per_rank * 2, MPI_DOUBLE, dest_rank, stage,
                     recv_buf, count_per_rank * 2, MPI_DOUBLE, src_rank, stage,
                     comm, MPI_STATUS_IGNORE);
        timing->mpi_time += MPI_Wtime() - mpi_start;
        
        /* Receive goes to position src_rank in output (same as naive) */
        int recv_block_idx = src_rank;
        int recv_offset = recv_block_idx * count_per_rank;
        memcpy_start = MPI_Wtime();
        memcpy(all_output + recv_offset, recv_buf, count_per_rank * sizeof(fftw_complex));
        timing->memcpy_time += MPI_Wtime() - memcpy_start;
    }
    
    memcpy_start = MPI_Wtime();
    memcpy(out, all_output, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    fftw_free(send_buf);
    fftw_free(recv_buf);
    fftw_free(all_data);
    fftw_free(all_output);
}

static void alltoall_hypercube(fftw_complex *in, fftw_complex *out,
                               int local_size, int count_per_rank, int P, int rank,
                               MPI_Comm comm, AlltoallTiming *timing)
{
    fftw_complex *send_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    fftw_complex *recv_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    fftw_complex *exchange_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));

    double memcpy_start = MPI_Wtime();
    memcpy(send_buf, in, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;

    int num_steps = 0;
    int temp = P;
    while (temp > 1) { num_steps++; temp >>= 1; }

    for (int step = 0; step < num_steps; step++) {
        int mask     = 1 << step;
        int low_mask = (1 << (step + 1)) - 1;   // bits 0..step
        int partner  = rank ^ mask;

        double mpi_start = MPI_Wtime();
        MPI_Sendrecv(send_buf,    local_size * 2, MPI_DOUBLE, partner, step,
                     exchange_buf, local_size * 2, MPI_DOUBLE, partner, step,
                     comm, MPI_STATUS_IGNORE);
        timing->mpi_time += MPI_Wtime() - mpi_start;

        memcpy_start = MPI_Wtime();
        for (int i = 0; i < P; i++) {
            if ((i & low_mask) == (rank & low_mask)) {
                // Block i belongs to our "neighborhood" — keep our copy
                memcpy(recv_buf + i * count_per_rank,
                       send_buf  + i * count_per_rank,
                       count_per_rank * sizeof(fftw_complex));
            } else {
                // Take it from the partner
                memcpy(recv_buf + i * count_per_rank,
                       exchange_buf + i * count_per_rank,
                       count_per_rank * sizeof(fftw_complex));
            }
        }
        timing->memcpy_time += MPI_Wtime() - memcpy_start;

        memcpy_start = MPI_Wtime();
        memcpy(send_buf, recv_buf, local_size * sizeof(fftw_complex));
        timing->memcpy_time += MPI_Wtime() - memcpy_start;
    }

    memcpy_start = MPI_Wtime();
    memcpy(out, send_buf, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    fftw_free(exchange_buf);
    fftw_free(send_buf);
    fftw_free(recv_buf);
}

static void alltoall_fattree(fftw_complex *in, fftw_complex *out,
                             int local_size, int count_per_rank, int P, int rank,
                             MPI_Comm comm, AlltoallTiming *timing)
{
    double mpi_start = MPI_Wtime();
    MPI_Alltoall(in,  count_per_rank * 2, MPI_DOUBLE,
                 out, count_per_rank * 2, MPI_DOUBLE,
                 comm);
    timing->mpi_time += MPI_Wtime() - mpi_start;
}


static void alltoall_torus(fftw_complex *in, fftw_complex *out,
                           int local_size, int count_per_rank, int P, int rank,
                           MPI_Comm comm, AlltoallTiming *timing)
{
    // For 16 nodes: decompose as 4x4 mesh (matching 2x2x4 torus)
    // This is topology-specific; in practice, query MPI for actual dimensions
    
    int P1 = 4, P2 = 4;  // For 16 nodes
    if (P == 8) { P1 = 2; P2 = 4; }      // For 8 nodes
    if (P == 32) { P1 = 4; P2 = 8; }     // For 32 nodes
    if (P == 64) { P1 = 8; P2 = 8; }     // For 64 nodes
    if (P == 128) { P1 = 8; P2 = 16; }   // For 128 nodes
    
    int rank1 = rank / P2;
    int rank2 = rank % P2;
    
    fftw_complex *temp_buf = (fftw_complex *)fftw_malloc(local_size * sizeof(fftw_complex));
    
    // After dim1 alltoall, block at position i within comm1 belongs to
    // global rank (i * P2 + rank2). Repack to global order for dim2 pass.
    double memcpy_start = MPI_Wtime();
    for (int i = 0; i < P1; i++) {
        int global_src = i * P2 + rank2;
        memcpy(temp_buf + global_src * count_per_rank,
               out + i * count_per_rank,
               count_per_rank * sizeof(fftw_complex));
    }
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    // Step 1: All-to-all along dimension 1 (fix rank2, vary rank1)
    {
        MPI_Comm comm1;
        MPI_Comm_split(comm, rank2, rank1, &comm1);
        double mpi_start = MPI_Wtime();
        MPI_Alltoall(temp_buf, count_per_rank * 2, MPI_DOUBLE,
                     out, count_per_rank * 2, MPI_DOUBLE,
                     comm1);
        timing->mpi_time += MPI_Wtime() - mpi_start;
        MPI_Comm_free(&comm1);
    }
    
    memcpy_start = MPI_Wtime();
    memcpy(temp_buf, out, local_size * sizeof(fftw_complex));
    timing->memcpy_time += MPI_Wtime() - memcpy_start;
    
    // Step 2: All-to-all along dimension 2 (fix rank1, vary rank2)
    {
        MPI_Comm comm2;
        MPI_Comm_split(comm, rank1, rank2, &comm2);
        double mpi_start = MPI_Wtime();
        MPI_Alltoall(temp_buf, count_per_rank * 2, MPI_DOUBLE,
                     out, count_per_rank * 2, MPI_DOUBLE,
                     comm2);
        timing->mpi_time += MPI_Wtime() - mpi_start;
        MPI_Comm_free(&comm2);
    }
    
    fftw_free(temp_buf);
}

static void alltoall_dragonfly_indirect(fftw_complex *in, fftw_complex *out,
                                        int local_size, int count_per_rank, int P, int rank,
                                        MPI_Comm comm)
{
    int groups          = (int)round(sqrt((double)P));
    int nodes_per_group = P / groups;

    int my_group   = rank / nodes_per_group;
    int local_rank = rank % nodes_per_group;

    /* Each rank is responsible for forwarding data to one destination group.
       Rank local_rank within a group is the "ambassador" for group local_rank. */
    int my_dest_group = local_rank % groups;

    int intragroup_size = nodes_per_group * count_per_rank;
    int intergroup_size = groups          * count_per_rank;

    fftw_complex *phase1_out = fftw_malloc(intragroup_size * sizeof(fftw_complex));
    fftw_complex *phase2_out = fftw_malloc(intergroup_size * sizeof(fftw_complex));
    fftw_complex *phase2_in  = fftw_malloc(intergroup_size * sizeof(fftw_complex));

    /* ----------------------------------------------------------------
       Phase 1: intra-group all-to-all
       Goal: rank local_rank r ends up holding all data that needs to
       cross to group (r % groups). Each rank sends count_per_rank
       elements to every peer in its group.
    ---------------------------------------------------------------- */
    {
        MPI_Comm group_comm;
        MPI_Comm_split(comm, my_group, local_rank, &group_comm);
        MPI_Alltoall(in,         count_per_rank * 2, MPI_DOUBLE,
                     phase1_out, count_per_rank * 2, MPI_DOUBLE,
                     group_comm);
        MPI_Comm_free(&group_comm);
    }

    /* Pack into phase2_in: collect the blocks from phase1_out that are
       destined for each group g. After phase 1, slot i in phase1_out
       came from local peer i within my_group. That peer originally held
       data for global ranks across all groups. We want to forward to
       group my_dest_group, so pack the block from each local peer i
       that is destined for (my_dest_group * nodes_per_group + i).     */
    for (int g = 0; g < groups; g++) {
        /* For global link to group g, take the block from local peer
           (g % nodes_per_group) — this is the intra-group slot that
           holds data originally destined for group g.                 */
        int local_slot = g % nodes_per_group;
        memcpy(phase2_in + g * count_per_rank,
               phase1_out + local_slot * count_per_rank,
               count_per_rank * sizeof(fftw_complex));
    }

    /* ----------------------------------------------------------------
       Phase 2: inter-group all-to-all
       One rank per group exchanges across the single global link.
       comm is split so that rank local_rank r talks to its counterpart
       in every other group — this is the "ambassador" pattern.
    ---------------------------------------------------------------- */
    {
        MPI_Comm intergroup_comm;
        MPI_Comm_split(comm, local_rank, my_group, &intergroup_comm);
        MPI_Alltoall(phase2_in,  count_per_rank * 2, MPI_DOUBLE,
                     phase2_out, count_per_rank * 2, MPI_DOUBLE,
                     intergroup_comm);
        MPI_Comm_free(&intergroup_comm);
    }

    /* ----------------------------------------------------------------
       Phase 3: intra-group all-to-all for final delivery
       After phase 2, rank r in group g holds data from group r for all
       local peers. Redistribute within group so each rank gets the
       elements meant for it.
    ---------------------------------------------------------------- */
    {
        MPI_Comm group_comm;
        MPI_Comm_split(comm, my_group, local_rank, &group_comm);

        fftw_complex *phase3_out = fftw_malloc(intragroup_size * sizeof(fftw_complex));
        MPI_Alltoall(phase2_out, count_per_rank * 2, MPI_DOUBLE,
                     phase3_out, count_per_rank * 2, MPI_DOUBLE,
                     group_comm);
        MPI_Comm_free(&group_comm);

        /* Repack into output indexed by global rank:
           slot i in phase3_out came from local peer i,
           global rank = my_group * nodes_per_group + i              */
        for (int i = 0; i < nodes_per_group; i++) {
            int global_rank = my_group * nodes_per_group + i;
            memcpy(out + global_rank * count_per_rank,
                   phase3_out + i * count_per_rank,
                   count_per_rank * sizeof(fftw_complex));
        }

        fftw_free(phase3_out);
    }

    fftw_free(phase1_out);
    fftw_free(phase2_in);
    fftw_free(phase2_out);
}


/* Topology selector - dispatches to appropriate implementation
 * Set TOPOLOGY_STRATEGY environment variable or detect from platform file
 */
enum TopologyType {
    TOPOLOGY_NAIVE_STAGED,       // Naive staged: completely sequential, no optimization (O(P) stages)
    TOPOLOGY_RECURSIVE_DOUBLING, // Binary tree / recursive doubling: logarithmic stages (O(log P))
    TOPOLOGY_NAIVE_MPI,          // Baseline: standard MPI_Alltoall (optimized by MPI library)
    TOPOLOGY_RING,               // Ring optimization
    TOPOLOGY_HYPERCUBE,          // Hypercube optimization
    TOPOLOGY_FATTREE,            // Fat-tree optimization
    TOPOLOGY_TORUS,              // Torus optimization
    TOPOLOGY_DRAGONFLY           // Dragonfly optimization
};

static TopologyType detect_topology()
{
    const char *topo_str = getenv("TOPOLOGY_STRATEGY");
    if (!topo_str) return TOPOLOGY_NAIVE_MPI;
    
    if (strcmp(topo_str, "naive_staged") == 0) return TOPOLOGY_NAIVE_STAGED;
    if (strcmp(topo_str, "recursive_doubling") == 0) return TOPOLOGY_RECURSIVE_DOUBLING;
    if (strcmp(topo_str, "ring") == 0) return TOPOLOGY_RING;
    if (strcmp(topo_str, "hypercube") == 0) return TOPOLOGY_HYPERCUBE;
    if (strcmp(topo_str, "fat_tree") == 0) return TOPOLOGY_FATTREE;
    if (strcmp(topo_str, "torus") == 0) return TOPOLOGY_TORUS;
    if (strcmp(topo_str, "dragonfly") == 0) return TOPOLOGY_DRAGONFLY;
    
    return TOPOLOGY_NAIVE_MPI;
}

static void transpose_alltoall_3d_optimized(fftw_complex *in, fftw_complex *out,
                                             int local_slices, int N, int P,
                                             TopologyType topology,
                                             MPI_Comm comm, AlltoallTiming *timing)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    int local_size = local_slices * N * N;
    int count_per_rank = local_size / P;
    
    switch (topology) {
        case TOPOLOGY_NAIVE_STAGED:
            if (rank == 0) printf("[Naive Staged] Using completely unoptimized staged all-to-all (P sequential stages)\n");
            alltoall_naive(in, out, local_size, count_per_rank, P, rank, comm, timing);
            break;
        case TOPOLOGY_RECURSIVE_DOUBLING:
            if (rank == 0) printf("[Recursive Doubling] Using binary tree all-to-all (log P stages)\n");
            alltoall_recursive_doubling(in, out, local_size, count_per_rank, P, rank, comm, timing);
            break;
        case TOPOLOGY_RING:
            if (rank == 0) printf("[Optimization] Using RING topology strategy\n");
            alltoall_ring(in, out, local_size, count_per_rank, P, rank, comm, timing);
            if(rank == 0){
            printf("Using RING topology strategy\n");
            }
            break;
        case TOPOLOGY_HYPERCUBE:
            if (rank == 0) printf("Using HYPERCUBE topology strategy increased latency but decreased traffic\n");
            alltoall_hypercube(in, out, local_size, count_per_rank, P, rank, comm, timing);
            break;
        case TOPOLOGY_FATTREE:
            if (rank == 0) printf("[Optimization] Using FAT-TREE topology strategy\n");
            alltoall_fattree(in, out, local_size, count_per_rank, P, rank, comm, timing);
            break;
        case TOPOLOGY_TORUS:
            if (rank == 0) printf("[Optimization] Using TORUS topology strategy\n");
            alltoall_torus(in, out, local_size, count_per_rank, P, rank, comm, timing);
            break;
        // case TOPOLOGY_DRAGONFLY:
            if (rank == 0) printf("[Optimization] Using DRAGONFLY topology strategy\n");
            alltoall_dragonfly_indirect(in, out, local_size, count_per_rank, P, rank, comm);
            break;
        case TOPOLOGY_NAIVE_MPI:
        default:
            if (rank == 0) printf("[Optimized Library] Using standard MPI_Alltoall (library-optimized)\n");
            transpose_alltoall_3d(in, out, local_slices, N, P, comm, timing);
            break;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int N = (argc > 1) ? atoi(argv[1]) : 64;
    
    // Detect and select topology-aware optimization strategy
    TopologyType topology = detect_topology();

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
    
    /* Compute input energy for verification via Parseval's theorem */
    double local_input_energy = 0.0;
    for (int i = 0; i < local_slices * N * N; i++) {
        double mag_sq = data[i][0] * data[i][0] + data[i][1] * data[i][1];
        local_input_energy += mag_sq;
    }
    
    if(rank == 1){
        // printf("Data after scattering (local slice on rank 0):\n");
        // print3dMatrix(data, N, local_slices);
    }
    // }

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
    double t_end = MPI_Wtime();

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
    
    AlltoallTiming alltoall_timing = {0.0, 0.0};
    double t_comm_start = MPI_Wtime();
    transpose_alltoall_3d_optimized(data_t, recvd, local_slices, N, P, topology, MPI_COMM_WORLD, &alltoall_timing);
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
    
    /* Compute output energy for Parseval verification DIRECTLY from FFT output
     * (before inverse alltoall). This tests the FFT's energy preservation,
     * independent of alltoall permutation correctness.
     * Energy should be preserved by FFT regardless of data rearrangement.
     */
    double local_output_energy = 0.0;
    for (int i = 0; i < local_slices * N * N; i++) {
        double mag_sq = fft3out[i][0] * fft3out[i][0] + fft3out[i][1] * fft3out[i][1];
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
    
    // printf("FFT along x done on rank %d, now doing inverse all-to-all transpose to restore original distribution...\n", rank);
    /* Inverse all-to-all transpose - using self-inverse block-based redistribution */
    // fftw_complex *data_restored = (fftw_complex *)fftw_malloc(local_size);
    // transpose_alltoall_3d_optimized(fft3out, data_restored, local_slices, N, P, topology, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("FFT computation complete. Energy verification via Parseval's theorem shown above.\n");
        printf("=== 3D FFT MPI Results ===\n");
        printf("N=%d (N³=%d), P=%d\n", N, N*N*N, P);
        printf("Total time: %.6f s\n", t_end - t_start);
        printf("FFT compute time: %.6f s\n", fft_time);
        printf("Alltoall comm (total): %.6f s\n", comm_time);
        printf("  - MPI call time: %.6f s\n", alltoall_timing.mpi_time);
        printf("  - Memcpy time: %.6f s\n", alltoall_timing.memcpy_time);
        printf("Internal transpose time: %.6f s\n", intl_transpose_time);
        // printf("RMS magnitude: %.6e\n", rms_magnitude);
}

    
    fftw_free(data);
    fftw_free(data_t);

    MPI_Finalize();
    return 0;
}