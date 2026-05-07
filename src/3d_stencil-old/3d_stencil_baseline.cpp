#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

// BASELINE (TOPOLOGY-AGNOSTIC) 3D STENCIL:
// Generic periodic 3D Cartesian decomposition with MPI_Dims_create and no topology-specific
// rank remapping, pinning, or topology-aware communicator keys.
// Usage: prog [N] | prog NX NY NZ   (default N = 256 cubic if no args)

constexpr double W_CENTER   = 0.5;
constexpr double W_NEIGHBOR = 0.5 / 6.0;
constexpr int ITERATIONS = 30;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int NX = 256, NY = 256, NZ = 256;
    if (argc == 2) {
        NX = NY = NZ = std::atoi(argv[1]);
    } else if (argc == 4) {
        NX = std::atoi(argv[1]);
        NY = std::atoi(argv[2]);
        NZ = std::atoi(argv[3]);
    } else if (argc != 1) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " [N]\n       "
                      << argv[0] << " NX NY NZ\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (NX <= 0 || NY <= 0 || NZ <= 0) {
        if (rank == 0) std::cerr << "Error: NX, NY, NZ must be positive.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[3]    = {0, 0, 0};
    int periods[3] = {1, 1, 1};
    MPI_Dims_create(size, 3, dims);

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int coords[3];
    MPI_Cart_get(cart_comm, 3, dims, periods, coords);

    if (NX % dims[0] != 0 || NY % dims[1] != 0 || NZ % dims[2] != 0) {
        if (rank == 0)
            std::cerr << "Error: NX/NY/NZ must be divisible by Cartesian dims.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank_xdown, rank_xup, rank_ydown, rank_yup, rank_zdown, rank_zup;
    MPI_Cart_shift(cart_comm, 0, 1, &rank_xdown, &rank_xup);
    MPI_Cart_shift(cart_comm, 1, 1, &rank_ydown, &rank_yup);
    MPI_Cart_shift(cart_comm, 2, 1, &rank_zdown, &rank_zup);

    int local_nx = NX / dims[0];
    int local_ny = NY / dims[1];
    int local_nz = NZ / dims[2];
    int x_start = coords[0] * local_nx;
    int y_start = coords[1] * local_ny;
    int z_start = coords[2] * local_nz;

    int halo_nx = local_nx + 2, halo_ny = local_ny + 2, halo_nz = local_nz + 2;
    std::vector<double> grid(halo_nx * halo_ny * halo_nz, 0.0);
    std::vector<double> new_grid(halo_nx * halo_ny * halo_nz, 0.0);

    auto idx = [&](int x, int y, int z) {
        return x * halo_ny * halo_nz + y * halo_nz + z;
    };

    for (int x = 1; x <= local_nx; x++)
        for (int y = 1; y <= local_ny; y++)
            for (int z = 1; z <= local_nz; z++)
                grid[idx(x, y, z)] = (double)((x_start + x - 1) + (y_start + y - 1) + (z_start + z - 1));

    int yz_plane = local_ny * local_nz, xz_plane = local_nx * local_nz, xy_plane = local_nx * local_ny;
    std::vector<double> send_xdown(yz_plane), recv_xdown(yz_plane), send_xup(yz_plane), recv_xup(yz_plane);
    std::vector<double> send_ydown(xz_plane), recv_ydown(xz_plane), send_yup(xz_plane), recv_yup(xz_plane);

    auto apply_stencil = [&](int x, int y, int z) {
        new_grid[idx(x, y, z)] =
            W_CENTER * grid[idx(x, y, z)] +
            W_NEIGHBOR * grid[idx(x - 1, y, z)] +
            W_NEIGHBOR * grid[idx(x + 1, y, z)] +
            W_NEIGHBOR * grid[idx(x, y - 1, z)] +
            W_NEIGHBOR * grid[idx(x, y + 1, z)] +
            W_NEIGHBOR * grid[idx(x, y, z - 1)] +
            W_NEIGHBOR * grid[idx(x, y, z + 1)];
    };

    auto step = [&]() {
        MPI_Request reqs[12];
        int nreqs = 0;

        MPI_Irecv(&grid[idx(1, 1, 0)],          xy_plane, MPI_DOUBLE, rank_zdown, 0, cart_comm, &reqs[nreqs++]);
        MPI_Irecv(&grid[idx(1, 1, local_nz+1)], xy_plane, MPI_DOUBLE, rank_zup,   1, cart_comm, &reqs[nreqs++]);
        MPI_Isend(&grid[idx(1, 1, 1)],          xy_plane, MPI_DOUBLE, rank_zdown, 1, cart_comm, &reqs[nreqs++]);
        MPI_Isend(&grid[idx(1, 1, local_nz)],   xy_plane, MPI_DOUBLE, rank_zup,   0, cart_comm, &reqs[nreqs++]);

        for (int y = 0; y < local_ny; y++)
            for (int z = 0; z < local_nz; z++) {
                send_xdown[y * local_nz + z] = grid[idx(1, y + 1, z + 1)];
                send_xup[y * local_nz + z]   = grid[idx(local_nx, y + 1, z + 1)];
            }
        MPI_Irecv(recv_xdown.data(), yz_plane, MPI_DOUBLE, rank_xdown, 2, cart_comm, &reqs[nreqs++]);
        MPI_Irecv(recv_xup.data(),   yz_plane, MPI_DOUBLE, rank_xup,   3, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_xdown.data(), yz_plane, MPI_DOUBLE, rank_xdown, 3, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_xup.data(),   yz_plane, MPI_DOUBLE, rank_xup,   2, cart_comm, &reqs[nreqs++]);

        for (int x = 0; x < local_nx; x++)
            for (int z = 0; z < local_nz; z++) {
                send_ydown[x * local_nz + z] = grid[idx(x + 1, 1, z + 1)];
                send_yup[x * local_nz + z]   = grid[idx(x + 1, local_ny, z + 1)];
            }
        MPI_Irecv(recv_ydown.data(), xz_plane, MPI_DOUBLE, rank_ydown, 4, cart_comm, &reqs[nreqs++]);
        MPI_Irecv(recv_yup.data(),   xz_plane, MPI_DOUBLE, rank_yup,   5, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_ydown.data(), xz_plane, MPI_DOUBLE, rank_ydown, 5, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_yup.data(),   xz_plane, MPI_DOUBLE, rank_yup,   4, cart_comm, &reqs[nreqs++]);

        if (local_nx >= 3 && local_ny >= 3 && local_nz >= 3) {
            for (int x = 2; x <= local_nx - 1; x++)
                for (int y = 2; y <= local_ny - 1; y++)
                    for (int z = 2; z <= local_nz - 1; z++) apply_stencil(x, y, z);
        }

        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        for (int y = 0; y < local_ny; y++)
            for (int z = 0; z < local_nz; z++) {
                grid[idx(0, y + 1, z + 1)] = recv_xdown[y * local_nz + z];
                grid[idx(local_nx + 1, y + 1, z + 1)] = recv_xup[y * local_nz + z];
            }
        for (int x = 0; x < local_nx; x++)
            for (int z = 0; z < local_nz; z++) {
                grid[idx(x + 1, 0, z + 1)] = recv_ydown[x * local_nz + z];
                grid[idx(x + 1, local_ny + 1, z + 1)] = recv_yup[x * local_nz + z];
            }
        for (int x = 1; x <= local_nx; x++)
            for (int y = 1; y <= local_ny; y++)
                for (int z = 1; z <= local_nz; z++) {
                    if (x >= 2 && x <= local_nx - 1 && y >= 2 && y <= local_ny - 1 && z >= 2 && z <= local_nz - 1)
                        continue;
                    apply_stencil(x, y, z);
                }
        std::swap(grid, new_grid);
    };

    step();
    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) step();
    MPI_Barrier(cart_comm);
    double t_end = MPI_Wtime();

    if (rank == 0) std::cout << "3D stencil time: " << (t_end - t_start) << " s\n";

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
