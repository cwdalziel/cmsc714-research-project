#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// ---------------------------
// 1D Z DECOMPOSITION - RING (periodic Z via MPI; periodic X/Y locally)
// Same global periodic 7-point stencil as Cartesian torus when NX,NY,NZ match.
// Usage: prog [N] | prog NX NY NZ   (default N = 256 cubic if no args)
// ---------------------------

constexpr double W_CENTER = 0.5;
constexpr double W_NEIGHBOR = 0.5 / 6.0;

constexpr int ITERATIONS = 30;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int NX = 256;
    int NY = 256;
    int NZ = 256;
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
        if (rank == 0)
            std::cerr << "Error: NX, NY, NZ must be positive.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (NZ % size != 0) {
        if (rank == 0)
            std::cerr << "Error: NZ must be divisible by number of ranks.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_nz = NZ / size;
    int z_start = rank * local_nz;

    int rank_below = (rank > 0) ? rank - 1 : size - 1;
    int rank_above = (rank < size - 1) ? rank + 1 : 0;

    int slab_size = NX * NY;

    std::vector<double> grid((local_nz + 2) * slab_size, 0.0);
    std::vector<double> new_grid((local_nz + 2) * slab_size, 0.0);

    auto idx = [&](int z, int y, int x) {
        return z * slab_size + y * NX + x;
    };

    for (int z = 1; z <= local_nz; z++)
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
                grid[idx(z, y, x)] =
                    (double)((z_start + z - 1) + y + x);

    auto apply_stencil = [&](int z, int y, int x) {
        int xm = (x + NX - 1) % NX;
        int xp = (x + 1) % NX;
        int ym = (y + NY - 1) % NY;
        int yp = (y + 1) % NY;
        new_grid[idx(z, y, x)] =
            W_CENTER * grid[idx(z, y, x)] +
            W_NEIGHBOR * grid[idx(z - 1, y, x)] +
            W_NEIGHBOR * grid[idx(z + 1, y, x)] +
            W_NEIGHBOR * grid[idx(z, ym, x)] +
            W_NEIGHBOR * grid[idx(z, yp, x)] +
            W_NEIGHBOR * grid[idx(z, y, xm)] +
            W_NEIGHBOR * grid[idx(z, y, xp)];
    };

    auto step = [&]() {
        MPI_Request reqs[4];

        MPI_Irecv(&grid[idx(0, 0, 0)], slab_size, MPI_DOUBLE,
                  rank_below, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&grid[idx(local_nz + 1, 0, 0)], slab_size, MPI_DOUBLE,
                  rank_above, 1, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(&grid[idx(1, 0, 0)], slab_size, MPI_DOUBLE,
                  rank_below, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&grid[idx(local_nz, 0, 0)], slab_size, MPI_DOUBLE,
                  rank_above, 0, MPI_COMM_WORLD, &reqs[3]);

        if (local_nz >= 3) {
            for (int z = 2; z <= local_nz - 1; z++)
                for (int y = 0; y < NY; y++)
                    for (int x = 0; x < NX; x++)
                        apply_stencil(z, y, x);
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        for (int z = 1; z <= local_nz; z++) {
            if (z >= 2 && z <= local_nz - 1)
                continue;
            for (int y = 0; y < NY; y++)
                for (int x = 0; x < NX; x++)
                    apply_stencil(z, y, x);
        }

        std::swap(grid, new_grid);
    };

    step();

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < ITERATIONS; iter++)
        step();

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "3D stencil time: "
                  << (t_end - t_start)
                  << " s\n";

    MPI_Finalize();
    return 0;
}
