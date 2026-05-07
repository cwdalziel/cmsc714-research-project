#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// RING OPTIMIZATIONS VS A GENERIC 3D STENCIL:
// 1) Parse node-* host ids and reorder ranks into physical ring order.
// 2) Build a 3D periodic Cartesian communicator on that ring-ordered rank set.
// 3) Use a ring-focused dims search that prefers a long ring-contiguous axis while
//    keeping local blocks reasonably balanced (vs pure 1D slab).
// 4) Overlap interior compute with nonblocking 6-face halo exchange every iteration.
// Usage: prog [N] | prog NX NY NZ   (default N = 256 cubic if no args)

constexpr double W_CENTER = 0.5;
constexpr double W_NEIGHBOR = 0.5 / 6.0;

constexpr int ITERATIONS = 30;

static int extract_node_id(const char* name) {
    const std::string s(name ? name : "");
    const std::string needle = "node-";
    const size_t p = s.find(needle);
    if (p == std::string::npos) return -1;
    size_t i = p + needle.size();
    int v = 0;
    bool any = false;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        any = true;
        v = v * 10 + (s[i] - '0');
        ++i;
    }
    return any ? v : -1;
}

static void choose_ring_dims(int size, int NX, int NY, int NZ, int dims[3]) {
    int best0 = 1, best1 = 1, best2 = size;
    int best_d2 = -1;
    double best_score = std::numeric_limits<double>::infinity();

    for (int d0 = 1; d0 <= size; ++d0) {
        if (size % d0 != 0) continue;
        int rem = size / d0;
        for (int d1 = 1; d1 <= rem; ++d1) {
            if (rem % d1 != 0) continue;
            int d2 = rem / d1;
            if (NX % d0 || NY % d1 || NZ % d2) continue;

            double lx = static_cast<double>(NX) / d0;
            double ly = static_cast<double>(NY) / d1;
            double lz = static_cast<double>(NZ) / d2;
            double surface = (ly * lz) + (lx * lz) + (lx * ly);
            double balance = std::max({lx, ly, lz}) / std::min({lx, ly, lz});
            double score = surface + 0.3 * balance;

            if (d2 > best_d2 || (d2 == best_d2 && score < best_score)) {
                best_d2 = d2;
                best_score = score;
                best0 = d0; best1 = d1; best2 = d2;
            }
        }
    }
    dims[0] = best0; dims[1] = best1; dims[2] = best2;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int proc_len = 0;
    MPI_Get_processor_name(proc_name, &proc_len);
    int node_id = extract_node_id(proc_name);
    if (node_id < 0) node_id = world_rank;

    MPI_Comm ring_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, node_id, &ring_comm);

    int rank;
    MPI_Comm_rank(ring_comm, &rank);

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
        if (world_rank == 0)
            std::cerr << "Usage: " << argv[0] << " [N]\n       "
                      << argv[0] << " NX NY NZ\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (NX <= 0 || NY <= 0 || NZ <= 0) {
        if (world_rank == 0)
            std::cerr << "Error: NX, NY, NZ must be positive.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[3] = {0, 0, 0};
    choose_ring_dims(size, NX, NY, NZ, dims);
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        if (rank == 0)
            std::cerr << "Error: could not choose valid ring decomposition dims.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int periods[3] = {1, 1, 1};
    MPI_Comm cart_comm;
    // Keep reorder=0 so coordinate order follows ring-ordered rank sequence.
    MPI_Cart_create(ring_comm, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_get(cart_comm, 3, dims, periods, coords);

    int local_nx = NX / dims[0];
    int local_ny = NY / dims[1];
    int local_nz = NZ / dims[2];
    int x_start = coords[0] * local_nx;
    int y_start = coords[1] * local_ny;
    int z_start = coords[2] * local_nz;

    int rank_xdown, rank_xup, rank_ydown, rank_yup, rank_zdown, rank_zup;
    MPI_Cart_shift(cart_comm, 0, 1, &rank_xdown, &rank_xup);
    MPI_Cart_shift(cart_comm, 1, 1, &rank_ydown, &rank_yup);
    MPI_Cart_shift(cart_comm, 2, 1, &rank_zdown, &rank_zup);

    int halo_nx = local_nx + 2;
    int halo_ny = local_ny + 2;
    int halo_nz = local_nz + 2;

    std::vector<double> grid(halo_nx * halo_ny * halo_nz, 0.0);
    std::vector<double> new_grid(halo_nx * halo_ny * halo_nz, 0.0);

    auto idx = [&](int x, int y, int z) {
        return x * halo_ny * halo_nz + y * halo_nz + z;
    };

    for (int x = 1; x <= local_nx; x++)
        for (int y = 1; y <= local_ny; y++)
            for (int z = 1; z <= local_nz; z++)
                grid[idx(x, y, z)] =
                    (double)((x_start + x - 1) + (y_start + y - 1) + (z_start + z - 1));

    int yz_plane = local_ny * local_nz;
    int xz_plane = local_nx * local_nz;
    int xy_plane = local_nx * local_ny;
    std::vector<double> send_xdown(yz_plane), recv_xdown(yz_plane);
    std::vector<double> send_xup  (yz_plane), recv_xup  (yz_plane);
    std::vector<double> send_ydown(xz_plane), recv_ydown(xz_plane);
    std::vector<double> send_yup  (xz_plane), recv_yup  (xz_plane);

    auto apply_stencil = [&](int x, int y, int z) {
        new_grid[idx(x, y, z)] =
            W_CENTER   * grid[idx(x,   y,   z  )] +
            W_NEIGHBOR * grid[idx(x-1, y,   z  )] +
            W_NEIGHBOR * grid[idx(x+1, y,   z  )] +
            W_NEIGHBOR * grid[idx(x,   y-1, z  )] +
            W_NEIGHBOR * grid[idx(x,   y+1, z  )] +
            W_NEIGHBOR * grid[idx(x,   y,   z-1)] +
            W_NEIGHBOR * grid[idx(x,   y,   z+1)];
    };

    auto step = [&]() {
        MPI_Request reqs[12];
        int nreqs = 0;

        MPI_Irecv(&grid[idx(1, 1, 0)],          xy_plane, MPI_DOUBLE,
                  rank_zdown, 0, cart_comm, &reqs[nreqs++]);
        MPI_Irecv(&grid[idx(1, 1, local_nz+1)], xy_plane, MPI_DOUBLE,
                  rank_zup,   1, cart_comm, &reqs[nreqs++]);
        MPI_Isend(&grid[idx(1, 1, 1)],          xy_plane, MPI_DOUBLE,
                  rank_zdown, 1, cart_comm, &reqs[nreqs++]);
        MPI_Isend(&grid[idx(1, 1, local_nz)],   xy_plane, MPI_DOUBLE,
                  rank_zup,   0, cart_comm, &reqs[nreqs++]);

        for (int y = 0; y < local_ny; y++)
            for (int z = 0; z < local_nz; z++) {
                send_xdown[y * local_nz + z] = grid[idx(1, y + 1, z + 1)];
                send_xup  [y * local_nz + z] = grid[idx(local_nx, y + 1, z + 1)];
            }
        MPI_Irecv(recv_xdown.data(), yz_plane, MPI_DOUBLE,
                  rank_xdown, 2, cart_comm, &reqs[nreqs++]);
        MPI_Irecv(recv_xup.data(),   yz_plane, MPI_DOUBLE,
                  rank_xup,   3, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_xdown.data(), yz_plane, MPI_DOUBLE,
                  rank_xdown, 3, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_xup.data(),   yz_plane, MPI_DOUBLE,
                  rank_xup,   2, cart_comm, &reqs[nreqs++]);

        for (int x = 0; x < local_nx; x++)
            for (int z = 0; z < local_nz; z++) {
                send_ydown[x * local_nz + z] = grid[idx(x + 1, 1, z + 1)];
                send_yup  [x * local_nz + z] = grid[idx(x + 1, local_ny, z + 1)];
            }
        MPI_Irecv(recv_ydown.data(), xz_plane, MPI_DOUBLE,
                  rank_ydown, 4, cart_comm, &reqs[nreqs++]);
        MPI_Irecv(recv_yup.data(),   xz_plane, MPI_DOUBLE,
                  rank_yup,   5, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_ydown.data(), xz_plane, MPI_DOUBLE,
                  rank_ydown, 5, cart_comm, &reqs[nreqs++]);
        MPI_Isend(send_yup.data(),   xz_plane, MPI_DOUBLE,
                  rank_yup,   4, cart_comm, &reqs[nreqs++]);

        if (local_nx >= 3 && local_ny >= 3 && local_nz >= 3) {
            for (int x = 2; x <= local_nx - 1; x++)
                for (int y = 2; y <= local_ny - 1; y++)
                    for (int z = 2; z <= local_nz - 1; z++)
                        apply_stencil(x, y, z);
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
                    if (x >= 2 && x <= local_nx - 1 &&
                        y >= 2 && y <= local_ny - 1 &&
                        z >= 2 && z <= local_nz - 1)
                        continue;
                    apply_stencil(x, y, z);
                }

        std::swap(grid, new_grid);
    };

    step();

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < ITERATIONS; iter++)
        step();

    MPI_Barrier(cart_comm);
    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "3D stencil time: "
                  << (t_end - t_start)
                  << " s\n";

    MPI_Comm_free(&cart_comm);
    MPI_Comm_free(&ring_comm);
    MPI_Finalize();
    return 0;
}
