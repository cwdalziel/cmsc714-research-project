// Hypercube-optimized sample sort (Phase 3 — topology-aware variant).
// Owner: Sonok Mahapatra.
//
// Baseline to beat (from ../RESULTS.md):
//   strong np=256: 0.0054 s
//   weak   np=256: 0.0342 s   (already the best of all topologies — high bar)
//
// Optimization idea: recursive halving / hyperquicksort. Instead of one
// global alltoall, do log2(p) rounds of pairwise exchanges across one
// hypercube dimension at a time. In dimension d (d = 0..log2(p)-1) every
// rank pairs with rank XOR (1 << d), partitions its data around a pivot,
// and exchanges the halves. After log p rounds, each rank holds its final
// sorted bucket.
//
// Note: this departs significantly from the regular-sampling structure of
// the agnostic algorithm. Steps 2-4 (sample + allgather + global splitters)
// might not apply at all — see hyperquicksort literature for pivot-selection
// strategies. The simplest version picks the global median at each round
// (via MPI_Allreduce on a sample) and uses it as the partition pivot.
//
// Constraint: p must be a power of 2. All NP values in NP_LIST satisfy this
// (16, 32, 64, 128, 256).
//
// This file is a SCAFFOLD: it currently runs the agnostic algorithm. Step 7
// is marked as the replacement point.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using Key = int64_t;
static const MPI_Datatype MPI_KEY = MPI_INT64_T;

static void print_usage(const char* prog) {
    std::cerr <<
      "Usage: " << prog << " [options] [TOTAL_N]\n"
      "Hypercube-optimized sample sort. Same CLI as the agnostic variant.\n"
      "  --per-rank-N=N   N keys per rank (TOTAL_N = N * num_ranks).\n"
      "  --seed=S         RNG seed (default 0 = rank-derived).\n"
      "  --verify         Verify globally sorted (slow; for dev).\n"
      "  -h, --help       Print this help and exit.\n";
}

static bool starts_with(const std::string& s, const char* prefix, std::string& value) {
    size_t plen = std::strlen(prefix);
    if (s.size() < plen || s.compare(0, plen, prefix) != 0) return false;
    value = s.substr(plen);
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    long long total_N    = (1LL << 24);
    long long per_rank_N = -1;
    uint64_t  seed       = 0;
    bool      verify     = false;
    bool      saw_pos_N  = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        std::string val;
        if (a == "-h" || a == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (starts_with(a, "--per-rank-N=", val)) {
            per_rank_N = std::atoll(val.c_str());
        } else if (starts_with(a, "--seed=", val)) {
            seed = std::strtoull(val.c_str(), nullptr, 10);
        } else if (a == "--verify") {
            verify = true;
        } else if (!a.empty() && a[0] != '-') {
            total_N = std::atoll(a.c_str());
            saw_pos_N = true;
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << a << "\n";
                print_usage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (per_rank_N > 0 && saw_pos_N) {
        if (rank == 0) std::cerr << "Error: pass either TOTAL_N or --per-rank-N=, not both.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (per_rank_N > 0) total_N = per_rank_N * p;
    if (total_N <= 0 || total_N % p != 0) {
        if (rank == 0)
            std::cerr << "Error: total_N (" << total_N << ") must be positive and divisible by ranks ("
                      << p << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long local_n = total_N / p;
    if (local_n < p) {
        if (rank == 0) std::cerr << "Error: local_n (" << local_n << ") must be >= ranks (" << p << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    uint64_t rng_seed = seed
        ? (seed ^ (static_cast<uint64_t>(rank) * 0x9E3779B97F4A7C15ULL))
        : (static_cast<uint64_t>(rank) * 0x9E3779B97F4A7C15ULL + 1);
    std::vector<Key> local(local_n);
    std::mt19937_64 rng(rng_seed);
    for (auto& x : local) x = static_cast<Key>(rng());

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // ===== Steps 1-6: identical to agnostic (placeholder; the hyperquicksort
    //                  variant may discard most of this) =====
    std::sort(local.begin(), local.end());

    std::vector<Key> samples(p - 1);
    for (int i = 0; i < p - 1; i++)
        samples[i] = local[(i + 1) * local_n / p];

    std::vector<Key> all_samples(static_cast<size_t>(p) * (p - 1));
    MPI_Allgather(samples.data(),     p - 1, MPI_KEY,
                  all_samples.data(), p - 1, MPI_KEY,
                  MPI_COMM_WORLD);

    std::sort(all_samples.begin(), all_samples.end());
    std::vector<Key> splitters(p - 1);
    for (int i = 0; i < p - 1; i++)
        splitters[i] = all_samples[(i + 1) * p - 1];

    std::vector<int> send_counts(p, 0);
    std::vector<int> send_displs(p, 0);
    {
        long long prev = 0;
        for (int i = 0; i < p - 1; i++) {
            auto it = std::lower_bound(local.begin() + prev, local.end(), splitters[i]);
            long long next = it - local.begin();
            send_counts[i] = static_cast<int>(next - prev);
            send_displs[i] = static_cast<int>(prev);
            prev = next;
        }
        send_counts[p - 1] = static_cast<int>(local_n - prev);
        send_displs[p - 1] = static_cast<int>(prev);
    }

    std::vector<int> recv_counts(p, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // ===== STEP 7 (HYPERCUBE OPTIMIZATION POINT) =====
    //
    // Replace the global MPI_Alltoallv with hyperquicksort: log2(p) rounds
    // of pairwise exchange, one per hypercube dimension.
    //
    // Skeleton (the "splitters" / "send_counts" arrays from steps 4-6 do
    // not directly apply — hyperquicksort uses a different partitioning
    // scheme):
    //
    //   int d = log2(p);  // assumes p is a power of 2
    //   for (int dim = d - 1; dim >= 0; dim--) {
    //       int partner = rank ^ (1 << dim);
    //       Key pivot   = pick_global_median(local, dim);  // sample-based
    //       auto mid    = std::partition(local.begin(), local.end(),
    //                                    [pivot](Key k){ return k < pivot; });
    //       // Lower half stays here if (rank & (1<<dim)) == 0, else send.
    //       // Symmetric rule for upper half.
    //       MPI_Sendrecv send the half going to partner, receive partner's;
    //       merge into local (it's still sorted because we partitioned
    //       a sorted array around the pivot).
    //   }
    //   // After log p rounds, `local` IS the rank's final sorted bucket.
    //   // No separate `received` vector needed; rename or std::swap.
    //
    // Pivot selection: at each round, ranks within the same active sub-cube
    // need to agree on the partition pivot. Cheapest: each rank contributes
    // one sample, MPI_Allreduce within sub-cube to get global samples,
    // pick the median.
    //
    // TODO: replace the placeholder MPI_Alltoallv below with hyperquicksort.
    //       Note that you may want to re-shape steps 1-6 above as well —
    //       hyperquicksort doesn't need the agnostic regular-sampling.

    std::vector<int> recv_displs(p, 0);
    long long recv_total = recv_counts[0];
    for (int i = 1; i < p; i++) {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        recv_total += recv_counts[i];
    }
    std::vector<Key> received(static_cast<size_t>(recv_total));

    // ----- PLACEHOLDER (matches agnostic; replace this whole alltoallv) -----
    MPI_Alltoallv(local.data(),    send_counts.data(), send_displs.data(), MPI_KEY,
                  received.data(), recv_counts.data(), recv_displs.data(), MPI_KEY,
                  MPI_COMM_WORLD);
    // ------------------------------------------------------------------------

    // ===== Step 8 =====
    std::sort(received.begin(), received.end());

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "Sample sort time: " << (t_end - t_start) << " s\n";

    if (verify) {
        int local_ok = std::is_sorted(received.begin(), received.end()) ? 1 : 0;
        int all_local_ok = 0;
        MPI_Reduce(&local_ok, &all_local_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

        Key my_min = received.empty() ? std::numeric_limits<Key>::max() : received.front();
        Key my_max = received.empty() ? std::numeric_limits<Key>::min() : received.back();
        int has_data = received.empty() ? 0 : 1;

        std::vector<Key> all_mins, all_maxs;
        std::vector<int> all_has;
        if (rank == 0) { all_mins.resize(p); all_maxs.resize(p); all_has.resize(p); }
        MPI_Gather(&my_min,   1, MPI_KEY, all_mins.data(), 1, MPI_KEY, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_max,   1, MPI_KEY, all_maxs.data(), 1, MPI_KEY, 0, MPI_COMM_WORLD);
        MPI_Gather(&has_data, 1, MPI_INT, all_has.data(),  1, MPI_INT, 0, MPI_COMM_WORLD);

        long long my_count = static_cast<long long>(received.size());
        long long total_count = 0;
        MPI_Reduce(&my_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            bool monotonic = true;
            Key last_max = std::numeric_limits<Key>::min();
            bool seen = false;
            for (int r = 0; r < p; r++) {
                if (!all_has[r]) continue;
                if (seen && last_max > all_mins[r]) { monotonic = false; break; }
                last_max = all_maxs[r];
                seen = true;
            }
            bool count_ok = (total_count == total_N);
            if (all_local_ok && monotonic && count_ok) {
                std::cerr << "VERIFY: ok\n";
            } else {
                std::cerr << "VERIFY: FAIL  local_sorted=" << all_local_ok
                          << "  monotonic=" << monotonic
                          << "  total_count=" << total_count
                          << "  expected=" << total_N << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
