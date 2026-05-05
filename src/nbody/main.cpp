#include <algorithm>
#include <cassert>
#include <complex>
#include <fstream>
#include <cstdint>
#include <execution>
#include <unordered_map>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>
#include <optional>
#include <random>
#include <mpi.h>
#include <thread>
#include <mutex>
#define KEY_REQUEST 61
#define KEY_RESPONSE 80

int debug_children_counter =0;
int debug_encoded_node_children_counter = 0;
struct Point {

    float x, y, z;

    uint64_t morton_key() const {
        const uint64_t scale = 1ull << 21;
        uint64_t x_int = static_cast<uint64_t>(x * scale);
        uint64_t y_int = static_cast<uint64_t>(y * scale);
        uint64_t z_int = static_cast<uint64_t>(z * scale);

        uint64_t key = 0;
        for (uint64_t i = 0; i < 21; ++i) {
            key |= ((x_int >> i) & 1ull) << (3 * i + 0);
            key |= ((y_int >> i) & 1ull) << (3 * i + 1);
            key |= ((z_int >> i) & 1ull) << (3 * i + 2);
        }
        return key;
    }
    static Point from_morton(uint64_t key) {
        const uint64_t scale = 1ull << 21;

        uint64_t x_int = 0;
        uint64_t y_int = 0;
        uint64_t z_int = 0;

        for (uint64_t i = 0; i < 21; ++i) {
            x_int |= ((key >> (3 * i + 0)) & 1ull) << i;
            y_int |= ((key >> (3 * i + 1)) & 1ull) << i;
            z_int |= ((key >> (3 * i + 2)) & 1ull) << i;
        }

        Point p;
        p.x = static_cast<float>(x_int) / scale;
        p.y = static_cast<float>(y_int) / scale;
        p.z = static_cast<float>(z_int) / scale;

        return p;
    }

    Point operator+(const Point& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Point operator-(const Point& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Point operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    Point operator/(float scalar) const { return {x / scalar, y / scalar, z / scalar}; }

    Point& operator+=(const Point& other) { x += other.x; y += other.y; z += other.z; return *this; }
    Point& operator-=(const Point& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
    Point& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
    Point& operator/=(float scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

    bool operator==(const Point& other) const { return x == other.x && y == other.y && z == other.z; }
    bool operator!=(const Point& other) const { return !(*this == other); }

    float operator[](int i) const { return (&x)[i]; }
    float& operator[](int i) { return (&x)[i]; }
};
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1,T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

struct Request{
    uint64_t morton_key;
    int level;
};
MPI_Datatype create_request_datatype() {
    MPI_Datatype custom_type;

    int count = 2;

    int blocklengths[2] = {1, 1};

    MPI_Datatype types[2] = {MPI_UINT64_T, MPI_INT};

    // Calculate displacements
    MPI_Aint offsets[2];
    offsets[0] = offsetof(Request, morton_key);
    offsets[1] = offsetof(Request, level);

    MPI_Type_create_struct(count, blocklengths, offsets, types, &custom_type);

    MPI_Type_commit(&custom_type);

    return custom_type;
}
Point force_on(Point p1, Point p2, float m2) {
    const float G = 0.01f;
    const float softening = 0.01f; // Prevents "infinity" glitches

    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;

    float r_sq = dx*dx + dy*dy + dz*dz + (softening * softening);
    float r = std::sqrt(r_sq);

    float factor = (G * m2) / (r_sq * r);

    return {dx * factor, dy * factor, dz * factor};
}
bool is_within(uint64_t key, int level, bool is_leaf, uint64_t lower, uint64_t upper, int size, int rank)
{
    level = level*3;
    if (rank == 0)
    {
        return is_leaf ||
            ((key>>(63-level)) < (upper >> (63-level)));
    }
    if (rank == size-1)
    {
        return is_leaf || (key >> (63-level)) > (lower >> (63-level));
    }
    return is_leaf || ((key >> (63-level)) > (lower >> (63-level)) &&
        (key>> (63-level)) < (upper >> (63-level)));
}
struct Particle {
    Point position;
    float mass;
};
struct WeightedParticle {
    Point weighted_position;
    float total_mass;
};
struct EncodedNode
{
    float weighted_x;
    float weighted_y;
    float weighted_z;
    float mass;
    bool is_leaf;
    uint64_t morton_key;
    int level;
    int rank;
};
struct OctreeNode {
    Point weighted_position;
    float total_mass;
    uint64_t morton_key;
    int level;
    int remote_level;
    std::optional<uint64_t> children[8];
    bool is_leaf;
    std::optional<int> owner;
    // a print function that prints data for debug
    void print() const {
        //print remote level
        std::cout << "Remote Level: " << remote_level << std::endl;

        std::cout << "Node Level: " << level
                  << " Morton: " << morton_key
                  << " Mass: " << total_mass
                  << " Pos: (" << weighted_position.x / total_mass << ", "
                  << weighted_position.y / total_mass << ", "
                  << weighted_position.z / total_mass << ")"
                  << " Leaf: " << (is_leaf ? "Yes" : "No")
                  << " Owner: " << (owner.has_value() ? std::to_string(owner.value()) : "Local")
                  << std::endl;
    }

    [[nodiscard]] EncodedNode encode(int rank) const
    {
        return EncodedNode{
            .weighted_x = weighted_position.x,
            .weighted_y = weighted_position.y,
            .weighted_z = weighted_position.z,
            .mass = total_mass,
            .is_leaf = is_leaf,
            .morton_key = morton_key,
            .level = level,
            .rank = rank

        };
    }
    static MPI_Datatype create_encoded_node_type() {
        MPI_Datatype t;

        const int    nblocks         = 8;
        int          blocklengths[nblocks] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        MPI_Datatype types[nblocks]        = {
            MPI_FLOAT,
            MPI_FLOAT,
            MPI_FLOAT,
            MPI_FLOAT,
            MPI_C_BOOL,
            MPI_UINT64_T,
            MPI_INT,
            MPI_INT
        };
        MPI_Aint disps[nblocks] = {
            offsetof(EncodedNode, weighted_x),
            offsetof(EncodedNode, weighted_y),
            offsetof(EncodedNode, weighted_z),
            offsetof(EncodedNode, mass),
            offsetof(EncodedNode, is_leaf),
            offsetof(EncodedNode, morton_key),
            offsetof(EncodedNode, level),
            offsetof(EncodedNode, rank)
        };

        MPI_Type_create_struct(nblocks, blocklengths, disps, types, &t);

        MPI_Datatype t_resized;
        MPI_Type_create_resized(t, 0, sizeof(EncodedNode), &t_resized);
        MPI_Type_free(&t);
        MPI_Type_commit(&t_resized);
        return t_resized;
    }
};
MPI_Datatype octree_node_datatype;
MPI_Datatype request_datatype;
class Octree {
public:
    int debug_count(uint64_t morton_key, int level)
    {
        if (octree_map.contains({morton_key, level})) {
            auto& node = octree_map.at({morton_key, level});
            if (node.is_leaf) return 1;
            int count = 0;
            for (auto& child : node.children) {
                if (child.has_value()) {
                    count += debug_count(child.value(), level + 1);
                }
            }
            return count;
        }
        return 0;
    }
    OctreeNode* debug_find(uint64_t morton_key, int level)
    {
        if (octree_map.contains({morton_key,level}))
            return &octree_map.at({morton_key,level});
        return nullptr;
    }
    Octree(std::vector<Particle>& particles){
        for(const auto& particle : particles) {
            insert_unweighted_particle(particle,std::nullopt);
        }
    }
    Octree(std::vector<float>& particles, MPI_Comm communication_communicator_):
    communcation_communicator(communication_communicator_)
    {
        for (int i = 0;i<particles.size();i+=4)
        {
            Particle particle = {
                .position = {
                    .x = particles[i],
                    .y = particles[i+1],
                    .z = particles[i+2]
                },
                .mass = particles[i+3]
            };
            insert_unweighted_particle(particle,std::nullopt);
        }
        compute_mass_and_center_of_mass(head_key,0);
    }
    // write a function that performs sanity checkso n the tree, assuming owner = nullopt. in toher words, no remote nodes
    // your sanity checks should make sure there are no nonsensical states
    void sanity_check(uint64_t key, int level) {
        auto& node = octree_map.at({key, level});
        if (node.is_leaf) return;

        float total_m = 0;
        Point weighted_p = {0, 0, 0};
        for (int i = 0; i < 8; ++i) {
            if (node.children[i].has_value()) {
                uint64_t child_key = node.children[i].value();
                sanity_check(child_key, level + 1);
                const auto& child_node = octree_map.at({child_key, level + 1});
                total_m += child_node.total_mass;
                weighted_p += child_node.weighted_position;
            }
        }
        assert(std::abs(node.total_mass - total_m) < 1e-3);
        assert(std::abs(node.weighted_position.x - weighted_p.x) < 1e-3);
    }



    void add_branches(std::vector<EncodedNode>& input, int disp, int count)
    {
        assert(head_key == 0);
        for (int i = 0;i<input.size();i++)
        {
            if (i>= disp && i < disp+count)
                continue;
            Point weighted_position{ input[i].weighted_x, input[i].weighted_y, input[i].weighted_z};
            Particle p{.position =weighted_position/input[i].mass , .mass = input[i].mass };
            if (input[i].is_leaf)
            {
                debug_children_counter++;
                insert_at(p, input[i].morton_key, &octree_map[{head_key, 0}], nullptr, 0, std::nullopt, input[i].level);
            }else
            {
                insert_at(p, input[i].morton_key, &octree_map[{head_key, 0}], nullptr, 0, input[i].rank, input[i].level);
            }
        }
        compute_mass_and_center_of_mass(head_key,0);
    }
    void local_branch_helper(uint64_t key, int level ,uint64_t lower, uint64_t upper, int optimal_level, std::vector<EncodedNode>& branches, int size, int rank)
    {
        auto& node = octree_map[{key,level}];
        if (is_within(key,level,node.is_leaf,lower,upper, size, rank))
        {
            debug_encoded_node_children_counter+= debug_count(node.morton_key,node.level);
            branches.push_back(node.encode(rank));
            return;
        }
        for (auto& child : node.children)
        {
            if (!child.has_value())
                continue;
            local_branch_helper(child.value(),level+1,lower,upper,optimal_level,branches,size,rank);
        }
    }
    std::vector<EncodedNode> local_branches(uint64_t lower, uint64_t upper,int size, int rank)
    {
        std::vector<EncodedNode> branches;
        local_branch_helper(head_key,0,lower,upper,20,branches, size, rank);
        return branches;
    }
    void insert_unweighted_particle(const Particle& particle, std::optional<int> owner) {
        uint64_t morton_key = particle.position.morton_key();
        if(octree_map.empty()){
            head_key = morton_key;
            std::unique_lock lock(octree_mutex);
            octree_map[{head_key,0}] = OctreeNode{
                {
                    .x = particle.position.x*particle.mass,
                    .y = particle.position.y*particle.mass,
                    .z = particle.position.z*particle.mass
                },
                particle.mass,
                head_key,
                0,
                -1,
                {std::nullopt},
                true,
               owner
            };
            return;
        }
        auto& start_node = octree_map[{head_key,0}];
        insert_at(particle, morton_key, &start_node, nullptr, 0, owner, -1);
    }
    void insert_at(const Particle& particle, uint64_t morton_key, OctreeNode* current_node, OctreeNode* parent_node, uint64_t parent_rel_dir,  std::optional<int> owner, int target_level) {
        int level = current_node->level;
        if(current_node->is_leaf){
            if(level == 20){
                current_node->total_mass += particle.mass;
                current_node->weighted_position.x += particle.position.x * particle.mass;
                current_node->weighted_position.y += particle.position.y * particle.mass;
                current_node->weighted_position.z += particle.position.z * particle.mass;
                return;
            }
            auto rel_dir  = (current_node->morton_key >> (3 * (20-level))) & 0x7;

            OctreeNode new_node =*current_node;
            new_node.level++;


            current_node->children[rel_dir] =std::make_optional(current_node->morton_key);
            current_node->is_leaf = false;
            current_node->morton_key = (current_node->morton_key >> (3 * (21-level))) << (3 * (21-level));
            current_node->owner.reset();
            uint64_t parent_morton = 0;
            if(level == 0){
                head_key = current_node->morton_key;
            }else
            {
                parent_node->children[parent_rel_dir].emplace(current_node->morton_key);
                parent_morton = parent_node->morton_key;
            }


            auto currnt_node_value = *current_node;

            std::unique_lock write_lock(octree_mutex);
            octree_map[{current_node->morton_key,currnt_node_value.level}]= currnt_node_value;
            octree_map[{new_node.morton_key,new_node.level}] = new_node;
            write_lock.unlock();

            OctreeNode* new_parent_node = nullptr;
            if (parent_node)
            {
                new_parent_node = &octree_map[{parent_morton,level-1}];
            }

            insert_at(particle, morton_key, &octree_map[{currnt_node_value.morton_key,currnt_node_value.level}],new_parent_node,parent_rel_dir, owner, target_level);
            return;
        }
        auto rel_dir  = (morton_key >> (3 * (20-level))) & 0x7;

        if(!current_node->children[rel_dir].has_value()){
            current_node->children[rel_dir].emplace(morton_key);
            octree_map[{morton_key,level+1}] = OctreeNode{
            {
                .x = particle.position.x*particle.mass,
                .y = particle.position.y*particle.mass,
                .z = particle.position.z*particle.mass
            },
                particle.mass,
                morton_key,
                level+1,
                target_level,
                {std::nullopt}, true, owner
            };
            return;
        }
        auto& child_node = octree_map[{current_node->children[rel_dir].value(),current_node->level+1}];
        insert_at(particle, morton_key, &child_node, current_node, rel_dir, owner, target_level);
    }
    WeightedParticle compute_mass_and_center_of_mass(uint64_t morton_key,int level)
    {
        auto& head = octree_map[{morton_key,level}];
        assert(head.level == level);
        if (head.is_leaf)
        {
            return {head.weighted_position,head.total_mass};
        }
        WeightedParticle weighted_particle = {0};
        for (auto& child : head.children)
        {
            if (!child.has_value())
                continue;
            auto child_particle = compute_mass_and_center_of_mass(child.value(),level+1);
            weighted_particle.total_mass+= child_particle.total_mass;
            weighted_particle.weighted_position.x+= child_particle.weighted_position.x;
            weighted_particle.weighted_position.y+= child_particle.weighted_position.y;
            weighted_particle.weighted_position.z+= child_particle.weighted_position.z;
        }
        head.total_mass = weighted_particle.total_mass;
        head.weighted_position= weighted_particle.weighted_position;
        return {head.weighted_position,head.total_mass};
    }
    void insert_encoded_node_into_map(EncodedNode& encoded_node, int level)
    {
        std::unique_lock write_lock(octree_mutex);
        std::cout<<"encoding "<<encoded_node.morton_key<<" at "<<level<<" with remote "<<encoded_node.level<<std::endl;
        octree_map[{encoded_node.morton_key,level}] = {
            .weighted_position = {encoded_node.weighted_x, encoded_node.weighted_y, encoded_node.weighted_z},
            .total_mass = encoded_node.mass,
            .morton_key = encoded_node.morton_key,
            .level = level,
            .remote_level = encoded_node.level,
            .children = {},
            .is_leaf = true,
            .owner = encoded_node.is_leaf ? std::nullopt : std::make_optional(encoded_node.rank)
        };
        // octree_map[{encoded_node.morton_key,level}].print();
    }
    Point compute_force_node(Particle& particle,uint64_t particle_morton_key,OctreeNode* current_node)
    {
        if (debug_find(2305843009213693952,1)->owner.has_value())
        {
            std::cout<<"became nonnull"<<std::endl;
        }
        if (particle_morton_key == current_node->morton_key && current_node->is_leaf)
        {
            if (std::abs(particle.mass-current_node->total_mass) < 0.001)
                return {0,0,0};

            Point adjusted_weighted_position = {
                .x = current_node->weighted_position.x - particle.position.x * particle.mass,
                .y = current_node->weighted_position.y - particle.position.y * particle.mass,
                .z = current_node->weighted_position.z - particle.position.z * particle.mass
            };
            float adjusted_mass = current_node->total_mass - particle.mass;
            return force_on(particle.position, adjusted_weighted_position/ adjusted_mass, adjusted_mass);

        }
        float s =  1.0f / (float)(1ull << current_node->level);
        Point unweighted_position {
            .x=current_node->weighted_position.x/current_node->total_mass,
            .y=current_node->weighted_position.y/current_node->total_mass,
            .z=current_node->weighted_position.z/current_node->total_mass
            };
        float dx = unweighted_position.x - particle.position.x;
        float dy = unweighted_position.y - particle.position.y;
        float dz = unweighted_position.z - particle.position.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        float d = std::sqrt(d2);

        float theta = 0.5f;
        // float theta = 0.f;
        bool particle_in_node = (particle_morton_key >> (3 * (21 - current_node->level))) ==
                        (current_node->morton_key >> (3 * (21 - current_node->level)));
        if ((current_node->is_leaf&& !current_node->owner.has_value())
            ||( s/(d+1e-9f) < theta && !particle_in_node ))
        {
            return force_on(particle.position,unweighted_position, current_node->total_mass);
        }
        uint64_t current_morton = current_node->morton_key;
        int current_level = current_node->level;
        int current_remote = current_node->remote_level;
        if (current_node->owner.has_value())
        {
            if (current_node->remote_level < 0)
            {
                std::cout<<"we're going down"<<std::endl;
            }
            assert(current_node->remote_level >= 0);
            EncodedNode encoded_nodes[8]; // NOLINT(*-pro-type-member-init)
            Request request = {.morton_key = current_node->morton_key, .level =current_node->remote_level};
            std::cout<<"requesting "<<current_node->morton_key<< current_node->remote_level<<std::endl;
            MPI_Sendrecv(
                &request,
                1,
                request_datatype,
                current_node->owner.value(),
                KEY_REQUEST,
                &encoded_nodes,
                8,
                octree_node_datatype,
                current_node->owner.value(),
                KEY_RESPONSE,
                communcation_communicator,
                MPI_STATUS_IGNORE);
            current_node->owner.reset();
            current_node->is_leaf = false;
            for (int i = 0;i < 8; i++)
            {
                auto& encoded_node = encoded_nodes[i];
                if (encoded_node.rank == -1)
                    continue;
                if (encoded_node.is_leaf)
                    debug_children_counter++;
                current_node->children[i].emplace(encoded_node.morton_key);
                if (encoded_node.level != current_remote+1)
                {
                    std::cout <<"funny assertion failed" <<std::endl;
                }
            }
            for (int i =0 ;i<8;i++)
            {
                auto& encoded_node = encoded_nodes[i];
                if (encoded_node.rank == -1)
                    continue;
                insert_encoded_node_into_map(encoded_node,current_level+1);
            }
        }
        Point total_force{0};

        for (int i =0 ;i<8;i++)
        {
            auto& child = octree_map.at({current_morton, current_level}).children[i];
            if (!child.has_value())
                continue;
            auto& child_node=octree_map[{child.value(),current_level+1}];
            auto force = compute_force_node(particle, particle_morton_key,&child_node);
            total_force.x += force.x;
            total_force.y += force.y;
            total_force.z += force.z;
        }
        return total_force;
    }
    Point compute_force(Particle& particle)
    {
        auto head_node =  &octree_map.at({head_key,0});
        return compute_force_node(particle, particle.position.morton_key(), head_node);
    }
    uint64_t head_key;
    MPI_Comm communcation_communicator;
    mutable std::mutex octree_mutex;
    std::unordered_map<std::pair<uint64_t,int>,OctreeNode,pair_hash> octree_map;
    // std::map<uint64_t,OctreeNode> octree_leafs;
};
class Simulation
{
public:
    std::vector<Particle> particles;
    std::vector<Point> velocities;
    Simulation(std::vector<Particle> particles_): particles(particles_)
    {
        velocities.resize(particles.size());

    }
    Simulation(std::vector<Particle> particles_, std::vector<Point> velocities_):
    particles(particles_),
    velocities(velocities_)
    {

    }
    Point compute_force_node(Particle& particle,uint64_t particle_morton_key,OctreeNode* current_node, Octree& octree)
    {
        float s =  1/((float)(1<<current_node->level));
        Point unweighted_position {
            .x=current_node->weighted_position.x/current_node->total_mass,
            .y=current_node->weighted_position.y/current_node->total_mass,
            .z=current_node->weighted_position.z/current_node->total_mass
            };
        float dx = unweighted_position.x - particle.position.x;
        float dy = unweighted_position.y - particle.position.y;
        float dz = unweighted_position.z - particle.position.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        float d = std::sqrt(d2 + 0.001f);

        // float theta = 0.5f;
        float theta = 0.5f;
        bool particle_in_node = (particle_morton_key >> (3 * (21 - current_node->level))) ==
                        (current_node->morton_key >> (3 * (21 - current_node->level)));
        if (current_node->is_leaf
            ||( s/d < theta ))
        {
            if ((particle_in_node ||current_node->morton_key == particle_morton_key )&& (current_node->total_mass-particle.mass)> 0.001)
            {
                Point  adjusted_position= (current_node->weighted_position-particle.position*particle.mass)/(current_node->total_mass-particle.mass);
                return force_on(particle.position,adjusted_position, current_node->total_mass-particle.mass);

            }
            return force_on(particle.position,unweighted_position, current_node->total_mass);
        }
        Point total_force{0};
        for (auto& child : current_node->children)
        {
            if (!child.has_value())
                continue;
            auto& child_node= octree.octree_map[{child.value(),current_node->level+1}];
            auto force = compute_force_node(particle, particle_morton_key,&child_node,octree);
            total_force.x += force.x;
            total_force.y += force.y;
            total_force.z += force.z;
        }
        return total_force;
    }
    std::vector<Point> compute_forces()
    {
        Octree tree(particles);
        tree.compute_mass_and_center_of_mass(tree.head_key,0);
        auto& head = tree.octree_map[{tree.head_key,0}];
        std::vector<Point> forces;
        for (auto& particle: particles)
        {
            forces.emplace_back(compute_force_node(particle,particle.position.morton_key(),&head,tree));
        }
        return forces;

    }
    void debug_net_force() {
        Octree tree(particles);
        tree.compute_mass_and_center_of_mass(tree.head_key, 0);
        auto& head = tree.octree_map[{tree.head_key, 0}];

        // Just check first 5 particles
        for (int i = 0; i < 5; i++) {
            auto& p = particles[i];
            auto f_bh = compute_force_node(p, p.position.morton_key(), &head, tree);

            Point f_brute{0, 0, 0};
            for (int j = 0; j < particles.size(); j++) {
                if (i == j) continue;
                auto f = force_on(p.position, particles[j].position, particles[j].mass);
                f_brute.x += f.x;
                f_brute.y += f.y;
                f_brute.z += f.z;
            }

            std::cout << "particle " << i << "\n";
            std::cout << "  brute: " << f_brute.x << " " << f_brute.y << " " << f_brute.z << "\n";
            std::cout << "  bh:    " << f_bh.x    << " " << f_bh.y    << " " << f_bh.z    << "\n";
        }
    }
    void step(float dt)
    {
        auto f =  compute_forces();
        for (int i = 0; i < particles.size(); i++) {
            velocities[i].x += f[i].x * dt * 0.5f;
            velocities[i].y += f[i].y * dt * 0.5f;
            velocities[i].z += f[i].z * dt * 0.5f;
        }
        for (int i = 0; i < particles.size(); i++) {
            particles[i].position.x += velocities[i].x * dt;
            particles[i].position.y += velocities[i].y * dt;
            particles[i].position.z += velocities[i].z * dt;
        }
        auto f2 = compute_forces();  // recompute at new positions
        for (int i = 0; i < particles.size(); i++) {
            velocities[i].x += f2[i].x * dt * 0.5f;
            velocities[i].y += f2[i].y * dt * 0.5f;
            velocities[i].z += f2[i].z * dt * 0.5f;
        }
    }
    void write_state(std::filesystem::path const& path,bool do_clear) const
    {
        std::ios_base::openmode mode;
        if (do_clear)
        {
            mode = std::ios_base::out;
        }else
        {
            mode = std::ios_base::app;
        }
        std::ofstream out(path,mode);
        for (auto& particle: particles)
        {
            out<< particle.position.x << "," << particle.position.y << "," << particle.position.z << std::endl;
        }
        out<<"break"<<std::endl;
        out.close();
    }
};
std::pair<std::vector<Particle>, std::vector<Point>> generateBinaryGalaxy(int numParticles, float center = 0.5f) {
    std::vector<Particle> particles;
    std::vector<Point> velocities;
    std::mt19937 gen(42);

    // Distribution parameters
    std::normal_distribution<float> distRadius(0.0f, 0.02f); // Slightly tighter clusters
    std::uniform_real_distribution<float> distAngle(0.0f, 2.0f * 3.14159);
    std::normal_distribution<float> distHeight(0.0f, 0.003f);

    float spinStrength = 12.0f;       // Internal rotation of each galaxy
    float interactionStrength = 5.0f; // Velocity of the two galaxies orbiting each other
    float offset = 0.1f;              // Distance from the total center (0.5 +/- 0.1)

    for (int i = 0; i < numParticles; ++i) {
        // Determine which galaxy this particle belongs to (-1 or 1)
        float side = (i < numParticles / 2) ? -1.0f : 1.0f;

        // 1. Local Position (relative to galaxy center)
        float r = std::abs(distRadius(gen));
        float theta = distAngle(gen);
        float z_off = distHeight(gen);

        // 2. Global Position
        // Galaxy A at 0.4, Galaxy B at 0.6
        float galCenterX = center + (side * offset);
        float galCenterY = center;

        float px = galCenterX + r * std::cos(theta);
        float py = galCenterY + r * std::sin(theta);
        float pz = center + z_off;

        // 3. Velocity
        // Part A: Internal Spin (tangent to local center)
        float internalVelMag = std::sqrt(r) * spinStrength;
        float vx_internal = -std::sin(theta) * internalVelMag;
        float vy_internal = std::cos(theta) * internalVelMag;

        // Part B: Orbital Velocity (tangent to global center 0.5)
        // Since galaxies are on the X-axis relative to 0.5,
        // their orbital motion should be primarily along the Y-axis.
        float vx_orbital = 0.0f;
        float vy_orbital = side * interactionStrength;

        particles.push_back({.position = {px, py, pz}, .mass = 1.0f});
        velocities.push_back({vx_internal + vx_orbital, vy_internal + vy_orbital, 0.01f * distHeight(gen)});
    }

    return {particles, velocities};
}
std::pair<std::vector<Particle>,std::vector<Point>>generateGalaxy(int numParticles, float center=0.5f) {
    std::vector<Particle> particles;
    std::vector<Point> velocities;
    std::mt19937 gen(42);
    std::normal_distribution<float> distRadius(0.0f, 0.035f); // Spread of the galaxy
    std::uniform_real_distribution<float> distAngle(0.0f, 2.0f *3.14159);
    std::normal_distribution<float> distHeight(0.0f, 0.005f); // Thinness of the disk

    float spinStrength = 15.5f;

    for (int i = 0; i < numParticles; ++i) {
        // 1. Position: Generate r, theta, z
        // Using abs(normal dist) gives a dense core and a tapering edge
        float r = std::abs(distRadius(gen));
        float theta = distAngle(gen);
        float z_off = distHeight(gen);

        float px = center + r * std::cos(theta);
        float py = center + r * std::sin(theta);
        float pz = center + z_off;

        // 2. Velocity: Tangent to the radius for that "spinning" look
        // The vector (-sin(theta), cos(theta)) is perpendicular to the radius
        // We scale velocity by sqrt(r) or similar to mimic orbital mechanics
        float velocityMag = std::sqrt(r) * spinStrength;
        float vx = -std::sin(theta) * velocityMag;
        float vy = std::cos(theta) * velocityMag;
        float vz = 0.01f * distHeight(gen); // Tiny random vertical drift
        particles.push_back({
        .position = {px,py,pz},
        .mass = 1.0f});
        velocities.push_back({vx,vy,vz});
        // particles.push_back({{px, py, pz}, 1.0f, {vx, vy, vz}});
    }

    return {particles,velocities};
}

std::pair<std::vector<Particle>, std::vector<Point>> generateBinaryGalaxys(int numParticles, float center = 0.5f) {
    std::vector<Particle> particles;
    std::vector<Point> velocities;
    std::mt19937 gen(42);

    // Tightened spread so they don't bleed out of the 0.3-0.7 range
    std::normal_distribution<float> distRadius(0.0f, 0.025f);
    std::uniform_real_distribution<float> distAngle(0.0f, 2.0f *3.14159);
    std::normal_distribution<float> distHeight(0.0f, 0.003f);

    float internalSpinStrength = 12.0f;
    float orbitalVelocityStrength = 3.0f;
    float galaxyOffset = 0.1f; // Distance from the system center (0.5 +/- 0.1 = 0.4 and 0.6)

    for (int i = 0; i < numParticles; ++i) {
        // Determine which galaxy this particle belongs to (-1 or 1)
        float side = (i < numParticles / 2) ? 1.0f : -1.0f;

        // 1. Local Position (relative to the galaxy's own center)
        float r = std::abs(distRadius(gen));
        float theta = distAngle(gen);
        float z_off = distHeight(gen);

        float localX = r * std::cos(theta);
        float localY = r * std::sin(theta);
        float localZ = z_off;

        // 2. Global Position
        // Galaxy A centered at center + offset, Galaxy B at center - offset
        float galCenterX = center + (side * galaxyOffset);
        float galCenterY = center;
        float galCenterZ = center;

        float px = galCenterX + localX;
        float py = galCenterY + localY;
        float pz = galCenterZ + localZ;

        // 3. Internal Spin Velocity (Rotation around local center)
        float spinMag = std::sqrt(r) * internalSpinStrength;
        float vSpinX = -std::sin(theta) * spinMag;
        float vSpinY =  std::cos(theta) * spinMag;

        // 4. Orbital Velocity (The whole galaxy moving around 0.5, 0.5)
        // For a circular orbit, velocity is perpendicular to the vector from system center to galaxy center
        // Since centers are on the X-axis, orbital velocity is purely on the Y-axis
        float vOrbX = 0.0f;
        float vOrbY = side * orbitalVelocityStrength;

        // Combine velocities
        float vx = vSpinX + vOrbX;
        float vy = vSpinY + vOrbY;
        float vz = 0.01f * distHeight(gen);

        particles.push_back({.position = {px, py, pz}, .mass = 1.0f});
        velocities.push_back({vx, vy, vz});
    }

    return {particles, velocities};
}
bool first_compare(std::pair<uint64_t,int>& a, std::pair<uint64_t,int>& b )
{
    return a.first < b.first;
}
std::tuple<std::vector<float>, uint64_t, uint64_t> scatter_particles_children(int size, int rank)
{
    int local_particle_count;
    uint64_t bound_keys[2];
    std::vector<float> local_particle_floats;
    MPI_Scatter(NULL, 2, MPI_UINT64_T, bound_keys, 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(NULL, 0, MPI_INT, &local_particle_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_particle_floats.resize(local_particle_count);
    std::cout<<local_particle_count<<" "<<rank<<std::endl;
    MPI_Scatterv(NULL,NULL,NULL,MPI_FLOAT,local_particle_floats.data(),local_particle_count,MPI_FLOAT,0,MPI_COMM_WORLD);
    return {std::move(local_particle_floats), bound_keys[0], bound_keys[1]};
}
std::tuple<std::vector<float>, uint64_t, uint64_t> scatter_particles_root(std::vector<float>& particle_data, int size, int rank)
{

    int local_particle_count;
    std::vector<float> local_particle_floats;

    auto particle_count = particle_data.size()/4;
    std::vector<std::pair<uint64_t, int>> morton_keys;
    morton_keys.reserve(particle_count);
    for (int i = 0;i< particle_count;i++)
    {
        morton_keys.emplace_back(
            Point{
                particle_data[i*4],
                particle_data[i*4+1],
                particle_data[i*4+2]
            }.morton_key(),i);
    }
    std::sort(morton_keys.begin(),morton_keys.end(),first_compare);

    std::vector<int> counts(size,(particle_count/size)*4);
    auto remainder=particle_count%size;
    for (int i = 0;i<remainder;i++)
    {
        counts[i]+=4;
    }
    std::vector<int> displacements;
    displacements.reserve(size);
    int disp = 0;
    for (int count : counts)
    {
        displacements.push_back(disp);
        disp+= count;
    }


    std::vector<uint64_t> morton_ranges(size*2);
    morton_ranges[0] = 0;
    morton_ranges[morton_ranges.size()-1] = 0;
    assert(particle_count > size);
    for (int p = 0; p < size-1;p++)
    {
        morton_ranges[p*2+1] = morton_keys[(displacements[p+1]/4)].first;
    }
    for (int p = 1; p < size; p++)
    {
        morton_ranges[p*2] = morton_keys[(displacements[p]/4)-1].first;
    }
    std::vector<float> sorted_particle_data;
    sorted_particle_data.reserve(particle_data.size());
    for (auto& p : morton_keys) {
        int idx = p.second;
        sorted_particle_data.push_back(particle_data[idx * 4 + 0]);
        sorted_particle_data.push_back(particle_data[idx * 4 + 1]);
        sorted_particle_data.push_back(particle_data[idx * 4 + 2]);
        sorted_particle_data.push_back(particle_data[idx * 4 + 3]);
    }
    particle_data = std::move(sorted_particle_data);

    uint64_t bound_keys[2];
    MPI_Scatter(morton_ranges.data(),2,MPI_UINT64_T, bound_keys,2, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(counts.data(),1,MPI_INT, &local_particle_count,1,MPI_INT,0,MPI_COMM_WORLD);
    local_particle_floats.resize(local_particle_count);
    MPI_Scatterv(particle_data.data(),counts.data(),displacements.data(),MPI_FLOAT,local_particle_floats.data(),local_particle_count,MPI_FLOAT,0,MPI_COMM_WORLD);
    return {std::move(local_particle_floats), bound_keys[0],bound_keys[1]};
}
std::tuple<std::vector<EncodedNode>,int, int> allgather_encoded_nodes(std::vector<EncodedNode>& input,int size, int rank)
{
    int local_count =  input.size();
    std::vector<int> counts(size);
    std::cout<<local_count<<"-"<< rank<<std::endl;
    MPI_Allgather(&local_count,1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displacements(size);
    displacements[0] = 0;
    for (int i = 1; i<size;i++)
    {
        displacements[i] = displacements[i-1]+ counts[i-1];
    }

    std::vector<EncodedNode> gather_nodes(displacements[size-1] + counts[size-1]);
    MPI_Allgatherv(input.data(), local_count, octree_node_datatype,gather_nodes.data(),counts.data(),displacements.data(),octree_node_datatype,MPI_COMM_WORLD);
    return {std::move(gather_nodes), displacements[rank], displacements[rank]+counts[rank]};
}

Octree* current_octree;
void communication_thread_loop(int rank, MPI_Comm communication_communicator)
{
    for (;;)
    {
        Request request; // NOLINT(*-pro-type-member-init)
        MPI_Status status;
        MPI_Recv(&request,1,request_datatype, MPI_ANY_SOURCE, KEY_REQUEST, communication_communicator, &status);
        if (request.level == -1)
        {
            break;
        }
        assert(current_octree != nullptr);
        std::unique_lock unique_lock(current_octree->octree_mutex);
        auto requested_node = current_octree->debug_find(request.morton_key,request.level);
        if (requested_node == nullptr)
        {
            std::cout<<"failed to find node"<<std::endl;
            current_octree->octree_map.at({request.morton_key,request.level});
        }
        EncodedNode encoded_nodes[8];
        for (int i = 0;i < 8; i++)
        {
            if (requested_node->children[i].has_value())
            {
                encoded_nodes[i] = current_octree->octree_map.at({requested_node->children[i].value(), request.level+1}).encode(rank);
                std::cout<<"sending "<<encoded_nodes[i].morton_key<<" at "<<encoded_nodes[i].level<<std::endl;
            }
            else
            {
                encoded_nodes[i].rank = -1;
            }
        }
        unique_lock.unlock();
        MPI_Send(encoded_nodes,8,octree_node_datatype,status.MPI_SOURCE, KEY_RESPONSE, communication_communicator);
    }
}
int main(int argc, char** argv){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename.txt>" << std::endl;
        return 1;
    }

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Error: MPI implementation does not support MPI_THREAD_MULTIPLE" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    octree_node_datatype = OctreeNode::create_encoded_node_type();
    request_datatype = create_request_datatype();

    int size;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm communicator_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &communicator_comm);
    std::thread communication_thread(communication_thread_loop, rank, communicator_comm);

    uint64_t lower, higher;
    std::vector<float> local_particle_data;
    if (rank == 0)
    {
        std::string filename = argv[1];
        std::ifstream inFile(filename);
        if (!inFile) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return 1;
        }
        std::vector<float> particle_data;
        float px, py, pz, vx,vy,vz, m ;
        while (inFile >> px >> py >> pz>>vx>>vy>>vz >> m) {
            particle_data.push_back(px);
            particle_data.push_back(py);
            particle_data.push_back(pz);
            particle_data.push_back(m);
        }
        inFile.close();
    std::cout<<"read file"<<rank<<std::endl;
        std::tie(local_particle_data,lower,higher) = scatter_particles_root(particle_data,size,rank);

        // std::vector<float> local_particle_floats =   scatter_particles_root(particle_data,size,rank);
    }else
    {
        std::tie(local_particle_data,lower,higher) = scatter_particles_children(size,rank);
    }
    Octree octree(local_particle_data, communicator_comm);

    // octree.sanity_check(0,0);
    auto encoded_branches = octree.local_branches(lower, higher, size, rank);
    auto [gathered_branches, disp, count] = allgather_encoded_nodes(encoded_branches, size, rank);



    current_octree = &octree;
    octree.add_branches(gathered_branches, disp, count);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<Point> forces;
    forces.resize(local_particle_data.size()/4);
    for (int i = 0; i < local_particle_data.size(); i+=4)
    {
        float px, py, pz,m;
        px = local_particle_data[i+0];
        py = local_particle_data[i+1];
        pz = local_particle_data[i+2];
        m = local_particle_data[i+3];
        Particle particle = {
            .position = {px,py,pz},
            .mass = m
        };
        forces[i/4] = octree.compute_force(particle);
    }

    std::cout<<"octree has size: "<<octree.debug_count(0,0)<<" "<<rank<<" debug counter: "<<debug_children_counter<<std::endl;
    // output into file named octree_[rank].txt containing every morton key of  leaf node in octree
    std::ofstream leaf_out("octree_" + std::to_string(rank) + ".txt");
    for (auto const& [key_pair, node] : octree.octree_map) {
        if (node.is_leaf) {
            leaf_out << key_pair.first << "\n";
        }
    }
    leaf_out.close();


    MPI_Barrier(MPI_COMM_WORLD);

    Request stop_request = {.morton_key = 0, .level = -1};
    MPI_Send(&stop_request, 1, request_datatype, rank, KEY_REQUEST, communicator_comm);
    communication_thread.join();
    std::vector<int> recv_counts(size);
    int local_force_count = forces.size() * 3;
    MPI_Gather(&local_force_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displacements(size, 0);
    std::vector<float> all_forces_flat;
    if (rank == 0) {
        int total_floats = 0;
        for (int i = 0; i < size; ++i) {
            displacements[i] = total_floats;
            total_floats += recv_counts[i];
        }
        all_forces_flat.resize(total_floats);
    }

    std::vector<float> local_forces_flat;
    for (const auto& f : forces) {
        local_forces_flat.push_back(f.x);
        local_forces_flat.push_back(f.y);
        local_forces_flat.push_back(f.z);
    }

    MPI_Gatherv(local_forces_flat.data(), local_force_count, MPI_FLOAT,
                all_forces_flat.data(), recv_counts.data(), displacements.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream force_out("forces.txt");
        for (size_t i = 0; i < all_forces_flat.size(); i += 3) {
            force_out << all_forces_flat[i] << " "
                      << all_forces_flat[i+1] << " "
                      << all_forces_flat[i+2] << "\n";
        }
        force_out.close();
        std::cout << "Forces written to forces.txt" << std::endl;
    }
    std::cout<< "my branches cover "<<debug_encoded_node_children_counter<<" as rank "<<rank<<std::endl;
    MPI_Comm_free(&communicator_comm);
    MPI_Type_free(&octree_node_datatype);
    MPI_Type_free(&request_datatype);
    MPI_Finalize();
    return 0;

}