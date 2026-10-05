#pragma once
#include "rules.hpp"

namespace roulette {
// Uniform finite constraint set, not a claim of uniform reachable game states.
struct SamplingDomain {
    int max_items_per_player = 1;
    int max_initial_shell_type_count = 2;
    bool include_effects = false;
    bool include_replenishment_boundaries = true;
};
void validate_sampling_domain(const SamplingDomain &domain);
State sample_boundary_state(const SamplingDomain &domain, RandomEngine &rng);
std::vector<State> sample_roots(const SamplingDomain &domain, std::size_t count,
                                std::uint64_t seed);
std::vector<State> sample_trajectory_roots(std::size_t count, std::uint64_t seed);
} // namespace roulette
