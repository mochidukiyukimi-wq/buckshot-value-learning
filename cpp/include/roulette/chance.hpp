#pragma once
#include "state.hpp"

namespace roulette {
struct InventoryOutcome {
    double probability;
    Inventory inventory;
};
struct ReloadOutcome {
    double probability;
    AmmoBelief ammo;
};
std::vector<InventoryOutcome> item_draw_outcomes(const Inventory &inventory);
std::vector<ReloadOutcome> reload_outcomes();
std::vector<std::pair<double, AmmoBelief>> observation_outcomes(const AmmoBelief &belief);
std::size_t sample_index(const std::vector<double> &probabilities, RandomEngine &rng);
State sample_initial_state(std::uint64_t seed);
} // namespace roulette
