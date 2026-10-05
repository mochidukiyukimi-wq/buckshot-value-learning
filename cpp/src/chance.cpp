#include "roulette/chance.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace roulette {
int inventory_size(const Inventory &inventory) {
    return std::accumulate(inventory.begin(), inventory.end(), 0);
}
std::vector<InventoryOutcome> item_draw_outcomes(const Inventory &inventory) {
    const int occupied = inventory_size(inventory);
    if (occupied > inventory_limit)
        throw std::invalid_argument("inventory capacity exceeded");
    const int draws = std::min(2, inventory_limit - occupied);
    if (!draws)
        return {{1.0, inventory}};
    std::vector<InventoryOutcome> outcomes;
    for (int first = 0; first < item_count; ++first) {
        if (draws == 1) {
            Inventory drawn = inventory;
            ++drawn[first];
            outcomes.push_back({1.0 / 8.0, drawn});
        } else {
            for (int second = first; second < item_count; ++second) {
                Inventory drawn = inventory;
                ++drawn[first];
                ++drawn[second];
                outcomes.push_back({first == second ? 1.0 / 64.0 : 2.0 / 64.0, drawn});
            }
        }
    }
    return outcomes;
}
std::vector<ReloadOutcome> reload_outcomes() {
    std::vector<ReloadOutcome> outcomes;
    for (int live = 1; live <= 4; ++live)
        for (int blank = 1; blank <= 4; ++blank)
            outcomes.push_back({1.0 / 16.0, make_reload_belief(live, blank)});
    return outcomes;
}
std::vector<std::pair<double, AmmoBelief>> observation_outcomes(const AmmoBelief &belief) {
    std::vector<std::pair<double, AmmoBelief>> outcomes;
    for (const auto round : {Round::Blank, Round::Live}) {
        auto observation = observe_current_round(belief, round);
        if (observation.first > 0.0)
            outcomes.push_back(std::move(observation));
    }
    return outcomes;
}
std::size_t sample_index(const std::vector<double> &probabilities, RandomEngine &rng) {
    if (probabilities.empty())
        throw std::invalid_argument("empty outcome distribution");
    double total = 0.0;
    for (const double probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0)
            throw std::invalid_argument("invalid outcome probability");
        total += probability;
    }
    if (std::abs(total - 1.0) > 1e-10)
        throw std::invalid_argument("probability mass is not one");
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double target = uniform(rng);
    double cumulative = 0;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        cumulative += probabilities[index];
        if (target < cumulative)
            return index;
    }
    return probabilities.size() - 1;
}
State sample_initial_state(std::uint64_t seed) {
    RandomEngine rng(seed);
    State state;
    auto p0_draws = item_draw_outcomes(state.players[0].inventory);
    std::vector<double> probabilities;
    for (const auto &draw : p0_draws)
        probabilities.push_back(draw.probability);
    state.players[0].inventory = p0_draws[sample_index(probabilities, rng)].inventory;
    // Initial P2 distribution has exactly one item; ordinary turn start uses two.
    probabilities.assign(item_count, 1.0 / item_count);
    ++state.players[1].inventory[sample_index(probabilities, rng)];
    const auto reloads = reload_outcomes();
    probabilities.assign(reloads.size(), 1.0 / reloads.size());
    state.ammo = reloads[sample_index(probabilities, rng)].ammo;
    return state;
}
} // namespace roulette
