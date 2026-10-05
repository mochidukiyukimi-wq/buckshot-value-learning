#include "roulette/sampling.hpp"
#include "roulette/state_key.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_set>

namespace roulette {
void validate_sampling_domain(const SamplingDomain &domain) {
    if (domain.max_items_per_player < 0 || domain.max_items_per_player > inventory_limit)
        throw std::invalid_argument("sampling inventory bound must be 0..7");
    if (domain.max_initial_shell_type_count < 1 || domain.max_initial_shell_type_count > 4)
        throw std::invalid_argument("sampling ammunition bound must be 1..4");
}
namespace {
std::vector<Inventory> inventory_catalogue(int max_items) {
    std::vector<Inventory> inventories;
    Inventory inventory{};
    std::function<void(int, int)> visit = [&](int item, int remaining) {
        if (item == item_count) {
            inventories.push_back(inventory);
            return;
        }
        for (int count = 0; count <= remaining; ++count) {
            inventory[item] = static_cast<std::uint8_t>(count);
            visit(item + 1, remaining - count);
        }
    };
    visit(0, max_items);
    return inventories;
}
struct BoundaryTemplate {
    State state;
    bool reload_compatible;
};
std::vector<BoundaryTemplate> boundary_catalogue(const SamplingDomain &domain) {
    std::vector<BoundaryTemplate> templates;
    std::unordered_set<StateKey, StateKeyHash> keys;
    const int bound = domain.max_initial_shell_type_count;
    auto append = [&](State state, bool reload_compatible) {
        if (keys.insert(make_state_key(state)).second)
            templates.push_back({state, reload_compatible});
    };
    for (int live = 1; live <= bound; ++live) {
        for (int blank = 1; blank <= bound; ++blank) {
            State state;
            state.ammo = make_reload_belief(live, blank);
            for (int knife = 0; knife <= (domain.include_effects ? 1 : 0); ++knife) {
                // skip=>blocked, but blocked can survive after the skip is consumed.
                for (int cuff_state = 0; cuff_state < (domain.include_effects ? 3 : 1);
                     ++cuff_state) {
                    state.players[0].knife = knife != 0;
                    state.players[0].skip_opponent = cuff_state == 2;
                    state.players[1].cuff_blocked = cuff_state != 0;
                    append(state, true);
                }
            }
        }
    }
    if (domain.include_replenishment_boundaries && domain.max_items_per_player >= 2) {
        for (int rounds = 1; rounds <= 2 * bound; ++rounds)
            for (int live = 0; live <= rounds; ++live) {
                State state;
                state.ammo = make_reload_belief(live, rounds - live);
                const bool reload_compatible =
                    live >= 1 && live <= bound && rounds - live >= 1 && rounds - live <= bound;
                append(state, reload_compatible);
            }
    }
    return templates;
}
} // namespace
std::vector<State> sample_roots(const SamplingDomain &domain, std::size_t count,
                                std::uint64_t seed) {
    validate_sampling_domain(domain);
    const auto inventories = inventory_catalogue(domain.max_items_per_player);
    const auto templates = boundary_catalogue(domain);
    RandomEngine rng(seed);
    std::uniform_int_distribution<std::size_t> inventory_choice(0, inventories.size() - 1);
    std::uniform_int_distribution<std::size_t> template_choice(0, templates.size() - 1);
    std::uniform_int_distribution<int> hp_choice(1, hp_limit), actor_choice(0, 1);
    std::vector<State> states;
    states.reserve(count);
    while (states.size() < count) {
        const auto &boundary = templates[template_choice(rng)];
        State state = boundary.state;
        state.players[0].inventory = inventories[inventory_choice(rng)];
        state.players[1].inventory = inventories[inventory_choice(rng)];
        if (!boundary.reload_compatible && inventory_size(state.players[0].inventory) < 2)
            continue;
        state.players[0].hp = hp_choice(rng);
        state.players[1].hp = hp_choice(rng);
        if (actor_choice(rng))
            state = swap_players(state);
        validate_state(state);
        states.push_back(std::move(state));
    }
    return states;
}
State sample_boundary_state(const SamplingDomain &domain, RandomEngine &rng) {
    return sample_roots(domain, 1, rng()).front();
}
std::vector<State> sample_trajectory_roots(std::size_t count, std::uint64_t seed) {
    RandomEngine rng(seed);
    State state = sample_initial_state(rng());
    std::vector<State> states;
    while (states.size() < count) {
        auto actions = legal_actions(state);
        std::uniform_int_distribution<std::size_t> choose(0, actions.size() - 1);
        auto outcome = sample_outcome(successors(state, actions[choose(rng)]), rng);
        if (outcome.kind != SuccessorKind::Internal && outcome.kind != SuccessorKind::Terminal)
            states.push_back(outcome.state);
        state = is_terminal(outcome.state) ? sample_initial_state(rng()) : outcome.state;
    }
    return states;
}
} // namespace roulette
