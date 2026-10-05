#include "roulette/shortcuts.hpp"
#include "roulette/state_key.hpp"
#include <unordered_map>

namespace roulette {
std::vector<Successor> merge_identical_outcomes(const std::vector<Successor> &outcomes) {
    std::unordered_map<StateKey, std::size_t, StateKeyHash> indices;
    std::vector<Successor> merged;
    for (const auto &outcome : outcomes) {
        auto key = make_state_key(outcome.state);
        key.push_back(static_cast<char>(outcome.kind));
        const auto [entry, inserted] = indices.emplace(key, merged.size());
        if (inserted)
            merged.push_back(outcome);
        else
            merged[entry->second].probability += outcome.probability;
    }
    return merged;
}
bool prove_immediate_win(const State &state) {
    for (const auto &action : legal_actions(state)) {
        bool all_win = true;
        for (const auto &successor : collect_successors(state, action))
            if (!is_terminal(successor.state) ||
                successor.state.players[other_player(state.actor)].hp != 0)
                all_win = false;
        if (all_win)
            return true;
    }
    return false;
}
} // namespace roulette
