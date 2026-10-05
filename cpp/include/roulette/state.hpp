#pragma once
#include "ammo_belief.hpp"

namespace roulette {
struct PlayerState {
    int hp = hp_limit;
    Inventory inventory{};
    bool knife = false;
    bool skip_opponent = false;
    bool cuff_blocked = false;
    bool operator==(const PlayerState &) const = default;
};
struct State {
    std::array<PlayerState, 2> players{};
    PlayerId actor = 0;
    AmmoBelief ammo = make_reload_belief(1, 1);
    bool operator==(const State &) const = default;
};
PlayerId current_player(const State &state);
PlayerId other_player(PlayerId player);
bool is_terminal(const State &state);
void validate_state(const State &state);
std::array<float, feature_count> state_features(const State &state);
State swap_players(State state);
} // namespace roulette
