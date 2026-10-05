#include "roulette/state_key.hpp"

namespace roulette {
StateKey make_state_key(const State &source) {
    State state = source;
    state.ammo = canonicalize_belief(state.ammo);
    StateKey key;
    key.reserve(175);
    auto byte = [&](int value) { key.push_back(static_cast<char>(value)); };
    byte(state.actor);
    byte(state.ammo.rounds);
    for (const auto &player : state.players) {
        byte(player.hp);
        byte(player.knife);
        byte(player.skip_opponent);
        byte(player.cuff_blocked);
        for (const auto count : player.inventory)
            byte(count);
    }
    for (const auto weight : state.ammo.weights)
        for (int shift = 0; shift < 64; shift += 8)
            byte(static_cast<int>((weight >> shift) & 255));
    return key;
}
std::size_t StateKeyHash::operator()(const StateKey &key) const {
    return std::hash<std::string>{}(key);
}
bool equal_state_keys(const StateKey &left, const StateKey &right) { return left == right; }
} // namespace roulette
