#pragma once
#include "state.hpp"

namespace roulette {
using StateKey = std::string;
StateKey make_state_key(const State &state);
struct StateKeyHash {
    std::size_t operator()(const StateKey &key) const;
};
bool equal_state_keys(const StateKey &left, const StateKey &right);
} // namespace roulette
