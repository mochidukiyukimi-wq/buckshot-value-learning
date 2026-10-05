#pragma once
#include "rules.hpp"

namespace roulette {
std::vector<Successor> merge_identical_outcomes(const std::vector<Successor> &outcomes);
bool prove_immediate_win(const State &state);
} // namespace roulette
