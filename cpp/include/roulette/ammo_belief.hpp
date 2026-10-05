#pragma once
#include "types.hpp"
#include <utility>

namespace roulette {
struct AmmoBelief {
    int rounds = 0;
    std::array<std::uint64_t, 18> weights{};
    bool operator==(const AmmoBelief &) const = default;
};
std::uint64_t checked_add(std::uint64_t left, std::uint64_t right);
std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right);
std::uint64_t total_weight(const AmmoBelief &belief);
AmmoBelief canonicalize_belief(AmmoBelief belief);
void validate_belief(const AmmoBelief &belief, bool allow_empty = false);
AmmoBelief make_reload_belief(int live, int blank);
AmmoBelief invert_current_round(const AmmoBelief &belief);
std::pair<double, AmmoBelief> observe_current_round(const AmmoBelief &belief, Round round);
AmmoBelief consume_current_round(const AmmoBelief &belief);
} // namespace roulette
