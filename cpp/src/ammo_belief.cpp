#include "roulette/ammo_belief.hpp"
#include <limits>
#include <numeric>
#include <stdexcept>

namespace roulette {
std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("belief weight addition overflow");
    return left + right;
}
std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right)
        throw std::overflow_error("belief weight multiplication overflow");
    return left * right;
}
std::uint64_t total_weight(const AmmoBelief &belief) {
    std::uint64_t total = 0;
    for (const auto weight : belief.weights)
        total = checked_add(total, weight);
    return total;
}
void validate_belief(const AmmoBelief &belief, bool allow_empty) {
    if (belief.rounds < 0 || belief.rounds > max_round_count ||
        (!allow_empty && belief.rounds == 0))
        throw std::invalid_argument("round count must be 1..8 for a decision state");
    for (int live = 0; live <= max_round_count; ++live) {
        for (int current = 0; current <= 1; ++current) {
            const bool feasible = belief.rounds > 0 && live <= belief.rounds && live >= current &&
                                  belief.rounds - live >= 1 - current;
            if (!feasible && belief.weights[2 * live + current] != 0)
                throw std::invalid_argument("impossible ammunition hypothesis");
        }
    }
    if (belief.rounds > 0 && total_weight(belief) == 0)
        throw std::invalid_argument("zero probability mass");
}
AmmoBelief canonicalize_belief(AmmoBelief belief) {
    validate_belief(belief, true);
    std::uint64_t divisor = 0;
    for (const auto weight : belief.weights)
        divisor = std::gcd(divisor, weight);
    if (divisor != 0)
        for (auto &weight : belief.weights)
            weight /= divisor;
    return belief;
}
AmmoBelief make_reload_belief(int live, int blank) {
    if (live < 0 || blank < 0 || live + blank < 1 || live + blank > max_round_count)
        throw std::invalid_argument("invalid ammunition counts");
    AmmoBelief belief;
    belief.rounds = live + blank;
    belief.weights[2 * live] = static_cast<std::uint64_t>(blank);
    belief.weights[2 * live + 1] = static_cast<std::uint64_t>(live);
    return canonicalize_belief(belief);
}
AmmoBelief invert_current_round(const AmmoBelief &belief) {
    validate_belief(belief);
    AmmoBelief inverted;
    inverted.rounds = belief.rounds;
    for (int live = 0; live <= max_round_count; ++live) {
        for (int current = 0; current <= 1; ++current) {
            const auto weight = belief.weights[2 * live + current];
            if (!weight)
                continue;
            const int new_live = live + 1 - 2 * current;
            const int index = 2 * new_live + 1 - current;
            inverted.weights[index] = checked_add(inverted.weights[index], weight);
        }
    }
    return canonicalize_belief(inverted);
}
std::pair<double, AmmoBelief> observe_current_round(const AmmoBelief &belief, Round round) {
    validate_belief(belief);
    const int current = static_cast<int>(round);
    if (current < 0 || current > 1)
        throw std::invalid_argument("invalid round type");
    AmmoBelief posterior = belief;
    for (int live = 0; live <= max_round_count; ++live)
        posterior.weights[2 * live + 1 - current] = 0;
    const auto observed_weight = total_weight(posterior);
    if (!observed_weight)
        return {0.0, AmmoBelief{}};
    const double probability = static_cast<double>(observed_weight) / total_weight(belief);
    return {probability, canonicalize_belief(posterior)};
}
AmmoBelief consume_current_round(const AmmoBelief &belief) {
    validate_belief(belief);
    AmmoBelief next;
    next.rounds = belief.rounds - 1;
    if (!next.rounds)
        return next;
    // All hypotheses share the same denominator n-1. Integer numerators preserve exact keys.
    for (int live = 0; live <= max_round_count; ++live) {
        for (int current = 0; current <= 1; ++current) {
            const auto weight = belief.weights[2 * live + current];
            if (!weight)
                continue;
            const int remaining_live = live - current;
            const int remaining_blank = next.rounds - remaining_live;
            const int index = 2 * remaining_live;
            next.weights[index] =
                checked_add(next.weights[index], checked_multiply(weight, remaining_blank));
            next.weights[index + 1] =
                checked_add(next.weights[index + 1], checked_multiply(weight, remaining_live));
        }
    }
    return canonicalize_belief(next);
}
} // namespace roulette
