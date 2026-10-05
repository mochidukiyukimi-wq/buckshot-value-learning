#pragma once
#include "state.hpp"

namespace roulette {
double terminal_value_p0(const State &state);
double chance_expectation(const std::vector<std::pair<double, double>> &weighted_values);
std::size_t choose_best_action(PlayerId actor, const std::vector<double> &action_values_p0);
double to_actor_value(double win_probability_p0, PlayerId actor);
} // namespace roulette
