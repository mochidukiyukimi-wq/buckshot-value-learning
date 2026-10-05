#include "roulette/backup.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace roulette {
double terminal_value_p0(const State &state) {
    if (!is_terminal(state))
        throw std::invalid_argument("state is not terminal");
    return state.players[1].hp == 0 ? 1.0 : 0.0;
}
double chance_expectation(const std::vector<std::pair<double, double>> &weighted_values) {
    double probability_mass = 0, expectation = 0;
    for (const auto &[probability, value] : weighted_values) {
        if (!std::isfinite(probability) || !std::isfinite(value) || probability < 0 || value < 0 ||
            value > 1)
            throw std::invalid_argument("invalid chance backup");
        probability_mass += probability;
        expectation += probability * value;
    }
    if (std::abs(probability_mass - 1) > 1e-9)
        throw std::logic_error("chance mass is not one");
    // Correct only floating-point accumulation drift; every child and the probability mass
    // have already been validated, so this cannot conceal an invalid branch value.
    return std::clamp(expectation, 0.0, 1.0);
}
std::size_t choose_best_action(PlayerId actor, const std::vector<double> &action_values_p0) {
    if (action_values_p0.empty())
        throw std::invalid_argument("no actions");
    other_player(actor);
    std::size_t best = 0;
    // Exact ties keep the first legal action: self shot, opponent shot, then item enum order.
    for (std::size_t index = 1; index < action_values_p0.size(); ++index)
        if (actor == 0 ? action_values_p0[index] > action_values_p0[best]
                       : action_values_p0[index] < action_values_p0[best])
            best = index;
    return best;
}
double to_actor_value(double win_probability_p0, PlayerId actor) {
    other_player(actor);
    return actor == 0 ? win_probability_p0 : 1.0 - win_probability_p0;
}
} // namespace roulette
