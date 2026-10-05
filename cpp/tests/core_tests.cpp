#include "roulette/sampling.hpp"
#include "roulette/search_session.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace roulette;
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main() {
    for (const auto &outcome : reload_outcomes()) {
        require(invert_current_round(invert_current_round(outcome.ammo)) == outcome.ammo,
                "double inversion");
        double mass = 0;
        for (const auto &observation : observation_outcomes(outcome.ammo))
            mass += observation.first;
        require(std::abs(mass - 1) < 1e-12, "observation mass");
    }
    double mass = 0;
    for (const auto &outcome : item_draw_outcomes({}))
        mass += outcome.probability;
    require(std::abs(mass - 1) < 1e-12, "item distribution mass");
    State state;
    state.ammo = make_reload_belief(1, 0);
    state.players[1].hp = 1;
    SearchSession search(state, SearchConfig{}, 0);
    for (;;) {
        auto progress = search.advance(10);
        if (progress.status == SearchStatus::NeedsValues)
            search.submit_values(progress.request->request_id,
                                 std::vector<double>(progress.request->states.size(), 0.5));
        else if (progress.status == SearchStatus::NeedsLabelDrain)
            search.take_labels(4096);
        else if (progress.status == SearchStatus::Complete)
            break;
        else
            require(progress.status == SearchStatus::Yielded, "unexpected resource limit");
    }
    require(search.result().win_probability_actor == 1, "certain immediate win");
    require(search.result().actions.size() == 2, "all legal actions");
    std::cout << "core invariants passed\n";
}
