#pragma once
#include "state.hpp"

namespace roulette {
enum class ActionKind : int { Shoot, UseItem, Steal };
enum class ShotTarget : int { Self, Opponent };
struct Action {
    ActionKind kind = ActionKind::Shoot;
    ShotTarget target = ShotTarget::Opponent;
    Item item = Item::Beer;
    // -1 explicitly consumes Drug without stealing.
    int stolen_item = -1;
    bool operator==(const Action &) const = default;
};
Action make_shot(ShotTarget target);
Action make_item_use(Item item);
Action make_steal(int item);
std::vector<Action> legal_actions(const State &state);
} // namespace roulette
