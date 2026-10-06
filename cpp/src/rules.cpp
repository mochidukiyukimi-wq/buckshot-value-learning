#include "roulette/rules.hpp"
#include <algorithm>
#include <stdexcept>

namespace roulette {
PlayerId current_player(const State &state) { return state.actor; }
PlayerId other_player(PlayerId player) {
    if (player != 0 && player != 1)
        throw std::invalid_argument("invalid player id");
    return 1 - player;
}
bool is_terminal(const State &state) {
    return state.players[0].hp == 0 || state.players[1].hp == 0;
}
void validate_state(const State &state) {
    other_player(state.actor);
    for (const auto &player : state.players) {
        if (player.hp < 0 || player.hp > hp_limit)
            throw std::invalid_argument("invalid HP");
        if (inventory_size(player.inventory) > inventory_limit)
            throw std::invalid_argument("inventory capacity exceeded");
    }
    if (state.players[0].hp == 0 && state.players[1].hp == 0)
        throw std::invalid_argument("simultaneous deaths are not possible");
    validate_belief(state.ammo, is_terminal(state));
    const auto &actor = state.players[state.actor];
    const auto &opponent = state.players[other_player(state.actor)];
    if (opponent.knife || opponent.skip_opponent || actor.cuff_blocked)
        throw std::invalid_argument("effects inconsistent with current turn");
    if (actor.skip_opponent && !opponent.cuff_blocked)
        throw std::invalid_argument("skip requires cuff reapplication restriction");
}
State swap_players(State state) {
    std::swap(state.players[0], state.players[1]);
    state.actor = other_player(state.actor);
    return state;
}
Action make_shot(ShotTarget target) {
    Action action;
    action.target = target;
    return action;
}
Action make_item_use(Item item) {
    if (item == Item::Drug)
        return make_steal(-1);
    Action action;
    action.kind = ActionKind::UseItem;
    action.item = item;
    return action;
}
Action make_steal(int item) {
    if (item < -1 || item >= item_count || item == static_cast<int>(Item::Drug))
        throw std::invalid_argument("Drug cannot steal Drug");
    Action action;
    action.kind = ActionKind::Steal;
    action.item = Item::Drug;
    action.stolen_item = item;
    return action;
}
std::vector<Action> legal_actions(const State &state) {
    validate_state(state);
    if (is_terminal(state))
        return {};
    std::vector<Action> actions{make_shot(ShotTarget::Self), make_shot(ShotTarget::Opponent)};
    const auto &actor = state.players[state.actor];
    const auto &opponent = state.players[other_player(state.actor)];
    for (int item = 0; item < item_count; ++item) {
        if (!actor.inventory[item])
            continue;
        if (item == static_cast<int>(Item::Drug)) {
            actions.push_back(make_steal(-1));
            for (int target = 0; target < item_count - 1; ++target)
                if (opponent.inventory[target])
                    actions.push_back(make_steal(target));
        } else
            actions.push_back(make_item_use(static_cast<Item>(item)));
    }
    return actions;
}
PendingTransition begin_action(const State &state, const Action &action) {
    const auto actions = legal_actions(state);
    if (std::find(actions.begin(), actions.end(), action) == actions.end())
        throw std::invalid_argument("illegal action");
    const PlayerId actor_id = state.actor, opponent_id = other_player(actor_id);
    State consumed = state;
    if (action.kind != ActionKind::Shoot)
        --consumed.players[actor_id].inventory[static_cast<int>(action.item)];
    PendingTransition pending;
    auto append = [&](double probability, State next, bool change_turn) {
        if (is_terminal(next)) {
            // Terminal takes priority over draws, reloads and effect cleanup.
            pending.branches.push_back({probability, next, false, false});
            return;
        }
        if (change_turn) {
            next.actor = opponent_id;
            next.players[opponent_id].cuff_blocked = false;
        }
        pending.branches.push_back({probability, next, next.ammo.rounds == 0, change_turn});
    };
    const bool observes_ammo = action.kind == ActionKind::Shoot ||
                               (action.kind == ActionKind::UseItem &&
                                (action.item == Item::Beer || action.item == Item::Magnifier));
    if (observes_ammo) {
        for (const auto round : {Round::Blank, Round::Live}) {
            auto [probability, posterior] = observe_current_round(consumed.ammo, round);
            if (probability == 0)
                continue;
            State next = consumed;
            next.ammo = posterior;
            if (action.kind == ActionKind::UseItem && action.item == Item::Magnifier) {
                append(probability, next, false);
                continue;
            }
            next.ammo = consume_current_round(posterior);
            bool change_turn = false;
            if (action.kind == ActionKind::Shoot) {
                auto &actor = next.players[actor_id];
                const int damage = round == Round::Live ? (actor.knife ? 2 : 1) : 0;
                actor.knife = false;
                if (action.target == ShotTarget::Self) {
                    actor.hp = std::max(0, actor.hp - damage);
                    if (round == Round::Live) {
                        actor.skip_opponent = false;
                        change_turn = true;
                    }
                } else {
                    auto &opponent = next.players[opponent_id];
                    opponent.hp = std::max(0, opponent.hp - damage);
                    if (actor.skip_opponent)
                        actor.skip_opponent = false;
                    else
                        change_turn = true;
                }
            }
            append(probability, next, change_turn);
        }
        return pending;
    }
    auto &actor = consumed.players[actor_id];
    auto &opponent = consumed.players[opponent_id];
    if (action.kind == ActionKind::Steal) {
        if (action.stolen_item >= 0) {
            --opponent.inventory[action.stolen_item];
            if (inventory_size(actor.inventory) < inventory_limit)
                ++actor.inventory[action.stolen_item];
        }
        append(1.0, consumed, false);
        return pending;
    }
    switch (action.item) {
    case Item::Cigarette:
        actor.hp = std::min(hp_limit, actor.hp + 1);
        break;
    case Item::Knife:
        actor.knife = true;
        break;
    case Item::Handcuffs:
        if (!opponent.cuff_blocked) {
            actor.skip_opponent = true;
            opponent.cuff_blocked = true;
        }
        break;
    case Item::Inverter:
        consumed.ammo = invert_current_round(consumed.ammo);
        break;
    case Item::Medicine: {
        State healed = consumed;
        healed.players[actor_id].hp = std::min(hp_limit, actor.hp + 2);
        State harmed = consumed;
        --harmed.players[actor_id].hp;
        append(0.6, healed, false);
        append(0.4, harmed, false);
        return pending;
    }
    case Item::Beer:
    case Item::Magnifier:
    case Item::Drug:
        throw std::logic_error("item branch should have been handled already");
    }
    append(1.0, consumed, false);
    return pending;
}
SuccessorCursor::SuccessorCursor(PendingTransition pending) : pending_(std::move(pending)) {}
void SuccessorCursor::prepare_branch() {
    const auto &branch = pending_.branches[branch_];
    inventory_outcomes_ =
        branch.replenish ? item_draw_outcomes(branch.state.players[branch.state.actor].inventory)
                         : std::vector<InventoryOutcome>{
                               {1.0, branch.state.players[branch.state.actor].inventory}};
    reload_outcomes_ =
        branch.reload ? reload_outcomes() : std::vector<ReloadOutcome>{{1.0, branch.state.ammo}};
    inventory_ = reload_ = 0;
    prepared_ = true;
}
std::optional<Successor> SuccessorCursor::next() {
    if (branch_ == pending_.branches.size())
        return std::nullopt;
    if (!prepared_)
        prepare_branch();
    const auto &branch = pending_.branches[branch_];
    const auto &inventory = inventory_outcomes_[inventory_];
    const auto &ammo = reload_outcomes_[reload_];
    State state = branch.state;
    state.players[state.actor].inventory = inventory.inventory;
    state.ammo = ammo.ammo;
    SuccessorKind kind = SuccessorKind::Internal;
    if (is_terminal(state))
        kind = SuccessorKind::Terminal;
    else if (branch.reload && branch.replenish)
        kind = SuccessorKind::BothBoundaries;
    else if (branch.reload)
        kind = SuccessorKind::ReloadBoundary;
    else if (branch.replenish)
        kind = SuccessorKind::ReplenishmentBoundary;
    const double probability = branch.probability * inventory.probability * ammo.probability;
    if (++reload_ == reload_outcomes_.size()) {
        reload_ = 0;
        if (++inventory_ == inventory_outcomes_.size()) {
            ++branch_;
            prepared_ = false;
        }
    }
    return Successor{probability, state, kind};
}
SuccessorCursor successors(const State &state, const Action &action) {
    return SuccessorCursor(begin_action(state, action));
}
SuccessorCursor advance_forced_events(PendingTransition pending) {
    return SuccessorCursor(std::move(pending));
}
std::vector<Successor> collect_successors(const State &state, const Action &action) {
    auto cursor = successors(state, action);
    std::vector<Successor> outcomes;
    while (auto outcome = cursor.next())
        outcomes.push_back(std::move(*outcome));
    return outcomes;
}
Successor sample_outcome(SuccessorCursor cursor, RandomEngine &rng) {
    std::vector<Successor> outcomes;
    std::vector<double> probabilities;
    while (auto outcome = cursor.next()) {
        probabilities.push_back(outcome->probability);
        outcomes.push_back(std::move(*outcome));
    }
    return outcomes[sample_index(probabilities, rng)];
}
} // namespace roulette
