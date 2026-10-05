#pragma once
#include "action.hpp"
#include "chance.hpp"
#include <optional>

namespace roulette {
enum class SuccessorKind : int {
    Terminal,
    Internal,
    ReplenishmentBoundary,
    ReloadBoundary,
    BothBoundaries
};
struct Successor {
    double probability;
    State state;
    SuccessorKind kind;
};
struct ForcedBranch {
    double probability;
    State state;
    bool reload = false;
    bool replenish = false;
};
struct PendingTransition {
    std::vector<ForcedBranch> branches;
};
PendingTransition begin_action(const State &state, const Action &action);
// The cursor streams the cartesian product; no giant product vector is allocated.
class SuccessorCursor {
  public:
    explicit SuccessorCursor(PendingTransition pending);
    std::optional<Successor> next();

  private:
    PendingTransition pending_;
    std::size_t branch_ = 0, inventory_ = 0, reload_ = 0;
    bool prepared_ = false;
    std::vector<InventoryOutcome> inventory_outcomes_;
    std::vector<ReloadOutcome> reload_outcomes_;
    void prepare_branch();
};
SuccessorCursor successors(const State &state, const Action &action);
SuccessorCursor advance_forced_events(PendingTransition pending);
std::vector<Successor> collect_successors(const State &state, const Action &action);
Successor sample_outcome(SuccessorCursor cursor, RandomEngine &rng);
} // namespace roulette
