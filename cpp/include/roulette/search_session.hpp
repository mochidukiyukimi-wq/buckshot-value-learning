#pragma once
#include "backup.hpp"
#include "rules.hpp"
#include "state_key.hpp"
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace roulette {
using NodeId = std::size_t;
struct SearchConfig {
    std::size_t max_internal_nodes = 200000;
    std::size_t max_frontier_nodes = 1000000;
    std::size_t value_batch_size = 256;
    std::size_t label_buffer_size = 4096;
    bool memoize = true;
};
struct SearchStats {
    std::size_t internal_nodes = 0, frontier_nodes = 0, cache_hits = 0;
    std::size_t transitions = 0, terminal_transitions = 0, boundary_transitions = 0;
    std::size_t emitted_labels = 0, maximum_stack = 0, approximate_memory_bytes = 0;
};
struct Label {
    State state;
    double win_probability_actor;
    StateKey key;
};
struct EvaluationRequest {
    std::uint64_t request_id;
    std::uint64_t evaluator_version;
    std::vector<State> states;
};
enum class SearchStatus : int { NeedsValues, NeedsLabelDrain, Yielded, Complete, ResourceLimit };
struct SearchProgress {
    SearchStatus status;
    std::optional<EvaluationRequest> request;
};
struct SearchResult {
    double win_probability_p0;
    double win_probability_actor;
    std::vector<Action> actions;
    std::vector<double> action_values_p0;
    std::size_t best_action_index;
    SearchStats stats;
};
class SearchSession {
  public:
    SearchSession(State root, SearchConfig config, std::uint64_t evaluator_version);
    SearchProgress advance(std::size_t work_budget);
    void submit_values(std::uint64_t request_id, const std::vector<double> &actor_values);
    std::vector<Label> take_labels(std::size_t max_rows);
    SearchResult result() const;
    SearchStats stats() const;

  private:
    struct Edge {
        double probability;
        NodeId child;
    };
    struct ActionEdges {
        Action action;
        std::vector<Edge> edges;
    };
    struct Node {
        State state;
        StateKey key;
        bool boundary = false, solved = false, expanded = false;
        double win_probability_p0 = 0.0;
        std::vector<ActionEdges> actions;
        std::vector<double> action_values_p0;
    };
    enum class Phase { Expand, Evaluate, Backup, Done, Failed };
    State root_;
    SearchConfig config_;
    std::uint64_t evaluator_version_, next_request_id_ = 1;
    std::vector<Node> nodes_;
    std::unordered_map<StateKey, NodeId, StateKeyHash> cache_;
    std::unordered_set<StateKey, StateKeyHash> emitted_keys_;
    std::deque<Label> labels_;
    std::vector<NodeId> frontier_, waiting_nodes_, backup_stack_;
    std::optional<EvaluationRequest> pending_request_;
    std::size_t expansion_cursor_ = 0, frontier_cursor_ = 0;
    Phase phase_ = Phase::Expand;
    SearchStats stats_;
    NodeId add_node(const State &state, bool boundary);
    bool expand_node(NodeId id);
    bool backup_node(NodeId id);
};
} // namespace roulette
