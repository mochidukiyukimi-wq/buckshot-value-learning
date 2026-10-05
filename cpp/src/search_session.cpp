#include "roulette/search_session.hpp"
#include "roulette/shortcuts.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace roulette {
SearchSession::SearchSession(State root, SearchConfig config, std::uint64_t evaluator_version)
    : root_(std::move(root)), config_(config), evaluator_version_(evaluator_version) {
    validate_state(root_);
    if (is_terminal(root_))
        throw std::invalid_argument("search root must be a decision state");
    if (!config_.max_internal_nodes || !config_.max_frontier_nodes || !config_.value_batch_size ||
        !config_.label_buffer_size)
        throw std::invalid_argument("search capacities must be positive");
    add_node(root_, false);
}
NodeId SearchSession::add_node(const State &state, bool boundary) {
    StateKey key = make_state_key(state);
    // Boundary and internal occurrences are distinct computations, even for an equal information
    // state.
    StateKey cache_key = key;
    cache_key.push_back(boundary ? 1 : 0);
    if (config_.memoize) {
        if (const auto entry = cache_.find(cache_key); entry != cache_.end()) {
            ++stats_.cache_hits;
            return entry->second;
        }
    }
    const bool terminal = is_terminal(state);
    if (!terminal && (boundary ? stats_.frontier_nodes >= config_.max_frontier_nodes
                               : stats_.internal_nodes >= config_.max_internal_nodes)) {
        phase_ = Phase::Failed;
        return std::numeric_limits<NodeId>::max();
    }
    const NodeId id = nodes_.size();
    Node node;
    node.state = state;
    node.key = std::move(key);
    node.boundary = boundary;
    if (terminal) {
        node.solved = true;
        node.win_probability_p0 = terminal_value_p0(state);
    } else if (boundary) {
        frontier_.push_back(id);
        ++stats_.frontier_nodes;
    } else
        ++stats_.internal_nodes;
    nodes_.push_back(std::move(node));
    if (config_.memoize)
        cache_.emplace(std::move(cache_key), id);
    return id;
}
bool SearchSession::expand_node(NodeId id) {
    // Copy the state before node insertion: vector growth invalidates references.
    const State state = nodes_[id].state;
    std::vector<ActionEdges> action_edges;
    for (const auto &action : legal_actions(state)) {
        ActionEdges collected;
        collected.action = action;
        const auto outcomes = merge_identical_outcomes(collect_successors(state, action));
        for (const auto &outcome : outcomes) {
            ++stats_.transitions;
            if (outcome.kind == SuccessorKind::Terminal)
                ++stats_.terminal_transitions;
            const bool boundary =
                outcome.kind != SuccessorKind::Internal && outcome.kind != SuccessorKind::Terminal;
            if (boundary)
                ++stats_.boundary_transitions;
            if (!boundary && !is_terminal(outcome.state)) {
                const auto before =
                    std::pair{state.ammo.rounds, inventory_size(state.players[0].inventory) +
                                                     inventory_size(state.players[1].inventory)};
                const auto after =
                    std::pair{outcome.state.ammo.rounds,
                              inventory_size(outcome.state.players[0].inventory) +
                                  inventory_size(outcome.state.players[1].inventory)};
                // Drug exchanges one item after consuming itself, so the combined count still
                // decreases.
                if (!(after < before))
                    throw std::logic_error("non-decreasing internal transition");
            }
            const NodeId child = add_node(outcome.state, boundary);
            if (phase_ == Phase::Failed)
                return false;
            collected.edges.push_back({outcome.probability, child});
        }
        action_edges.push_back(std::move(collected));
    }
    nodes_[id].actions = std::move(action_edges);
    nodes_[id].expanded = true;
    return true;
}
bool SearchSession::backup_node(NodeId id) {
    auto &node = nodes_[id];
    for (const auto &action : node.actions)
        for (const auto &edge : action.edges)
            if (!nodes_[edge.child].solved) {
                backup_stack_.push_back(edge.child);
                return false;
            }
    for (const auto &action : node.actions) {
        std::vector<std::pair<double, double>> weighted_values;
        for (const auto &edge : action.edges)
            weighted_values.emplace_back(edge.probability, nodes_[edge.child].win_probability_p0);
        node.action_values_p0.push_back(chance_expectation(weighted_values));
    }
    const auto best = choose_best_action(node.state.actor, node.action_values_p0);
    node.win_probability_p0 = node.action_values_p0[best];
    node.solved = true;
    if (emitted_keys_.insert(node.key).second) {
        labels_.push_back(
            {node.state, to_actor_value(node.win_probability_p0, node.state.actor), node.key});
        ++stats_.emitted_labels;
    }
    return true;
}
SearchProgress SearchSession::advance(std::size_t work_budget) {
    if (!work_budget)
        throw std::invalid_argument("work budget must be positive");
    if (phase_ == Phase::Failed)
        return {SearchStatus::ResourceLimit, std::nullopt};
    if (pending_request_)
        return {SearchStatus::NeedsValues, pending_request_};
    if (labels_.size() >= config_.label_buffer_size)
        return {SearchStatus::NeedsLabelDrain, std::nullopt};
    for (std::size_t work = 0; work < work_budget; ++work) {
        if (phase_ == Phase::Expand) {
            while (expansion_cursor_ < nodes_.size() &&
                   (nodes_[expansion_cursor_].boundary || nodes_[expansion_cursor_].solved))
                ++expansion_cursor_;
            if (expansion_cursor_ < nodes_.size()) {
                if (!expand_node(expansion_cursor_++))
                    return {SearchStatus::ResourceLimit, std::nullopt};
            } else
                phase_ = Phase::Evaluate;
        } else if (phase_ == Phase::Evaluate) {
            if (frontier_cursor_ < frontier_.size()) {
                EvaluationRequest request{next_request_id_++, evaluator_version_, {}};
                const auto end =
                    std::min(frontier_.size(), frontier_cursor_ + config_.value_batch_size);
                for (; frontier_cursor_ < end; ++frontier_cursor_) {
                    waiting_nodes_.push_back(frontier_[frontier_cursor_]);
                    request.states.push_back(nodes_[frontier_[frontier_cursor_]].state);
                }
                pending_request_ = std::move(request);
                return {SearchStatus::NeedsValues, pending_request_};
            }
            backup_stack_.push_back(0);
            phase_ = Phase::Backup;
        } else if (phase_ == Phase::Backup) {
            stats_.maximum_stack = std::max(stats_.maximum_stack, backup_stack_.size());
            if (backup_stack_.empty()) {
                phase_ = Phase::Done;
                return {SearchStatus::Complete, std::nullopt};
            }
            const NodeId id = backup_stack_.back();
            if (nodes_[id].solved || backup_node(id))
                backup_stack_.pop_back();
            if (labels_.size() >= config_.label_buffer_size)
                return {SearchStatus::NeedsLabelDrain, std::nullopt};
        } else if (phase_ == Phase::Done)
            return {SearchStatus::Complete, std::nullopt};
        else
            return {SearchStatus::ResourceLimit, std::nullopt};
    }
    return {SearchStatus::Yielded, std::nullopt};
}
void SearchSession::submit_values(std::uint64_t request_id,
                                  const std::vector<double> &actor_values) {
    if (!pending_request_ || pending_request_->request_id != request_id)
        throw std::invalid_argument("stale evaluation request id");
    if (actor_values.size() != waiting_nodes_.size())
        throw std::invalid_argument("value count mismatch");
    for (const auto value : actor_values)
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument("actor value outside finite [0,1]");
    for (std::size_t index = 0; index < waiting_nodes_.size(); ++index) {
        auto &node = nodes_[waiting_nodes_[index]];
        node.win_probability_p0 = to_actor_value(actor_values[index], node.state.actor);
        node.solved = true;
    }
    waiting_nodes_.clear();
    pending_request_.reset();
}
std::vector<Label> SearchSession::take_labels(std::size_t max_rows) {
    if (phase_ == Phase::Failed) {
        labels_.clear();
        return {};
    }
    std::vector<Label> labels;
    while (!labels_.empty() && labels.size() < max_rows) {
        labels.push_back(std::move(labels_.front()));
        labels_.pop_front();
    }
    return labels;
}
SearchResult SearchSession::result() const {
    if (phase_ != Phase::Done)
        throw std::logic_error("search is incomplete");
    const auto &node = nodes_.front();
    std::vector<Action> actions;
    for (const auto &action : node.actions)
        actions.push_back(action.action);
    return {node.win_probability_p0,
            to_actor_value(node.win_probability_p0, node.state.actor),
            actions,
            node.action_values_p0,
            choose_best_action(node.state.actor, node.action_values_p0),
            stats()};
}
SearchStats SearchSession::stats() const {
    auto statistics = stats_;
    statistics.approximate_memory_bytes = nodes_.capacity() * sizeof(Node);
    for (const auto &node : nodes_) {
        statistics.approximate_memory_bytes += node.key.capacity();
        for (const auto &action : node.actions)
            statistics.approximate_memory_bytes += action.edges.capacity() * sizeof(Edge);
    }
    // This is an estimate; process peak RSS is separately measured in Python.
    statistics.approximate_memory_bytes += cache_.size() * (sizeof(NodeId) + 175 + 32);
    return statistics;
}
} // namespace roulette
