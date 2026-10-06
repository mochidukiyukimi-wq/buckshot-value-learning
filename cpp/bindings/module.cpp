#include "roulette/sampling.hpp"
#include "roulette/search_session.hpp"
#include "roulette/shortcuts.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace roulette;

namespace {
py::array_t<float> to_numpy_batch(const std::vector<State> &states) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(states.size()), static_cast<py::ssize_t>(feature_count)});
    auto rows = array.mutable_unchecked<2>();
    for (std::size_t row = 0; row < states.size(); ++row) {
        const auto features = state_features(states[row]);
        for (int column = 0; column < feature_count; ++column)
            rows(row, column) = features[column];
    }
    return array;
}
py::dict labels_to_numpy(const std::vector<Label> &labels) {
    std::vector<State> states;
    states.reserve(labels.size());
    py::array_t<float> values(labels.size());
    py::list keys;
    for (std::size_t row = 0; row < labels.size(); ++row) {
        states.push_back(labels[row].state);
        values.mutable_data()[row] = static_cast<float>(labels[row].win_probability_actor);
        keys.append(py::bytes(labels[row].key));
    }
    py::dict batch;
    batch["features"] = to_numpy_batch(states);
    batch["values"] = values;
    batch["keys"] = keys;
    return batch;
}
std::vector<double> read_value_array(const py::array &array) {
    if (array.ndim() != 1 || !(array.flags() & py::array::c_style))
        throw std::invalid_argument("values must be a contiguous rank-one NumPy array");
    const bool fp32 = array.dtype().is(py::dtype::of<float>());
    const bool fp64 = array.dtype().is(py::dtype::of<double>());
    if (!fp32 && !fp64)
        throw std::invalid_argument("values dtype must be float32 or float64");
    std::vector<double> values(array.size());
    for (py::ssize_t row = 0; row < array.size(); ++row)
        values[row] = fp32 ? static_cast<const float *>(array.data())[row]
                           : static_cast<const double *>(array.data())[row];
    return values;
}
void bind_feature_schema(py::module_ &module) {
    py::class_<FeatureRange>(module, "FeatureRange")
        .def_readonly("start", &FeatureRange::start)
        .def_readonly("count", &FeatureRange::count)
        .def_property_readonly("stop", &FeatureRange::stop)
        .def_property_readonly("columns", [](const FeatureRange &range) {
            return py::slice(range.start, range.stop(), 1);
        });
    py::class_<FeatureSchema>(module, "FeatureSchema")
        .def_readonly_static("global_features", &FeatureSchema::global_features)
        .def_readonly_static("belief_probabilities", &FeatureSchema::belief_probabilities)
        .def_readonly_static("player_features", &FeatureSchema::player_features)
        .def_readonly_static("item_features", &FeatureSchema::item_features)
        .def_readonly_static("numeric_features", &FeatureSchema::numeric_features)
        .def_readonly_static("feature_count", &FeatureSchema::feature_count)
        .def_readonly_static("round_count_column", &FeatureSchema::round_count_column)
        .def_readonly_static("player_feature_count", &FeatureSchema::player_feature_count)
        .def_readonly_static("player_count", &FeatureSchema::player_count)
        .def_readonly_static("actor_player_index", &FeatureSchema::actor_player_index)
        .def_readonly_static("opponent_player_index", &FeatureSchema::opponent_player_index)
        .def_readonly_static("hp_field", &FeatureSchema::hp_field)
        .def_readonly_static("inventory_count_field", &FeatureSchema::inventory_count_field)
        .def_readonly_static("round_count_limit", &FeatureSchema::round_count_limit)
        .def_readonly_static("hp_limit", &FeatureSchema::hp_limit)
        .def_readonly_static("inventory_limit", &FeatureSchema::inventory_limit)
        .def_readonly_static("empty_item_id", &FeatureSchema::empty_item_id)
        .def_readonly_static("first_item_id", &FeatureSchema::first_item_id)
        .def_readonly_static("item_vocabulary_size", &FeatureSchema::item_vocabulary_size)
        .def_readonly_static("global_token_index", &FeatureSchema::global_token_index)
        .def_readonly_static("token_count", &FeatureSchema::token_count)
        .def_readonly_static("token_type_count", &FeatureSchema::token_type_count)
        .def_readonly_static("owner_role_count", &FeatureSchema::owner_role_count)
        .def_readonly_static("token_types", &token_types)
        .def_readonly_static("token_owner_roles", &token_owner_roles)
        .def("player_columns", [](const FeatureSchema &, int relative_player_index) {
            other_player(relative_player_index); // Reject an index outside actor/opponent.
            return FeatureSchema::player_columns(relative_player_index);
        })
        .def("inventory_columns", [](const FeatureSchema &, int relative_player_index) {
            other_player(relative_player_index);
            return FeatureSchema::inventory_columns(relative_player_index);
        });
    module.attr("FEATURE_SCHEMA") = py::cast(FeatureSchema{});
}
void bind_types(py::module_ &module) {
    py::enum_<Item>(module, "Item")
        .value("BEER", Item::Beer)
        .value("CIGARETTE", Item::Cigarette)
        .value("KNIFE", Item::Knife)
        .value("MAGNIFIER", Item::Magnifier)
        .value("HANDCUFFS", Item::Handcuffs)
        .value("INVERTER", Item::Inverter)
        .value("MEDICINE", Item::Medicine)
        .value("DRUG", Item::Drug);
    py::enum_<Round>(module, "Round").value("BLANK", Round::Blank).value("LIVE", Round::Live);
    py::enum_<ActionKind>(module, "ActionKind")
        .value("SHOOT", ActionKind::Shoot)
        .value("USE_ITEM", ActionKind::UseItem)
        .value("STEAL", ActionKind::Steal);
    py::enum_<ShotTarget>(module, "ShotTarget")
        .value("SELF", ShotTarget::Self)
        .value("OPPONENT", ShotTarget::Opponent);
    py::enum_<SuccessorKind>(module, "SuccessorKind")
        .value("TERMINAL", SuccessorKind::Terminal)
        .value("INTERNAL", SuccessorKind::Internal)
        .value("REPLENISHMENT", SuccessorKind::ReplenishmentBoundary)
        .value("RELOAD", SuccessorKind::ReloadBoundary)
        .value("BOTH", SuccessorKind::BothBoundaries);
    py::class_<AmmoBelief>(module, "AmmoBelief")
        .def(py::init<>())
        .def_readwrite("rounds", &AmmoBelief::rounds)
        .def_readwrite("weights", &AmmoBelief::weights)
        .def("__eq__", [](const AmmoBelief &a, const AmmoBelief &b) { return a == b; });
    py::class_<PlayerState>(module, "PlayerState")
        .def(py::init<>())
        .def_readwrite("hp", &PlayerState::hp)
        .def_readwrite("inventory", &PlayerState::inventory)
        .def_readwrite("knife", &PlayerState::knife)
        .def_readwrite("skip_opponent", &PlayerState::skip_opponent)
        .def_readwrite("cuff_blocked", &PlayerState::cuff_blocked);
    py::class_<State>(module, "State")
        .def(py::init<>())
        .def_readwrite("actor", &State::actor)
        .def_readwrite("ammo", &State::ammo)
        .def(
            "player",
            [](State &state, int player) -> PlayerState & {
                other_player(player);
                return state.players[player];
            },
            py::return_value_policy::reference_internal)
        .def("copy", [](const State &state) { return state; })
        .def("__eq__", [](const State &a, const State &b) { return a == b; });
    py::class_<Action>(module, "Action")
        .def_readonly("kind", &Action::kind)
        .def_readonly("target", &Action::target)
        .def_readonly("item", &Action::item)
        .def_readonly("stolen_item", &Action::stolen_item)
        .def("__eq__", [](const Action &a, const Action &b) { return a == b; })
        .def("__repr__", [](const Action &action) {
            return "Action(kind=" + std::to_string(static_cast<int>(action.kind)) +
                   ", target=" + std::to_string(static_cast<int>(action.target)) +
                   ", item=" + std::to_string(static_cast<int>(action.item)) +
                   ", stolen_item=" + std::to_string(action.stolen_item) + ")";
        });
    py::class_<Successor>(module, "Successor")
        .def_readonly("probability", &Successor::probability)
        .def_readonly("state", &Successor::state)
        .def_readonly("kind", &Successor::kind);
    py::class_<PendingTransition>(module, "PendingTransition");
    py::class_<SuccessorCursor>(module, "SuccessorCursor").def("next", &SuccessorCursor::next);
    py::class_<InventoryOutcome>(module, "InventoryOutcome")
        .def_readonly("probability", &InventoryOutcome::probability)
        .def_readonly("inventory", &InventoryOutcome::inventory);
    py::class_<ReloadOutcome>(module, "ReloadOutcome")
        .def_readonly("probability", &ReloadOutcome::probability)
        .def_readonly("ammo", &ReloadOutcome::ammo);
    py::class_<SamplingDomain>(module, "SamplingDomain")
        .def(py::init<>())
        .def_readwrite("max_items_per_player", &SamplingDomain::max_items_per_player)
        .def_readwrite("max_initial_shell_type_count",
                       &SamplingDomain::max_initial_shell_type_count)
        .def_readwrite("include_effects", &SamplingDomain::include_effects)
        .def_readwrite("include_replenishment_boundaries",
                       &SamplingDomain::include_replenishment_boundaries);
}
void bind_search(py::module_ &module) {
    py::class_<SearchConfig>(module, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("max_internal_nodes", &SearchConfig::max_internal_nodes)
        .def_readwrite("max_frontier_nodes", &SearchConfig::max_frontier_nodes)
        .def_readwrite("value_batch_size", &SearchConfig::value_batch_size)
        .def_readwrite("label_buffer_size", &SearchConfig::label_buffer_size)
        .def_readwrite("memoize", &SearchConfig::memoize);
    py::class_<SearchStats>(module, "SearchStats")
        .def_readonly("internal_nodes", &SearchStats::internal_nodes)
        .def_readonly("frontier_nodes", &SearchStats::frontier_nodes)
        .def_readonly("cache_hits", &SearchStats::cache_hits)
        .def_readonly("transitions", &SearchStats::transitions)
        .def_readonly("terminal_transitions", &SearchStats::terminal_transitions)
        .def_readonly("boundary_transitions", &SearchStats::boundary_transitions)
        .def_readonly("emitted_labels", &SearchStats::emitted_labels)
        .def_readonly("maximum_stack", &SearchStats::maximum_stack)
        .def_readonly("approximate_memory_bytes", &SearchStats::approximate_memory_bytes);
    py::enum_<SearchStatus>(module, "SearchStatus")
        .value("NEEDS_VALUES", SearchStatus::NeedsValues)
        .value("NEEDS_LABEL_DRAIN", SearchStatus::NeedsLabelDrain)
        .value("YIELDED", SearchStatus::Yielded)
        .value("COMPLETE", SearchStatus::Complete)
        .value("RESOURCE_LIMIT", SearchStatus::ResourceLimit);
    py::class_<EvaluationRequest>(module, "EvaluationRequest")
        .def_readonly("request_id", &EvaluationRequest::request_id)
        .def_readonly("evaluator_version", &EvaluationRequest::evaluator_version)
        .def_property_readonly(
            "features",
            [](const EvaluationRequest &request) { return to_numpy_batch(request.states); })
        .def_property_readonly("keys", [](const EvaluationRequest &request) {
            py::list keys;
            for (const auto &state : request.states)
                keys.append(py::bytes(make_state_key(state)));
            return keys;
        });
    py::class_<SearchProgress>(module, "SearchProgress")
        .def_readonly("status", &SearchProgress::status)
        .def_readonly("request", &SearchProgress::request);
    py::class_<SearchResult>(module, "SearchResult")
        .def_readonly("win_probability_p0", &SearchResult::win_probability_p0)
        .def_readonly("win_probability_actor", &SearchResult::win_probability_actor)
        .def_readonly("actions", &SearchResult::actions)
        .def_readonly("action_values_p0", &SearchResult::action_values_p0)
        .def_readonly("best_action_index", &SearchResult::best_action_index)
        .def_readonly("stats", &SearchResult::stats);
    py::class_<SearchSession>(module, "SearchSession")
        .def(py::init<State, SearchConfig, std::uint64_t>())
        .def("advance", &SearchSession::advance, py::call_guard<py::gil_scoped_release>())
        .def("submit_values",
             [](SearchSession &session, std::uint64_t request_id, const py::array &array) {
                 const auto values = read_value_array(array);
                 py::gil_scoped_release release;
                 session.submit_values(request_id, values);
             })
        .def("take_labels",
             [](SearchSession &session, std::size_t max_rows) {
                 return labels_to_numpy(session.take_labels(max_rows));
             })
        .def("result", &SearchSession::result)
        .def("stats", &SearchSession::stats);
}
} // namespace
PYBIND11_MODULE(_native, module) {
    bind_feature_schema(module);
    bind_types(module);
    bind_search(module);
    module.attr("RULES_VERSION") = rules_version;
    module.attr("INPUT_SCHEMA") = input_schema;
    module.attr("FEATURE_COUNT") = feature_count;
    module.def("make_reload_belief", &make_reload_belief)
        .def("canonicalize_belief", &canonicalize_belief)
        .def("invert_current_round", &invert_current_round)
        .def("observe_current_round", &observe_current_round)
        .def("consume_current_round", &consume_current_round)
        .def("checked_multiply", &checked_multiply)
        .def("checked_add", &checked_add)
        .def("total_weight", &total_weight)
        .def("inventory_size", &inventory_size)
        .def("validate_state", &validate_state)
        .def("is_terminal", &is_terminal)
        .def("current_player", &current_player)
        .def("other_player", &other_player)
        .def("swap_players", &swap_players)
        .def("make_shot", &make_shot)
        .def("make_item_use", &make_item_use)
        .def("make_steal", &make_steal)
        .def("legal_actions", &legal_actions)
        .def("begin_action", &begin_action)
        .def("advance_forced_events", &advance_forced_events)
        .def("successors", &successors)
        .def("collect_successors", &collect_successors)
        .def("item_draw_outcomes", &item_draw_outcomes)
        .def("reload_outcomes", &reload_outcomes)
        .def("observation_outcomes", &observation_outcomes)
        .def("sample_initial_state", &sample_initial_state)
        .def("sample_roots", &sample_roots, py::call_guard<py::gil_scoped_release>())
        .def("sample_trajectory_roots", &sample_trajectory_roots,
             py::call_guard<py::gil_scoped_release>())
        .def("validate_sampling_domain", &validate_sampling_domain)
        .def("make_state_key",
             [](const State &state) {
                 validate_state(state);
                 return py::bytes(make_state_key(state));
             })
        .def("encode_states", &to_numpy_batch)
        .def("prove_immediate_win", &prove_immediate_win)
        .def("terminal_value_p0", &terminal_value_p0)
        .def("to_actor_value", &to_actor_value)
        .def("choose_best_action", &choose_best_action)
        .def("chance_expectation", &chance_expectation)
        .def("sample_transition", [](const State &state, const Action &action, std::uint64_t seed) {
            RandomEngine rng(seed);
            return sample_outcome(successors(state, action), rng);
        });
}
