#include "roulette/state.hpp"

namespace roulette {
std::array<float, feature_count> state_features(const State &state) {
    validate_state(state);
    std::array<float, feature_count> features{};
    features[FeatureSchema::round_count_column] =
        static_cast<float>(state.ammo.rounds) / FeatureSchema::round_count_limit;

    const double probability_mass = static_cast<double>(total_weight(state.ammo));
    if (probability_mass > 0)
        for (int hypothesis = 0; hypothesis < FeatureSchema::belief_probabilities.count; ++hypothesis)
            features[FeatureSchema::belief_probabilities.start + hypothesis] =
                static_cast<float>(state.ammo.weights[hypothesis] / probability_mass);

    // Relative player order is actor, opponent, regardless of the absolute player IDs.
    for (int relative_player = 0; relative_player < FeatureSchema::player_count; ++relative_player) {
        const auto player_id = relative_player == FeatureSchema::actor_player_index
                                   ? state.actor
                                   : other_player(state.actor);
        const auto &player = state.players[player_id];
        const auto player_columns = FeatureSchema::player_columns(relative_player);
        features[player_columns.start + PlayerFeature::hp] =
            static_cast<float>(player.hp) / FeatureSchema::hp_limit;
        features[player_columns.start + PlayerFeature::inventory_count] =
            static_cast<float>(inventory_size(player.inventory)) / FeatureSchema::inventory_limit;
        features[player_columns.start + PlayerFeature::knife] = player.knife;
        features[player_columns.start + PlayerFeature::skip_opponent] = player.skip_opponent;
        features[player_columns.start + PlayerFeature::cuff_blocked] = player.cuff_blocked;

        const auto inventory_columns = FeatureSchema::inventory_columns(relative_player);
        int slot = 0;
        for (int item = 0; item < item_count; ++item)
            for (int copy = 0; copy < player.inventory[item]; ++copy)
                features[inventory_columns.start + slot++] =
                    static_cast<float>(item + FeatureSchema::first_item_id);
    }
    return features;
}
} // namespace roulette
