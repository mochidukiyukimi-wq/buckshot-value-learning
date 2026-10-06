#pragma once
#include "types.hpp"

namespace roulette {
struct FeatureRange {
    int start;
    int count;
    constexpr int stop() const { return start + count; }
};

struct PlayerFeature {
    static constexpr int hp = 0;
    static constexpr int inventory_count = hp + 1;
    static constexpr int knife = inventory_count + 1;
    static constexpr int skip_opponent = knife + 1;
    static constexpr int cuff_blocked = skip_opponent + 1;
    static constexpr int count = cuff_blocked + 1;
};

enum class TokenType : int { Global, Player, Item, Count };
enum class OwnerRole : int { Global, Actor, Opponent, Count };

// Native serialization and Python tokenization use this same layout.
// Changes to column or token meanings require a new input_schema version.
struct FeatureSchema {
    static constexpr int round_count_column = 0;
    static constexpr FeatureRange belief_probabilities{round_count_column + 1,
                                                     ammo_hypothesis_count};
    static constexpr FeatureRange global_features{0, belief_probabilities.stop()};
    static constexpr int player_feature_count = PlayerFeature::count;
    static constexpr int player_count = roulette::player_count;
    static constexpr FeatureRange player_features{global_features.stop(),
                                                 player_count * player_feature_count};
    static constexpr FeatureRange item_features{player_features.stop(),
                                               player_count * roulette::inventory_limit};
    static constexpr FeatureRange numeric_features{global_features.start,
                                                  player_features.stop() - global_features.start};
    static constexpr int feature_count = item_features.stop();

    static constexpr int actor_player_index = 0;
    static constexpr int opponent_player_index = actor_player_index + 1;
    static constexpr int hp_field = PlayerFeature::hp;
    static constexpr int inventory_count_field = PlayerFeature::inventory_count;
    static constexpr int round_count_limit = max_round_count;
    static constexpr int hp_limit = roulette::hp_limit;
    static constexpr int inventory_limit = roulette::inventory_limit;
    static constexpr int empty_item_id = 0;
    static constexpr int first_item_id = empty_item_id + 1;
    static constexpr int item_vocabulary_size = first_item_id + item_count;

    static constexpr int global_token_index = 0;
    static constexpr int player_token_start = global_token_index + 1;
    static constexpr int item_token_start = player_token_start + player_count;
    static constexpr int token_count = item_token_start + item_features.count;
    static constexpr int token_type_count = static_cast<int>(TokenType::Count);
    static constexpr int owner_role_count = static_cast<int>(OwnerRole::Count);

    static constexpr FeatureRange player_columns(int relative_player_index) {
        return {player_features.start + relative_player_index * player_feature_count,
                player_feature_count};
    }
    static constexpr FeatureRange inventory_columns(int relative_player_index) {
        return {item_features.start + relative_player_index * inventory_limit, inventory_limit};
    }
};

inline constexpr auto token_types = [] {
    std::array<int, FeatureSchema::token_count> types{};
    types[FeatureSchema::global_token_index] = static_cast<int>(TokenType::Global);
    for (int player = 0; player < FeatureSchema::player_count; ++player) {
        types[FeatureSchema::player_token_start + player] = static_cast<int>(TokenType::Player);
        for (int slot = 0; slot < FeatureSchema::inventory_limit; ++slot)
            types[FeatureSchema::item_token_start + player * FeatureSchema::inventory_limit + slot] =
                static_cast<int>(TokenType::Item);
    }
    return types;
}();

inline constexpr auto token_owner_roles = [] {
    std::array<int, FeatureSchema::token_count> roles{};
    roles[FeatureSchema::global_token_index] = static_cast<int>(OwnerRole::Global);
    for (int player = 0; player < FeatureSchema::player_count; ++player) {
        const auto role = player == FeatureSchema::actor_player_index ? OwnerRole::Actor
                                                                    : OwnerRole::Opponent;
        roles[FeatureSchema::player_token_start + player] = static_cast<int>(role);
        for (int slot = 0; slot < FeatureSchema::inventory_limit; ++slot)
            roles[FeatureSchema::item_token_start + player * FeatureSchema::inventory_limit + slot] =
                static_cast<int>(role);
    }
    return roles;
}();

inline constexpr int feature_count = FeatureSchema::feature_count;
} // namespace roulette
