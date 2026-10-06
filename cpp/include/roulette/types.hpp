#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace roulette {
inline constexpr int hp_limit = 4;
inline constexpr int inventory_limit = 7;
inline constexpr int item_count = 8;
inline constexpr int player_count = 2;
inline constexpr int max_round_count = 8;
inline constexpr int ammo_hypothesis_count = (max_round_count + 1) * 2;
inline constexpr const char *rules_version = "vrchat-pvp-1.5-default-immediate-20261005";
inline constexpr const char *input_schema = "actor-normalized-43f-17tokens-v1";
enum class Item : int { Beer, Cigarette, Knife, Magnifier, Handcuffs, Inverter, Medicine, Drug };
enum class Round : int { Blank = 0, Live = 1 };
using PlayerId = int;
using Inventory = std::array<std::uint8_t, item_count>;
using RandomEngine = std::mt19937_64;
int inventory_size(const Inventory &inventory);
} // namespace roulette
