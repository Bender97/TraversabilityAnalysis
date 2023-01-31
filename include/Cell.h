#pragma once

struct Cell {
  std::vector<int> points_idx;
  int8_t status; // predictable / unpredictable
  int8_t label;  // ground truth: traversable / not traversable
  float predicted_label; // prediction: traversable / not traversable
};