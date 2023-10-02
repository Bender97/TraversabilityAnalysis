#pragma once
#include <vector>

struct Cell {
  std::vector<int> points_idx;
  int status; // predictable / unpredictable
  int label;  // ground truth: traversable / not traversable
  float predicted_label; // prediction: traversable / not traversable
  float area_inverse;
};