#pragma once

struct Cell {
  std::vector<int> points_idx;
  int label;
  float predicted_label;
};