#pragma once

struct Cell {
  //std::vector<pcl::PointXYZRGB *> points;
  std::vector<int> points_idx;
  int label;
  float predicted_label;
  float avg_label;
};