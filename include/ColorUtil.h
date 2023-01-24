#pragma once

#ifndef COLOR_UTIL
#define COLOR_UTIL

#include <vector>
#include <cmath>
#include <Eigen/Dense>

struct Color {
  int r, g, b;
};

class ColorUtil {
public:
  std::vector<Color> colors;

  ColorUtil();

//  void setColor(pcl::PointXYZRGB &point, int &label);

  void setColor(Eigen::Vector3d &point, int &label);

//  void setColorBasedOnCoordinates(pcl::PointXYZRGB &point);

};

#endif // COLOR_UTIL