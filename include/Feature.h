#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "Cell.h"
#include "common_macro.hpp"

class Feature {
  public:
    float linearity;
    float planarity;
    float sphericity;
    float omnivariance;
    float anisotropy;
    float eigenentropy;
    float sum_of_eigenvalues;
    float curvature;
    float angle;
    float goodness_of_fit;
    float roughness;
    float nvx;
    float nvy;
    float nvz;
    float unevenness;
    float surface_density;
    float z_diff;

    float cx, cy, cz, d1, d2, d3, d, normal_magnitude, numpoints;
    float a11, a12, a13, a22, a23, a33, ax, ay, az;

    Eigen::MatrixXd matA1;
    Eigen::Vector3d *p;

    std::vector<float> derived_features;

    void computeCorrelationMatrix(std::vector<int> &points_idx, std::vector<Eigen::Vector3d> &points);

    // Feature();

    std::string toString();

    std::vector<float> toVector();

    std::vector<float> toVectorTransformed();

    void toFile(std::ofstream &out);

    int fromFileLine(std::ifstream &in, int derived_features_num);
    int ignoreFeatureFromFile(std::ifstream &in, int derived_features_num);

    int computeFeatures(Cell *cell, Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points, float area);
};
