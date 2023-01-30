#pragma once

#ifndef CYLINDER
#define CYLINDER

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <math.h>

#include "Normalizer.h"
#include "DataLoader.h"
#include "Feature.h"
#include "common_macro.hpp"
#include "Metric.h"
#include "yaml-cpp/yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

inline bool check_file_exists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else return false;
}
inline int int_floor(double x)
{
  int i = (int)x; /* truncate */
  return i - ( i > x ); /* convert trunc to floor */
}
inline std::string sanitize(std::string path) {
  if (path.back()=='/') return path;
  return path + "/";
}

class Cylinder {
public:
  float start_radius;   // [m]
  float end_radius;     // [m]
  int steps_num;        // [steps]
  int yaw_steps;        // [steps]
  int yaw_steps_half;   // [steps]
  float radius_step;    // [m]
  float yaw_res;        // [radiants]
  float z;              // [m]

  int tot_cells;  // redundant but for optimization (often used)
  int level;

  ExpMode expmode;

  std::vector<Cell>    grid;
  std::vector<Feature> features;
  std::vector<float>   area;
  
  Normalizer normalizer;
  Metric tmetric, gmetric;
  cv::Mat full_featMatrix, GT_labels_vector, featMatrix, predictions_vector;
  cv::PCA pca;
  std::vector<int> remap_idxs;
  std::vector<int> re_idx;

  int inherited_labels_size;

  double M_DOUBLE_PI = M_PI*2.0;

  std::string store_features_ofname;

  std::vector<int> inherit_idxs;
  int prevfeats_num;

  std::string load_path, save_path;
  cv::Ptr<cv::ml::SVM> model;
  std::vector<std::string> modes = {"geom", "geom_label", "geom_all", "geom_pca", "geom_pca_label", "geom_pca_all_label"};
  int max_feats_num, tot_geom_features_across_all_levels;
  int mode, pca_mode, trick_mode;

  Cylinder(YAML::Node &node);
  Cylinder(YAML::Node &node, Cylinder *cyl_, ExpMode expmode);
  
  void printSummary();

  void computeFeaturesCols();

  void loadPCAConfigs(YAML::Node &node);

  void sortBins_cyl(std::vector<Eigen::Vector3d> &points);

  void resetGrid();

  virtual void computeTravGT(std::vector<int> &labels);

  void computePredictedLabel(std::vector<int> &labels);

  void computeFeatures(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points);


  void loadSVM(YAML::Node &node);
  void process(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points);
  void computeAccuracy();
  void filterOutliers();

  void storeFeaturesToFile();

  void inheritFeatures(Cylinder *cyl_);
  void inheritGTFeatures(Cylinder *cyl_);

  void produceFeaturesRoutine(DataLoader &dl, Cylinder *back_cyl);
  void OnlineRoutine(DataLoader &dl, Cylinder *cyl_);


};


class Cylinder_SemKITTI : public Cylinder {
public:
  Cylinder_SemKITTI(YAML::Node &node);
  Cylinder_SemKITTI(YAML::Node &node,  Cylinder *cyl_, ExpMode expmode);
  void computeTravGT(std::vector<int> &labels);
};
class Cylinder_NuSc : public Cylinder {
public:
  Cylinder_NuSc(YAML::Node &node);
  Cylinder_NuSc(YAML::Node &node,  Cylinder *cyl_, ExpMode expmode);
  void computeTravGT(std::vector<int> &labels);
};

#endif // CYLINDER