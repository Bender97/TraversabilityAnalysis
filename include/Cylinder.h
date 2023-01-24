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
#include "Feature.h"
#include "Metric.h"
#include "Synchro.h"
#include "yaml-cpp/yaml.h"
#include "open3d/Open3D.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

class Cylinder {
public:
  float start_radius;   // [m]
  float end_radius;     // [m]
  int steps_num;        // [steps]
  int yaw_steps;        // [steps]
  int yaw_steps_half;        // [steps]
  float radius_step;
  float yaw_res;
  float z;

  int tot_cells;  // redundant but for optimization (often used)
  std::shared_ptr<open3d::geometry::TriangleMesh> mesh;

  std::vector<Cell> grid;
  std::vector<Feature> features;
  std::vector<float> area;
  
  Normalizer normalizer;
  Metric tmetric, gmetric;
  cv::Mat X_train, y_train, tmp, predictions_vector;
  cv::PCA pca;
  std::vector<int> remap_idxs;

  int der_feats_size;

  double M_DOUBLE_PI = M_PI*2.0;

  std::string store_features_ofname;
  Synchro *synchro_;

  std::vector<int> idxs;
  std::vector<int> temp_color;

  int level;

  cv::Ptr<cv::ml::SVM> model;
  std::vector<std::string> modes;
  int training_cols, tot_geom_features, tot_geom_features_across_all_levels;
  int mode=-1, pca_mode=-1, trick_mode=-1, ref_label_idx;
  std::vector<int> re_idx;

  Cylinder(YAML::Node &node, int tot_geom_features_);
  Cylinder(YAML::Node &node, Synchro *synchro__, Cylinder *cyl_, int tot_geom_features_, int produce_features=0);
  Cylinder(YAML::Node &node,  Cylinder *cyl_, int tot_geom_features_, int produce_features=0);
  
  void computeFeaturesCols();

  void createTriang();

  void updateTriang();
  void updateTriangGT();

  void sortBins_cyl(std::vector<Eigen::Vector3d> &points);

  void resetGrid();

  void computeTravGT_SemKITTI(std::vector<int> &labels);
  void computeTravGT_NuSc(std::vector<int> &labels);

  // void (Cylinder::*computeTravGT)(std::vector<int> &);
  std::function<void(std::vector<int> &)> computeTravGT;


  void computePredictedLabel(std::vector<int> &labels);


  void computeFeatures(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points);


  void loadSVM(std::string path);
  void process(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points);
  void computeAccuracy();
  void filterOutliers();

  void storeFeatures();
  void storeFeaturesToFile(std::string name);

  void inheritFeatures(Cylinder *cyl_);
  void inheritGTFeatures(Cylinder *cyl_);

  void produceFeaturesRoutine(std::vector<Eigen::Vector3d> &points, std::vector<int> &labels, Eigen::MatrixXd &scene_normal, Cylinder *cyl_);


};

#endif // CYLINDER