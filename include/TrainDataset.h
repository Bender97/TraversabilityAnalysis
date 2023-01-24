#pragma once
#ifndef TRAINDATASET_H
#define TRAINDATASET_H

#include <iostream>
#include <fstream>
#include <vector>

#include <algorithm>
#include <random>
#include <stdexcept>
#include "Feature.h"
#include "Metric.h"
#include "Normalizer.h"
#include "yaml-cpp/yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <sstream>

class TrainDataset {
public:
  // std::vector<Feature> features;
  // std::vector<int> labels;
  std::string filename;

  std::vector<std::string> modes;

  // std::vector<std::vector<float>> features_v;
  // std::vector<int> labels_v;

  cv::Mat X_train, y_train;

  int num_entries_to_train_on;
  int counters[2];
  bool need_to_balance_flag;
  int level;
  int tot_cells;
  std::vector<float> nu_vec, C_vec, gamma_vec;

  int tot_geom_features, tot_geom_features_across_all_levels;
  int mode, pca_mode, trick_mode, ref_label_idx;

  std::vector<bool> taken;
  Normalizer normalizer;

  Metric post_train_metric;
  Metric post_valid_metric;

  int cont_trav, cont_nontrav, dfs, tot_samples_found;

  cv::PCA pca;
  int training_cols;
  std::vector<int> re_idx;

  std::vector<int> sampled_idxs;

  TrainDataset(std::string filename_, YAML::Node &node, int level_, int tot_geom_features_, int tot_cells_);
  TrainDataset(std::string filename_, YAML::Node &node, Normalizer &normalizer_, cv::PCA &pca_, int level_, int tot_geom_features_, int tot_cells_);
  void load();
  void init(std::string filename_, YAML::Node &node, int level_, int tot_geom_features_, int tot_cells_);
  void summary();
  void computeTrainingCols();

  void readLabels();
  void loadData();
  void sampleIdxs();
  void checkFileAndConfigAreValid(int feats_size, int tot_entries, int cont_trav, int cont_nontrav);
};

#endif // TRAINDATASET_H