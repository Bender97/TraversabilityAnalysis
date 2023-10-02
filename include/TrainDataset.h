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
#include "Cylinder.h"
#include "yaml-cpp/yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include <sstream>

class Dataset4 : public Cylinder {
public:
  cv::Mat X_train, y_train;
  cv::Mat fullX, fullY;

  int num_entries_to_train_on;
  int counters[2];
  bool need_to_balance_flag;
  std::vector<float> nu_vec, C_vec, gamma_vec;

  int ref_label_idx;
  int32_t seed;

  std::vector<bool> taken;

  Metric post_train_metric;
  Metric post_valid_metric;

  int cont_trav, cont_nontrav, inherited_labels_size, tot_samples_found;

  std::vector<int> sampled_idxs;

  Dataset4();
  Dataset4(YAML::Node &node_cyl, int level_);
  Dataset4(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_);
  void load();
  virtual void parseYAMLConfig(YAML::Node &node_cyl, int level_);
  void summary();

  void createPCAOnAll();

  void readLabels();
  void loadData();
  virtual void sampleIdxs();
  void checkFileAndConfigAreValid(int feats_size, int tot_entries, int cont_trav, int cont_nontrav);
};

class Dataset4Train : public Dataset4 {
public:
  Dataset4Train();
  Dataset4Train(YAML::Node &node_cyl, int level_);
  void parseYAMLConfig(YAML::Node &node_cyl, int level_);
  void sampleIdxs();
};

class Dataset4Valid : public Dataset4 {
public:
  Dataset4Valid();
  Dataset4Valid(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_);
  void parseYAMLConfig(YAML::Node &node_cyl, int level_);
  void sampleIdxs();
};

class Dataset4Test : public Dataset4 {
public:
  Dataset4Test();
  Dataset4Test(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_);
  void parseYAMLConfig(YAML::Node &node_cyl, int level_);
  void sampleIdxs();
};

#endif // TRAINDATASET_H