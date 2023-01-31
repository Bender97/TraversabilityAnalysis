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

class TrainDataset : public Cylinder {
public:
  cv::Mat X_train, y_train;

  int num_entries_to_train_on;
  int counters[2];
  bool need_to_balance_flag;
  std::vector<float> nu_vec, C_vec, gamma_vec;

  int ref_label_idx;

  std::vector<bool> taken;

  Metric post_train_metric;
  Metric post_valid_metric;

  int cont_trav, cont_nontrav, inherited_labels_size, tot_samples_found;

  std::vector<int> sampled_idxs;

  TrainDataset(YAML::Node &node_cyl, int level_, bool train_flag);
  TrainDataset(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_, bool train_flag);
  void load();
  void init(YAML::Node &node_cyl, int level_, bool train_flag=true);
  void summary();

  void readLabels();
  void loadData();
  void sampleIdxs();
  void checkFileAndConfigAreValid(int feats_size, int tot_entries, int cont_trav, int cont_nontrav);
};

#endif // TRAINDATASET_H