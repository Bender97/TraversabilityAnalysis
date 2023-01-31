#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include "cv_ext.h"
#include "yaml-cpp/yaml.h"

class Metric {
public:
  uint64_t tp, tn, fp, fn, tot;
  cv_ext::BasicTimer bt;
  uint64_t checkpointTime_;

  Metric();
  void update(float pred, float gt);
  Metric& operator+=(const Metric& rhs);
  float avgTP() const;
  float avgTN() const;
  float avgFP() const;
  float avgFN() const;
  float acc() const ;

  void resetTime();
  void resetAcc();
  void resetAll();
  void checkpointTime();

  void log2YAML(float nu, float gamma, float C, int pca, int row, int tot_cells, std::string folderpath);

  void print(const char *msg, int tot_cells=1, int tot_workers=1) const;
  void printLight(const char *msg, int tot_cells=1, int tot_workers=1) const;
  std::string getresults() const;
};