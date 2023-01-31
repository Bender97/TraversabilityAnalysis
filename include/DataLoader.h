#pragma once

#ifndef DATA_LOADER
#define DATA_LOADER


#include <iostream>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <eigen3/Eigen/Dense>
#include "yaml-cpp/yaml.h"
#include "ColorUtil.h"
#include "common_macro.hpp"
#include "open3d/Open3D.h"

class DataLoader {
protected:
  float f;
  int c;
  Eigen::Vector3d p;
  ColorUtil color_util;

public:
  DataLoader();
  std::vector<Eigen::Vector3d> points;
  std::vector<int> labels;
  Eigen::MatrixXd scene_normal;

  open3d::geometry::PointCloud pc;
  std::shared_ptr<open3d::geometry::VoxelGrid> voxel;

  void computeSceneNormal();
  int count_samples(YAML::Node &sample_data, int seq);


  virtual void readData(int seq, int idx, YAML::Node &sample_data);
  virtual void readPredicted(int seq, int idx, YAML::Node &sample_data);
  virtual std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  virtual std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);

  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud_DL();
  std::shared_ptr<const open3d::geometry::PointCloud> getUniformPaintedCloud();
};

class DataLoader_SemKITTI : public DataLoader {
protected:
  ColorUtil_SemKITTI color_util;
public:
  DataLoader_SemKITTI();
  void readData(int seq, int idx, YAML::Node &sample_data);
  void readPredicted(int seq, int idx, YAML::Node &sample_data);
  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);
};

class DataLoader_NuSc : public DataLoader {
protected:
  ColorUtil_NuSC color_util;
public:
  std::vector<std::string> pts_s, lab_s;
  DataLoader_NuSc();
  void readData(int, int idx, YAML::Node &);
  void readPredicted(int seq, int idx, YAML::Node &sample_data);
  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);
};

#endif // DATA_LOADER