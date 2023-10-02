#pragma once

#ifndef DATA_LOADER
#define DATA_LOADER


#include <iostream>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "yaml-cpp/yaml.h"
#include "ColorUtil.h"
#include "common_macro.hpp"

#if OPEN3D == 1
#include "open3d/Open3D.h"
#endif

class DataLoader {
protected:
  float f;
  int c;
  Eigen::Vector3d p;
  ColorUtil color_util;
  std::string dataset_path;

public:
  DataLoader(std::string dataset_path_);
  std::vector<Eigen::Vector3d> points;
  std::vector<int> labels;
  std::vector<int> pred_labels;
  Eigen::MatrixXd scene_normal;

#if OPEN3D == 1
  open3d::geometry::PointCloud pc;
  std::shared_ptr<open3d::geometry::VoxelGrid> voxel;
#endif

  void computeSceneNormal();
  virtual int count_samples(int seq);
  void assertConsistency();
  void assertDLConsistency();

  int count_files_in_folder(std::string folder_path);

  virtual void readData(int seq, int idx, YAML::Node &sample_data);
  virtual void readPredicted(int seq, int idx, YAML::Node &sample_data);
#if OPEN3D == 1
  virtual std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  virtual std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);

  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud_DL();
  std::shared_ptr<const open3d::geometry::PointCloud> getUniformPaintedCloud();
#endif
};

class DataLoader_SemKITTI : public DataLoader {
protected:
  ColorUtil_SemKITTI color_util;
public:
  DataLoader_SemKITTI(std::string dataset_path);
  void readData(int seq, int idx, YAML::Node &sample_data);
  void readPredicted(int seq, int idx, YAML::Node &sample_data);
  int count_samples(int seq);

#if OPEN3D == 1
  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);
#endif
};

class DataLoader_PandaSet : public DataLoader {
protected:
  ColorUtil_PandaSet color_util;
public:
  DataLoader_PandaSet(std::string dataset_path);
  void readData(int seq, int idx, YAML::Node &sample_data);
  void readPredicted(int seq, int idx, YAML::Node &sample_data);
  int count_samples(int seq);

#if OPEN3D == 1
  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  // std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);
#endif
};

class DataLoader_NuSc : public DataLoader {
protected:
  ColorUtil_NuSC color_util;
public:
  std::vector<std::string> pts_s, lab_s;
  DataLoader_NuSc(std::string recipe_path);
  void readData(int, int idx, YAML::Node &);
  void readPredicted(int seq, int idx, YAML::Node &sample_data);
  int count_samples(int);
#if OPEN3D == 1
  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);
#endif

  int size();

};

class DataLoader_PLY : public DataLoader {
protected:
  //DataLoader_PLY color_util;
  std::string ply_path;
public:
  DataLoader_PLY(std::string ply_path_);
  void readData(int, int idx, YAML::Node &);
  void readPLY(std::string ply_path);
  void readBin(std::string ply_path);
  // void readPredicted(int seq, int idx, YAML::Node &sample_data);
#if OPEN3D == 1
  std::shared_ptr<const open3d::geometry::PointCloud> getPaintedCloud();
  // std::shared_ptr<const open3d::geometry::VoxelGrid> getPaintedCloudVoxeled(float voxel_size);
#endif

};

#endif // DATA_LOADER