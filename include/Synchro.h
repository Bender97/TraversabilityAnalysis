#pragma once

#ifndef SYNCHRO
#define SYNCHRO

#include <thread>
// #include <mutex>
#include <queue>
#include <eigen3/Eigen/Dense>
#include "open3d/Open3D.h"
#include "yaml-cpp/yaml.h"

#include "cv_ext.h"
#include "Cell.h"
#include "DataLoader.h"
#include "common_macro.hpp"

struct Info {
  float start_radius;   // [m]
  float end_radius;     // [m]
  int steps_num;        // [steps]
  int yaw_steps;        // [steps]
  int yaw_steps_half;        // [steps]
  float radius_step;
  float yaw_res;
  float z;
  int tot_cells;
};

class Synchro {
protected:
  void createTriangs();
  void updateTriang(int level, std::vector<Cell> &grid);
  void updateTriangPred(int level, std::vector<Cell> &grid);
  void updateTriangGT(int level, std::vector<Cell> &grid);

public:
  // std::mutex mu;
  bool update_geometry, reset_view_flag, spin_flag;
  std::thread workerThread;

  std::queue<std::shared_ptr<const open3d::geometry::PointCloud>> pointclouds_ptr;
  std::queue<std::shared_ptr<const open3d::geometry::VoxelGrid>> pointclouds_v_ptr;
  std::queue<std::shared_ptr<const open3d::geometry::TriangleMesh>> polar_grids_ptr;

  // one for each cylinder
  std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> meshes;
  std::vector<Info> infos;


  cv_ext::BasicTimer bt;

  Synchro(YAML::Node &sample_data, bool graphic=true);
  ~Synchro();

  void reset();

  void update();

  void resetViewFlag(bool reset_view_flag_=true);

  void addPointCloud(DataLoader &dl);
  void addPointCloudVoxeled(DataLoader &dl, float voxel_size);
  void addPolarGrid (int level, std::vector<Cell> &grid);
  void addPolarGridPred (int level, std::vector<Cell> &grid);
  void delay(int vis_offset);
  void join();


};

#endif // SYNCHRO