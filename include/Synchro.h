#pragma once

#ifndef SYNCHRO
#define SYNCHRO

#include <thread>
#include <condition_variable>
#include <mutex>
#include "open3d/Open3D.h"


class Synchro {
public:
  std::mutex mu;
  std::condition_variable cv;
  open3d::visualization::Visualizer vis;
  bool update_geometry;
  std::thread workerThread;

  Synchro(bool graphic=true);
  ~Synchro();

  void update();
  void pauseGeometryUpdate();

  void resetView();

  void addPointCloud(std::shared_ptr<const open3d::geometry::Geometry>     pointcloud_ptr);
  void  addPolarGrid(std::shared_ptr<const open3d::geometry::TriangleMesh> polar_grid_ptr);

  void join();
};

#endif // SYNCHRO