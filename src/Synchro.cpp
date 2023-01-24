#include "Synchro.h"


Synchro::Synchro(bool graphic) {

  if (graphic) {
    update_geometry=false;
      std::unique_lock<std::mutex> lk(mu);
      vis.CreateVisualizerWindow("Open3D", 2000, 2000);
    lk.unlock();

    workerThread = std::thread(&Synchro::update, this); 
  }
}

Synchro::~Synchro() {
	if (workerThread.joinable()) workerThread.join();
}

void Synchro::update() {
  while(1) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk);
    if (update_geometry) {
      vis.UpdateGeometry();
      vis.UpdateRender();
    }
    vis.PollEvents();

    lk.unlock();
  }
}

void Synchro::pauseGeometryUpdate() {
	update_geometry = false;
}

void Synchro::resetView() {
	mu.lock();
  vis.ResetViewPoint(true);
  mu.unlock();
}

void Synchro::addPointCloud(std::shared_ptr<const open3d::geometry::Geometry> pointcloud_ptr) {
  mu.lock();
  vis.ClearGeometries();
  vis.AddGeometry(pointcloud_ptr, false);
  update_geometry=true;
  mu.unlock();
}

void Synchro::addPolarGrid(std::shared_ptr<const open3d::geometry::TriangleMesh> polar_grid_ptr) {
  mu.lock();
  vis.AddGeometry(polar_grid_ptr, false);
  mu.unlock();
  cv.notify_one();
}

void Synchro::join() {
	vis.Run();
}