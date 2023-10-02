#ifndef MAIN
#define MAIN

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "Feature.h"
#include "Synchro.h"
#include "Cylinder.h"
#include "cv_ext.h"
#include "common_funcs.hpp"


#include "yaml-cpp/yaml.h"

using namespace std::chrono_literals;

YAML::Node sample_data = YAML::LoadFile("../models/test.yaml");

std::vector<Eigen::Vector3d> points;
std::vector<int> labels;

Synchro synchro_;

std::vector<Cylinder> cyls;

int main (int argc, char** argv) {
  
  std::vector<int> seqs = sample_data["general"]["split"]["train"].as<std::vector<int>>();
  int visualization_offset = sample_data["general"]["vis_offset"].as<int>();
  int sample_idx_start = 0, sample_idx_end = 150; 
  if (sample_data["general"]["sample_idx_start"]) sample_idx_start = sample_data["general"]["sample_idx_start"].as<int>();
  if (sample_data["general"]["sample_idx_end"]) sample_idx_end = sample_data["general"]["sample_idx_end"].as<int>();

  loadCyls(cyls, &synchro_, sample_data, true);

  cv_ext::BasicTimer bt;

  bool reset_view;

  ProgressBar pBar(100, "read seq ");

  for (auto &seq : seqs) {

    std::cout << "parsing seq " << seq << std::endl;
    reset_view=true;
    
    for (int sample_idx=sample_idx_start; sample_idx < sample_idx_end; sample_idx++) {
      bt.reset();

      //std::cout << "." << std::flush;
      //pBar.update((float)sample_idx / 4071.0f);

      points.clear();
      labels.clear();
      synchro_.pauseGeometryUpdate();

      readData(seq, sample_idx, points, labels, sample_data);

      Eigen::MatrixXd scene_normal = computeSceneNormal(points);
      
      for (size_t i=0; i<cyls.size(); i++)
        cyls[i].produceFeaturesRoutine(points, labels, scene_normal, 
                                       (!i) ? nullptr : &(cyls[i-1]));

      auto pointcloud = open3d::geometry::PointCloud(points);
      paintCloud_cyl(pointcloud);

      auto pointcloud_ptr = std::make_shared<open3d::geometry::PointCloud>(pointcloud);
      
      synchro_.addPointCloud(pointcloud_ptr);

      for (auto &cyl: cyls) cyl.updateTriangGT();

      synchro_.cv.notify_one();

      if (visualization_offset>0) {
        int rem = MAX(0, visualization_offset-bt.elapsedTimeMs());
        std::this_thread::sleep_for(std::chrono::milliseconds(rem));
      }

      if (reset_view) { synchro_.resetView(); reset_view=false; }
    }
  }

  synchro_.join();

  return 0;
}
#endif
