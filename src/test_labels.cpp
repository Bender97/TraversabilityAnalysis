
#ifndef TEST_LABELS
#define TEST_LABELS

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "Feature.h"
#include "Synchro.h"
#include "Cylinder.h"
#include "cv_ext.h"
#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include "open3d/Open3D.h"

#include "common_funcs.hpp"

#include "yaml-cpp/yaml.h"

std::vector<Eigen::Vector3d> points;
std::vector<int> labels, pred_labels;
bool invert = false;

using namespace std::chrono_literals;
Synchro synchro_;

std::vector<Cylinder> cyls;

std::string path;

bool already_written = false;

YAML::Node sample_data = YAML::LoadFile("test.yaml");


void handleOut(std::vector<Cylinder> &cyls, float step) {
  if (already_written) {
    std::cout << "\e[A"; // step
    std::cout << "\e[A";
    std::cout << "\e[A"; 
    std::cout << "\e[A"; 
  }
  
  std::cout << "progress: " << std::setw(6) << std::setprecision(4) << step*100 << std::endl;
  
  std::string msg = "acc" + std::to_string(cyls[0].level) +" : ";
  cyls[0].tmetric.printLight(msg.c_str(), -1, 1);
  msg = "glob acc" + std::to_string(cyls[0].level) +" : ";

  cyls[0].gmetric.printLight(msg.c_str(), -1, 1);
  
  already_written = true;
}


int main (int argc, char** argv)
{

  std::vector<int> seqs = sample_data["general"]["split"]["test"].as<std::vector<int>>();
  int visualization_offset = sample_data["general"]["vis_offset"].as<int>();
  std::string dataset_path = sample_data["general"]["dataset_path"].as<std::string>();
  std::string path = sample_data["general"]["predicted_path"].as<std::string>();
  
  cyls.clear();
  int tot_geom_features = sample_data["tot_geom_features"].as<int>();

  YAML::Node node = sample_data["general"]["cyl02"];
  
  Cylinder cyl = Cylinder(node, &synchro_, nullptr, tot_geom_features, -1);    
  cyls.push_back(cyl);


  cv_ext::BasicTimer bt;
  
  bool reset_view;

  // std::vector<std::string> pts_s, lab_s;
  // std::ifstream file("recipe.txt");
  // std::string str; 
  // while (std::getline(file, str)) {
  //   pts_s.push_back(str);
  //   std::getline(file, str);
  //   lab_s.push_back(str);
  // }

  for (auto &seq : seqs) {

    int tot_samples = count_samples(sample_data, seq);  

    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;
    reset_view=true;
    std::ofstream fileo("metrics_pvkd.txt");

    for (int sample_idx=sample_data["general"]["sample_idx_start"].as<int>(); sample_idx < tot_samples; sample_idx++) {

      bt.reset();

      points.clear();
      labels.clear();
      synchro_.pauseGeometryUpdate();

      // std::cout << pts_s[sample_idx] << std::endl;

      readData(seq, sample_idx, points, labels, sample_data);
      // readDataNu(pts_s[sample_idx], lab_s[sample_idx], points, labels, sample_data);
      readPredicted(seq, sample_idx, pred_labels, path);

      std::cout << "points " << points.size() << std::endl;
      std::cout << "labels " << labels.size() << std::endl;
      std::cout << "pred_labels " << pred_labels.size() << std::endl;

      
      cyls[0].resetGrid();
      cyls[0].sortBins_cyl(points);
      cyls[0].computeGridGroundTruth(labels);
      cyls[0].computePredictedLabel(pred_labels);
      //cyls[0].filterOutliers();
      cyls[0].computeAccuracy();
      handleOut(cyls, sample_idx / tot_samples);
      
      fileo << sample_idx << " " << std::to_string(cyls[0].tmetric.acc()) << std::endl;

      auto pointcloud = open3d::geometry::PointCloud(points);
      paintCloud_cyl(pointcloud, pred_labels);
      // paintCloud_cyl_NuDL(pointcloud, pred_labels);
      
      // auto voxel = open3d::geometry::VoxelGrid::CreateFromPointCloud(pointcloud, 0.05);
      
       auto pointcloud_ptr = std::make_shared<open3d::geometry::PointCloud>(pointcloud);
      // synchro_.addPointCloud(voxel);
      synchro_.addPointCloud(pointcloud_ptr);
      
      cyls[0].updateTriang();

      synchro_.cv.notify_one();

      Eigen::Vector3d      zerov(0.0f, 0.0f, 0.0f);

      if (reset_view) { synchro_.resetView(); reset_view=false; }

      if (visualization_offset>0) {
        int rem = MAX(0, visualization_offset-bt.elapsedTimeMs());
        while (rem>0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          synchro_.vis.GetViewControl().SetLookat(zerov);
          synchro_.vis.GetViewControl().SetConstantZNear(10.0f);
          rem -= 5;
          synchro_.cv.notify_one();
        }
      }

    }

  }

  cyls[0].gmetric.print("final lv 2: ", cyls[2].tot_cells, 1);

  
  synchro_.join();
  return 0;
}

#endif //TEST_LABELS