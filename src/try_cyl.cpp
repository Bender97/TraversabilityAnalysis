
#ifndef TEST_CYL
#define TEST_CYL

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

#include "common_funcs.hpp"

#include "yaml-cpp/yaml.h"
using namespace std::chrono_literals;


std::vector<Eigen::Vector3d> points;
std::vector<int> labels;

YAML::Node sample_data = YAML::LoadFile("test.yaml");

Synchro synchro_(true);

std::vector<Cylinder> cyls;

bool already_written = false;

void handleOut(std::vector<Cylinder> &cyls, float step) {
  //if (already_written) {
    //std::cout << "\e[A"; // step
    //for (auto &c __attribute__ ((unused)): cyls) std::cout << "\e[A";
    //// std::cout << "\e[A"; 
    //std::cout << "\e[A"; // total_latency
  //}
  
  std::cout << "progress: " << std::setw(4) << step << " " << std::setw(6) << std::setprecision(4) << (step *100.0f / 4071.0f) << std::endl;

  for (int i=0; i<(int)cyls.size(); i++) {
    std::string msg = "acc" + std::to_string(cyls[i].level) +" : ";
    cyls[i].tmetric.printLight(msg.c_str(), cyls[i].tot_cells, 1);
  }
  std::string msg = "glob acc" + std::to_string(cyls[2].level) +" : ";
  cyls[2].gmetric.printLight(msg.c_str(), cyls[2].tot_cells, 1);

  already_written = true;
}

int main (int argc, char** argv)
{

  std::vector<int> seqs = sample_data["general"]["split"]["test"].as<std::vector<int>>();
  int visualization_offset = sample_data["general"]["vis_offset"].as<int>();
  std::string dataset_path = sample_data["general"]["dataset_path"].as<std::string>();
  
  loadCyls(cyls, &synchro_, sample_data);

  cv_ext::BasicTimer bt;
  cv_ext::BasicTimer bt0;
  
  bool reset_view;

  // std::vector<std::string> pts_s, lab_s;
  // std::ifstream file("recipe.txt");
  // std::string str; 
  // while (std::getline(file, str)) {
  //   pts_s.push_back(str);
  //   std::getline(file, str);
  //   lab_s.push_back(str);
  // }
  // file.close();

  for (auto &seq : seqs) {

    int tot_samples = 3; //count_samples(sample_data, seq);  

    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;
    reset_view=true;

    for (int sample_idx=sample_data["general"]["sample_idx_start"].as<int>(); sample_idx < tot_samples; sample_idx++) {

      // std::cout << pts_s[sample_idx] << std::endl;

      points.clear();
      labels.clear();
      synchro_.pauseGeometryUpdate();

      //std::cout << pts_s[sample_idx] << std::endl;

      readData(seq, sample_idx, points, labels, sample_data);
      //readDataNu(pts_s[sample_idx], lab_s[sample_idx], points, labels, sample_data);

      bt.reset();

      Eigen::MatrixXd scene_normal = computeSceneNormal(points);
      
      // bt0.reset();
      cyls[0].resetGrid();
      cyls[0].sortBins_cyl(points);
      cyls[0].computeTravGT_SemKITTI(labels);
      // cyls[0].computeTravGT_NuSc(labels);
      cyls[0].computeFeatures(scene_normal, points);
      cyls[0].process(scene_normal, points);
      //cyls[0].filterOutliers();
       cyls[0].computeAccuracy();
      // std::cout << "cyl0: " << bt0.elapsedTimeMs() << " ms" << std::endl;
      
      // bt0.reset();
      cyls[1].resetGrid();
      cyls[1].sortBins_cyl(points);
      cyls[1].computeTravGT_SemKITTI(labels);
      // cyls[1].computeTravGT_NuSc(labels);
      cyls[1].computeFeatures(scene_normal, points);
      cyls[1].inheritFeatures(&(cyls[0]));
      cyls[1].process(scene_normal, points);
      //cyls[1].filterOutliers();
       cyls[1].computeAccuracy();
      // // std::cout << "cyl1: " << bt0.elapsedTimeMs() << " ms" << std::endl;
      
      // bt0.reset(); 
      cyls[2].resetGrid();
      // std::cout << "cyl0 - resetGrid " << bt0.elapsedTimeMs() << " ms" << std::endl; bt0.reset();
      cyls[2].sortBins_cyl(points);
      // std::cout << "cyl0 - sortBins_cyl " << bt0.elapsedTimeMs() << " ms" << std::endl; bt0.reset();
      cyls[2].computeTravGT_SemKITTI(labels);
      // cyls[2].computeTravGT_NuSc(labels);
      // std::cout << "cyl0 - computeTravGT_SemKITTI " << bt0.elapsedTimeMs() << " ms" << std::endl; bt0.reset();
      cyls[2].computeFeatures(scene_normal, points);
      // std::cout << "cyl0 - computeFeatures " << bt0.elapsedTimeMs() << " ms" << std::endl; bt0.reset();
      cyls[2].inheritFeatures(&(cyls[1]));
      // std::cout << "cyl0 - deriveAllGTFeatures " << bt0.elapsedTimeMs() << " ms" << std::endl; bt0.reset();
      cyls[2].process(scene_normal, points);
      // std::cout << "cyl0 - process " << bt0.elapsedTimeMs() << " ms" << std::endl; bt0.reset();
      // cyls[2].filterOutliers();
       cyls[2].computeAccuracy();
      // std::cout << "cyl2: " << bt0.elapsedTimeMs() << " ms" << std::endl;

      std::cout << "idx " << sample_idx << " total latency: " << bt.elapsedTimeMs() << " ms" << std::endl;
     
     
      handleOut(cyls, sample_idx  );
      
      auto pointcloud = open3d::geometry::PointCloud(points);
      paintCloud_cyl(pointcloud, labels);
      // paintCloud_cylB(pointcloud);
      
       //auto voxel = open3d::geometry::VoxelGrid::CreateFromPointCloud(pointcloud, 0.2);
      
      auto pointcloud_ptr = std::make_shared<open3d::geometry::PointCloud>(pointcloud);
      //  synchro_.addPointCloud(pointcloud_ptr);
      //synchro_.addPointCloud(voxel);
      
      // cyls[0].updateTriang();
      // cyls[1].updateTriang();
      cyls[2].updateTriang();

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

  cyls[0].gmetric.print("final lv 0: ", cyls[0].tot_cells, 1);
  cyls[1].gmetric.print("final lv 1: ", cyls[1].tot_cells, 1);
  cyls[2].gmetric.print("final lv 2: ", cyls[2].tot_cells, 1);
  
   synchro_.join();
  return 0;
}

#endif //TEST_CYL