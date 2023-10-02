
#ifndef TEST_CYL
#define TEST_CYL

#include <iostream>
#include <vector>

#include "yaml-cpp/yaml.h"

#if OPEN3D == 1
  #include "Synchro.h"
#endif

#include "Cylinder.h"
#include "DataLoader.h"
#include "common_macro.hpp"
#include "cv_ext.h"

using namespace std::chrono_literals;


YAML::Node sample_data = YAML::LoadFile("../models/test.yaml");
cv_ext::BasicTimer bt;

std::vector<Cylinder*> cyls;
std::vector<int> seqs;
int visualization_offset;
std::string dataset_mode;

bool already_written = false;
uint64_t runtime, tot_runtime=0;
double avg_runtime;

#if OPEN3D == 1
Synchro synchro(sample_data, true);
#endif

template<typename T>
void loadCyls(std::vector<T*> &cyls, YAML::Node &sample_data) {
  cyls.clear();
  int level;

  std::cout << "DATA: " << dataset_mode << std::endl;
  std::cout << "#######################################" << std::endl;
  for (level=0; ; level++) {
    auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(level).length()), '0') + std::to_string(level);
    YAML::Node node = sample_data["general"][cyl_s.c_str()];
    if (!node) break;

    node["dataset"] = dataset_mode;
    node["load_path"] = sample_data["general"]["load_path"].as<std::string>();
    node["save_path"] = sample_data["general"]["save_path"].as<std::string>();
    
    T *back_cyl = (level>0) ? (cyls[level-1]) : nullptr;
    
    auto cyl = new Cylinder_SinglePLY(node, back_cyl, ExpMode::test); 
    cyl->printSummary();   
    cyls.push_back(cyl);

  }
  std::cout << "#######################################" << std::endl;

  if (!level) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. please provide"
                  " at least a cylinder in yaml config file.\n"));
}


int main (int argc, char** argv)
{
  
  seqs                 = sample_data["general"]["split"]["test"].as<std::vector<int>>();
  visualization_offset = sample_data["general"]["vis_offset"].as<int>();
  dataset_mode         = sample_data["general"]["dataset"].as<std::string>();
  
  loadCyls(cyls, sample_data);

#if OPEN3D == 1
  synchro.resetViewFlag(); 
#endif

  // std::string single_cloud_path = "/media/fusy/Windows-SSD/Users/orasu/Downloads/LiDAR-GTA-V-1.2/LiDAR-GTA-V-1.2/samples/LiDAR - Traffic.ply";
  std::string single_cloud_path = "/home/fusy/repos/velodyne/";
  DataLoader_PLY dl_ply(single_cloud_path);

  for (int i=0; i<10; i++) {
  // while(true) {

    std::string idx_s = std::to_string(i);
    auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

#if OPEN3D == 1
    synchro.reset();
#endif

    // dl_ply.readPLY(single_cloud_path);
    // dl_ply.readBin(single_cloud_path + new_idx_s + ".bin");
    dl_ply.readBin("/home/fusy/repos/pandaset_lidar/sweeps/" + new_idx_s + ".bin");
    //dl_nusc.readPredicted(0, sample_idx, sample_data);

    std::cout << "points " << dl_ply.points.size() << std::endl;
    std::cout << "labels " << dl_ply.labels.size() << std::endl;
    // std::cout << "pred_labels " << dl_ply.pred_labels.size() << std::endl;
    
    for (auto cyl : cyls) {
    // cyl->OnlineRoutine(dl, cyl->level ? cyls[cyl->level-1] : nullptr);
    runtime = cyl->OnlineRoutineProfile(dl_ply, cyl->level ? cyls[cyl->level-1] : nullptr);
    tot_runtime += runtime;
    }
    std::cout << " total latency: " << runtime << " ms" << std::endl;
    
#if OPEN3D == 1
    synchro.addPointCloud(dl_ply); // synchro.addPointCloudVoxeled(dl, 0.02f);
    synchro.addPolarGridPred(2, cyls[2]->grid);
    synchro.delay(visualization_offset);
#endif



//   for (size_t l=0; l<cyls.size(); l++)
//     cyls[l]->gmetric.print(
//         std::string("final lv ") + std::to_string(l) + std::string(": "), 
//         cyls[l]->tot_cells, 1);
  
}
#if OPEN3D == 1
  synchro.join();
#endif
//   for (auto cyl: cyls) free(cyl);

  return 0;
}

#endif //TEST_CYL