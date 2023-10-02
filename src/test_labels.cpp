
#ifndef TEST_LABELS
#define TEST_LABELS

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#if OPEN3D == 1
#include "Synchro.h"
#endif

#include "Feature.h"
#include "Cylinder.h"
#include "cv_ext.h"

#include "yaml-cpp/yaml.h"
using namespace std::chrono_literals;

YAML::Node sample_data = YAML::LoadFile("../models/test.yaml");

#if OPEN3D == 1
Synchro synchro(sample_data, true);
#endif

std::vector<Cylinder*> cyls;
std::vector<int> seqs;
int visualization_offset;
std::string dataset_mode, predicted_path, dataset_path;

bool already_written = false;


template<typename T>
void handleOut(std::vector<T> &cyls, float step) {
  if (already_written) {
    std::cout << "\e[A"; // step
    std::cout << "\e[A";
    std::cout << "\e[A"; 
    std::cout << "\e[A"; 
  }
  
  std::cout << "progress: " << std::setw(6) << std::setprecision(4) << step*100 << std::endl;
  
  std::string msg = "acc" + std::to_string(cyls[0]->level) +" : ";
  cyls[0]->tmetric.printLight(msg.c_str(), -1, 1);
  msg = "glob acc" + std::to_string(cyls[0]->level) +" : ";

  cyls[0]->gmetric.printLight(msg.c_str(), -1, 1);
  
  already_written = true;
}

void processSemanticKITTI() {
  std::string SemKITTI_dataset_path = sample_data["general"]["SemKITTI_dataset_path"].as<std::string>();
  DataLoader_SemKITTI dl(SemKITTI_dataset_path);

  for (auto &seq : seqs) {

    int tot_samples = 4071; //(sample_data, seq);  

    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;
    // std::ofstream fileo("metrics_pvkd.txt");
#if OPEN3D == 1
    synchro.resetViewFlag();
#endif

    for (int sample_idx=sample_data["general"]["sample_idx_start"].as<int>(); sample_idx < tot_samples; sample_idx++) {

#if OPEN3D == 1
      synchro.reset();
#endif
      // std::cout << pts_s[sample_idx] << std::endl;
      dl.readData(seq, sample_idx, sample_data);
      dl.readPredicted(seq, sample_idx, sample_data);

      dl.assertDLConsistency();
      
      cyls[0]->OnlineDLRoutineProfile(dl);
      handleOut(cyls, sample_idx / tot_samples);
      
      // fileo << sample_idx << " " << std::to_string(cyls[0].tmetric.acc()) << std::endl;
#if OPEN3D == 1
      synchro.addPointCloud(dl); // synchro.addPointCloudVoxeled(dl, 0.02f);
      synchro.addPolarGrid(2, cyls[0]->grid);
      synchro.delay(visualization_offset);
#endif
    }

  }
}

void processnuScenes() {
  std::string nuscenes_recipe_path = sample_data["general"]["nuScenes_path"].as<std::string>();
  DataLoader_NuSc dl_nusc(nuscenes_recipe_path);
#if OPEN3D == 1
  synchro.resetViewFlag();
#endif

  for (auto &seq : seqs) {

    int tot_samples = 4071; //(sample_data, seq);  

    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;
    // std::ofstream fileo("metrics_pvkd.txt");

#if OPEN3D == 1
    synchro.resetViewFlag();
#endif

    for (int sample_idx=sample_data["general"]["sample_idx_start"].as<int>(); sample_idx < tot_samples; sample_idx++) {


#if OPEN3D == 1
      synchro.reset();
#endif
      dl_nusc.readData(seq, sample_idx, sample_data);
      dl_nusc.readPredicted(seq, sample_idx, sample_data);

      dl_nusc.assertDLConsistency();
      
      cyls[0]->OnlineDLRoutineProfile(dl_nusc);
      handleOut(cyls, sample_idx / tot_samples);
      
      // fileo << sample_idx << " " << std::to_string(cyls[0].tmetric.acc()) << std::endl;
#if OPEN3D == 1
      synchro.addPointCloud(dl_nusc); // synchro.addPointCloudVoxeled(dl, 0.02f);
      synchro.addPolarGrid(2, cyls[0]->grid);
      synchro.delay(visualization_offset);
#endif
    }

  }
}


void processPandaSet() {

  if (!sample_data["general"]["PandaSet_dataset_path"])
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. PandaSet dataset mode needs" \
                      " PandaSet_dataset_path. Set it in YAML config file. Exit.\n") );

  std::string PandaSet_dataset_path = sample_data["general"]["PandaSet_dataset_path"].as<std::string>();
  DataLoader_PandaSet dl(PandaSet_dataset_path);

  int tot_samples = dl.count_samples(0);
  std::cout << "(main) parsing samples " << tot_samples << std::endl;

#if OPEN3D == 1
    synchro.resetViewFlag();
#endif

    for (int sample_idx=0; sample_idx < tot_samples; sample_idx++) {

#if OPEN3D == 1
      synchro.reset();
#endif

      dl.readData(0, sample_idx, sample_data);
      dl.readPredicted(0, sample_idx, sample_data);

      dl.assertDLConsistency();
      
      cyls[0]->OnlineDLRoutineProfile(dl);
      
      handleOut(cyls, sample_idx / tot_samples);
      
      #if OPEN3D == 1
            synchro.addPointCloud(dl); // synchro.addPointCloudVoxeled(dl, 0.02f);
            synchro.addPolarGrid(2, cyls[0]->grid);
            synchro.delay(visualization_offset);
      #endif
  }
}




int main (int argc, char** argv)
{

  seqs                 = sample_data["general"]["split"]["test"].as<std::vector<int>>();
  visualization_offset = sample_data["general"]["vis_offset"].as<int>();
  dataset_mode         = sample_data["general"]["dataset"].as<std::string>();
  predicted_path = sample_data["general"]["predicted_path"].as<std::string>();
  
  cyls.clear();

  YAML::Node node = sample_data["general"]["cyl02"];
  
  if (dataset_mode=="SemKITTI") {
    auto cyl = new Cylinder_SemKITTI(node, nullptr, ExpMode::DL); 
    cyl->printSummary();   
    cyls.push_back(cyl);
  }
  else if (dataset_mode=="nuScenes") {
    auto cyl = new Cylinder_NuSc(node, nullptr, ExpMode::DL); 
    cyl->printSummary();   
    cyls.push_back(cyl);
  }
  else if (dataset_mode=="PandaSet") {
    auto cyl = new Cylinder_PandaSet(node, nullptr, ExpMode::DL); 
    cyl->printSummary();   
    cyls.push_back(cyl);
  }
  else
    throw std::runtime_error(
      "Dataset not recognized! Available choices are\n - SemKITTI\n - nuScenes\n -PandaSet\nExit.");

  if (dataset_mode == "SemKITTI")
    processSemanticKITTI();
  else if (dataset_mode == "nuScenes")
    processnuScenes();
  else if (dataset_mode == "PandaSet")
    processPandaSet();
  else
    throw std::runtime_error(\
        "Dataset not recognized! Available choices are\n - SemKITTI\n - nuScenes\n - PandaSet\nExit.");


  cyls[0]->gmetric.print("final lv 2: ", cyls[0]->tot_cells, 1);

#if OPEN3D == 1
  synchro.join();
#endif

  return 0;
}

#endif //TEST_LABELS