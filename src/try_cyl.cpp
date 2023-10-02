
#ifndef TEST_CYL
#define TEST_CYL

#include <iostream>
#include <vector>

#include "yaml-cpp/yaml.h"

#if OPEN3D 
#include "Synchro.h"
#endif

#include "Cylinder.h"
#include "DataLoader.h"
#include "common_macro.hpp"
#include "cv_ext.h"

using namespace std::chrono_literals;


YAML::Node sample_data = YAML::LoadFile("../models/test.yaml");

#if OPEN3D 
Synchro synchro(sample_data, true);
#endif

std::vector<Cylinder*> cyls;
std::vector<int> seqs;
int visualization_offset;
std::string dataset_mode;

bool already_written = false;
uint64_t runtime, frame_runtime, tot_runtime=0;
double avg_runtime;

template<typename T>
void handleOut(std::vector<T> &cyls, float step) {
  // if (already_written) {
  //   std::cout << "\e[A"; // step
  //   for (auto &c __attribute__ ((unused)): cyls) std::cout << "\e[A";
  //   // std::cout << "\e[A"; 
  //   std::cout << "\e[A"; 
  //   std::cout << "\e[A"; // total_latency
  // }
  
  std::cout << "progress: " << std::setw(4) << step << " " << std::setw(6) << std::setprecision(4) << (step *100.0f / 4071.0f) << std::endl;

  for (int i=0; i<(int)cyls.size(); i++) {
    std::string msg = "acc" + std::to_string(cyls[i]->level) +" : ";
    cyls[i]->tmetric.printLight(msg.c_str(), cyls[i]->tot_cells, 1);
  }
  std::string msg = "avg acc" + std::to_string(cyls[cyls.size()-1]->level) +" : ";
  cyls[cyls.size()-1]->gmetric.printLight(msg.c_str(), cyls[cyls.size()-1]->tot_cells, 1);
  // std::string msg = "avg acc" + std::to_string(cyls[0]->level) +" : ";
  // cyls[0]->gmetric.printLight(msg.c_str(), cyls[0]->tot_cells, 1);

  already_written = true;
}

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
    
    if (dataset_mode=="SemKITTI") {
      auto cyl = new Cylinder_SemKITTI(node, back_cyl, ExpMode::test); 
      cyl->printSummary();   
      cyls.push_back(cyl);
    }
    
    else if (dataset_mode=="PandaSet") {
      auto cyl = new Cylinder_PandaSet(node, back_cyl, ExpMode::test); 
      cyl->printSummary();   
      cyls.push_back(cyl);
    }
    
    else if (dataset_mode=="nuScenes") {
      auto cyl = new Cylinder_NuSc(node, back_cyl, ExpMode::test); 
      cyl->printSummary();   
      cyls.push_back(cyl);
    }
    else
      throw std::runtime_error(
        "Dataset not recognized! Available choices are\n - SemKITTI\n -nuScenes\nExit.");
  }
  std::cout << "#######################################" << std::endl;

  if (!level) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. please provide"
                  " at least a cylinder in yaml config file.\n"));
}

void processSemanticKITTI() {

  if (!sample_data["general"]["SemKITTI_dataset_path"])
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. SemKITTI dataset mode needs" \
                      " SemKITTI_dataset_path. Set it in YAML config file. Exit.\n") );

  std::string SemKITTI_dataset_path = sample_data["general"]["SemKITTI_dataset_path"].as<std::string>();
  DataLoader_SemKITTI dl(SemKITTI_dataset_path);

  std::ofstream out("runtimes.txt", std::ios::out);

  int cont=0;
  for (auto &seq : seqs) {

    int tot_samples = dl.count_samples(seq);  

    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;

#if OPEN3D 
    synchro.resetViewFlag();
#endif

    for (int sample_idx=sample_data["general"]["sample_idx_start"].as<int>(); sample_idx < tot_samples; sample_idx++) {

#if OPEN3D 
      synchro.reset();
#endif

      dl.readData(seq, sample_idx, sample_data);
      dl.assertConsistency();

      frame_runtime = 0;
      for (auto cyl : cyls) {
        // cyl->OnlineRoutine(dl, cyl->level ? cyls[cyl->level-1] : nullptr);
        if (cyl->trick_mode>0)
          runtime = cyl->OnlineRoutineProfile_wt_Trick(dl, cyl->level ? cyls[cyl->level-1] : nullptr);
        else
          runtime = cyl->OnlineRoutineProfile(dl, cyl->level ? cyls[cyl->level-1] : nullptr);
        
        frame_runtime += runtime;
      }
      
      tot_runtime += frame_runtime;

      std::cout << "idx " << sample_idx << "  current frame_ runtime: " << frame_runtime << " ms" << std::endl;
      out<< std::to_string(cont) << " " << std::to_string(frame_runtime) << " " << std::to_string(cyls[2]->tmetric.acc()) << "\n";
      
      cont++;
      
      
      handleOut(cyls, sample_idx);

#if OPEN3D == 1
      synchro.addPointCloud(dl); // synchro.addPointCloudVoxeled(dl, 0.02f);
      // synchro.addPolarGrid(0, cyls[0]->grid);
      // synchro.addPolarGrid(1, cyls[1]->grid);
      synchro.addPolarGrid(cyls.size()-1, cyls[cyls.size()-1]->grid);
      synchro.delay(visualization_offset);
#endif
    }
  }
  std::cout << "tot_runtime" << tot_runtime << "\n";
  std::cout << "cont" << cont << "\n";

  avg_runtime = (double) tot_runtime / (double) cont;
  out.close();
}

void processnuScenes() {

  if (!sample_data["general"]["nuScenes_path"])
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. nuScenes dataset mode needs" \
                      " nuScenes_path. Set it in YAML config file. Exit.\n") );
  std::string nuscenes_recipe_path = sample_data["general"]["nuScenes_path"].as<std::string>();
  DataLoader_NuSc dl_nusc(nuscenes_recipe_path);
  
#if OPEN3D == 1
  synchro.resetViewFlag();
#endif

  int cont=0;
  for (int sample_idx=0; sample_idx < dl_nusc.count_samples(0); sample_idx++) {

#if OPEN3D == 1
      synchro.reset();
#endif

      dl_nusc.readData(0, sample_idx, sample_data);
      dl_nusc.assertConsistency();
      
      for (auto cyl : cyls) {
        // cyl->OnlineRoutine(dl, cyl->level ? cyls[cyl->level-1] : nullptr);
        runtime = cyl->OnlineRoutineProfile(dl_nusc, cyl->level ? cyls[cyl->level-1] : nullptr);
        tot_runtime += runtime;
      }
      std::cout << "idx " << sample_idx << " total latency: " << runtime << " ms" << std::endl;
      cont++;
      
      handleOut(cyls, sample_idx);

#if OPEN3D == 1
      synchro.addPointCloud(dl_nusc); // synchro.addPointCloudVoxeled(dl, 0.02f);
      synchro.addPolarGrid(2, cyls[2]->grid);
      synchro.delay(visualization_offset);
#endif
    }

  avg_runtime = (double) tot_runtime / (double) cont;

}

void processPandaSet() {

  if (!sample_data["general"]["PandaSet_dataset_path"])
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. PandaSet dataset mode needs" \
                      " PandaSet_dataset_path. Set it in YAML config file. Exit.\n") );

  std::string PandaSet_dataset_path = sample_data["general"]["PandaSet_dataset_path"].as<std::string>();
  DataLoader_PandaSet dl(PandaSet_dataset_path);
  std::ofstream out("runtimes.txt", std::ios::out);

  int cont=0;
  for (auto &seq : seqs) {

    int tot_samples = dl.count_samples(seq);
    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;

#if OPEN3D
    synchro.resetViewFlag();
#endif

    for (int sample_idx=1280; sample_idx < tot_samples; sample_idx++) {
#if OPEN3D == 1
      synchro.reset();
#endif

      dl.readData(seq, sample_idx, sample_data);
      dl.assertConsistency();

      for (auto cyl : cyls) {
        // cyl->OnlineRoutine(dl, cyl->level ? cyls[cyl->level-1] : nullptr);
        runtime = cyl->OnlineRoutineProfile(dl, cyl->level ? cyls[cyl->level-1] : nullptr);
        tot_runtime += runtime;
      }
      out<< std::to_string(cont) << " " << std::to_string(cyls[2]->tmetric.acc()) << "\n";
      
      std::cout << "idx " << sample_idx << " total runtime: " << runtime << " ms" << std::endl;
      cont++;
      
      handleOut(cyls, sample_idx);

#if OPEN3D == 1
      synchro.addPointCloud(dl); // synchro.addPointCloudVoxeled(dl, 0.02f);
      synchro.addPolarGrid(2, cyls[2]->grid);
      synchro.delay(visualization_offset);
#endif
    }
  }
  avg_runtime = (double) tot_runtime / (double) cont;
}

int main (int argc, char** argv)
{
  
  seqs                 = sample_data["general"]["split"]["test"].as<std::vector<int>>();
  visualization_offset = sample_data["general"]["vis_offset"].as<int>();
  dataset_mode         = sample_data["general"]["dataset"].as<std::string>();
  
  loadCyls(cyls, sample_data);

  if      (dataset_mode == "SemKITTI") processSemanticKITTI();
  else if (dataset_mode == "nuScenes") processnuScenes();
  else if (dataset_mode == "PandaSet") processPandaSet();
  else
    throw std::runtime_error(
        "Dataset not recognized! Available choices are\n - SemKITTI\n -nuScenes\n -PandaSet\nExit.");


  for (size_t l=0; l<cyls.size(); l++)
    cyls[l]->gmetric.print(
        std::string("final lv ") + std::to_string(l) + std::string(": "), 
        cyls[l]->tot_cells, 1);
  
  std::cout << "AVG_RUNTIME " << avg_runtime << std::endl;
  
#if OPEN3D == 1
  synchro.join();
#endif
  // for (auto cyl: cyls) free(cyl);

  return 0;
}

#endif //TEST_CYL