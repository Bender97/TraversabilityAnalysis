
#ifndef TEST_CYL
#define TEST_CYL

#include <iostream>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "Synchro.h"
#include "Cylinder.h"
#include "DataLoader.h"
#include "common_macro.hpp"
#include "cv_ext.h"

using namespace std::chrono_literals;


YAML::Node sample_data = YAML::LoadFile("test.yaml");
Synchro synchro(sample_data, true);
DataLoader_SemKITTI dl;
  cv_ext::BasicTimer bt;

bool already_written = false;

template<typename T>
void handleOut(std::vector<T> &cyls, float step) {
  //if (already_written) {
    //std::cout << "\e[A"; // step
    //for (auto &c __attribute__ ((unused)): cyls) std::cout << "\e[A";
    //// std::cout << "\e[A"; 
    //std::cout << "\e[A"; // total_latency
  //}
  
  std::cout << "progress: " << std::setw(4) << step << " " << std::setw(6) << std::setprecision(4) << (step *100.0f / 4071.0f) << std::endl;

  for (int i=0; i<(int)cyls.size(); i++) {
    std::string msg = "acc" + std::to_string(cyls[i]->level) +" : ";
    cyls[i]->tmetric.printLight(msg.c_str(), cyls[i]->tot_cells, 1);
  }
  std::string msg = "avg acc" + std::to_string(cyls[2]->level) +" : ";
  cyls[2]->gmetric.printLight(msg.c_str(), cyls[2]->tot_cells, 1);

  already_written = true;
}

template<typename T>
void loadCyls(std::vector<T*> &cyls, YAML::Node &sample_data) {
  cyls.clear();
  int level;

  std::string data_ = sample_data["general"]["dataset"].as<std::string>();

  std::cout << "DATA: " << data_ << std::endl;
  std::cout << "#######################################" << std::endl;
  for (level=0; ; level++) {
    auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(level).length()), '0') + std::to_string(level);
    YAML::Node node = sample_data["general"][cyl_s.c_str()];
    if (!node) break;

    node["dataset"] = data_;
    node["load_path"] = sample_data["general"]["load_path"].as<std::string>();
    node["save_path"] = sample_data["general"]["save_path"].as<std::string>();
    
    T *back_cyl = (level>0) ? (cyls[level-1]) : nullptr;
    
    if (data_=="SemKITTI") {
      auto cyl = new Cylinder_SemKITTI(node, back_cyl, ExpMode::test); 
      cyl->printSummary();   
      cyls.push_back(cyl);
    }
  }
  std::cout << "#######################################" << std::endl;

  if (!level) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. please provide"
                  " at least a cylinder in yaml config file.\n"));
}

int main (int argc, char** argv)
{
  std::vector<Cylinder*> cyls;
  
  std::vector<int> seqs    = sample_data["general"]["split"]["test"].as<std::vector<int>>();
  int visualization_offset = sample_data["general"]["vis_offset"].as<int>();
  std::string dataset_path = sample_data["general"]["dataset_path"].as<std::string>();
  
  loadCyls(cyls, sample_data);

  for (auto &seq : seqs) {

    int tot_samples = sample_data["general"]["sample_idx_end"].as<int>(); //count_samples(sample_data, seq);  

    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;
    synchro.resetViewFlag();

    for (int sample_idx=sample_data["general"]["sample_idx_start"].as<int>(); sample_idx < tot_samples; sample_idx++) {

      synchro.reset();

      dl.readData(seq, sample_idx, sample_data);
      bt.reset();
      for (auto cyl : cyls)
        cyl->OnlineRoutine(dl, cyl->level ? cyls[cyl->level-1] : nullptr); 
      std::cout << "idx " << sample_idx << " total latency: " << bt.elapsedTimeMs() << " ms" << std::endl;
     
      handleOut(cyls, sample_idx);
      
      synchro.addPointCloud(dl); // synchro.addPointCloudVoxeled(dl, 0.02f);
      synchro.addPolarGrid(2, cyls[2]->grid);
      synchro.delay(visualization_offset);
    }

  }

  cyls[0]->gmetric.print("final lv 0: ", cyls[0]->tot_cells, 1);
  cyls[1]->gmetric.print("final lv 1: ", cyls[1]->tot_cells, 1);
  cyls[2]->gmetric.print("final lv 2: ", cyls[2]->tot_cells, 1);
  
  synchro.join();

  for (auto cyl: cyls) free(cyl);

  return 0;
}

#endif //TEST_CYL