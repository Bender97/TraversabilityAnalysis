
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
#include <Eigen/Dense>
#include "open3d/Open3D.h"

#include "yaml-cpp/yaml.h"

std::vector<Eigen::Vector3d> points;
std::vector<int> labels, pred_labels;
bool invert = false;
YAML::Node sample_data = YAML::LoadFile("test.yaml");

using namespace std::chrono_literals;
Synchro synchro(sample_data, true);

std::vector<Cylinder_SemKITTI> cyls;

std::string path;

bool already_written = false;



void handleOut(std::vector<Cylinder_SemKITTI> &cyls, float step) {
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

  YAML::Node node = sample_data["general"]["cyl02"];
  
  Cylinder_SemKITTI cyl = Cylinder_SemKITTI(node, nullptr, ExpMode::DL);    
  cyls.push_back(cyl);

DataLoader_SemKITTI dl;

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

    int tot_samples = 4071; //(sample_data, seq);  

    std::cout << "(main) parsing seq " << seq << " upto " << tot_samples << std::endl;
    // std::ofstream fileo("metrics_pvkd.txt");
    synchro.resetViewFlag();

    for (int sample_idx=sample_data["general"]["sample_idx_start"].as<int>(); sample_idx < tot_samples; sample_idx++) {

      bt.reset();
      synchro.reset();

      // std::cout << pts_s[sample_idx] << std::endl;
      dl.readData(seq, sample_idx, sample_data);
      dl.readPredicted(seq, sample_idx, sample_data);

      // readDataNu(pts_s[sample_idx], lab_s[sample_idx], points, labels, sample_data);

      std::cout << "points " << dl.points.size() << std::endl;
      std::cout << "labels " << dl.labels.size() << std::endl;
      std::cout << "pred_labels " << dl.pred_labels.size() << std::endl;
      
      cyls[0].resetGrid();
      cyls[0].sortBins_cyl(dl.points);
      cyls[0].computeTravGT(dl.labels);
      cyls[0].computePredictedLabel(dl.pred_labels);
      //cyls[0].filterOutliers();
      cyls[0].computeAccuracy();
      handleOut(cyls, sample_idx / tot_samples);
      
      // fileo << sample_idx << " " << std::to_string(cyls[0].tmetric.acc()) << std::endl;

      synchro.addPointCloud(dl); // synchro.addPointCloudVoxeled(dl, 0.02f);
      synchro.addPolarGrid(2, cyls[0].grid);
      synchro.delay(visualization_offset);

    }

  }

  cyls[0].gmetric.print("final lv 2: ", cyls[2].tot_cells, 1);

  
  synchro.join();

  return 0;
}

#endif //TEST_LABELS