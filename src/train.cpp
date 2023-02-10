
#ifndef TRAIN
#define TRAIN

// # level #0

// shape[:, 18] = shape[:, 17 + 1(gt)]

// # level #1
// shape[:, 36] = shape[:, 17(geom) + 17(geom0) + 1(label0) + 1(gt)]

// # level #2
// shape[:, 54] = shape[:, 17(geom) + 17(geom0) + 1(label0) + 17(geom1) + 1(label1) + 1(gt)]
//                                                    34                    52         53

#include <iostream>
#include <fstream>

#include <vector>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <future>
#include <functional>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Metric.h"
#include "Results.h"
#include "TrainDataset.h"

#include "yaml-cpp/yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

struct stat st = {0};

int train_duration_secs;
std::string train_config_path;
YAML::Node sample_data = YAML::LoadFile("test.yaml");

int train_level, train_mode, pca_mode=-1, max_iter=1000, trick_mode=0;

std::vector<std::string> modes = std::vector<std::string>({"geom", "geom_label", "geom_all", "geom_pca", "geom_pca_label", "geom_pca_all_label"});

std::vector<TrainDataset> train_datasets, train_datasets2;
std::vector<TrainDataset> val_datasets, val_datasets2;

void computeAcc(TrainDataset *td, cv::Mat &pred, Metric &metric, std::string msg) {
  double plab;

  for (int r=0; r < td->y_train.rows; r++) {
    plab = pred.at<float>(r, 0) > 0 ? 1.0f : -1.0f;
    metric.update(plab, td->y_train.at<float>(r, 0));
  }
  if (pred.rows > 50000 && metric.acc()>0.9075)
    metric.printV(msg.c_str(), td->tot_cells, 1);
  else metric.print(msg.c_str(), td->tot_cells, 1);
}

void analyze(std::vector<int> &labs) {
  int trav=0, nontrav=0;
  for (int i=0; i<(int)labs.size(); i++)
      if (labs[i]==NOT_TRAV_CELL_LABEL) nontrav++;
      else trav++;

  std::cout << "trav: " << trav 
            << " (" << std::setw(4) << std::setprecision(4) << (trav/(float)labs.size()) << ") -- "
            << "nontrav: " << nontrav
            << " (" << std::setw(4) << std::setprecision(4) << (nontrav/(float)labs.size()) << ") "
            << std::endl;
}


void parseConfig(std::string path) {

  std::vector<std::string> modes 
          = std::vector<std::string>({"geom", "geom_label", "geom_all", "geom_pca", "geom_pca_label", "geom_pca_all_label"});

  if (sample_data["general"]["train_level"]) train_level = sample_data["general"]["train_level"].as<int>();
  else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. please provide which level to train on.\n"));

  
  auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(train_level).length()), '0') + std::to_string(train_level);
  YAML::Node node = sample_data["general"][cyl_s.c_str()];
  if (!node) throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Cylinder (at train_level) not found!\n"));


  if (node["max_iter"]) max_iter = node["max_iter"].as<int>();

  if (node["mode"]) train_mode = node["mode"].as<int>();
  else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. please provide which level to train on.\n"));

  if (node["pca"]) pca_mode = node["pca"].as<int>();
  if (node["trick"]) trick_mode = node["trick"].as<int>();

  if (train_mode<3) pca_mode=-1;

  std::string savepath = sanitize(sample_data["general"]["save_path"].as<std::string>());

  if(stat(savepath.c_str(), &st) == -1) {
      mkdir(savepath.c_str(), 0700);
      std::cerr<< "folder created: "<< savepath << std::endl;
  }

  savepath += "lv" + std::to_string(train_level);
  if(stat(savepath.c_str(), &st) == -1) {
      mkdir(savepath.c_str(), 0700);
      std::cerr<< "folder created: "<< savepath << std::endl;
  }

  if (trick_mode) savepath += "/" + modes[train_mode] + "_trick/";
  else savepath += "/" + modes[train_mode] + "/";
  if(stat(savepath.c_str(), &st) == -1) {
      mkdir(savepath.c_str(), 0700);
      std::cerr<< "folder created: "<< savepath << std::endl;
  }
  std::cout << "SAVEPATH: " << savepath << std::endl;

  node["load_path"] = sample_data["general"]["load_path"];
  node["save_path"] = sample_data["general"]["save_path"];
  node["store_features_filename"] = sample_data["general"]["store_features_filename"];
  node["seed"] = sample_data["general"]["seed"];

  for (int a=0; a<1; a++) {
    TrainDataset td(node, train_level, true);
    std::cout << " -- Will train on:" << std::endl;
    train_datasets.push_back(td);
    td.summary();
  }


  // TrainDataset vd(node, train_level, train_datasets[0].normalizer, train_datasets[0].pca, false);
  // std::cout << " -- Will valid on:" << std::endl;
  // val_datasets.push_back(vd);
  // vd.summary();

  node["store_features_filename"] = "features_data_4p_notgt";
  TrainDataset vd2(node, train_level, train_datasets[0].normalizer, train_datasets[0].pca, false);
  std::cout << " -- Will valid on:" << std::endl;
  val_datasets2.push_back(vd2);
  vd2.summary();

}


int main (int argc, char** argv)
{
  parseConfig(train_config_path);

  if (train_duration_secs<0) train_duration_secs = 10000;

  std::vector<Results> results;
  cv::Mat predictions_vector;
  Metric tmetric, vmetric, vmetric2, gvmetric, bestvmetric;
  TrainDataset *vd, *td2, *vd2;


  // td = &(train_datasets[i]);
  vd = &(val_datasets[0]);
  vd2 = &(val_datasets2[0]);

  bool skipped=false;

  for (auto &C: train_datasets[0].C_vec) {
    for (auto &nu: train_datasets[0].nu_vec) {
      for (auto &gamma: train_datasets[0].gamma_vec) {

        auto iters = std::vector<int>({ 3400}); //, 4800, 500});
        // std::vector<int> iters;
        // for (int i=1900; i<2000; i+=5) iters.push_back(i);
        for (auto iter: iters) {
          max_iter = iter;
          std::cout << " ##############################      ###########################" << std::endl
                    // << " training " << td->X_train.rows << " rows " << td->X_train.cols << " cols"
                    << " max_iter: " << max_iter << "  ##  nu: " << nu 
                    << " C: " << C << " gamma: " << gamma << std::endl;
          
          

          // TRAIN MODEL: SET PARAMS
          cv::Ptr<cv::ml::SVM> local_model;
          local_model.reset();
          local_model = cv::ml::SVM::create();
          local_model->setType(cv::ml::SVM::NU_SVR);
          local_model->setKernel(cv::ml::SVM::RBF);
          local_model->setNu(nu);
          local_model->setGamma(gamma);
          local_model->setC(C);

          /*int follow=1;
          double eps = 1e-4;
          while(follow) {
            std::cout << "set term criteria: " << eps << std::endl;
            local_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS, 100, eps)); // it was 1e-6

            // TRAIN
            local_model->train(td->X_train, cv::ml::ROW_SAMPLE, td->y_train);

            predictions_vector = cv::Mat::zeros(td->y_train.rows, 1, CV_32F); 
            tmetric.resetAll();
            local_model->predict(td->X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
            tmetric.checkpointTime();
            computeAcc(td, predictions_vector, tmetric, "train: ");

            std::cout << "continue? (yes 1, no 0) ";
            std::cin >> follow;
            eps /= 1.2;
          }*/

          for (size_t i=0; i<train_datasets.size(); i++) {
            
            td2 = &(train_datasets[i]);

            td2->svm_gamma = gamma;
            td2->svm_nu = nu;
            
            local_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, max_iter, 1e-6)); // it was 1e-6
            local_model->train(td2->X_train, cv::ml::ROW_SAMPLE, td2->y_train);

            // TRAIN ERROR        
            predictions_vector = cv::Mat::zeros(td2->y_train.rows, 1, CV_32F); 
            tmetric.resetAll();
            local_model->predict(td2->X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
            tmetric.checkpointTime();
            std::string msg1 = "train " + std::to_string(i) + "/" + std::to_string(train_datasets.size()) + ": ";
            computeAcc(td2, predictions_vector, tmetric, msg1.c_str());
            float acc = tmetric.acc();
            skipped = false;
            // if (acc<0.941) {// || acc>=0.95) {
            if (acc<0.908) {// || acc>=0.95) {
              std::cout << "skip!" << std::endl;
              skipped = true;
              continue;
            }

            // VALID ERROR
            uint32_t block_size = 100000;
            gvmetric.resetAll();
            for (int i=0; i<val_datasets2[0].X_train.rows+block_size; i+=block_size) {

              uint32_t end = MIN(val_datasets2[0].X_train.rows-1, i+block_size);
              uint32_t size = end-i;
              if (size<=0 || i > val_datasets2[0].X_train.rows-1) break;
              
              float checkp = (float) end/(val_datasets2[0].X_train.rows-1);

              // VALID ERROR
              predictions_vector = cv::Mat::zeros(size, 1, CV_32F);
              vmetric2.resetAll();
              local_model->predict(val_datasets2[0].X_train.rowRange(i, end), predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
              vmetric2.checkpointTime();
              // computeAcc(&(val_datasets2[0]), predictions_vector, vmetric2, msg3.c_str());
              double plab;

              for (int r=0; r < size; r++) {
                plab = predictions_vector.at<float>(r, 0) > 0 ? 1.0f : -1.0f;
                vmetric2.update(plab, val_datasets2[0].y_train.at<float>(r+i, 0));
              }
              gvmetric += vmetric2;
              if (i>0) {
                std::cout << "\e[A";
                std::cout << "\e[A";
              }
              // std::cout << "                                                                          ";
              std::cout << "valid2 : " << std::setw(9) << std::setprecision(5) << checkp*100 << "%: " << vmetric2.acc()*100 << std::endl;
              std::cout << " glob  : " << std::setw(12) << " "                                         << gvmetric.acc()*100 << std::endl;
            }

            if (gvmetric > bestvmetric) bestvmetric = gvmetric;
            // if ( gvmetric.acc()>0.9075)
            if ( gvmetric.acc()>0.9391)
              gvmetric.printV("valid2 : ", val_datasets2[0].tot_cells, 1);
            else gvmetric.print("valid2 : ", val_datasets2[0].tot_cells, 1);

            std::cout << "   \033[1;34m$$ best result up to now: " << bestvmetric.acc() << "\033[0m" << std::endl;

          }



          // TRAIN ERROR        
          // predictions_vector = cv::Mat::zeros(td->y_train.rows, 1, CV_32F); 
          // tmetric.resetAll();
          // local_model->predict(td->X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
          // tmetric.checkpointTime();
          // computeAcc(td, predictions_vector, tmetric, "train1: ");

          // VALID ERROR
          // predictions_vector = cv::Mat::zeros(vd->y_train.rows, 1, CV_32F);
          // vmetric.resetAll();
          // local_model->predict(vd->X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
          // vmetric.checkpointTime();
          // computeAcc(vd, predictions_vector, vmetric, "valid1: ");


          std::string model_file_name_ = train_datasets[0].getSVMName(train_datasets[0].save_path);
          local_model->save(model_file_name_);
          if (skipped) continue;
          results.push_back(Results(nu, C, gamma, tmetric, gvmetric));
          gvmetric.seed = train_datasets[0].seed;
          //LOG
          gvmetric.log2YAML(nu, gamma, C, pca_mode, 
                          vd2->X_train.rows, vd2->tot_cells, 
                          train_datasets[0].getYAMLMetricsName());
        }
      }
    }
  }

  // PRINT results sorted (top to bottom: best to worse)

  std::sort(results.begin(), results.end()/*, std::greater<Results>()*/);

  // for (auto &res: results) {
  for  (int i=0; i<MIN(10, results.size()); i++) {
      std::cout << results[i] << std::endl;   
  }

  return 0;
}

#endif
