
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
YAML::Node sample_data = YAML::LoadFile("../models/test.yaml");

std::string nuscenes = "/home/fusy/repos/code2_paper/code2/build/results05/test_features_data_4p2_nuscenes.bin";
std::string pandaset = "/home/fusy/repos/code2_paper/code2/build/results05/test_features_data_4p2_pandaset.bin";

int train_level, train_mode, pca_mode=-1, max_iter=1000, trick_mode=0;

std::vector<std::string> modes = std::vector<std::string>({"geom", "geom_label", "geom_all", "geom_pca", "geom_pca_label", "geom_pca_all_label"});

Dataset4Train train_dataset;
Dataset4Valid valid_dataset;
Dataset4Test test_nuscenes, test_pandaset;

std::string journal_path, timems;

void notRefreshingProgressBar(int barWidth, float progress) {
  int pos = barWidth * progress;
  std::cout << "[";
  for (int i = 0; i < barWidth; ++i) {
      if (i < pos) std::cout << "=";
      else if (i == pos) std::cout << ">";
      else std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();
}

void computeAcc(Dataset4 &d, cv::Mat &pred, Metric &metric, std::string msg) {
  double plab;

  for (int r=0; r < d.y_train.rows; r++) {
    plab = pred.at<float>(r, 0) > 0 ? 1.0f : -1.0f;
    metric.update(plab, d.y_train.at<float>(r, 0));
  }
  if (pred.rows > 50000 && metric.acc()>0.9734) //0.9075)
    metric.printV(msg.c_str(), d.tot_cells, 1);
  else metric.print(msg.c_str(), d.tot_cells, 1);
}

void handleOut(Dataset4 &d, cv::Mat &pred, Metric &vmetric, Metric &gvmetric, std::string msg1, std::string msg2, 
                uint32_t size, int32_t offset, float checkp) {
  double plab;

  for (uint32_t r=0; r < size; r++) {
    plab = pred.at<float>(r, 0) > 0 ? 1.0f : -1.0f;
    vmetric.update(plab, valid_dataset.y_train.at<float>(r+offset, 0));
  }
  gvmetric += vmetric;
  if (offset>0) {
    std::cout << "\e[A";
    std::cout << "\e[A";
  }
  // std::cout << "                                                                          ";
  std::cout << msg1 << std::setw(9) << std::setprecision(5) << checkp*100 << "%: " << vmetric.acc()*100 << std::endl;
  std::cout << msg2 << std::setw(12) << " "                                         << gvmetric.acc()*100 << std::endl;
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

  journal_path = sanitize(sample_data["general"]["save_path"].as<std::string>()) + "/journal" + std::to_string(train_level) + "/";
  if(stat(journal_path.c_str(), &st) == -1) {
      mkdir(journal_path.c_str(), 0700);
      std::cerr<< "folder created: "<< journal_path << std::endl;
  }

  node["load_path"] = sample_data["general"]["load_path"];
  node["save_path"] = sample_data["general"]["save_path"];
  node["store_features_filename"] = sample_data["general"]["store_features_filename"];
  node["seed"] = sample_data["general"]["seed"];

  train_dataset = Dataset4Train(node, train_level);
  std::cout << " -- Will train on:" << std::endl;
  train_dataset.summary();


  // node["store_features_filename"] = "features_data_4p_notgt";
  valid_dataset = Dataset4Valid(node, train_level, train_dataset.normalizer, train_dataset.pca);
  std::cout << " -- Will valid on:" << std::endl;
  valid_dataset.summary();

  // node["store_features_filename"] = nuscenes;
  // test_nuscenes = Dataset4Test(node, train_level, train_dataset.normalizer, train_dataset.pca);
  // std::cout << " -- Will test on nuscenes as:" << std::endl;
  // test_nuscenes.summary();

  // node["store_features_filename"] = pandaset;
  // test_pandaset = Dataset4Test(node, train_level, train_dataset.normalizer, train_dataset.pca);
  // std::cout << " -- Will test on pandaset as:" << std::endl;
  // test_pandaset.summary();

}

void journaling(Dataset4& traindataset, Metric &tmetric, Dataset4 &validdataset, Metric &gvmetric, bool skipped) {
  auto path = journal_path + "/" + timems + ".yaml";

  std::ofstream out(path.c_str());
  YAML::Emitter outyaml(out);
  outyaml << YAML::BeginMap;
  outyaml << YAML::Key << "nu" << YAML::Flow << std::to_string(train_dataset.svm_nu);
  outyaml << YAML::Key << "gamma" << YAML::Flow << std::to_string(train_dataset.svm_gamma);
  outyaml << YAML::Key << "C" << YAML::Flow << std::to_string(train_dataset.svm_C);
  outyaml << YAML::Key << "iters" << YAML::Flow << std::to_string(train_dataset.svm_iters);
  outyaml << YAML::Key << "seed" << YAML::Flow << std::to_string(train_dataset.seed);
  outyaml << YAML::Key << "mode" << YAML::Flow << std::to_string(train_dataset.mode);
  outyaml << YAML::Key << "pca_mode" << YAML::Flow << std::to_string(train_dataset.pca_mode);

  outyaml << YAML::Key << "train_rows" << YAML::Flow << std::to_string(train_dataset.X_train.rows);
  outyaml << YAML::Key << "train_tn" << YAML::Flow << std::to_string(tmetric.tn);
  outyaml << YAML::Key << "train_fp" << YAML::Flow << std::to_string(tmetric.fp);
  outyaml << YAML::Key << "train_fn" << YAML::Flow << std::to_string(tmetric.fn);
  outyaml << YAML::Key << "train_tp" << YAML::Flow << std::to_string(tmetric.tp);
  outyaml << YAML::Key << "train_latency" << YAML::Flow << std::to_string((tmetric.checkpointTime_  * train_dataset.tot_cells / tmetric.tot));
  outyaml << YAML::Key << "train_acc" << YAML::Flow << std::to_string((tmetric.tp+tmetric.tn)/(float)tmetric.tot);

  if (skipped) {
    outyaml << YAML::Key << "valid_rows" << YAML::Flow << std::to_string(-1);
    outyaml << YAML::Key << "valid_tn" << YAML::Flow << std::to_string(-1);
    outyaml << YAML::Key << "valid_fp" << YAML::Flow << std::to_string(-1);
    outyaml << YAML::Key << "valid_fn" << YAML::Flow << std::to_string(-1);
    outyaml << YAML::Key << "valid_tp" << YAML::Flow << std::to_string(-1);
    outyaml << YAML::Key << "valid_latency" << YAML::Flow << std::to_string(-1);
    outyaml << YAML::Key << "valid_acc" << YAML::Flow << std::to_string(-1);
  }
  else {
    gvmetric.compute();
    outyaml << YAML::Key << "valid_rows" << YAML::Flow << std::to_string(valid_dataset.X_train.rows);
    outyaml << YAML::Key << "valid_tn" << YAML::Flow << std::to_string(gvmetric.tn);
    outyaml << YAML::Key << "valid_fp" << YAML::Flow << std::to_string(gvmetric.fp);
    outyaml << YAML::Key << "valid_fn" << YAML::Flow << std::to_string(gvmetric.fn);
    outyaml << YAML::Key << "valid_tp" << YAML::Flow << std::to_string(gvmetric.tp);
    outyaml << YAML::Key << "valid_iouT" << YAML::Flow << std::to_string(gvmetric.iouT);
    outyaml << YAML::Key << "valid_iouF" << YAML::Flow << std::to_string(gvmetric.iouF);
    outyaml << YAML::Key << "valid_f1" << YAML::Flow << std::to_string(gvmetric.f1);
    outyaml << YAML::Key << "valid_cohen" << YAML::Flow << std::to_string(gvmetric.cohen);
    outyaml << YAML::Key << "valid_latency" << YAML::Flow << std::to_string((gvmetric.checkpointTime_  * valid_dataset.tot_cells / gvmetric.tot));
    outyaml << YAML::Key << "valid_acc" << YAML::Flow << std::to_string((gvmetric.tp+gvmetric.tn)/(float)gvmetric.tot);
  }
  outyaml << YAML::EndMap;
  out.close();
}

int main (int argc, char** argv)
{
  parseConfig(train_config_path);

  std::vector<Results> results;
  cv::Mat predictions_vector;
  Metric tmetric, vmetric, gvmetric, bestvmetric;

  Metric testmetric_nuscenes, testmetric_pandaset, gtestmetric_pandaset;


  bool skipped=false;
  std::vector<int> iters = sample_data["general"]["iters"].as<std::vector<int>>();

  int step_cont=0, tot_steps=train_dataset.C_vec.size()*train_dataset.nu_vec.size()
                            *train_dataset.gamma_vec.size() * iters.size();
  // bool first = true;
  // for (auto &C: train_dataset.C_vec) {
  //   for (auto &nu: train_dataset.nu_vec) {
  //     for (auto &gamma: train_dataset.gamma_vec) {
    auto seed = ((int32_t) (time(NULL))); // % 3000000000000; // 27813
    std::default_random_engine rand_dev{seed};
    std::mt19937               generator(rand_dev());
    std::cout << "SEeEeEeeEeeeED: " << seed << std::endl;
    std::uniform_int_distribution<>  distr(0, 10000000);

    // std::string load_path = "/home/fusy/repos/code2_paper/code2/build/results06/lv0/geom_pca/svm_model_17_0.2895_0.1095_7000.bin_1685106887928.params";
    // std::ifstream ifs(load_path, std::ios::binary);
    // if(!ifs.is_open()){
    //   std::cout << "file not found " << load_path << "\n";
    //   return 1;
    // }
    // float nu, gamma;
    // nu = 0.2488;
    // gamma = 0.0873;
    // iters = std::vector<int>();
    // iters.push_back(7000);
    // iters.push_back(5500);

	  // ifs.read((char*)(&nu), sizeof(float));
	  // ifs.read((char*)(&gamma), sizeof(float));
    // ifs.close();

    // std::cout << "read\nnu:" << std::setprecision(20) << nu << "\ngamma:\n" << std::setprecision(20) << gamma << "\n";


    while(true) {
        
        float nu = (distr(generator) / 10000000.0f * (0.30-0.16)) + 0.16;
        float gamma = (distr(generator) / 10000000.0f * (0.105-0.065)) + 0.065;

        nu = ((int) (nu * 10000)) / 10000.0f;
        gamma = ((int) (gamma * 10000)) / 10000.0f;

        float C = 1.0f;
        for (auto &iter: iters) {
          max_iter = iter;
          std::cout << " ##############################      ###########################" << std::endl;
          std::cout << "epoch " << std::setw(5) << (step_cont+1) << "/" << std::setw(5) << tot_steps << " " << std::endl;
          notRefreshingProgressBar(100, ((float)(step_cont+1)) / tot_steps);
          std::cout << " max_iter: " << max_iter << "  ##  nu: " << std::setprecision(50) << nu 
                    << " C: " << C << " gamma: " << std::setprecision(50) << gamma << std::endl
                    << " ##  train rows: " << train_dataset.X_train.rows
                    << " ##        seed: " << train_dataset.seed
                    << std::endl;
          
          // if (step_cont<5) {step_cont++; continue;}

          // TRAIN MODEL: SET PARAMS
          cv::Ptr<cv::ml::SVM> local_model;
          local_model.reset();
          local_model = cv::ml::SVM::create();
          local_model->setType(cv::ml::SVM::NU_SVR);
          local_model->setKernel(cv::ml::SVM::RBF);
          local_model->setNu(nu);
          local_model->setGamma(gamma);
          local_model->setC(C);

          train_dataset.svm_gamma = gamma;
          train_dataset.svm_nu = nu;
          train_dataset.svm_C = C;
          train_dataset.svm_iters = max_iter;
          
          local_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, max_iter, 1e-6)); // it was 1e-6
          local_model->train(train_dataset.X_train, cv::ml::ROW_SAMPLE, train_dataset.y_train);
          
          step_cont++;

          timems = std::to_string(std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count());

          // std::ofstream xtrainfs("x_train.bin", std::ios::binary);
          // for (int row_d=0; row_d<train_dataset.X_train.rows; row_d++) {
          //   for (int col_d=0; col_d<train_dataset.X_train.cols; col_d++) {
          //     xtrainfs.write((const char*)(&(train_dataset.X_train.at<float>(row_d, col_d))), sizeof(float));
          //   }
          // }
          // xtrainfs.close();

          // std::ofstream xvalidfs("x_valid.bin", std::ios::binary);
          // for (int row_d=0; row_d<valid_dataset.X_train.rows; row_d++) {
          //   for (int col_d=0; col_d<valid_dataset.X_train.cols; col_d++) {
          //     xvalidfs.write((const char*)(&(valid_dataset.X_train.at<float>(row_d, col_d))), sizeof(float));
          //   }
          // }
          // xvalidfs.close();
          // assert(false);


          // TRAIN ERROR        
          predictions_vector = cv::Mat::zeros(train_dataset.y_train.rows, 1, CV_32F); 
          tmetric.resetAll();
          local_model->predict(train_dataset.X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
          tmetric.checkpointTime();
          std::string msg1 = "train: ";
          computeAcc(train_dataset, predictions_vector, tmetric, msg1.c_str());
          float acc = tmetric.acc();
          skipped = false;
          // if (acc<0.943) {// || acc>=0.95) {
          if (acc<0.92) {// || acc>=0.95) {
            std::cout << "skip!" << std::endl;
            skipped = true;
            journaling(train_dataset, tmetric, valid_dataset, gvmetric, true);
            continue;
          }

          // VALID ERROR
          uint32_t block_size = 100000, end, size;
          float checkp;
          gvmetric.resetAll(); //GLOBAL Validation metric
          // Compute the  the validation error in blocks: in this way you can see the progression
          for (size_t offset=0; offset<valid_dataset.X_train.rows+block_size; offset+=block_size) {

            end = MIN(valid_dataset.X_train.rows-1, offset+block_size);
            size = end-offset;
            if (size<=0 || offset >= (size_t)valid_dataset.X_train.rows) break;

            predictions_vector = cv::Mat::zeros(size, 1, CV_32F);
            vmetric.resetAll(); // LOCAL validation metric (in this chunk/block)
            local_model->predict(valid_dataset.X_train.rowRange(offset, end), predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
            vmetric.checkpointTime();
            
            checkp = (float) end/(valid_dataset.X_train.rows-1);
            handleOut(valid_dataset, predictions_vector, vmetric, gvmetric, 
                        "valid2 : ", "glob   : ", size, offset, checkp);
            
            // for (int c=0; c<predictions_vector.cols; c++) {
            //   for (int r=0; r<100; r++) {
            //     std::cout << predictions_vector.at<float>(r, c) << " ";
            //   }
            // }
            // std::cout << "\n";
            // assert(false);

            if (checkp > 0.45 && gvmetric.acc()<0.85) {
              std::cout << "skip!" << std::endl;
              skipped = true;
              journaling(train_dataset, tmetric, valid_dataset, gvmetric, true);
              break;
            }
            
            // if (checkp > 0.7 && gvmetric.acc()<0.904) {
            //   std::cout << "skip!" << std::endl;
            //   skipped = true;  
            //   journaling(train_dataset, tmetric, valid_dataset, gvmetric, true);
            //   break;
            // }
          }
          if (skipped) continue;

          if (gvmetric > bestvmetric) bestvmetric = gvmetric;
          // if ( gvmetric.acc()>0.9075)
          if ( gvmetric.acc()>0.91) {
            gvmetric.print("valid2 : ", valid_dataset.tot_cells, 1);
            /*
            if ( gvmetric.acc()>=0.914) {

              // TEST NUSCENES
              predictions_vector = cv::Mat::zeros(test_nuscenes.y_train.rows, 1, CV_32F); 
              testmetric_nuscenes.resetAll();
              local_model->predict(test_nuscenes.X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
              testmetric_nuscenes.checkpointTime();
              msg1 = "test on nuscenes: ";
              computeAcc(test_nuscenes, predictions_vector, testmetric_nuscenes, msg1.c_str());

              // TEST PANDASET
              predictions_vector = cv::Mat::zeros(test_pandaset.y_train.rows, 1, CV_32F); 

              uint32_t block_size = 100000, end, size;
              float checkp;
              gtestmetric_pandaset.resetAll(); //GLOBAL test metric
              // Compute the  the test error in blocks: in this way you can see the progression
              for (size_t offset=0; offset<test_pandaset.X_train.rows+block_size; offset+=block_size) {

                end = MIN(test_pandaset.X_train.rows-1, offset+block_size);
                size = end-offset;
                if (size<=0 || offset >= (size_t)test_pandaset.X_train.rows) break;

                predictions_vector = cv::Mat::zeros(size, 1, CV_32F);
                gtestmetric_pandaset.resetAll(); // LOCAL test metric (in this chunk/block)
                local_model->predict(test_pandaset.X_train.rowRange(offset, end), predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
                gtestmetric_pandaset.checkpointTime();
                
                checkp = (float) end/(test_pandaset.X_train.rows-1);
                handleOut(test_pandaset, predictions_vector, gtestmetric_pandaset, gtestmetric_pandaset, 
                            "pandaset : ", "glob     : ", size, offset, checkp);
              }

            }
            */
          }
          else gvmetric.print("valid2 : ", valid_dataset.tot_cells, 1);

          std::cout << "   \033[1;34m$$ best result up to now: " << std::setprecision(10) << bestvmetric.acc() << "\033[0m" << std::endl;

          std::string model_file_name_ = train_dataset.getSVMName(train_dataset.save_path);
          local_model->save(model_file_name_);
          std::cout << "model saved at " << model_file_name_ << "\n";

          std::string params_filename = model_file_name_ + "_" + timems + ".params";
          std::ofstream ofs(params_filename, std::ios::binary);
          ofs.write((const char*)(&nu), sizeof(float));
          ofs.write((const char*)(&gamma), sizeof(float));
          ofs.close();


          if (skipped) continue;
          results.push_back(Results(nu, C, gamma, tmetric, gvmetric));
          gvmetric.seed = train_dataset.seed;
          
          //LOG
          gvmetric.log2YAML(nu, gamma, C, pca_mode, 
                          valid_dataset.X_train.rows, valid_dataset.tot_cells,
                          train_dataset.getYAMLMetricsName());
          journaling(train_dataset, tmetric, valid_dataset, gvmetric, false);
          
        }
  //     }
    }
    //}

  // PRINT results sorted (top to bottom: best to worse)

  std::sort(results.begin(), results.end()/*, std::greater<Results>()*/);

  // for (auto &res: results) {
  for  (size_t i=0; i<MIN(10, results.size()); i++) {
      std::cout << results[i] << std::endl;   
  }

  return 0;
}

#endif
