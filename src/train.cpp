
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
//#include <Eigen/Dense>

#include <vector>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <future>
#include <functional>

#include "svm.h"
#include "Metric.h"
#include "Results.h"
#include "TrainDataset.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "common_funcs.hpp"
#include "yaml-cpp/yaml.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

struct svm_parameter param;   // set by parse_command_line

int train_duration_secs;
std::string train_config_path;
YAML::Node sample_data = YAML::LoadFile("test.yaml");

int tot_geom_features = sample_data["general"]["tot_geom_features"].as<int>();
int train_level, train_mode, pca_mode=-1, max_iter=1000, trick_mode=0;

std::string savepath;

std::vector<std::string> modes = std::vector<std::string>({"geom", "geom_label", "geom_all", "geom_pca", "geom_pca_label", "geom_pca_all_label"});

int parseCommandLineArguments(int argc, char** argv) {
    try {
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
        ("help", "produce help message")
        ("svm_type,t",       boost::program_options::value<int>(&param.svm_type)->default_value(NU_SVR), " C_SVC:0, NU_SVC:1, ONE_CLASS:2, EPSILON_SVR:3,      NU_SVR:4")
        ("kernel_type,k",    boost::program_options::value<int>(&param.kernel_type)->default_value(RBF), "LINEAR:0,   POLY:1,       RBF:2,     SIGMOID:3, PRECOMPUTED:4")
        ("degree,d",         boost::program_options::value<int>(&param.degree)->default_value(5),        "for poly")
        ("gamma,g",          boost::program_options::value<double>(&param.gamma)->default_value(0.5),    "for poly/rbf/sigmoid")
        ("coef0,c0",         boost::program_options::value<double>(&param.coef0)->default_value(0),      "for poly/sigmoid")
        ("nu",               boost::program_options::value<double>(&param.nu)->default_value(0.02),      "for NU_SVC, ONE_CLASS, and NU_SVR")
        ("cache_size,cs",    boost::program_options::value<double>(&param.cache_size)->default_value(1000000),   " in MB ")
        ("C",                boost::program_options::value<double>(&param.C)->default_value(0),          "for C_SVC, EPSILON_SVR and NU_SVR")
        ("eps,e",            boost::program_options::value<double>(&param.eps)->default_value(1e-2),     "stopping criteria")
        ("p",                boost::program_options::value<double>(&param.p)->default_value(0),          "for EPSILON_SVR")
        ("shrinking,s",      boost::program_options::value<int>(&param.shrinking)->default_value(0),     "use the shrinking heuristics")
        ("probability,prob", boost::program_options::value<int>(&param.probability)->default_value(0),   "do probability estimates")
        ("nr_weight,nr",     boost::program_options::value<int>(&param.nr_weight)->default_value(0),     "for C_SVC")
        ("train_duration,td",     boost::program_options::value<int>(&train_duration_secs)->default_value(-1),     "Training max duration time in seconds (int64_t)")
        ("train_config,tc",     boost::program_options::value<std::string>(&train_config_path)->default_value(""),     "Training config path");

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

    }
    catch(std::exception& e) {
        std::cout << "error: " << e.what() << "\n";
        return 0;
    }
    catch(...) {
        std::cout << "Exception of unknown type!\n";
        return 0;
    }

    return 1;
}

std::vector<TrainDataset> train_datasets;
std::vector<TrainDataset> val_datasets;

void computeAcc(TrainDataset *td, cv::Mat &pred, Metric &metric, std::string msg) {
  double plab;

  for (int r=0; r < td->y_train.rows; r++) {
    plab = pred.at<float>(r, 0) > 0 ? 1.0f : -1.0f;
    metric.update(plab, td->y_train.at<float>(r, 0));
  }

  metric.print(msg.c_str(), td->tot_cells, 1);
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

  //int level;
  int tot_geom_features = sample_data["general"]["tot_geom_features"].as<int>();

  if (sample_data["general"]["train_level"]) train_level = sample_data["general"]["train_level"].as<int>();
  else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. please provide which level to train on.\n"));

  
  auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(train_level).length()), '0') + std::to_string(train_level);
  YAML::Node node = sample_data["general"][cyl_s.c_str()];
  if (!node) throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Cylinder (at train_level) not found!\n"));


  std::string filename = node["store_path"].as<std::string>();

  if (node["max_iter"]) max_iter = node["max_iter"].as<int>();

  YAML::Node tnode = node["train"];
  YAML::Node vnode = node["valid"];

  if (node["mode"]) train_mode = node["mode"].as<int>();
  else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. please provide which level to train on.\n"));

  if (node["pca"]) pca_mode = node["pca"].as<int>();
  if (node["trick"]) trick_mode = node["trick"].as<int>();

  if (train_mode<3) pca_mode=-1;

  tnode["mode"] = train_mode; tnode["pca"] = pca_mode; tnode["trick"] = trick_mode;
  vnode["mode"] = train_mode; vnode["pca"] = pca_mode; tnode["trick"] = trick_mode;


  savepath = "results/";
  if (node["yamlsavepath"]) savepath = node["yamlsavepath"].as<std::string>();
  if (trick_mode) savepath += modes[train_mode]+std::string("_trick/");
  else savepath += modes[train_mode]+std::string("/");
  std::cout << "SAVEPATH: " << savepath << std::endl;

  boost::filesystem::path dir(savepath);
  if(boost::filesystem::create_directories(dir))
  {
      std::cerr<< "folder created: "<< savepath << std::endl;
  }

  TrainDataset td(filename, tnode, train_level, tot_geom_features, node["steps_num"].as<int>() * node["yaw_steps"].as<int>());
  std::cout << " -- Will train on:" << std::endl;
  train_datasets.push_back(td);
  td.summary();

  TrainDataset vd(std::string("val_") + filename, vnode, td.normalizer, td.pca, train_level, tot_geom_features, node["steps_num"].as<int>() * node["yaw_steps"].as<int>());
  std::cout << " -- Will valid on:" << std::endl;
  val_datasets.push_back(vd);
  vd.summary();

}


int main (int argc, char** argv)
{
  if (!parseCommandLineArguments(argc, argv)) return 0;

  parseConfig(train_config_path);

  if (train_duration_secs<0) train_duration_secs = 10000;

  //printModelParameters(param);

  std::vector<Results> results;

  for (size_t i=0; i<train_datasets.size(); i++) {

    float C = 1.0f;
    param.C = C;

    cv::Mat predictions_vector;

    for (auto &nu: train_datasets[i].nu_vec) {
      for (auto &gamma: train_datasets[i].gamma_vec) {

        std::cout << " ##############################      ###########################" << std::endl;
        std::cout << " training " << train_datasets[i].X_train.rows << " rows " << train_datasets[i].X_train.cols << " cols max_iter: " << max_iter << "  ##  nu: " << nu << " C: " << C << " gamma: " << gamma << std::endl;

        param.secs_to_wait = train_duration_secs;

        cv::Ptr<cv::ml::SVM> local_model;
        local_model.reset();
        local_model = cv::ml::SVM::create();
        local_model->setType(cv::ml::SVM::NU_SVR);
        local_model->setKernel(cv::ml::SVM::RBF);
        local_model->setNu(nu);
        local_model->setGamma(gamma);
        local_model->setC(1.0f);
        local_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, max_iter, 1e-6));

        local_model->train(train_datasets[i].X_train, cv::ml::ROW_SAMPLE, train_datasets[i].y_train);

        std::cout << "trained? " << local_model->isTrained() << std::endl;

        Metric tmetric;
        predictions_vector = cv::Mat::zeros(train_datasets[i].y_train.rows, 1, CV_32F); 
        tmetric.resetAll();
        local_model->predict(train_datasets[i].X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
        tmetric.checkpointTime();
        computeAcc(&(train_datasets[i]), predictions_vector, tmetric, "train: ");

        Metric vmetric;
        predictions_vector = cv::Mat::zeros(val_datasets[i].y_train.rows, 1, CV_32F);
        vmetric.resetAll();
        local_model->predict(val_datasets[i].X_train, predictions_vector, cv::ml::StatModel::RAW_OUTPUT);
        vmetric.checkpointTime();
        computeAcc(&(val_datasets[i]), predictions_vector, vmetric, "valid: ");


        std::string nu_s = std::to_string(nu);
        nu_s.erase ( nu_s.find_last_not_of('0') + 1, std::string::npos );
        nu_s.erase ( nu_s.find_last_not_of('.') + 1, std::string::npos );

        std::string gamma_s = std::to_string(gamma);
        gamma_s.erase ( gamma_s.find_last_not_of('0') + 1, std::string::npos );
        gamma_s.erase ( gamma_s.find_last_not_of('.') + 1, std::string::npos );

        std::string model_file_name_ = 
              "results/lv" + std::to_string(train_datasets[i].level) + "/"
               + modes[train_mode] + (train_datasets[i].trick_mode ? "_trick" : "") + "/svm_model"
               + ((train_mode>=3) ? (std::string("_") + std::to_string(pca_mode)) : "")
               + std::string("_") + nu_s
               + std::string("_") + gamma_s
               + std::string(".bin");

               std::cout << "train_mode " << train_mode << std::endl;

        local_model->save(model_file_name_);
        //svm_save_model(model_file_name_.c_str(), model);
        std::cout << "svm model stored at " << model_file_name_ << std::endl << std::endl;

        results.push_back(Results(nu, C, gamma, tmetric, vmetric));

        //log
        vmetric.toYaml(nu, gamma, C, pca_mode, val_datasets[i].X_train.rows, val_datasets[i].tot_cells, savepath);

      }
    }
  }

  std::sort(results.begin(), results.end()/*, std::greater<Results>()*/);

  for (auto &res: results) {
      std::cout << res << std::endl;   
  }

  return 0;
}

#endif
