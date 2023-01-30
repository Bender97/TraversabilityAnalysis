#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <stdexcept>
#include <sstream>

#include "Feature.h"
#include "Cylinder.h"
#include "DataLoader.h"

#include "yaml-cpp/yaml.h"

using namespace std::chrono_literals;

YAML::Node sample_data = YAML::LoadFile("test.yaml");


static float *progress;
static int *stat;
static int *seq_stat;

static int tot_pools;
static int barWidth = 70;
static bool already_written = false, threadPrint_continue_flag=true;
static std::stringstream ss;

std::mutex mu, mu2;
std::condition_variable cond_var;
std::vector<int> seqs;

template<typename T>
void loadCyls(std::vector<T*> &cyls, YAML::Node &sample_data, bool verbose=false) {
  cyls.clear();
  int level;

  std::string data_ = sample_data["general"]["dataset"].as<std::string>();

  if (verbose) {
    std::cout << "DATA: " << data_ << std::endl;
    std::cout << "#######################################" << std::endl;
  }
  for (level=0; ; level++) {
    auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(level).length()), '0') + std::to_string(level);
    YAML::Node node = sample_data["general"][cyl_s.c_str()];
    if (!node) break;

    node["dataset"] = data_;
    node["load_path"] = sample_data["general"]["load_path"].as<std::string>();
    node["save_path"] = sample_data["general"]["save_path"].as<std::string>();
    node["store_features_filename"] = sample_data["general"]["store_features_filename"].as<std::string>();
    
    T *back_cyl = (level>0) ? (cyls[level-1]) : nullptr;
    
    if (data_=="SemKITTI") {
      auto cyl = new Cylinder_SemKITTI(node, back_cyl, ExpMode::produce); 
      if (verbose) cyl->printSummary();   
      cyls.push_back(cyl);
    }
  }
  if (verbose) std::cout << "#######################################" << std::endl;

  if (!level) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. please provide"
                  " at least a cylinder in yaml config file.\n"));
}

void threadPrint() {

  while(threadPrint_continue_flag) {
    std::unique_lock<std::mutex> lk(mu);
    cond_var.wait(lk);
    if (!threadPrint_continue_flag) break;

    if (already_written) 
      for (int i=0; i<tot_pools; i++) std::cout << "\e[A";

    for (int i=0; i<tot_pools; i++) {
      ss = std::stringstream("[");
      int pos = barWidth * progress[i];
      for (int i = 0; i < barWidth; ++i)
            if (i < pos) ss << "=";
            else if (i == pos) ss << ">";
            else ss << " ";
      ss << "]  ";
      ss << std::setw(5) << ((int)(progress[i]* 10000))/100.0f << " %";

      std::cout << std::setw(2) << seq_stat[i] << " "
                << std::setw(5) << stat[i] << " "
                << ss.str() << std::endl;
    }
    already_written = true;
    lk.unlock();
  }
}


// threaded!
void produceFeatures(int thread_idx, int start, int end) {
  
  std::vector<Cylinder*> cyls;
  mu2.lock();
  loadCyls(cyls, sample_data);
  mu2.unlock();
  for (auto &cyl: cyls) cyl->store_features_ofname += std::string("_") + std::to_string(thread_idx);
  
  DataLoader_SemKITTI dl;
  int tot = end - start;
  
  for (auto &seq : seqs) {
    seq_stat[thread_idx] = seq;

    for (int sample_idx=start; sample_idx < end; sample_idx++) {

      stat[thread_idx] = sample_idx;
      progress[thread_idx] = (float)(sample_idx+1-start) / tot;
      mu2.lock();
      dl.readData(seq, sample_idx, sample_data);
      mu2.unlock();
      if (dl.scene_normal.isZero()) continue;
      
      for (size_t i=0; i<cyls.size(); i++) 
        cyls[i]->produceFeaturesRoutine(dl, (!i) ? nullptr : cyls[i-1]);

      cond_var.notify_one();
    }
  }
}




int main (int argc, char** argv) {
  
  seqs = sample_data["mtthread"]["sequences"].as<std::vector<int>>();
  tot_pools = sample_data["mtthread"]["tot_pools"].as<int>();
  int sample_idx_start, sample_idx_end; 
  
  if (sample_data["mtthread"]["sample_idx_start"]) sample_idx_start = sample_data["mtthread"]["sample_idx_start"].as<int>();
  else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Please provide sample_idx_start!\n"));
  if (sample_data["mtthread"]["sample_idx_end"])   sample_idx_end   = sample_data["mtthread"]["sample_idx_end"].as<int>();
  else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Please provide sample_idx_end!\n"));

  std::thread workerThread(threadPrint);

  progress = MALLOC_(float, tot_pools);
  stat = MALLOC_(int, tot_pools);
  seq_stat = MALLOC_(int, tot_pools);

  int step = (int)std::ceil((float)(sample_idx_end-sample_idx_start)/tot_pools);

  std::cout << "start: " << sample_idx_start << " end: " << sample_idx_end << " step: " << step << std::endl;

  std::vector<std::thread> ts(tot_pools);

  for (int i=0; i<tot_pools; i++) {
    int start = sample_idx_start + i*step, end=sample_idx_start + (i+1)*step;
    if (end>sample_idx_end) end=sample_idx_end;

    progress[i]=0;
    stat[i]=start;
    seq_stat[i]=0;

    int thread_idx = i;

    // std::cout << "starting test pool [" << i << "] from " << start << " to " << end << std::endl;
    ts[i] = std::thread(produceFeatures, thread_idx, start, end);
  }

  for (auto &t: ts) t.join();
  
  threadPrint_continue_flag = false;
  cond_var.notify_one();
  workerThread.join();

  free(progress);
  free(stat);
  free(seq_stat);


  //////////////////////////////////////////////////////////////////
  /// CLEAN UP and MERGE to single files
  FILE *fp;
  int BUFFER_SIZE = 20000;
  char buffer[BUFFER_SIZE];

  std::vector<Cylinder_SemKITTI *> cyls;
  loadCyls(cyls, sample_data, true);

  for (int i=0; i<(int)cyls.size(); i++) {
    // prepare command statement
    std::string cat_cmd = "cat";
    std::string  rm_cmd = "rm ";
      
    for (int thread_idx=0; thread_idx<tot_pools; thread_idx++) {
      cat_cmd += std::string(" ") + cyls[i]->store_features_ofname + std::string("_") + std::to_string(thread_idx);
       rm_cmd += std::string(" ") + cyls[i]->store_features_ofname + std::string("_") + std::to_string(thread_idx);
    }
    cat_cmd += std::string(" > ") + cyls[i]->store_features_ofname;

    // execute cat (merge all temp file into one)
    fp = popen(cat_cmd.c_str(), "r");
    if (fp != NULL) {
        while (fgets(buffer, BUFFER_SIZE, fp) != NULL)
            printf("%s", buffer);
        pclose(fp);
    }

    // execute rm (remove temp files)
    fp = popen( rm_cmd.c_str(), "r");
    if (fp != NULL) {
        while (fgets(buffer, BUFFER_SIZE, fp) != NULL)
            printf("%s", buffer);
        pclose(fp);
    }

  }

  return 0;
}
