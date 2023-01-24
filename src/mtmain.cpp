#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>

#include "Feature.h"
#include "Cylinder.h"
#include "cv_ext.h"
#include "common_funcs.hpp"
#include "yaml-cpp/yaml.h"

using namespace std::chrono_literals;

YAML::Node sample_data = YAML::LoadFile("test.yaml");


static float *progress;
static int *stat;
static int *seq_stat;
static int tot_pools;
static int barWidth = 70;
static bool already_written = false, threadPrint_continue_flag=true;
std::mutex mu;
std::condition_variable cond_var;
std::vector<int> seqs;


void threadPrint() {
  while(threadPrint_continue_flag) {
    std::unique_lock<std::mutex> lk(mu);
    cond_var.wait(lk);
    if (!threadPrint_continue_flag) break;

    if (already_written) 
      for (int i=0; i<tot_pools; i++) std::cout << "\e[A";

    
    for (int i=0; i<tot_pools; i++) {
      std::string s="[";
      int pos = barWidth * progress[i];
      for (int i = 0; i < barWidth; ++i)
            if (i < pos) s+= "=";
            else if (i == pos) s+= ">";
            else s+= " ";
      s += "]  ";
      s+=std::to_string(((int)(progress[i]* 10000))/100.0f);

      std::cout << std::setw(2) << seq_stat[i] << " ";
      std::cout << std::setw(5) << stat[i] << " ";

      std::cout << s << std::endl;
    }
    already_written = true;
    lk.unlock();
  }
}



void threadAcc(int thread_idx, int start, int end) {
  std::vector<Cylinder> cyls;
  loadCyls(cyls, sample_data, true);
  for (auto &cyl: cyls) cyl.store_features_ofname += std::string("_") + std::to_string(thread_idx);
  
  std::vector<Eigen::Vector3d> points;
  std::vector<int> labels;
    
  int tot = end - start;
  
  for (auto &seq : seqs) {
    seq_stat[thread_idx] = seq;

    for (int sample_idx=start; sample_idx < end; sample_idx++) {

      stat[thread_idx] = sample_idx;

      progress[thread_idx] = (float)(sample_idx+1-start) / tot;

      points.clear();
      labels.clear();

      readData(seq, sample_idx, points, labels, sample_data);

      Eigen::MatrixXd scene_normal = computeSceneNormal(points);
      if (scene_normal.isZero()) continue;
      
      for (size_t i=0; i<cyls.size(); i++) 
        cyls[i].produceFeaturesRoutine(points, labels, scene_normal, 
                                       (!i) ? nullptr : &(cyls[i-1]));

      cond_var.notify_one();
    }
  }
}


int main (int argc, char** argv) {
  
  seqs = sample_data["mtthread"]["train"].as<std::vector<int>>();
  tot_pools = sample_data["mtthread"]["tot_pools"].as<int>();
  int sample_idx_start = 0, sample_idx_end = 150; 
  if (sample_data["mtthread"]["sample_idx_start"]) sample_idx_start = sample_data["mtthread"]["sample_idx_start"].as<int>();
  if (sample_data["mtthread"]["sample_idx_end"]) sample_idx_end = sample_data["mtthread"]["sample_idx_end"].as<int>();

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

    std::cout << "starting test pool [" << i << "] from " << start << " to " << end << std::endl;
    ts[i] = std::thread(threadAcc, thread_idx, start, end);
  }

  for (auto &t: ts) t.join();
  
  threadPrint_continue_flag = false;
  cond_var.notify_one();
  workerThread.join();

  free(progress);

  std::vector<Cylinder> cyls;
  loadCyls(cyls, sample_data, true);

  std::vector<std::string> prev_names(cyls.size());
    for (int i=0; i<(int)cyls.size(); i++)
      prev_names[i] = cyls[i].store_features_ofname;

  
  for (int i=0; i<(int)cyls.size(); i++) {
    std::string cat_cmd = "cat";
    std::string  rm_cmd = "rm ";
      
    for (int thread_idx=0; thread_idx<tot_pools; thread_idx++) {
      cat_cmd += std::string(" ") + prev_names[i] + std::string("_") + std::to_string(thread_idx);
       rm_cmd += std::string(" ") + prev_names[i] + std::string("_") + std::to_string(thread_idx);
    }

    cat_cmd += std::string(" > ") + prev_names[i];

    //execute cat
    FILE *fp;
    int BUFFER_SIZE = 20000;
    char buffer[20000];

    fp = popen(cat_cmd.c_str(), "r");
    if (fp != NULL)
    {
        while (fgets(buffer, BUFFER_SIZE, fp) != NULL)
            printf("%s", buffer);
        pclose(fp);
    }

    fp = popen( rm_cmd.c_str(), "r");
    if (fp != NULL)
    {
        while (fgets(buffer, BUFFER_SIZE, fp) != NULL)
            printf("%s", buffer);
        pclose(fp);
    }

  }

  return 0;
}
