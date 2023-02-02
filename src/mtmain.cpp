#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <stdexcept>
#include <sstream>

#include <sys/types.h>
#include <dirent.h>

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
      std::cout << "creating cyl at level " << level << std::endl;
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


class Pair {
public:
  int seq, idx;
  Pair(int s, int id) {seq=s; idx=id;}
  Pair() {}
};


// threaded!
void produceFeatures(int thread_idx, int start, int end, std::vector<Pair> &pairs) {
  
  std::vector<Cylinder*> cyls;
  mu2.lock();
  std::string temp = sample_data["general"]["store_features_filename"].as<std::string>();
  sample_data["general"]["store_features_filename"] = temp + "_" + std::to_string(thread_idx);
  std::cout << "HI FROM PRODUCE " + sample_data["general"]["store_features_filename"].as<std::string>() << std::endl;
  loadCyls(cyls, sample_data);
  sample_data["general"]["store_features_filename"] = temp;
  mu2.unlock();
  
  DataLoader_SemKITTI dl;
  int tot = end - start;
  
  // for (auto &seq : seqs) {
  //   seq_stat[thread_idx] = seq;

  //   for (int sample_idx=start; sample_idx < end; sample_idx++) {
  
  Pair pair;
  int seq, sample_idx;
  for (int i=start; i < end; i++) {
    pair=pairs[i];
    seq = pair.seq;
    sample_idx = pair.idx;

    stat[thread_idx] = i;
    seq_stat[thread_idx] = seq;
    progress[thread_idx] = (float)(i+1-start) / tot;
    mu2.lock();
    dl.readData(seq, sample_idx, sample_data);
    mu2.unlock();
    if (dl.scene_normal.isZero()) continue;

    for (size_t a=0; a<cyls.size(); a++) 
      cyls[a]->produceFeaturesRoutine(dl, (!a) ? nullptr : cyls[a-1]);

    cond_var.notify_one();
  }
}
int count_samples(YAML::Node &sample_data, int seq) {

  auto seq_s = std::string(2 - MIN(2, std::to_string(seq).length()), '0') + std::to_string(seq);
  std::string path = sample_data["general"]["dataset_path"].as<std::string>()
                    +"sequences/"+seq_s+"/labels/";

  int fileCount = 0;
  DIR *dp;
  struct dirent *ep;     
  dp = opendir (path.c_str());
  if (dp != NULL) {
    while ((ep = readdir (dp))) fileCount++;
    (void) closedir (dp);
  }
  else {
    std::cout << path << std::endl; 
    perror ("Couldn't open the directory");
  }
  return fileCount;
}


int main (int argc, char** argv) {
  
  seqs = sample_data["mtthread"]["sequences"].as<std::vector<int>>();
  tot_pools = sample_data["mtthread"]["tot_pools"].as<int>();
  int sample_idx_start, sample_idx_end; 


  std::vector<Cylinder_SemKITTI *> cyls;
  loadCyls(cyls, sample_data, true);
  
  // if (sample_data["mtthread"]["sample_idx_start"]) sample_idx_start = sample_data["mtthread"]["sample_idx_start"].as<int>();
  // else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Please provide sample_idx_start!\n"));
  // if (sample_data["mtthread"]["sample_idx_end"])   sample_idx_end   = sample_data["mtthread"]["sample_idx_end"].as<int>();
  // else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Please provide sample_idx_end!\n"));

  std::thread workerThread(threadPrint);

  progress = MALLOC_(float, tot_pools);
  stat = MALLOC_(int, tot_pools);
  seq_stat = MALLOC_(int, tot_pools);

  // int step = (int)std::ceil((float)(sample_idx_end-sample_idx_start)/tot_pools);
  // std::cout << "start: " << sample_idx_start << " end: " << sample_idx_end << " step: " << step << std::endl;

  std::vector<Pair> pairs;
  for (auto seq: seqs) {
    int tot = 10; //count_samples(sample_data, seq);
    for (int i=0; i<tot; i++) pairs.push_back(Pair(seq, i));
  }

  sample_idx_start = 0;
  sample_idx_end = (int) pairs.size();
  int step = (int)std::ceil((float)pairs.size()/tot_pools);

  std::vector<std::thread> ts(tot_pools);

  for (int i=0; i<tot_pools; i++) {
    int start = sample_idx_start + i*step, end=sample_idx_start + (i+1)*step;
    if (end>sample_idx_end) end=sample_idx_end;

    progress[i]=0;
    stat[i]=start;
    seq_stat[i]=0;

    int thread_idx = i;

    // std::cout << "starting test pool [" << i << "] from " << start << " to " << end << std::endl;
    ts[i] = std::thread(produceFeatures, thread_idx, start, end, std::ref(pairs));
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

  for (int level=0; level<(int)cyls.size(); level++) {
    // prepare command statement
    std::string cat_cmd = "cat";
    std::string  rm_cmd = "rm ";

    std::string name = cyls[level]->store_features_filename;
    name.erase(name.length()-5); // remove "{LEVEL}.bin" // ASSUMING 0<LEVEL<10
    for (int thread_idx=0; thread_idx<tot_pools; thread_idx++) {
      
      cat_cmd += std::string(" ") + name + std::string("_") + std::to_string(thread_idx) + std::to_string(level) + ".bin";
       rm_cmd += std::string(" ") + name + std::string("_") + std::to_string(thread_idx) + std::to_string(level) + ".bin";
    }
    cat_cmd += std::string(" > ") + cyls[level]->store_features_filename;

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
