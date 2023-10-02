#include "TrainDataset.h"

class ProgressBar {
public:
    float progress = 0.0;
    int barWidth;
    std::string msg;
    ProgressBar(int barWidth_, const std::string &msg_) {barWidth = barWidth_; msg = msg_;}
    ProgressBar(int barWidth_) {barWidth = barWidth_; msg="";}
    ~ProgressBar() {update(1.0f);}

    void update(float progress) {

        int pos = barWidth * progress;
        std::cout << msg << " [";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        if (progress>=1.0f) std::cout << std::endl;
    }
};

Dataset4::Dataset4() : Cylinder() {}

Dataset4::Dataset4(YAML::Node &node_cyl, int level_)
    : Cylinder(node_cyl) 
{}

Dataset4::Dataset4(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_)
    : Cylinder(node_cyl)
{
  normalizer = normalizer_;
  pca = pca_;
}

Dataset4Train::Dataset4Train() : Dataset4() {}
Dataset4Train::Dataset4Train(YAML::Node &node_cyl, int level_)
  : Dataset4(node_cyl, level_) {

  parseYAMLConfig(node_cyl, level_);
  computeFeaturesCols();
  load();
  }

Dataset4Valid::Dataset4Valid() : Dataset4()  {}
Dataset4Valid::Dataset4Valid(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_)
  : Dataset4(node_cyl, level_, normalizer_, pca_) {

  parseYAMLConfig(node_cyl, level_);
  computeFeaturesCols();
  load();
  }


Dataset4Test::Dataset4Test() : Dataset4()  {}
Dataset4Test::Dataset4Test(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_)
  : Dataset4(node_cyl, level_, normalizer_, pca_) {

  parseYAMLConfig(node_cyl, level_);
  computeFeaturesCols();
  load();
  }

void Dataset4::parseYAMLConfig(YAML::Node &node_cyl, int level_) {std::cout << "not the right one\n";}

void Dataset4Train::parseYAMLConfig(YAML::Node &node_cyl, int level_) {
  trick_mode=0;

  YAML::Node node_mode = node_cyl["train"];
  nu_vec    = node_mode["nu_vec"].as<std::vector<float>>();
  C_vec     = node_mode["C_vec"].as<std::vector<float>>();
  gamma_vec = node_mode["gamma_vec"].as<std::vector<float>>();  

  num_entries_to_train_on = node_mode["feats_to_sample"].as<int>();
  need_to_balance_flag    = node_mode["balance_flag"].as<bool>();
  level = level_;

  save_path = sanitize(node_cyl["save_path"].as<std::string>());
  load_path = sanitize(node_cyl["load_path"].as<std::string>());

  seed       = node_cyl["seed"].as<int32_t>();
  mode       = node_cyl["mode"].as<int>();
  trick_mode = node_cyl["trick"].as<int>();
  pca_mode   = node_cyl["pca"].as<int>();
  store_features_filename = load_path + node_cyl["store_features_filename"].as<std::string>() 
                            + std::to_string(level_) + ".bin";

  // LOGGING to console
  std::cout << "train level: " << level
            << "  - " << modes[mode]
            << "  - trick " << (trick_mode ? " on  " : "off  ") 
            << "  - pca: " << pca_mode
            << "  - seed: " << seed << std::endl;

}


void Dataset4Valid::parseYAMLConfig(YAML::Node &node_cyl, int level_) {
  trick_mode=0;

  YAML::Node node_mode = node_cyl["valid"];

  num_entries_to_train_on = node_mode["feats_to_sample"].as<int>();
  need_to_balance_flag    = node_mode["balance_flag"].as<bool>();
  level = level_;

  save_path = sanitize(node_cyl["save_path"].as<std::string>());
  load_path = sanitize(node_cyl["load_path"].as<std::string>());

  seed       = node_cyl["seed"].as<int32_t>();
  mode       = node_cyl["mode"].as<int>();
  trick_mode = node_cyl["trick"].as<int>();
  pca_mode   = node_cyl["pca"].as<int>();
  store_features_filename = load_path + "val_"
                            + node_cyl["store_features_filename"].as<std::string>() 
                            + std::to_string(level_) + ".bin";

  // LOGGING to console
  std::cout << "train level: " << level
            << "  - " << modes[mode] << " -"
            << "  trick " << (trick_mode ? " on  " : "off  ") 
            << "  pca: " << pca_mode << std::endl;

}


void Dataset4Test::parseYAMLConfig(YAML::Node &node_cyl, int level_) {
  trick_mode=0;

  YAML::Node node_mode = node_cyl["valid"];

  num_entries_to_train_on = node_mode["feats_to_sample"].as<int>();
  need_to_balance_flag    = node_mode["balance_flag"].as<bool>();
  level = level_;

  save_path = sanitize(node_cyl["save_path"].as<std::string>());
  load_path = sanitize(node_cyl["load_path"].as<std::string>());

  seed       = node_cyl["seed"].as<int32_t>();
  mode       = node_cyl["mode"].as<int>();
  trick_mode = node_cyl["trick"].as<int>();
  pca_mode   = node_cyl["pca"].as<int>();
  store_features_filename = node_cyl["store_features_filename"].as<std::string>();

  // LOGGING to console
  std::cout << "test level: " << level
            << "  - " << modes[mode] << " -"
            << "  trick " << (trick_mode ? " on  " : "off  ") 
            << "  pca: " << pca_mode << std::endl;

}



void Dataset4::readLabels() {
  Feature feature;
  float label;
  int pos = 0;

  std::ifstream in(store_features_filename.c_str(), std::ios::in | std::ios::binary);
  if (level>0 && trick_mode) {
    while(true) {
        if (!feature.ignoreFeatureFromFile(in, inherited_labels_size)) break;
        in.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
        if (feature.derived_features[ref_label_idx] >= 0) continue; // skip all GT trav 
        y_train.at<float>(pos, 0) = static_cast<float>((float)label);
        if (label>0) cont_trav ++;
        else cont_nontrav++;
        pos++;
    }
  }
  else {
    ProgressBar pbar(70, "reading labels: ");
    while(true) {
        if (!feature.ignoreFeatureFromFile(in, inherited_labels_size)) break;
        in.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
        y_train.at<float>(pos, 0) = static_cast<float>((float)label);
        if (label>0) cont_trav ++;
        else cont_nontrav++;
        if (pos%10000==0) pbar.update(pos/(float)tot_samples_found);
        pos++;
    }
  }
  in.close();

  std::cout << "                source has: " << pos << " samples:  " 
            << cont_trav << " T and " << cont_nontrav << " nT" << std::endl;
}

void Dataset4::loadData() {
  Feature feature;
  std::vector<float> vec;
  float label;
  float *features_matrix_data_row;
  int pos=0, row, c;
  int step = num_entries_to_train_on / 100;

  ProgressBar pbar(70, "loading features: ");

  std::ifstream feat_file(store_features_filename.c_str(), std::ios::in | std::ios::binary);
  for (row=0; row<tot_samples_found && pos<num_entries_to_train_on; row++) {
    if (row<sampled_idxs[pos]) {
      if (!feature.ignoreFeatureFromFile(feat_file, inherited_labels_size)) break;
      feat_file.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
      continue;
    }
    feature.fromFileLine(feat_file, inherited_labels_size);
    feat_file.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));

    vec  = feature.toVectorTransformed();
    features_matrix_data_row = X_train.ptr<float>(pos);
    for (c=0; c<max_feats_num; c++) features_matrix_data_row[c] = vec[re_idx[c]];
    
    y_train.at<float>(pos, 0) = static_cast<float>((float)label);
    if (pos%step==0) pbar.update(pos / (float)num_entries_to_train_on);
    pos++;
  }
  feat_file.close();
}

void Dataset4::sampleIdxs() { throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. empty sample idxs") ); }

void Dataset4Valid::sampleIdxs() {

  if (num_entries_to_train_on < 0) {
    sampled_idxs = std::vector<int>(tot_samples_found);
    std::cout << "loading full " << tot_samples_found << " samples" << std::endl;
    for (int i=0; i<tot_samples_found; i++) sampled_idxs[i] = i;
    num_entries_to_train_on = tot_samples_found;
    return;
  }

  sampled_idxs = std::vector<int>(num_entries_to_train_on);

  // seed = (int32_t)time(0); // in case you want full randomness
  bool flag;
  
  std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};;
  std::mt19937               generator(rand_dev());
  std::uniform_int_distribution<>  distr(seed, tot_samples_found-1);

  // sample feature rows to select only a subsample of the whole training set
  taken.resize(tot_samples_found, false);
  counters[0] = 0; counters[1] = 0;

  float half_netto = num_entries_to_train_on/2.0f;

  int num;
  for (int t=0; t<num_entries_to_train_on; t++)  {
    flag = true;
    while(flag) {
       num = distr(generator);
       if (taken[num]==false && (!need_to_balance_flag || counters[((int)y_train.at<float>(num)+1)>>1]<half_netto) ) flag=false;
    }

    taken[num]=true;
    counters[((int)y_train.at<float>(num)+1)>>1]++;
    sampled_idxs[t] = num;
  }
}

void Dataset4Test::sampleIdxs() {
  sampled_idxs = std::vector<int>(tot_samples_found);
  std::cout << "loading full " << tot_samples_found << " samples" << std::endl;
  for (int i=0; i<tot_samples_found; i++) sampled_idxs[i] = i;
  num_entries_to_train_on = tot_samples_found;
  return;
}


/*
void Dataset4Train::sampleIdxs() {
  sampled_idxs = std::vector<int>(num_entries_to_train_on);

  int BLOCK_SIZE = 80000;
  //int to_sample = (int) std::floor((float)BLOCK_SIZE / tot_samples_found * num_entries_to_train_on);
  
  int b = tot_samples_found % BLOCK_SIZE;

  int k = (int) std::floor( (float) tot_samples_found / BLOCK_SIZE);

  double ratio = (double)b / (double)BLOCK_SIZE;

  std::cout << "b: " << b << std::endl;
  std::cout << "k: " << k << std::endl;
  std::cout << "ratio: " << ratio << std::endl;

  int to_sample = (int) ((float) num_entries_to_train_on / (k+ratio));
  int to_sample_b = (int) ((ratio) / (k+ratio));

  int num, cont=0;
  bool flag;

  std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};;
  std::mt19937               generator(rand_dev());
  std::uniform_int_distribution<>  distr(seed, BLOCK_SIZE-1);
  int offset;
  for (offset=0; offset<tot_samples_found-BLOCK_SIZE; offset+=BLOCK_SIZE) {
    // sample feature rows to select only a subsample of the whole training set
    taken.resize(BLOCK_SIZE, false);

    for (int t=0; t<to_sample; t++)  {
      flag = true;
      while(flag) {
        num = distr(generator);
        if (!taken[num]) flag=false;
      }

      taken[num]=true;
      sampled_idxs[cont++] = num+offset;
      if (cont==num_entries_to_train_on) break;
    }

  }

  taken.resize(b, false);
  std::uniform_int_distribution<>  distr2(seed, b-1);

  for (int t=0; t<to_sample_b; t++)  {
    flag = true;
    while(flag) {
      num = distr2(generator);
      if (!taken[num]) flag=false;
    }

    taken[num]=true;
    sampled_idxs[cont++] = num+offset;
    if (cont==num_entries_to_train_on) break;
  }

  // int max_=0;
  // for (int i=0; i<num_entries_to_train_on; i++) {
  //   if (max_<sampled_idxs[i]) max_=sampled_idxs[i];
  // }
  // std::cout << "MAX IS " << max_ << std::endl;

}
*/

void Dataset4Train::sampleIdxs() {
  sampled_idxs = std::vector<int>(num_entries_to_train_on);
  int to_sample = num_entries_to_train_on;

  int num, cont=0;
  bool flag;

  std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};
  //std::default_random_engine rand_dev{static_cast<long unsigned int>(seed)};
  std::mt19937               generator(rand_dev());
  std::uniform_int_distribution<>  distr(seed, tot_samples_found-1);

  // std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};;
  // std::mt19937               generator(rand_dev());
  // std::uniform_int_distribution<>  distr(seed, tot_samples_found-1);
  taken.resize(tot_samples_found, false);

  for (int t=0; t<to_sample; t++)  {
    flag = true;
    while(flag) {
      num = distr(generator);
      if (!taken[num]) flag=false;
    }

    taken[num]=true;
    sampled_idxs[cont++] = num;
    if (cont==num_entries_to_train_on) break;
  }

}

std::string exec(const char* cmd) {
    std::array<char, 2048> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != NULL) {
        result += buffer.data();
    }

    // pclose(pipe.get());

    return result;
}

std::string getSHA256(std::string path) {
  std::string result = exec(("shasum -a 256 " + path).c_str());
  return result.substr(0, 64);
}

#include <sys/stat.h>
#include <unistd.h>
bool shouldIComputePCA(std::string pca_path, std::string pca_sha_path) {
  // auto pca_path = getPCAConfigNameFull(save_path);

  struct stat buffer;   
  bool file_exists;
  
  // check pca file exists
  file_exists = (stat ((pca_path + "_eigenvectors").c_str(), &buffer) == 0);
  if (!file_exists) return true;
  return false;

  // check pca sha256 exists
  file_exists = (stat (pca_sha_path.c_str(), &buffer) == 0);
  if (!file_exists) return true;


  auto sha256 = getSHA256(pca_path);

  std::ifstream t(pca_sha_path);
  std::string sha256_from_checkpoint((std::istreambuf_iterator<char>(t)),
                 std::istreambuf_iterator<char>());
  
  if (sha256 == sha256_from_checkpoint) return false;
  return true;

}

void Dataset4::load() {

  std::cout << "loading file \033[1;33m" << store_features_filename << "\033[0m" << std::flush;
  
  cont_trav=0; cont_nontrav=0;
  inherited_labels_size = (1+TOT_GEOM_FEATURES)*level;  // derived features size of the produced features file
  tot_samples_found=0;
  
  std::ifstream in(store_features_filename.c_str(), std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m") + store_features_filename + std::string(" not found.\n"));

  tot_samples_found = in.tellg() / (sizeof(float)*(TOT_GEOM_FEATURES + inherited_labels_size+1));
  std::cout << " found: " << tot_samples_found << " samples\n";

  y_train = cv::Mat(tot_samples_found, 1, CV_32FC1);

  readLabels();
  sampleIdxs();

  std::sort(sampled_idxs.begin(), sampled_idxs.end());

  X_train = cv::Mat(num_entries_to_train_on, (int) max_feats_num, CV_32F);
  y_train = cv::Mat(num_entries_to_train_on, 1, CV_32FC1);

  loadData();

  if (!nu_vec.empty() && mode>=3) {

    std::string PCAConfigNameFull = getPCAConfigNameFull(save_path);
    std::string PCAConfigSHANameFull = getPCAConfigSHANameFull(save_path);

    std::cout << "PCAConfigNameFull: " << PCAConfigNameFull << "\n";

    // if (/*true || */shouldIComputePCA(PCAConfigNameFull, PCAConfigSHANameFull) && false) {
    if (shouldIComputePCA(PCAConfigNameFull, PCAConfigSHANameFull) ) {
      std::cout << "I shall compute PCA on all)\n";
      createPCAOnAll();
    } 
    else {
      std::cout << "I shall NOT compute PCA on all: I'm loading it from file!\n";
      std::cout << pca_mode << "\n";
      if (mode>=3) {
        auto PCAConfigName = getPCAConfigName(load_path);

        std::cout << "  loading   PCA from " << PCAConfigName+"_eigenvectors" << std::flush;
        LoadMatBinary(PCAConfigName+"_eigenvectors", pca.eigenvectors);
        std::cout << "done!\n  loading   PCA from " << PCAConfigName+"_mean" << std::flush;
        LoadMatBinary(PCAConfigName+"_mean", pca.mean);
        std::cout << " done!\n";

        normalizer = Normalizer(tot_geom_features_across_all_levels);
        normalizer.loadConfig(getNormalizerConfigName(save_path));
      }
    }
  }
  
  // checkFileAndConfigAreValid((int) tot_samples_found, num_entries_to_train_on, cont_trav, cont_nontrav);


  if (normalizer.empty()) {
    //normalizer = Normalizer(tot_geom_features_across_all_levels);
    Feature feature;
    std::vector<float> vec;
    float label;
    float *features_matrix_data_row;
    int pos=0, row, c;
    int step = num_entries_to_train_on / 100;


    fullX = cv::Mat(tot_samples_found, (int) max_feats_num, CV_32F);
    fullY = cv::Mat(tot_samples_found, 1, CV_32FC1);


    ProgressBar pbar(70, "creating full PCA: ");

    std::ifstream feat_file(store_features_filename.c_str(), std::ios::in | std::ios::binary);
    for (row=0; row<tot_samples_found; row++) {
      feature.fromFileLine(feat_file, inherited_labels_size);
      feat_file.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
      vec  = feature.toVectorTransformed();
      features_matrix_data_row = fullX.ptr<float>(pos);
      for (c=0; c<max_feats_num; c++) features_matrix_data_row[c] = vec[re_idx[c]];
      
      fullY.at<float>(pos, 0) = static_cast<float>((float)label);
      if (pos%step==0) pbar.update(pos / (float)tot_samples_found);
      pos++;
    }
    feat_file.close();
    
    normalizer = Normalizer(tot_geom_features_across_all_levels);
    normalizer.normalize_train_and_store(fullX, getNormalizerConfigName(save_path));
    normalizer.normalize(X_train);

    // std::string cname = getNormalizerConfigName(save_path);

    // normalizer.normalize_train_and_store(X_train, cname);
    // normalizer.normalize_train_and_store(fullX, cname);

    // if (!normalizer.check4NormalizationErrors(fullX)) 
    //   throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Found Not normalized data. abort."));

    // if (!normalizer.check4NormalizationErrors(X_train)) 
    //   throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Found Not normalized data. abort."));
  }
  else  { 
    std::cout << "using built normalizer\n";
    normalizer.normalize(X_train);
  }

  fullX = cv::Mat(1, 1, CV_32FC1);
  fullY = cv::Mat(1, 1, CV_32FC1);

  

  if (mode>=3) {

    int der_feats_size=(re_idx.size()-tot_geom_features_across_all_levels);

    cv::Mat tmp(X_train.rows, pca_mode+der_feats_size, CV_32F);
    
    if (level>0 && der_feats_size>0) 
      X_train(cv::Range(0, X_train.rows), cv::Range(tot_geom_features_across_all_levels, X_train.cols))
      .copyTo(tmp(cv::Range(0, X_train.rows), cv::Range(pca_mode, tmp.cols)));


    // project to PCA space (only geometric features, thus the firsts n features)
    pca.project(X_train(cv::Range(0, X_train.rows),cv::Range(0, tot_geom_features_across_all_levels)), 
                    tmp(cv::Range(0, X_train.rows), cv::Range(0, pca_mode)));
    X_train = tmp;

    std::cout << "now X_train has " << X_train.cols << std::endl;

    // std::cout << "\eigenvectors\n";
    // for (int r=0; r< pca.eigenvectors.rows; r++) {
    //   for (int c=0; c<pca.eigenvectors.cols; c++) {
    //     std::cout << pca.eigenvectors.at<float>(r, c) << " ";
    //   }
    //   std::cout << "\n";
    // }
    // std::cout << "\nmean\n";
    // for (int r=0; r< pca.mean.rows; r++) {
    //   for (int c=0; c<pca.mean.cols; c++) {
    //     std::cout << pca.mean.at<float>(r, c) << " ";
    //   }
    //   std::cout << "\n";
    // }
  
  }

  

}

void Dataset4::createPCAOnAll() {
  Feature feature;
  std::vector<float> vec;
  float label;
  float *features_matrix_data_row;
  int pos=0, row, c;
  int step = num_entries_to_train_on / 100;


  fullX = cv::Mat(tot_samples_found, (int) max_feats_num, CV_32F);
  fullY = cv::Mat(tot_samples_found, 1, CV_32FC1);

  ProgressBar pbar(70, "creating full PCA: ");

  std::ifstream feat_file(store_features_filename.c_str(), std::ios::in | std::ios::binary);
  for (row=0; row<tot_samples_found; row++) {
    feature.fromFileLine(feat_file, inherited_labels_size);
    feat_file.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
    vec  = feature.toVectorTransformed();
    features_matrix_data_row = fullX.ptr<float>(pos);
    for (c=0; c<max_feats_num; c++) features_matrix_data_row[c] = vec[re_idx[c]];
    
    fullY.at<float>(pos, 0) = static_cast<float>((float)label);
    if (pos%step==0) pbar.update(pos / (float)tot_samples_found);
    pos++;
  }
  feat_file.close();
  
  normalizer = Normalizer(tot_geom_features_across_all_levels);
  normalizer.normalize_train_and_store(fullX, getNormalizerConfigName(save_path));

  std::cout << " creating pca ..." << std::endl;
  pca = cv::PCA (fullX(cv::Range(0, fullX.rows),cv::Range(0, tot_geom_features_across_all_levels)), 
                cv::Mat(), // mean
                cv::ml::ROW_SAMPLE,
                pca_mode);

  std::cout << " done PCA! \n";



  SaveMatBinary(getPCAConfigNameFull(save_path)+"_eigenvectors", pca.eigenvectors);
  SaveMatBinary(getPCAConfigNameFull(save_path)+"_eigenvalues", pca.eigenvalues);
  SaveMatBinary(getPCAConfigNameFull(save_path)+"_mean", pca.mean);

  std::ofstream outpcasha(getPCAConfigSHANameFull(save_path), std::ios::out);
  outpcasha << getSHA256(getPCAConfigNameFull(save_path));
  outpcasha.close();

}

void Dataset4::summary() {
  std::cout << "    " << store_features_filename +" using " + std::to_string(num_entries_to_train_on) + " samples."
             << " Balance? " << need_to_balance_flag
             << " at level: "  << level
             << " nu size: " << nu_vec.size()
             << " C_size: "  << C_vec.size() 
             << " train mode: " << modes[mode]
             << " pca mode: " << pca_mode
             << std::endl;
}

void Dataset4::checkFileAndConfigAreValid(int feats_size, int tot_entries, int cont_trav, int cont_nontrav) {
  if (tot_entries > feats_size) {
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. need to sample more entries (") 
        + std::to_string(tot_entries) + std::string(") than features available (")
        + std::to_string(feats_size)
        + std::string("). abort."));
  }

  if (need_to_balance_flag && cont_trav < tot_entries/2) {
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. need to sample more \033[1;33mTRAV\033[0m entries (") 
        + std::to_string(tot_entries/2) + std::string(") than trav features available (")
        + std::to_string(cont_trav)
        + std::string("). abort."));
  }

  if (need_to_balance_flag && cont_nontrav < tot_entries/2) {
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. need to sample more \033[1;33mNOT TRAV\033[0m entries (") 
        + std::to_string(tot_entries/2) + std::string(") than trav features available (")
        + std::to_string(cont_nontrav)
        + std::string("). abort."));
  }
}