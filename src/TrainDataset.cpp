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

TrainDataset::TrainDataset(YAML::Node &node_cyl, int level_, bool train_flag)
    : Cylinder(node_cyl) 
{
  
  init(node_cyl, level_);
  computeFeaturesCols();
  load();
}

TrainDataset::TrainDataset(YAML::Node &node_cyl, int level_, Normalizer &normalizer_, cv::PCA &pca_, bool train_flag)
    : Cylinder(node_cyl)
{
  normalizer = normalizer_;
  pca = pca_;

  init(node_cyl, level_, false);
  computeFeaturesCols();
  load();
}

void TrainDataset::init(YAML::Node &node_cyl, int level_, bool train_flag) {
  store_features_filename = node_cyl["store_features_filename"].as<std::string>();

  YAML::Node node_mode;
  if (!train_flag) node_mode = node_cyl["valid"];
  else {
    node_mode = node_cyl["train"];
    nu_vec    = node_mode["nu_vec"].as<std::vector<float>>();
    C_vec     = node_mode["C_vec"].as<std::vector<float>>();
    gamma_vec = node_mode["gamma_vec"].as<std::vector<float>>();  
  }

  num_entries_to_train_on = node_mode["feats_to_sample"].as<int>();
  need_to_balance_flag = node_mode["balance_flag"].as<bool>();
  level = level_;


  save_path = sanitize(node_cyl["save_path"].as<std::string>());
  load_path = sanitize(node_cyl["load_path"].as<std::string>());

  seed = node_cyl["seed"].as<uint16_t>();

  store_features_filename = load_path + (!train_flag ? "val_" : "") + store_features_filename + std::to_string(level_) + ".bin";

  // if (train_flag) {
  //   store_features_filename = load_path + "features_data_notgt" + std::to_string(level_) + ".bin";
  // }
  // else {
  //   store_features_filename = load_path + "val_features_data" + std::to_string(level_) + ".bin";
  // }

  std::cout << "train level: " << level << "  " << std::flush;
  std::cout << "fname: " << store_features_filename << "  " << std::flush;

  mode = node_cyl["mode"].as<int>();
  std::cout << "training in " << modes[mode] << " mode  " << std::flush;

  trick_mode=0;
  if (node_cyl["trick"])      trick_mode = node_cyl["trick"].as<int>();
  std::cout << "trick_mode " << (trick_mode ? " on  " : "off  ") << std::endl;
  
  if (node_cyl["pca"])       pca_mode = node_cyl["pca"].as<int>();
  else std::cout << "NO PCA MODE FOUND!" << std::endl;
  std::cout << "pca_mode " << pca_mode << std::endl;
}

void TrainDataset::readLabels() {
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

  std::cout << " read Labels: found: " << pos << " samples:  " 
            << cont_trav << " T and " << cont_nontrav << " nT." << std::endl;
  in.close();
}

void TrainDataset::loadData() {
  Feature feature;
  std::vector<float> vec;
  float label;
  float *features_matrix_data_row;
  int pos=0, row, c;

  ProgressBar pbar(70, "loading features: ");

  int step = num_entries_to_train_on / 100;

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

void TrainDataset::sampleIdxs() {

  if (tot_samples_found == 402088) {
    std::cout << "NO SAMPLE. GOING WITH 402088\n";
    for (int i=0; i<402088; i++) sampled_idxs[i] = i;
    return;
  }
  if (tot_samples_found == 1348715) {
    std::cout << "NO SAMPLE. GOING WITH 1348715 (sampled_idxs size is " << sampled_idxs.size() << ")\n";
    for (int i=0; i<1348715; i++) sampled_idxs[i] = i;
    return;
  }
  std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};;
  std::mt19937               generator(rand_dev());
  // generate numbers
  if (seed==-1) {
    seed = (uint16_t)time(0);
  }
  // if (tot_samples_found>50063723) seed = 42414;
  // if (tot_samples_found>100000) seed = 61569;
  std::cout << " SEEEEEEEEED: " << seed << std::endl;
  std::uniform_int_distribution<>  distr(seed, tot_samples_found-1);
  // std::uniform_int_distribution<>  distr(0, tot_samples_found-1);
  ///// sample feature rows to select only a subsample of the whole training set
  taken.resize(tot_samples_found, false);
  counters[0] = 0; counters[1] = 0;

  float half_netto = num_entries_to_train_on/2.0f;

  int num;
  for (int t=0; t<num_entries_to_train_on; t++)  {
    bool flag = true;
    while(flag) {
       num = distr(generator);
       if (taken[num]==false && (!need_to_balance_flag || counters[((int)y_train.at<float>(num)+1)>>1]<half_netto) ) flag=false;
    }

    taken[num]=true;
    counters[((int)y_train.at<float>(num)+1)>>1]++;
    sampled_idxs[t] = num;
  }
}

void TrainDataset::sampleIdxs_ContextBased() {

  
  std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};;
  std::mt19937               generator(rand_dev());

  int BLOCK_SIZE = 500000;

  int to_sample = std::floor((float)BLOCK_SIZE / tot_samples_found * num_entries_to_train_on);

  int cont = 0;
    int num;

  // generate numbers
  std::cout << " SEEEEEEEEED (ContextBased): " << seed << std::endl;
  std::uniform_int_distribution<>  distr(seed, BLOCK_SIZE-1);
  for (int offset=0; offset<tot_samples_found; offset+=BLOCK_SIZE) {

    ///// sample feature rows to select only a subsample of the whole training set
    taken.resize(BLOCK_SIZE, false);

    for (int t=0; t<to_sample; t++)  {
      bool flag = true;
      while(flag) {
        num = distr(generator);
        if (!taken[num]) flag=false;
      }

      taken[num]=true;
      sampled_idxs[cont++] = num+offset;
      if (cont==num_entries_to_train_on) break;
    }
  }
}

void TrainDataset::load() {

  std::cout << "loading file \033[1;33m" << store_features_filename << "\033[0m" << std::flush;
  
  cont_trav=0; cont_nontrav=0;
  inherited_labels_size = (1+TOT_GEOM_FEATURES)*level;  // derived features size of the produced features file
  tot_samples_found=0;
  
  std::ifstream in(store_features_filename.c_str(), std::ios::binary | std::ios::ate);
  tot_samples_found = in.tellg() / (sizeof(float)*(TOT_GEOM_FEATURES + inherited_labels_size+1));
  std::cout << " found: " << tot_samples_found << " samples\n";


  y_train = cv::Mat(tot_samples_found, 1, CV_32FC1);

  readLabels();
  sampled_idxs = std::vector<int>(num_entries_to_train_on);
  
  // if (num_entries_to_train_on>30000)
  //   sampleIdxs();
  // else {
  //   std::fstream myfile("results3/pcas/idxs.txt", std::ios_base::in);
  //   int a=0;
  //   while (myfile >> sampled_idxs[a])
  //       //printf("%f ", a);
  //       a++;
  //   //for (int i=0; i<10; i++) std::cout << i << " " << sampled_idxs[i] << std::endl;
  //   //throw std::runtime_error(std::string("just finished."));
  // }
  if (num_entries_to_train_on<100000) sampleIdxs_ContextBased();
  else sampleIdxs();


  std::sort(sampled_idxs.begin(), sampled_idxs.end());

  X_train = cv::Mat(num_entries_to_train_on, (int) max_feats_num, CV_32F);
  y_train = cv::Mat(num_entries_to_train_on, 1, CV_32FC1);

  loadData();
  
  // simple checks
  if (num_entries_to_train_on<0) {
    std::cout << "all entries are selected for training!" << std::endl;
    return;
  }
  checkFileAndConfigAreValid((int) tot_samples_found, num_entries_to_train_on, cont_trav, cont_nontrav);


  if (normalizer.empty()) {
    normalizer = Normalizer(tot_geom_features_across_all_levels);

    std::string cname = getNormalizerConfigName(save_path);

    normalizer.normalize_train_and_store(X_train, cname);

    if (!normalizer.check4NormalizationErrors(X_train)) {
      throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Found Not normalized data. abort."));
    }
  }
  else  normalizer.normalize(X_train);
  // normalizer.print();

  if (mode>=3) {

    int der_feats_size=(re_idx.size()-tot_geom_features_across_all_levels);

    cv::Mat tmp(X_train.rows, pca_mode+der_feats_size, CV_32F);
    std::cout << tot_geom_features_across_all_levels << " " << X_train.cols << " "
              << pca_mode << " " << tmp.cols << std::endl;
    
    if (level>0 && der_feats_size>0) 
      X_train(cv::Range(0, X_train.rows), cv::Range(tot_geom_features_across_all_levels, X_train.cols))
      .copyTo(tmp(cv::Range(0, X_train.rows), cv::Range(pca_mode, tmp.cols)));

    if (!nu_vec.empty()) {
      pca = cv::PCA (X_train(cv::Range(0, X_train.rows),cv::Range(0, tot_geom_features_across_all_levels)), 
                cv::Mat(),
                cv::ml::ROW_SAMPLE,
                pca_mode);

      cv::FileStorage fs(getPCAConfigName(save_path), cv::FileStorage::WRITE);
      pca.write(fs);
      fs.release();
    }

    // project to PCA space (only geometric features, thus the firsts n features)
    pca.project(X_train(cv::Range(0, X_train.rows),cv::Range(0, tot_geom_features_across_all_levels)), 
                    tmp(cv::Range(0, X_train.rows), cv::Range(0, pca_mode)));
    X_train = tmp;

    std::cout << "now X_train has " << X_train.cols << std::endl;
  }

  // std::cout << X_train.rows << " " << X_train.cols << std::endl;
  // std::cout << y_train.rows << " " << y_train.cols << std::endl;

  // std::cout << X_train.row(0) << std::endl;


}

void TrainDataset::summary() {
  std::cout << "    " << store_features_filename +" using " + std::to_string(num_entries_to_train_on) + " samples."
             << " Balance? " << need_to_balance_flag
             << " at level: "  << level
             << " nu size: " << nu_vec.size()
             << " C_size: "  << C_vec.size() 
             << " train mode: " << modes[mode]
             << " pca mode: " << pca_mode
             << std::endl;
}

void TrainDataset::checkFileAndConfigAreValid(int feats_size, int tot_entries, int cont_trav, int cont_nontrav) {
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