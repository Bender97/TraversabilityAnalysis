#include "TrainDataset.h"

TrainDataset::TrainDataset(std::string filename_, YAML::Node &node, int level_, int tot_geom_features_, int tot_cells_) {
  
  if (node["nu_vec"])    nu_vec = node["nu_vec"].as<std::vector<float>>();
  if (node["C_vec"])     C_vec  = node["C_vec"].as<std::vector<float>>();
  if (node["gamma_vec"]) gamma_vec  = node["gamma_vec"].as<std::vector<float>>();
  
  init(filename_, node, level_, tot_geom_features_, tot_cells_);
  load();
}

TrainDataset::TrainDataset(std::string filename_, YAML::Node &node, Normalizer &normalizer_, cv::PCA &pca_, int level_, int tot_geom_features_, int tot_cells_)
 // :TrainDataset(node)
 {
  normalizer = normalizer_;
  pca = pca_;

  init(filename_, node, level_, tot_geom_features_, tot_cells_);
  load();
}

void TrainDataset::init(std::string filename_, YAML::Node &node, int level_, int tot_geom_features_, int tot_cells_) {
  filename = filename_;
  num_entries_to_train_on = node["feats_to_sample"].as<int>();
  need_to_balance_flag = node["balance_flag"].as<bool>();
  level = level_;

  modes = std::vector<std::string>({"geom", "geom_label", "geom_all", "geom_pca", "geom_pca_label", "geom_pca_all_label"});

  std::cout << "train level: " << level << "  " << std::flush;

  mode = node["mode"].as<int>();
  std::cout << "training in " << modes[mode] << " mode  " << std::flush;

  trick_mode=0;
  if (node["trick"])      trick_mode = node["trick"].as<int>();
  std::cout << "trick_mode " << (trick_mode ? " on  " : "off  ") << std::endl;
  
  if (node["pca"])       pca_mode = node["pca"].as<int>();
  else std::cout << "NO PCA MODE FOUND!" << std::endl;
  std::cout << "pca_mode " << pca_mode << std::endl;

  tot_cells = tot_cells_;
  tot_geom_features = tot_geom_features_;

  computeTrainingCols();

}

  /*
   * geom               #0 -> [:17]
   * geom_label         #0 -> /
   * geom_all           #0 -> /
   * geom_pca           #0 -> [:17] + pca
   * geom_pca_label     #0 -> /
   * geom_pca_all_label #0 -> /
   *
   * g  g  g  g  g  g  g  g  g  g  g  g  g  g  g  g  g  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  P  G
   * 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
   * geom               #1 -> [:17]
   * geom_label         #1 -> [:17] + [34]          // reorder
   * geom_all           #1 -> [:35]         
   * geom_pca           #1 -> [:17] + pca[:17]
   * geom_pca_label     #1 -> [:17] + [34] + pca[:17]   // reorder
   * geom_pca_all_label #1 -> [:35] + pca[:34]
   *
   * g  g  g  g  g  g  g  g  g  g  g  g  g  g  g  g  g  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  P  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  L  P  G  
   * 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53
   * geom               #2 -> [:17]
   * geom_label         #2 -> [:17] + [34] + [52]   // reorder!
   * geom_all           #2 -> [:34] + [35:52]  + [34] + [52]   // reorder!
   * geom_pca           #2 -> [:17] + pca[:17]
   * geom_pca_label     #2 -> [:17] + [34] + [52] + pca[:17] // reorder
   * geom_pca_all_label #2 -> [:34] + [35:52]  + [34] + [52] + pca[:51] // reorder!
   */
void TrainDataset::computeTrainingCols() {
  switch(mode) {
    case 0: // geom
    case 3: // geom_pca
      training_cols = tot_geom_features;
      tot_geom_features_across_all_levels = tot_geom_features;
      for (int i=0; i<17; i++) re_idx.push_back(i);
      break;
    case 1: // geom_label
    case 4: // geom_pca_label
      if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
      training_cols = tot_geom_features + level;
      tot_geom_features_across_all_levels = tot_geom_features;
      if (level==1) {
        for (int i=0; i<17; i++) re_idx.push_back(i);
        re_idx.push_back(34);
      }
      else if (level==2) {
        for (int i=0; i<17; i++) re_idx.push_back(i);
        re_idx.push_back(34);
        re_idx.push_back(52);
      }
      break;
    case 2: // geom_all
    case 5: // geom_pca_all_label
      if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
      training_cols = (tot_geom_features*(level+1)) + level;
      if (level==1) {
        tot_geom_features_across_all_levels = tot_geom_features*2;
        for (int i=0; i<35; i++) re_idx.push_back(i);
      }
      else if (level==2) {
        tot_geom_features_across_all_levels = tot_geom_features*3;
        for (int i=0; i<34; i++) re_idx.push_back(i);
        for (int i=35; i<52; i++) re_idx.push_back(i);
        re_idx.push_back(34);
        re_idx.push_back(52);
      }
      break;
    default:
      throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Mode not recognized!"));
  }

  if (pca_mode<0 && mode>=3)
    throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Mode with pca but no pca_mode set!"));

  if (pca_mode>tot_geom_features_across_all_levels)
    throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m fouond pca_mode > tot_geom_features_across_all_levels!"));

  if (level==1) ref_label_idx = 17;
  else if (level==2) ref_label_idx = 35;


  std::cout << "computed Training Cols (before pca) : " << training_cols << std::endl;
  std::cout << "to check re_idx size                : " << re_idx.size() << std::endl;
}

void TrainDataset::readLabels() {
  Feature feature;
  float label;
  int pos = 0;
  std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
  if (level>0 && trick_mode) {
    while(true) {
        if (!feature.ignoreFeatureFromFile(in, dfs)) break;
        in.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
        if (feature.derived_features[ref_label_idx] >= 0) continue; // skip all GT trav 
        y_train.at<float>(pos, 0) = static_cast<float>((float)label);
        if (label>0) cont_trav ++;
        else cont_nontrav++;
        pos++;
    }
  }
  else {
    while(true) {
        if (!feature.ignoreFeatureFromFile(in, dfs)) break;
        in.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
        y_train.at<float>(pos, 0) = static_cast<float>((float)label);
        if (label>0) cont_trav ++;
        else cont_nontrav++;
        pos++;
    }
  }

  std::cout << " read Labels: found: " << pos << " samples\n";
  // read data from file
  // std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
  //in.seekg(0, std::ios::beg);
  in.close();
}

void TrainDataset::loadData() {
  Feature feature;
  float label;
  std::vector<float> vec; 

  float *features_matrix_data_row;

  int pos=0;

  std::ifstream in2(filename.c_str(), std::ios::in | std::ios::binary);

  for (int row=0; row<tot_samples_found, pos<num_entries_to_train_on; row++) {
    if (row<sampled_idxs[pos]) {
      if (!feature.ignoreFeatureFromFile(in2, dfs)) break;
      in2.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));
      continue;
    }
    feature.fromFileLine(in2, dfs);
    in2.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));

    vec  = feature.toVectorTransformed();
    features_matrix_data_row = X_train.ptr<float>(pos);
    for (int c=0; c<training_cols; c++) features_matrix_data_row[c] = vec[re_idx[c]];
    
    y_train.at<float>(pos, 0) = static_cast<float>((float)label);

    pos++;
  }

  // if (level>0 && trick_mode) {
  //   while(true) {
  //     if (!feature.fromFileLine(in2, dfs)) break;
  //     in2.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));

  //     if (feature.derived_features[ref_label_idx] >= 0) continue; // skip all GT trav 
  //     vec  = feature.toVectorTransformed();
  //     features_matrix_data_row = X_train.ptr<float>(pos);
  //     for (int c=0; c<training_cols; c++) features_matrix_data_row[c] = vec[re_idx[c]];
  //     y_train.at<float>(pos, 0) = static_cast<float>((float)label);

  //     pos++;

  //   }
  // }
  // else {
  //   while(true) {
  //     if (!feature.fromFileLine(in2, dfs)) break;
  //     in2.read( reinterpret_cast< char*>( &(label) ), sizeof( float ));

  //     vec  = feature.toVectorTransformed();
  //     features_matrix_data_row = X_train.ptr<float>(pos);
  //     for (int c=0; c<training_cols; c++) {
  //       features_matrix_data_row[c] = vec[re_idx[c]];
  //     }
  //     y_train.at<float>(pos, 0) = static_cast<float>((float)label);

  //     pos++;

  //   }
  // }
  in2.close();
}

void TrainDataset::sampleIdxs() {
  // generate numbers
  std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};;
  std::mt19937               generator(rand_dev());
  std::uniform_int_distribution<>  distr(0, tot_samples_found-1);
  ///// sample feature rows to select only a subsample of the whole training set
  taken.resize(tot_samples_found, false);
  counters[0] = 0; counters[1] = 0;

  float half_netto = num_entries_to_train_on/2.0f;
  float *features_matrix_data_row;

  int num;
  for (int t=0; t<num_entries_to_train_on; t++)  {
    bool flag = true;
    while(flag) {
       num = distr(generator);
       //std::cout << "from " << 0 << " to " << features_all.size() << " sampled: " << num << " " << taken[num] << " " << counters[0] << " " << counters[1] << std::endl;
       if (taken[num]==false && (!need_to_balance_flag || counters[((int)y_train.at<float>(num)+1)>>1]<half_netto) ) flag=false;
    }

    taken[num]=true;
    counters[((int)y_train.at<float>(num)+1)>>1]++;
    sampled_idxs[t] = num;
  }
}

void TrainDataset::load() {

  std::cout << "loading file \033[1;33m" << filename << "\033[0m" << std::flush;
  
  cont_trav=0; cont_nontrav=0;
  dfs = (1+tot_geom_features)*level;  // derived features size of the produced features file
  tot_samples_found=0;
  
  std::ifstream in(filename.c_str(), std::ios::binary | std::ios::ate);
  tot_samples_found = in.tellg() / (sizeof(float)*(tot_geom_features + dfs+1));
  std::cout << " found: " << tot_samples_found << " samples\n";

  y_train = cv::Mat(tot_samples_found, 1, CV_32FC1);

  readLabels();
  sampled_idxs = std::vector<int>(num_entries_to_train_on);
  std::cout << " found: " << tot_samples_found << " entries, " << cont_trav << " T and " << cont_nontrav << " nT." << std::endl;


  sampleIdxs();

  std::sort(sampled_idxs.begin(), sampled_idxs.end());

  X_train = cv::Mat(num_entries_to_train_on, (int) training_cols, CV_32F);
  y_train = cv::Mat(num_entries_to_train_on, 1, CV_32FC1);



  loadData();


  std::cout << y_train.rows << " " << y_train.cols << std::endl;
  
  // simple checks
  if (num_entries_to_train_on<0) {
    std::cout << "all entries are selected for training!" << std::endl;
    return;
  }
  checkFileAndConfigAreValid((int) tot_samples_found, num_entries_to_train_on, cont_trav, cont_nontrav);


  if (normalizer.empty()) {
    normalizer = Normalizer(tot_geom_features_across_all_levels);

    std::string cname = "results/lv" + std::to_string(level) + "/"
               + modes[mode] + (trick_mode ? "_trick" : "") + "/config_data" + std::to_string(level) + ".yaml";

    normalizer.normalize_train_and_store(X_train, cname);

    std::cout << "norm config saved to " << cname << std::endl;

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
      pca = cv::PCA (X_train(cv::Range(0, X_train.rows),cv::Range(0, tot_geom_features_across_all_levels)),                 //Input Array Data
                cv::Mat(),                //Mean of input array, if you don't want to pass it   simply put Mat()
                cv::ml::ROW_SAMPLE,   //int flag
                pca_mode);

      std::string cname = "results/lv" + std::to_string(level) + "/"
              + modes[mode] + (trick_mode ? "_trick" : "") + "/pca_config_data" + std::to_string(pca_mode) + ".yaml";
      cv::FileStorage fs(cname, cv::FileStorage::WRITE);
      pca.write(fs);
      fs.release();
    }

    // project to PCA space (only geometric features, thus the firsts n features)
    pca.project(X_train(cv::Range(0, X_train.rows),cv::Range(0, tot_geom_features_across_all_levels)), 
                    tmp(cv::Range(0, X_train.rows), cv::Range(0, pca_mode)));
    X_train = tmp;

    std::cout << "now X_train has " << X_train.cols << std::endl;
  }
  


  // ///// sample feature rows to select only a subsample of the whole training set
  // taken.resize(tot_samples_found, false);
  // counters[0] = 0; counters[1] = 0;

  // cv::Mat X_small(num_entries_to_train_on, (int) X_train.cols, CV_32F);
  // cv::Mat y_small(num_entries_to_train_on, 1, CV_32FC1);
  // float *trow;

  // // generate numbers
  // std::default_random_engine rand_dev{static_cast<long unsigned int>(1)};;
  // std::mt19937               generator(rand_dev());
  // std::uniform_int_distribution<>  distr(0, tot_samples_found-1);

  // float half_netto = num_entries_to_train_on/2.0f;
  // float *features_matrix_data_row;

  // int num;
  // for (int t=0; t<num_entries_to_train_on; t++)  {
  //   bool flag = true;
  //   while(flag) {
  //      num = distr(generator);
  //      //std::cout << "from " << 0 << " to " << features_all.size() << " sampled: " << num << " " << taken[num] << " " << counters[0] << " " << counters[1] << std::endl;
  //      if (taken[num]==false && (!need_to_balance_flag || counters[((int)y_train.at<float>(num)+1)>>1]<half_netto) ) flag=false;
  //   }

  //   taken[num]=true;
  //   trow = X_train.ptr<float>(num);
  //   features_matrix_data_row = X_small.ptr<float>(t);
  //   for (int c=0; c<X_train.cols; c++) features_matrix_data_row[c] = trow[c];
  //   y_small.at<float>(t) = y_train.at<float>(num);
  //   counters[((int)y_train.at<float>(num)+1)>>1]++;
  // }


  // X_train = X_small;
  // y_train = y_small;

  std::cout << X_train.rows << " " << X_train.cols << std::endl;
  std::cout << y_train.rows << " " << y_train.cols << std::endl;

  std::cout << X_train.row(0) << std::endl;


}

void TrainDataset::summary() {
  std::stringstream ss("    ");
  std::cout << "    " << filename +" using " + std::to_string(num_entries_to_train_on) + " samples."
             << " Balance? " << need_to_balance_flag
             << " at level: "  << level
             << " nu size: " << nu_vec.size()
             << " C_size: "  << C_vec.size() 
             << " train mode: " << modes[mode]
             << " pca mode: " << pca_mode
             << std::endl;
  //return ss.str(  );
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