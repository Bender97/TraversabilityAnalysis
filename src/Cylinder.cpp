#include "Cylinder.h"



Cylinder::Cylinder(YAML::Node &node) {
  start_radius = node["min_radius"].as<float>();
  end_radius   = node["max_radius"].as<float>();
  steps_num    = node["steps_num"].as<int>();
  yaw_steps    = node["yaw_steps"].as<int>();
  z            = node["z_level"].as<float>();

  yaw_steps_half = yaw_steps / 2;

  radius_step = ((end_radius - start_radius) * 100000 + 5) / steps_num / 100000.0f; // just a truncation
  yaw_res     = M_DOUBLE_PI / yaw_steps;

  tot_cells = steps_num*yaw_steps;

  svm_nu = node["nu"].as<float>();
  svm_gamma = node["gamma"].as<float>();

  grid       = std::vector<Cell>(tot_cells);
  features   = std::vector<Feature>(tot_cells);
  area       = std::vector<float>(tot_cells);
  remap_idxs = std::vector<int>(tot_cells, -1);

  GT_labels_vector = cv::Mat(tot_cells, 1, CV_32FC1);
  predictions_vector = cv::Mat(tot_cells, 1, CV_32F);
  
  tmetric.resetAll();
  gmetric.resetAll();

  // pre-compute area of each cell: it will be used in computeFeatures
  for (int row_idx = 0, idx=0; row_idx<steps_num; row_idx++) {
    for (int yaw_idx = 0; yaw_idx<yaw_steps; yaw_idx++, idx++) {
      float r = row_idx*radius_step, R = r+ radius_step;
      area[idx] = M_PI*(R*R-r*r) / yaw_steps;
    }
  }
  
}

Cylinder::Cylinder(YAML::Node &node,  Cylinder *cyl_, ExpMode expmode_) : Cylinder(node)
{

  level = 0; mode = -1; trick_mode=0; pca_mode=-1;
  if (cyl_!=nullptr) level = cyl_->level + 1;

  expmode = expmode_;

  if (node["mode"]) mode = node["mode"].as<int>();

  if (expmode == ExpMode::DL) {
    std::cout << "Deep Learning test mode! No need for other configurations" << std::endl;
    return;
  }
  

  if (!node["dataset"])
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. please set dataset mode.\n") );

  if (node["trick"]) trick_mode = node["trick"].as<int>();
  
  if (node["load_path"]) load_path = sanitize(node["load_path"].as<std::string>());
  if (node["save_path"]) save_path = sanitize(node["save_path"].as<std::string>());
  
  loadPCAConfigs(node);
  computeFeaturesCols();
  loadSVM(node);

  full_featMatrix = cv::Mat(tot_cells, max_feats_num, CV_32F);
  
  if (mode<3 || expmode==ExpMode::produce)
    featMatrix = cv::Mat(tot_cells, max_feats_num, CV_32F);
  else
    featMatrix = cv::Mat(tot_cells, pca_mode+level, CV_32F);

  
  for (auto &f : features)
    f.derived_features.resize((1+TOT_GEOM_FEATURES)*level);
  
  if (expmode == ExpMode::test) {
  // if (level<2) {
    std::string cname = getNormalizerConfigName(load_path);
    std::cout << "  loading  norm from " << cname << std::flush;
    normalizer = Normalizer(tot_geom_features_across_all_levels, cname.c_str());
    std::cout << " done!" << std::endl;
  }
  else {
    std::string ofname;
    if (node["store_features_filename"]) ofname = node["store_features_filename"].as<std::string>();
    else throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Please provide a valid outfile name. Found empty!\n"));

    store_features_filename = sanitize(save_path) + ofname + std::to_string(level) + ".bin";
    
    if (store_features_filename.empty()) 
      throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. Please provide a valid outfile path. Found empty!\n"));

    if (check_file_exists(store_features_filename)) 
      throw std::runtime_error(std::string("\033[1;93mWARNING!\033[0m FILE " + store_features_filename + " EXISTS! Please manually remove it.\n"));

    std::ofstream out(store_features_filename.c_str(), std::ios::out | std::ios::binary);
    out.close();

  } 
  // pre-compute the index from which to inherit features
  if (cyl_!=nullptr) {
    float angle, sampled_angle, sampled_radius;
    int row, col, sampled_idx, exdx;

    inherit_idxs.resize(grid.size());
    prevfeats_num = cyl_->features[0].derived_features.size();

    for (int row_idx = 0, i=0; row_idx<steps_num; row_idx++) {
      for (int yaw_idx = 0; yaw_idx<yaw_steps; yaw_idx++, i++) {
        
        angle          = yaw_idx*yaw_res;
        sampled_angle  = angle  + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(yaw_res)));
        sampled_radius = (start_radius + row_idx*radius_step) + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(radius_step)));

        if (sampled_angle>=M_DOUBLE_PI) sampled_angle = sampled_angle - M_DOUBLE_PI;
        else if (sampled_angle<0.0f)    sampled_angle = M_DOUBLE_PI + sampled_angle;
        
        row = (int) std::floor((sampled_radius-cyl_->start_radius)/cyl_->radius_step);
        col = ((int) std::floor((sampled_angle)/cyl_->yaw_res)+cyl_->yaw_steps/2)%cyl_->yaw_steps;

        sampled_idx = col*cyl_->steps_num + row;
        exdx = ((yaw_idx+yaw_steps/2)%yaw_steps)*steps_num + row_idx;

        inherit_idxs[exdx] = sampled_idx; //save the idx of the correspondent backsector

      } 
    }
  }
}

void Cylinder::printSummary() {
  std::cout << "level: " << std::setw(2) << level << " | "
            <<  "mode: " << std::setw(18) << modes[mode] << " | "
            <<   "pca: " << std::setw(4) << (pca_mode>0 ? std::to_string(pca_mode) : "OFF") << " | "
            << "trick: " << std::setw(4) << (trick_mode ? "ON" : "OFF") << std::endl;
}

Cylinder_SemKITTI::Cylinder_SemKITTI(YAML::Node &node)
   : Cylinder(node) {}
Cylinder_SemKITTI::Cylinder_SemKITTI(YAML::Node &node,  Cylinder *cyl_, ExpMode expmode)
   : Cylinder(node, cyl_, expmode) {}

Cylinder_NuSc::Cylinder_NuSc(YAML::Node &node)
   : Cylinder(node) {}
Cylinder_NuSc::Cylinder_NuSc(YAML::Node &node,  Cylinder *cyl_, ExpMode expmode)
   : Cylinder(node, cyl_, expmode) {}


void Cylinder::inheritFeatures(Cylinder *cyl_) {
  if (max_feats_num == TOT_GEOM_FEATURES) return;
  
  std::vector<float> prev_feats; //((TOT_GEOM_FEATURES*(level)) + level - 1);
  int i, j, c;
  
  for (i=0; i<tot_cells; i++) {
    if (grid[i].status==UNPREDICTABLE) continue;
    // if (grid[i].label==UNKNOWN_CELL_LABEL) continue;

    for (j=0; j<prevfeats_num; j++)
      features[i].derived_features[j] = cyl_->features[inherit_idxs[i]].derived_features[j];

    prev_feats = cyl_->features[inherit_idxs[i]].toVectorTransformed();
    for (j=prevfeats_num, c=0; j<prevfeats_num+TOT_GEOM_FEATURES; j++, c++)
      features[i].derived_features[j] = prev_feats[c];
    features[i].derived_features[j] = cyl_->grid[inherit_idxs[i]].predicted_label;
  }

}

void Cylinder::inheritGTFeatures(Cylinder *cyl_) {
  if (max_feats_num == TOT_GEOM_FEATURES) return;

  std::vector<float> prev_feats; //(TOT_GEOM_FEATURES + prevfeats_num);
  int i, j, c;

  for (i=0; i<tot_cells; i++) {
    if (grid[i].status==UNPREDICTABLE) continue;
    
    for (j=0; j<prevfeats_num; j++)
      features[i].derived_features[j] = cyl_->features[inherit_idxs[i]].derived_features[j];

    prev_feats = cyl_->features[inherit_idxs[i]].toVectorTransformed();
    for (j=prevfeats_num, c=0; j<prevfeats_num+TOT_GEOM_FEATURES; j++, c++)
      features[i].derived_features[j] = prev_feats[c];
    int lab = cyl_->grid[inherit_idxs[i]].label;
    if (lab>1.0f) {

      std::cout << " prev: " << cyl_->grid[inherit_idxs[i]].points_idx.size() << " points at idx " << inherit_idxs[i] << "\n";
      std::cout << " curr: " << grid[i].points_idx.size() << " points\n";

      

      throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m UNKNOWN INHERITED FEATURE! ") + std::to_string(level));
    }
    features[i].derived_features[j] = cyl_->grid[inherit_idxs[i]].label;
  }
  
}

void Cylinder::sortBins_cyl(std::vector<Eigen::Vector3d> &points) {
  int row, col, ps;
  float radius, yaw;
  Eigen::Vector3d *p;
  
  ps = (int) points.size();
  for (int i=0; i<ps; ++i) {
    p = &(points[i]);
    radius = p->norm();
    if (radius >= end_radius || radius<=start_radius) continue;

    // assumption: no x=y=0 since it's the laser origin
    yaw = std::atan2((*p)(0), (*p)(1)) + M_PI;

    col = int_floor(yaw / yaw_res);
    row = int_floor((radius-start_radius) / radius_step);
    col = (col + yaw_steps_half) % yaw_steps;

    grid[col*steps_num + row].points_idx.push_back(i);
  }
}

void Cylinder::resetGrid() {
  for (auto &cell: grid) {
    cell.status = UNPREDICTABLE;
    cell.label = UNKNOWN_CELL_LABEL;
    cell.predicted_label = UNKNOWN_CELL_LABEL;
    cell.points_idx.clear();
  }
}

void Cylinder::computeTravGT(std::vector<int> &labels) {}

void Cylinder_SemKITTI::computeTravGT(std::vector<int> &labels) {
  int trav_cont, non_trav_cont, road, sidewalk, label;
  Cell *cell;
  for (int r=0, valid_rows=0; r<tot_cells; r++) {
    cell = &(grid[r]);
    if (cell->points_idx.size() < MIN_NUM_POINTS_IN_CELL) {
      cell->label = UNKNOWN_CELL_LABEL;
      cell->status = UNPREDICTABLE;
      continue;
    }

    trav_cont = non_trav_cont = road = sidewalk = 0;

    for (auto &p_idx : cell->points_idx) {
      label = labels[p_idx];
      if (POINTBELONGSTOROAD(label)) trav_cont ++;
      else if (LABELED(label)) non_trav_cont ++;

      if (label==40) road++;
      else if (label==48) sidewalk++;
    }

    if (road>0 && sidewalk>0) cell->label = NOT_TRAV_CELL_LABEL;
    else {
        if (non_trav_cont > 3 ) cell->label = NOT_TRAV_CELL_LABEL;
        else if (trav_cont > 1) cell->label = TRAV_CELL_LABEL;
        else {
          cell->label = NOT_TRAV_CELL_LABEL;
        }
    }

    GT_labels_vector.at<float>(valid_rows, 0) = cell->label;
    valid_rows++;
  }


}

void Cylinder_NuSc::computeTravGT(std::vector<int> &labels) {
  int trav_cont, non_trav_cont;
  int road, sidewalk;
  Cell *cell;

  for (int r=0, valid_rows=0; r<tot_cells; r++) {
    cell = &(grid[r]);
    if (cell->points_idx.size() < MIN_NUM_POINTS_IN_CELL) {
      continue;
    }

    trav_cont = 0;
    non_trav_cont = 0;
    road=0;
    sidewalk=0;

    int label;

    for (auto &p_idx : cell->points_idx) {
      label = labels[p_idx];
      if (POINTBELONGSTOROAD_NU(label)) trav_cont ++;
      else if (LABELED_NU(label)) non_trav_cont ++;

      if (IS_DRIVEABLE_SURFACE_NU(label)) road++;
      else if (IS_SIDEWALK_NU(label)) sidewalk++;
    }

    if (road>0 && sidewalk>0) cell->label=NOT_TRAV_CELL_LABEL;
    else {
        if (non_trav_cont > 3 ) cell->label = NOT_TRAV_CELL_LABEL;
        else if (trav_cont > 1) cell->label = TRAV_CELL_LABEL;
        // else cell->label = UNKNOWN_CELL_LABEL;
    }

    GT_labels_vector.at<float>(valid_rows, 0) = cell->label;
    valid_rows++;

  }
}

void Cylinder::computePredictedLabel(std::vector<int> &labels) {
  int trav_cont, non_trav_cont, road, sidewalk;
  int unk=0;
  int label;

  for (auto &cell : grid) {
    if (cell.points_idx.size() < MIN_NUM_POINTS_IN_CELL)
      continue;
    
    cell.status = PREDICTABLE;

    trav_cont = non_trav_cont = road = sidewalk=0;

    for (auto &p_idx : cell.points_idx) {
      label = labels[p_idx];
      if (label==1 || label==3) trav_cont ++;
      else if (label==-1) non_trav_cont ++;

      if (label==1) {road++;}
      else if (label==3) {sidewalk++;}
      else if (label==0) {unk++;}
      else if (label>0) throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. provided a not valid label: ") 
        + std::to_string(label) + std::string("\n"));
    }

    if (road>0 && sidewalk>0) cell.predicted_label=NOT_TRAV_CELL_LABEL;
    else {
        if (non_trav_cont > 3 )
          cell.predicted_label = NOT_TRAV_CELL_LABEL;
        else if (trav_cont > 1)
          cell.predicted_label = TRAV_CELL_LABEL;
        // else cell.predicted_label = UNKNOWN_CELL_LABEL;
    }
  }

}

/*
 * 17 17 1 17 1 17 1 17 1 17 1
 *       34   52
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

void Cylinder::computeFeaturesCols() {
  int i, l;
  switch(mode) {
    case 0: // geom
    case 3: // geom_pca
      max_feats_num = TOT_GEOM_FEATURES;
      tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
      for (i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
      break;
    case 1: // geom_label
    case 4: // geom_pca_label
      if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
      max_feats_num = TOT_GEOM_FEATURES + level;
      tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
      for (i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
      for (l=1; l<=level; l++) re_idx.push_back(TOT_GEOM_FEATURES + (TOT_GEOM_FEATURES+1)*l -1);
      
      break;
    case 2: // geom_all
    case 5: // geom_pca_all_label
      if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
      max_feats_num = (TOT_GEOM_FEATURES*(level+1)) + level;
      for (i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);

      for (l=1; l<=level; l++) {
        int end_geom_feats = TOT_GEOM_FEATURES + (TOT_GEOM_FEATURES+1)*(l-1);
        for (i=end_geom_feats; i<end_geom_feats+TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
      }
      for (l=1; l<=level; l++) re_idx.push_back(TOT_GEOM_FEATURES + (TOT_GEOM_FEATURES+1)*l -1);

      break;
    default:
      throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Mode not recognized!"));
  }

  if (expmode == ExpMode::test && mode>=3 && pca_mode<0)
    throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Mode with pca but no pca_mode set!"));

  if (pca_mode>TOT_GEOM_FEATURES)
    throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m fouond pca_mode > tot_geom_features_across_all_levels!"));

  inherited_labels_size = re_idx.size()-tot_geom_features_across_all_levels;

}

void Cylinder::loadPCAConfigs(YAML::Node &node)
{
  if (expmode != ExpMode::test) return;
  if (node["pca"]) pca_mode = node["pca"].as<int>();

  if (mode>=3) {
    std::string PCAConfigName = getPCAConfigName(load_path);
    
    std::cout << "  loading   PCA from " << PCAConfigName << std::flush;
    
    cv::FileStorage fs(PCAConfigName, cv::FileStorage::READ);
    fs["mean"]    >> pca.mean;
    fs["vectors"] >> pca.eigenvectors;
    fs["values"]  >> pca.eigenvalues;
    fs.release();
    
    std::cout << " done!" << std::endl;
  }
}

void Cylinder::loadSVM(YAML::Node &node) {
  
  if (expmode != ExpMode::test) return;

  std::string path = getSVMName(load_path);
  
  std::cout << "  loading model from " << path << " ... " << std::flush;

  if (!check_file_exists(path)) 
    throw std::runtime_error("\033[1;31mERROR\033[0m. level "
        + std::to_string(level) + " has invalid SVM path: " + path + ".\n");

  model = cv::ml::SVM::load(path);

  if (!model->isTrained())
    throw std::runtime_error("\033[1;31mERROR\033[0m.  level "
        + std::to_string(level) + " has invalid SVM model: it's not trained!");

  std::cout << " done!" << std::endl;
}


void Cylinder::process(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points) {


  std::vector<float> feat;//((TOT_GEOM_FEATURES*(level+1)) + level);
  float *features_matrix_data_row;
  float plab, pred;
  int r, c, valid_rows=0;

  // fill cv::Mat full_featMatrix
  for (r=0; r<tot_cells; r++) {
    if (grid[r].status == UNPREDICTABLE)
    // if (grid[r].label == UNKNOWN_CELL_LABEL)
      continue;

    feat = features[r].toVectorTransformed();

    features_matrix_data_row = full_featMatrix.ptr<float>(valid_rows);
    for (c=0; c<max_feats_num; c++) features_matrix_data_row[c] = feat[re_idx[c]];
      
    remap_idxs[valid_rows++] = r;
  }

  normalizer.normalize(full_featMatrix);

  if (mode>=3) {

    featMatrix = cv::Mat(valid_rows, pca_mode+inherited_labels_size, CV_32F);
    
    if (level>0 && inherited_labels_size>0) 
      full_featMatrix(cv::Range(0, valid_rows), cv::Range(tot_geom_features_across_all_levels, full_featMatrix.cols))
      .copyTo(featMatrix(cv::Range(0, valid_rows), cv::Range(pca_mode, featMatrix.cols)));

    pca.project(full_featMatrix(cv::Range(0, valid_rows), cv::Range(0, tot_geom_features_across_all_levels)), 
                    featMatrix(cv::Range(0, valid_rows), cv::Range(0, pca_mode)));
  }
  else
    featMatrix = full_featMatrix(cv::Range(0, valid_rows), cv::Range(0, full_featMatrix.cols));
  
  tmetric.resetTime();
  model->predict(featMatrix, predictions_vector(cv::Range(0, valid_rows), cv::Range(0, 1)), cv::ml::StatModel::RAW_OUTPUT);
  tmetric.checkpointTime();

  for (r=0; r < valid_rows; r++) {
    pred = predictions_vector.at<float>(r, 0);
    plab = pred > 0 ? TRAV_CELL_LABEL : NOT_TRAV_CELL_LABEL;
    
    // trick!
    // if (level>0 && featMatrix.at<float>(r, pca_mode+level-1) > 0.5) pred = TRAV_CELL_LABEL;

    grid[remap_idxs[r]].predicted_label = plab; // if trick is wanted to be float, put pred

  }

}

void Cylinder::computeAccuracy() {
    tmetric.resetAcc();
    for (int c=0; c<tot_cells; c++) {
      // if (grid[c].label == UNKNOWN_CELL_LABEL) continue;
      if (grid[c].status == UNPREDICTABLE) continue;
      tmetric.update(grid[c].predicted_label, grid[c].label);
    }
    gmetric += tmetric;
}

void Cylinder::computeFeatures(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points) {
  bool status;
  for (int r=0; r<(int)features.size(); r++) {
    if (grid[r].points_idx.size()<MIN_NUM_POINTS_IN_CELL) continue;
    status = features[r].computeFeatures(&grid[r], scene_normal, points, area[r]);
    if (!status) {
      grid[r].label=UNKNOWN_CELL_LABEL; // even though it has more than 2 points, features are uncomputable.
      grid[r].status=UNPREDICTABLE;
      std::cout << "UNPREDICTABLE FEATURE!\n";
    }
    else
      grid[r].status=PREDICTABLE;

  }
}

void Cylinder::storeFeaturesToFile() {
  std::ofstream out(store_features_filename.c_str(), std::ios::binary | std::ios::app);
  if (!out) 
      throw std::runtime_error("\033[1;31mERROR\033[0m.  OUTSTREAM opening  "
        + store_features_filename);
  
  for (int i=0; i<(int) grid.size(); i++) {
    if (grid[i].label == UNKNOWN_CELL_LABEL) continue;
    // if (level==2) {
    //   std::cout << features[i].toString() << std::endl;
    //   throw std::runtime_error(std::string("just finished"));
    // }
    features[i].toFile(out);
    float lab = static_cast<float> (grid[i].label);
    out.write( reinterpret_cast<const char*>( &(lab) ), sizeof( float ));
  }
  out.close();
}

void Cylinder::produceFeaturesRoutine(DataLoader &dl, Cylinder *back_cyl) {
  if (back_cyl==nullptr && level>0) {
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. provided a cylinder of level ") 
        + std::to_string(level) + std::string(" with nullptr back cylinder.\n"));
  }

  resetGrid();
  sortBins_cyl(dl.points);
  computeTravGT(dl.labels);
  computeFeatures(dl.scene_normal, dl.points);
  // if (level>0) inheritGTFeatures(back_cyl);
  if (level>0) inheritFeatures(back_cyl);
  if (level<2) process(dl.scene_normal, dl.points);
  storeFeaturesToFile();
}

void Cylinder::OnlineRoutine(
                    DataLoader &dl, Cylinder *back_cyl) {
  resetGrid();
  sortBins_cyl(dl.points);
  computeTravGT(dl.labels);
  computeFeatures(dl.scene_normal, dl.points);
  if (level>0) inheritFeatures(back_cyl);
  process(dl.scene_normal, dl.points);
  computeAccuracy();
}

void Cylinder::OnlineRoutine_Profile(
                    DataLoader &dl, Cylinder *back_cyl) {
  cv_ext::BasicTimer bt;                      
  resetGrid();

  bt.reset();
  sortBins_cyl(dl.points);
  auto e = bt.elapsedTimeMs();
  std::cout << "lv " << level << " - sortbins: " << e << " ms" << std::endl;

  bt.reset();
  computeTravGT(dl.labels);
  e = bt.elapsedTimeMs();
  std::cout << "lv " << level << " - computeTravGT: " << e << " ms" << std::endl;

  bt.reset();
  computeFeatures(dl.scene_normal, dl.points);
  e = bt.elapsedTimeMs();
  std::cout << "lv " << level << " - computeFeatures: " << e << " ms" << std::endl;
  
  if (level>0) {
    bt.reset();
    inheritFeatures(back_cyl);
    e = bt.elapsedTimeMs();
    std::cout << "lv " << level << " - inheritFeatures: " << e << " ms" << std::endl;
  }

  bt.reset();
  process(dl.scene_normal, dl.points);
  e = bt.elapsedTimeMs();
  std::cout << "lv " << level << " - process: " << e << " ms" << std::endl;
  
  bt.reset();
  computeAccuracy();
  e = bt.elapsedTimeMs();
  std::cout << "lv " << level << " - computeAccuracy: " << e << " ms" << std::endl;

  std::cout << std::endl;
}

std::string Cylinder::getSVMName(std::string prefix) {
  return prefix + "lv" + std::to_string(level) + "/"
    + modes[mode] + (trick_mode ? "_trick" : "") + "/svm_model"
    + (mode>=3 ? "_" + std::to_string(pca_mode) : "") + "_" 
    + cleanFloatStr(svm_nu) + "_" 
    + cleanFloatStr(svm_gamma) + ".bin";
}
std::string Cylinder::getPCAConfigName(std::string prefix) {
  return prefix + "lv" + std::to_string(level) + "/"
    + modes[mode] + (trick_mode ? "_trick" : "") + "/pca_config"
    + std::to_string(pca_mode) + ".yaml";
}
std::string Cylinder::getNormalizerConfigName(std::string prefix) {
  return prefix + "lv" + std::to_string(level) + "/"
    + modes[mode] + (trick_mode ? "_trick" : "") 
    + "/norm_config" + std::to_string(pca_mode) + ".yaml";
}
std::string Cylinder::getYAMLMetricsName() {
  return save_path + "lv" + std::to_string(level) + "/"
    + modes[mode] + (trick_mode ? "_trick" : "") + "/"
    + (mode>=3 ? std::to_string(pca_mode) : "") + "_" 
    + cleanFloatStr(svm_nu) + "_" 
    + cleanFloatStr(svm_gamma) + ".yaml";
}