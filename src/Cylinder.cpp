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

  grid       = std::vector<Cell>(tot_cells);
  features   = std::vector<Feature>(tot_cells);
  area       = std::vector<float>(tot_cells);
  remap_idxs = std::vector<int>(tot_cells, -1);

  GT_labels_vector = cv::Mat(tot_cells, 1, CV_32FC1);
  predictions_vector = cv::Mat(tot_cells, 1, CV_32F);

  if (!node["dataset"])
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. please set dataset mode.\n") );
  
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

Cylinder::Cylinder(YAML::Node &node,  Cylinder *cyl_, ExpMode expmode) : Cylinder(node)
{

  level = 0; mode = -1; trick_mode=0; pca_mode=-1;
  if (cyl_!=nullptr) level = cyl_->level + 1;

  if (node["mode"]) mode = node["mode"].as<int>();

  if (expmode == ExpMode::DL) {
    std::cout << "Deep Learning test mode! No need for other configurations" << std::endl;
    return;
  }
  
  if (node["trick"]) trick_mode = node["trick"].as<int>();
  
  if (node["load_path"]) load_path = sanitize(node["load_path"].as<std::string>());
  if (node["save_path"]) save_path = sanitize(node["save_path"].as<std::string>());
  
  loadPCAConfigs(node);
  computeFeaturesCols();
  if (expmode == ExpMode::test) loadSVM(node);

  full_featMatrix = cv::Mat(tot_cells, training_cols, CV_32F);
  
  if (mode<3)
    featMatrix = cv::Mat(tot_cells, training_cols, CV_32F);
  else
    featMatrix = cv::Mat(tot_cells, pca_mode+level, CV_32F);

  
  for (auto &f : features)
    f.derived_features.resize((1+TOT_GEOM_FEATURES)*level);
  
  if (expmode == ExpMode::test) {
    std::string cname = load_path + "lv" + std::to_string(level) + "/"
               + modes[mode] + (trick_mode ? "_trick" : "") + "/config_data" + std::to_string(level) + ".yaml";
    normalizer = Normalizer(tot_geom_features_across_all_levels, cname.c_str());
  }
  else {
    store_features_ofname = node["store_path"].as<std::string>();
    if (store_features_ofname.empty()) 
      throw std::runtime_error(
            std::string("\033[1;31mERROR\033[0m. please provide a valid outfile path. Found empty!\n"));
    std::ofstream out(store_features_ofname.c_str(), std::ios::out | std::ios::binary);
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
  if (training_cols == TOT_GEOM_FEATURES) return;
  
  std::vector<float> prev_feats;
  int i, j, c;
  
  for (i=0; i<tot_cells; i++) {
    if (grid[i].label==UNKNOWN_CELL_LABEL) continue;
    
    for (j=0; j<prevfeats_num; j++)
      features[i].derived_features[j] = cyl_->features[inherit_idxs[i]].derived_features[j];

    prev_feats = cyl_->features[inherit_idxs[i]].toVectorTransformed();
    for (j=prevfeats_num, c=0; j<prevfeats_num+TOT_GEOM_FEATURES; j++, c++)
      features[i].derived_features[j] = prev_feats[c];
    features[i].derived_features[j] = cyl_->grid[inherit_idxs[i]].predicted_label;
  }

}

void Cylinder::inheritGTFeatures(Cylinder *cyl_) {
  if (training_cols == TOT_GEOM_FEATURES) return;

  std::vector<float> prev_feats;
  int i, j, c;

  for (i=0; i<tot_cells; i++) {
    if (grid[i].label==UNKNOWN_CELL_LABEL) continue;
    
    for (j=0; j<prevfeats_num; j++)
      features[i].derived_features[j] = cyl_->features[inherit_idxs[i]].derived_features[j];

    prev_feats = cyl_->features[inherit_idxs[i]].toVectorTransformed();
    for (j=prevfeats_num, c=0; j<prevfeats_num+TOT_GEOM_FEATURES; j++, c++)
      features[i].derived_features[j] = prev_feats[c];
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

    // assumption: no x=y=0 since it's the laser origin..
    yaw = std::atan2((*p)(0), (*p)(1)) + M_PI;

    row = int_floor(yaw / yaw_res);
    col = int_floor((radius-start_radius) / radius_step);
    row = (row + yaw_steps_half) % yaw_steps;

    grid[row*steps_num + col].points_idx.push_back(i);
  }
}

void Cylinder::resetGrid() {
  for (auto &cell: grid) {
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
        else cell->label = UNKNOWN_CELL_LABEL;
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
    if (cell->points_idx.size() < 2) {
      cell->label = UNKNOWN_CELL_LABEL;
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
        else cell->label = UNKNOWN_CELL_LABEL;
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
    if (cell.points_idx.size() < MIN_NUM_POINTS_IN_CELL) {
      cell.label = UNKNOWN_CELL_LABEL;
      continue;
    }

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
        else {
          cell.predicted_label = UNKNOWN_CELL_LABEL;
        }
    }
  }

}

// 17 17 1 17 1 17 1 17 1 17 1
//       34   52
void Cylinder::computeFeaturesCols() {
  int i, l;
  switch(mode) {
    case 0: // geom
    case 3: // geom_pca
      training_cols = TOT_GEOM_FEATURES;
      tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
      for (i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
      break;
    case 1: // geom_label
    case 4: // geom_pca_label
      if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
      training_cols = TOT_GEOM_FEATURES + level;
      tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
      for (i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
      for (l=1; l<=level; l++) re_idx.push_back(TOT_GEOM_FEATURES + (TOT_GEOM_FEATURES+1)*l -1);
      
      break;
    case 2: // geom_all
    case 5: // geom_pca_all_label
      if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
      training_cols = (TOT_GEOM_FEATURES*(level+1)) + level;
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

  if (pca_mode<0 && mode>=3)
    throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m Mode with pca but no pca_mode set!"));

  if (pca_mode>TOT_GEOM_FEATURES)
    throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m fouond pca_mode > tot_geom_features_across_all_levels!"));

  inherited_labels_size = re_idx.size()-tot_geom_features_across_all_levels;

}

void Cylinder::loadPCAConfigs(YAML::Node &node)
{
  if (node["pca"]) pca_mode = node["pca"].as<int>();

  if (mode>=3) {
    std::string cname = load_path + "lv" + std::to_string(level) + "/"
                + modes[mode] + (trick_mode ? "_trick" : "") + "/pca_config_data"
                + std::to_string(pca_mode) + ".yaml";
    
    std::cout << "  loading   PCA from " << cname << std::flush;
    
    cv::FileStorage fs(cname, cv::FileStorage::READ);
    fs["mean"]    >> pca.mean;
    fs["vectors"] >> pca.eigenvectors;
    fs["values"]  >> pca.eigenvalues;
    fs.release();
    
    std::cout << " done!" << std::endl;
  }
}

void Cylinder::loadSVM(YAML::Node &node) {
  std::string path = load_path + "lv" + std::to_string(level) + "/"
      + modes[mode] + (trick_mode ? "_trick" : "") + "/svm_model"
      + (mode>=3 ? "_" + node["pca"].as<std::string>() : "") + "_" 
      + node["nu"].as<std::string>() + "_" 
      + node["gamma"].as<std::string>() + ".bin";
  
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


  std::vector<float> feat;
  float *features_matrix_data_row;
  float plab, pred;
  int r, c, valid_rows=0;

  // fill cv::Mat full_featMatrix
  for (r=0; r<tot_cells; r++) {
    if (grid[r].label == UNKNOWN_CELL_LABEL)
      continue;

    feat = features[r].toVectorTransformed();
    features_matrix_data_row = full_featMatrix.ptr<float>(valid_rows);
    for (c=0; c<training_cols; c++) features_matrix_data_row[c] = feat[re_idx[c]];
      
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
    if (level>0 && featMatrix.at<float>(r, pca_mode+level-1) > 0.5) pred = TRAV_CELL_LABEL;

    grid[remap_idxs[r]].predicted_label = plab; // if trick is wanted to be float, put pred

  }

}

void Cylinder::computeAccuracy() {
    tmetric.resetAcc();
    for (int c=0; c<tot_cells; c++) {
      if (grid[c].label == UNKNOWN_CELL_LABEL) continue;
      tmetric.update(grid[c].predicted_label, grid[c].label);
    }
    gmetric += tmetric;
}

void Cylinder::filterOutliers() {
  int cell_idx, newrow, newcol, road_count, non_road_count;
  int pred_label;
  int predicted_label_weight = 1;
  int row;
  for (int col = 0; col < steps_num; ++col) {
      for (int rowa = 0; rowa < yaw_steps; ++rowa) {
          road_count = 0; non_road_count = 0;

          row = (rowa+yaw_steps/2)%yaw_steps;

          for (int o1=-1; o1<2; o1++) {
              for (int o2=-1; o2<2; o2++) {
                  if (o1 || o2) { // skip the cell with o1==0 && o2==0 (it's the current cell)
                      newrow = row + o1; if ( newrow < 0 || newrow >= steps_num ) continue;
                      newcol = col + o2; if ( newcol < 0 || newcol >= yaw_steps ) continue;
                      cell_idx = newrow*steps_num + newcol;
                      pred_label = grid[cell_idx].predicted_label;
                      if (pred_label==TRAV_CELL_LABEL) road_count++;
                      else if (pred_label==NOT_TRAV_CELL_LABEL) non_road_count++;
                  }
              }
          }
          cell_idx = row*steps_num + col;

          if ( grid[cell_idx].predicted_label < 2 ) {
              if (grid[cell_idx].predicted_label == TRAV_CELL_LABEL) road_count += predicted_label_weight;
              else non_road_count += predicted_label_weight;
              grid[cell_idx].predicted_label = (road_count > non_road_count ? TRAV_CELL_LABEL : NOT_TRAV_CELL_LABEL);
          }
      }
  }
}


void Cylinder::computeFeatures(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points) {
  bool status;
  for (int r=0; r<(int)features.size(); r++) {
    if (grid[r].points_idx.size()<2) continue;
    status = features[r].computeFeatures(&grid[r], scene_normal, points, area[r]);
    if (!status) grid[r].label=UNKNOWN_CELL_LABEL; 
  }
}

void Cylinder::storeFeatures() {
  std::ofstream out(store_features_ofname.c_str(), std::ios::app | std::ios::binary);
  if (!out) // didn't open, do some error reporting here
      std::cout << "OUTSTREAM opening error - " << store_features_ofname << std::endl;
  //int cont=0;
  for (int i=0; i<(int) grid.size(); i++) {
    if (grid[i].label == UNKNOWN_CELL_LABEL) continue;
    //cont++;
    features[i].toFile(out);
    float lab = static_cast< float > (grid[i].label);
    out.write( reinterpret_cast<const char*>( &(lab) ), sizeof( float ));
  }
  //std::cout << "written " << cont << std::endl;
  out.close();
}

void Cylinder::storeFeaturesToFile(std::string name) {
  std::ofstream out(name.c_str(), std::ios::binary);
  if (!out) // didn't open, do some error reporting here
      std::cout << "OUTSTREAM opening error - " << store_features_ofname << std::endl;
  //int cont=0;
  for (int i=0; i<(int) grid.size(); i++) {
    if (grid[i].label == UNKNOWN_CELL_LABEL) continue;
    //cont++;
    features[i].toFile(out);
    float lab = static_cast< float > (grid[i].label);
    out.write( reinterpret_cast<const char*>( &(lab) ), sizeof( float ));
  }
  //std::cout << "written " << cont << std::endl;
  out.close();
}

void Cylinder::produceFeaturesRoutine(
                    std::vector<Eigen::Vector3d> &points, std::vector<int> &labels, 
                    Eigen::MatrixXd &scene_normal, Cylinder *back_cyl) {
  if (back_cyl==nullptr && level>0) {
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. provided a cylinder of level ") 
        + std::to_string(level) + std::string(" with nullptr back cylinder.\n"));
  }

  resetGrid();
  sortBins_cyl(points);
  computeTravGT(labels);
  computeFeatures(scene_normal, points);
  if (level>0) inheritGTFeatures(back_cyl);
  storeFeatures();
}

void Cylinder::OnlineRoutine(
                    DataLoader &dl, Cylinder *back_cyl) {
  resetGrid();
  sortBins_cyl(dl.points);
  computeTravGT(dl.labels);
  computeFeatures(dl.scene_normal, dl.points);
  if (level>0) inheritGTFeatures(back_cyl);
  process(dl.scene_normal, dl.points);
  computeAccuracy();
}
