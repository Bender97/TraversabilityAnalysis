#include "Cylinder.h"


static Eigen::Vector3d      red(1.0f, 0.0f, 0.0f);
static Eigen::Vector3d  darkred(0.545f, 0.0f, 0.0f);
static Eigen::Vector3d lightred(1.0f, 0.7f, 0.7f);
static Eigen::Vector3d    white(1.0f, 1.0f, 1.0f);
static Eigen::Vector3d darkgray(0.92f, 0.92f, 0.92f);

static Eigen::Vector3d road(0.7f, 0.7f, 0.5f);


static Eigen::Vector3d limegreen(0.19f, 0.8f, 0.19f);
static Eigen::Vector3d yellow(0.94f, 0.91f, 0.64f);
static Eigen::Vector3d darkorange(1.0f, 0.54f, 0.0f);
static Eigen::Vector3d blue(0.0f, 0.0f, 1.0f);

static Eigen::Vector3d tp_color = limegreen; //road;
static Eigen::Vector3d tn_color = limegreen; //red;
static Eigen::Vector3d fp_color = darkred;
static Eigen::Vector3d fn_color = blue;

static Eigen::Vector3d      zerov(0.0f, 0.0f, 0.0f);

inline int int_floor(double x)
{
  int i = (int)x; /* truncate */
  return i - ( i > x ); /* convert trunc to floor */
}

Cylinder::Cylinder(YAML::Node &node, int tot_geom_features_) {
  start_radius = node["min_radius"].as<float>();
  end_radius   = node["max_radius"].as<float>();
  steps_num    = node["steps_num"].as<int>();
  yaw_steps    = node["yaw_steps"].as<int>();
  z            = node["z_level"].as<float>();

  yaw_steps_half    = yaw_steps / 2;
  tot_geom_features = tot_geom_features_;

  radius_step = ((end_radius - start_radius) * 100000 + 5) / steps_num / 100000.0f; // just a truncation
  yaw_res = M_DOUBLE_PI / yaw_steps;

  tot_cells = steps_num*yaw_steps;

  y_train = cv::Mat(tot_cells, 1, CV_32FC1);

  grid       = std::vector<Cell>(tot_cells);
  features   = std::vector<Feature>(tot_cells);
  area       = std::vector<float>(tot_cells);
  remap_idxs = std::vector<int>(tot_cells, -1);

  predictions_vector = cv::Mat(tot_cells, 1, CV_32F);

  if (!node["dataset"])
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. please set dataset mode.\n"));
  
  // if (node["dataset"].as<std::string>() == "SemKITTI")
    //  std::bind(&Cylinder::computeTravGT, computeTravGT_SemKITTI, std::placeholders::_1);
  // computeTravGT = std::bind(&Cylinder::computeTravGT_SemKITTI, this, std::placeholders::_1);

  // else 
    // computeTravGT = std::function<void(std::vector<int> &)>(&Cylinder::computeTravGT_NuSc, this);
    // computeTravGT = std::bind(&Cylinder::computeTravGT_NuSc, this, std::placeholders::_1);

  tmetric.resetAll();
  gmetric.resetAll();

  // pre-compute area of each cell: it will be used in computeFeatures
  for (int row_idx = 0, idx=0; row_idx<steps_num; row_idx++) {
    for (int yaw_idx = 0; yaw_idx<yaw_steps; yaw_idx++, idx++) {
      float r = row_idx*radius_step, R = r+ radius_step;
      area[idx] = M_PI*(R*R-r*r) / yaw_steps;
    }
  }
  
  // initialize mesh: it's the polar-grid graphics geometry
  createTriang();
  
}

Cylinder::Cylinder(YAML::Node &node,  Cylinder *cyl_, int tot_geom_features_, int produce_features)
: Cylinder(node, tot_geom_features_)
{

  level = 0;
  //if (cyl_!=nullptr) level = cyl_->features[0].derived_features.size()+1;
  if (cyl_!=nullptr) level = cyl_->level + 1;

  modes = std::vector<std::string>({"geom", "geom_label", "geom_all", "geom_pca", "geom_pca_label", "geom_pca_all_label"});

  std::cout << "train level: " << level << "  " << std::flush;

  mode = -1;
  if (node["mode"]) {
    mode = node["mode"].as<int>();
    std::cout << "training in " << modes[mode] << " mode  " << std::flush;
  }
  if (produce_features<0) {
    std::cout << "Test mode!" << std::endl;
    return;
  }

  trick_mode=0;
  if (node["trick"])      trick_mode = node["trick"].as<int>();
  std::cout << "trick_mode " << (trick_mode ? " on  " : "off  ") << std::endl;
  
  pca_mode = -1;
  if (node["pca"])       pca_mode = node["pca"].as<int>();
  else std::cout << "NO PCA MODE FOUND!" << std::endl;
  std::cout << "pca_mode " << pca_mode << std::endl;
  if (mode>=3) {

    std::string cname = "results/lv" + std::to_string(level) + "/"
              + modes[mode] + (trick_mode ? "_trick" : "") + "/pca_config_data" + std::to_string(pca_mode) + ".yaml";
    std::cout << "loading PCA from " << cname << std::endl;
    cv::FileStorage fs(cname, cv::FileStorage::READ);
    fs["mean"] >> pca.mean ;
    fs["vectors"] >> pca.eigenvectors ;
    fs["values"] >> pca.eigenvalues ;
    fs.release();

    std::cout << "eigenvalues: " << pca.eigenvalues.size() << std::endl;
  }

  computeFeaturesCols();

  X_train = cv::Mat(tot_cells, training_cols, CV_32F);
  if (mode<3)
    tmp = cv::Mat(tot_cells, training_cols, CV_32F);
  else
    tmp = cv::Mat(tot_cells, pca_mode+level, CV_32F);

  //std::cout << " adding " << level << " derived features. Check: " << std::flush;

  int dfs = (1+tot_geom_features)*level;  // derived features size
  for (auto &f : features) f.derived_features.resize(dfs);
  std::cout << " adding " << dfs << " derived features. Check: " << features[0].derived_features.size() << " but real used: " << (re_idx.size() - tot_geom_features) << std::endl;
  
  if (!produce_features) {
    std::string cname = "results/lv" + std::to_string(level) + "/"
               + modes[mode] + (trick_mode ? "_trick" : "") + "/config_data" + std::to_string(level) + ".yaml";
    normalizer = Normalizer(tot_geom_features_across_all_levels, cname.c_str());
    // normalizer.print();
  }
  else if (mode>=0) {
    store_features_ofname = node["store_path"].as<std::string>();
    if (store_features_ofname.empty()) 
      throw std::runtime_error(
            std::string("\033[1;31mERROR\033[0m. please provide a valid outfile path. Found empty!\n"));
    std::ofstream out(store_features_ofname.c_str(), std::ios::out | std::ios::binary);
    out.close();
  }

  //STORE IDX RELATION BTW CYLS
  idxs.resize(grid.size());
  temp_color.resize(grid.size());

  if (cyl_!=nullptr) {

    for (int row_idx = 0, i=0; row_idx<steps_num; row_idx++) {
      for (int yaw_idx = 0; yaw_idx<yaw_steps; yaw_idx++, i++) {
        
        float angle = yaw_idx*yaw_res;
        
        float sampled_angle  = angle  + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(yaw_res)));
        float sampled_radius = (start_radius + row_idx*radius_step) + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(radius_step)));

        if (sampled_angle>=M_DOUBLE_PI) sampled_angle = sampled_angle - M_DOUBLE_PI;
        else if (sampled_angle<0.0f)    sampled_angle = M_DOUBLE_PI + sampled_angle;
        
        int row, col;
        row = (int) std::floor((sampled_radius-cyl_->start_radius)/cyl_->radius_step);
        col = ((int) std::floor((sampled_angle)/cyl_->yaw_res)+cyl_->yaw_steps/2)%cyl_->yaw_steps;

        int sampled_idx = col*cyl_->steps_num + row;
          
        int exdx = ((yaw_idx+yaw_steps/2)%yaw_steps)*steps_num + row_idx;

        idxs[exdx] = sampled_idx; //save the idx of the correspondent backsector
        temp_color[exdx] = cyl_->temp_color[sampled_idx];

      } 
    }
  }
  else {
    for (int row_idx = 0, i=0; row_idx<steps_num; row_idx++) {
      for (int yaw_idx = 0; yaw_idx<yaw_steps; yaw_idx++, i++) {
        int sampled_idx = ((yaw_idx+yaw_steps/2)%yaw_steps)*steps_num + row_idx;
        idxs[i] = sampled_idx;
      }
    }

  }

  if (mode>=0 && !produce_features)
    loadSVM(node["svm_path"].as<std::string>() + std::string("lv") + std::to_string(level) + std::string("/")
               + modes[mode] + (trick_mode ? "_trick" : "") + std::string("/svm_model")
               + (mode>=3 ? std::string("_") + node["pca"].as<std::string>() : "")
               + std::string("_") + node["nu"].as<std::string>()
               + std::string("_") + node["gamma"].as<std::string>()
               + std::string(".bin")
               );
}


Cylinder::Cylinder(YAML::Node &node,  Synchro *synchro__, Cylinder *cyl_, int tot_geom_features_, int produce_features)
: Cylinder(node, cyl_, tot_geom_features_, produce_features)
{
  synchro_ = synchro__;
}

void Cylinder::createTriang() {

  // initialize data structure
  mesh = std::make_shared<open3d::geometry::TriangleMesh>();
  mesh->vertices_.resize(4*tot_cells);
  mesh->triangles_.resize(4*tot_cells);
  mesh->vertex_colors_.resize(4*tot_cells);

  // handy variables to store intermediate values
  double sina, cosa, sina_, cosa_;
  float angle, r0, r1;
  int off=0;

  for (float r=start_radius; r<end_radius; r+=radius_step) {
    r0 = r; r1 = r + radius_step;

    for (float realangle=0; realangle<2*M_PI; realangle+=yaw_res) {

      // clip angle in range [0, 2M_PI]
      angle = realangle+M_PI;
           if (angle>=M_DOUBLE_PI) angle = angle - M_DOUBLE_PI;
      else if (angle<0.0f)         angle = M_DOUBLE_PI + angle;

      sina = sin(angle); sina_ = sin(angle+yaw_res);
      cosa = cos(angle); cosa_ = cos(angle+yaw_res);

      mesh-> vertices_[off] = Eigen::Vector3d(r0*sina,  r0*cosa,  z);
      mesh->triangles_[off] = Eigen::Vector3i(off,   off+1, off+2);
      off++;
      mesh-> vertices_[off] = Eigen::Vector3d(r1*sina,  r1*cosa,  z);
      mesh->triangles_[off] = Eigen::Vector3i(off+1, off,   off-1);
      off++;
      mesh-> vertices_[off] = Eigen::Vector3d(r0*sina_, r0*cosa_, z);
      mesh->triangles_[off] = Eigen::Vector3i(off-1, off,   off+1);
      off++;
      mesh-> vertices_[off] = Eigen::Vector3d(r1*sina_, r1*cosa_, z);
      mesh->triangles_[off] = Eigen::Vector3i(off,   off-1, off-2);
      off++;

    }
  }
}

void Cylinder::updateTriang() {
  Eigen::Vector3d color_;
  int idx, label, gtlabel;
  int cont=0;

  for (int row_idx = 0; row_idx<steps_num; row_idx++) {
    for (int yaw_idx = 0; yaw_idx<yaw_steps; yaw_idx++) {

      idx = ((yaw_idx+yaw_steps/2)%yaw_steps)*steps_num + row_idx;
      label = grid[idx].predicted_label;//.predicted_label;
      gtlabel = grid[idx].label;

           if (label==UNKNOWN_CELL_LABEL) color_ = darkgray;
      else if (label>0) {
        if (label*gtlabel>0) color_ = tp_color;
        else color_ = fp_color;
      }
      else {
        if (label*gtlabel>0) color_ = tn_color;
        else color_ = fn_color;
      }                         

      // set the color for the cell
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
    }
  }

  mesh->ComputeVertexNormals();
  synchro_->addPolarGrid(mesh);

}

void Cylinder::updateTriangGT() {
  Eigen::Vector3d color_;
  int idx, label;
  int cont=0;

  for (int row_idx = 0; row_idx<steps_num; row_idx++) {
    for (int yaw_idx = 0; yaw_idx<yaw_steps; yaw_idx++, cont++) {

      idx = ((yaw_idx+yaw_steps/2)%yaw_steps)*steps_num + row_idx;
      label = grid[idx].label;

           if (label==UNKNOWN_CELL_LABEL) color_ = darkgray;
      else if (label==TRAV_CELL_LABEL)    color_ = tp_color;
      else                                color_ = tn_color;

      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
    }
  }

  mesh->ComputeVertexNormals();
  synchro_->addPolarGrid(mesh);
}

// avg_label is the average (trav/non trav) label of the POINTs inside each cell
// clearly is available only in ground truth evaluation
// during inference this should be replaced with the output of the svm SVR classifier

void Cylinder::inheritFeatures(Cylinder *cyl_) {
  // TODO: introduce error if level==0
  if (training_cols == tot_geom_features) return;

  int prevfeats = cyl_->features[0].derived_features.size();
  std::vector<float> prev_feats;
  int i, j, c;
  

  for (i=0; i<tot_cells; i++) {
    if (grid[i].label==UNKNOWN_CELL_LABEL) continue;
    
    for (j=0; j<prevfeats; j++)
      features[i].derived_features[j] = cyl_->features[idxs[i]].derived_features[j];

    prev_feats = cyl_->features[idxs[i]].toVectorTransformed();
    for (j=prevfeats, c=0; j<prevfeats+tot_geom_features; j++, c++)
      features[i].derived_features[j] = prev_feats[c];
    features[i].derived_features[j] = cyl_->grid[idxs[i]].predicted_label;
  }

}

void Cylinder::inheritGTFeatures(Cylinder *cyl_) {
  // TODO: introduce error if level==0
  if (training_cols == tot_geom_features) return;

  int prevfeats = cyl_->features[0].derived_features.size();
  std::vector<float> prev_feats;
  int i, j, c;

  for (i=0; i<tot_cells; i++) {
    if (grid[i].label==UNKNOWN_CELL_LABEL) continue;
    
    for (j=0; j<prevfeats; j++)
      features[i].derived_features[j] = cyl_->features[idxs[i]].derived_features[j];

    prev_feats = cyl_->features[idxs[i]].toVectorTransformed();
    for (j=prevfeats, c=0; j<prevfeats+tot_geom_features; j++, c++)
      features[i].derived_features[j] = prev_feats[c];
    features[i].derived_features[j] = cyl_->grid[idxs[i]].label;
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

void Cylinder::computeTravGT_SemKITTI(std::vector<int> &labels) {
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

    y_train.at<float>(valid_rows, 0) = cell->label;
    valid_rows++;

  }

  std::cout << "LEVEL " << level << std::endl;
}

void Cylinder::computeTravGT_NuSc(std::vector<int> &labels) {
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

    y_train.at<float>(valid_rows, 0) = cell->label;
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

void Cylinder::computeFeaturesCols() {
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

  der_feats_size=(re_idx.size()-tot_geom_features_across_all_levels);


  std::cout << "computed Training Cols (before pca) : " << training_cols << std::endl;
  std::cout << "to check re_idx size                : " << re_idx.size() << std::endl;
}

void Cylinder::loadSVM(std::string path) {
  std::cout << "Loading model from " << path << " ..." << std::flush;
  model = cv::ml::SVM::load(path);
  if (!model->isTrained())
    throw std::runtime_error(
          std::string("\033[1;31mERROR\033[0m. cyl of level ") 
        + std::to_string(level) + std::string(" has invalid SVM path: ")
        + path + std::string(".\n"));

  std::cout << " done!" << std::endl;
}



void Cylinder::process(Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points) {


  std::vector<float> feat;
  float *features_matrix_data_row;
  int r, valid_rows;

  // fill cv::Mat X_train
  for (r=0, valid_rows=0; r<tot_cells; r++) {
    if (grid[r].label == UNKNOWN_CELL_LABEL)
      continue;

    feat = features[r].toVectorTransformed();
    features_matrix_data_row = X_train.ptr<float>(valid_rows);
    for (int c=0; c<training_cols; c++) features_matrix_data_row[c] = feat[re_idx[c]];
      
    remap_idxs[valid_rows] = r;
    valid_rows++;
  }

  

  normalizer.normalize(X_train);

  if (mode>=3) {
    int der_feats_size=(re_idx.size()-tot_geom_features_across_all_levels);

    tmp = cv::Mat(valid_rows, pca_mode+der_feats_size, CV_32F);
    
    

    if (level>0 && der_feats_size>0) 
      X_train(cv::Range(0, valid_rows), cv::Range(tot_geom_features_across_all_levels, X_train.cols))
      .copyTo(tmp(cv::Range(0, valid_rows), cv::Range(pca_mode, tmp.cols)));

    pca.project(X_train(cv::Range(0, valid_rows), cv::Range(0, tot_geom_features_across_all_levels)), 
                    tmp(cv::Range(0, valid_rows), cv::Range(0, pca_mode)));
  }
  else
    tmp = X_train(cv::Range(0, valid_rows), cv::Range(0, X_train.cols));
  
  tmetric.resetAll();
  model->predict(tmp, predictions_vector(cv::Range(0, valid_rows), cv::Range(0, 1)), cv::ml::StatModel::RAW_OUTPUT);
  tmetric.checkpointTime();
  std::cout << "level " << level << " chekc: " << tmetric.checkpointTime_ << " ms" << std::endl;

  float plab, pred;
  for (int r=0; r < valid_rows; r++) {
    pred = predictions_vector.at<float>(r, 0);
    plab = pred > 0 ? 1.0f : -1.0f;
    
    // // trick!
    if (level>0 && tmp.at<float>(r, pca_mode+level-1) > 1.5) pred = 1.0f;

    grid[remap_idxs[r]].predicted_label = plab; // if trick is wanted to be float, put pred

    if (!(r%100)) (synchro_->cv).notify_one();
  }

  // remove when computing latency
  (synchro_->cv).notify_one();

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
  int status, tot=0;
  for (int r=0; r<(int)features.size(); r++) {
    if (grid[r].points_idx.size()<2) continue;
    status = features[r].computeFeatures(&grid[r], scene_normal, points, area[r]);
    
    if (!status) {
      grid[r].label=UNKNOWN_CELL_LABEL; tot++;
    }
  }
  std::cout << "TOT: " << tot << "\n";
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
  computeTravGT_SemKITTI(labels);
  computeFeatures(scene_normal, points);
  if (level>0) inheritGTFeatures(back_cyl);
  storeFeatures();
}
