#include "DataLoader.h"

#define MIN(a, b) (a>b ? b : a)


DataLoader::DataLoader(std::string dataset_path_) {
  dataset_path = dataset_path_;
}

DataLoader_SemKITTI::DataLoader_SemKITTI(std::string SemKITTI_dataset_path) : DataLoader(SemKITTI_dataset_path) {}

DataLoader_PandaSet::DataLoader_PandaSet(std::string PandaSet_dataset_path) : DataLoader(PandaSet_dataset_path) {}

DataLoader_NuSc::DataLoader_NuSc(std::string recipe_path) : DataLoader(recipe_path) {
  pts_s.clear();
  lab_s.clear();
  std::ifstream file(recipe_path);
  std::string str;
  while (std::getline(file, str)) {
    pts_s.push_back(str);
    std::getline(file, str);
    lab_s.push_back(str);
  }
  file.close();

  if (pts_s.size()!=lab_s.size())
    throw std::runtime_error(std::string("Error. DataLoader_NuSc found different sizes in points binary paths and\n")
              + std::string("labels binary paths.\nPlease check the recipe file at path ")
              + recipe_path + std::string("\nExit.\n"));

}


void DataLoader::readData(int seq, int idx, YAML::Node &sample_data) {}
void DataLoader::readPredicted(int seq, int idx, YAML::Node &sample_data) {}


DataLoader_PLY::DataLoader_PLY(std::string ply_path_) : DataLoader(ply_path_) {
}
void DataLoader_PLY::readData(int seq, int idx, YAML::Node &sample_data) {}
void DataLoader_PLY::readPLY(std::string ply_path) {
  points.clear();
  labels.clear();
  std::ifstream fin(ply_path);
  std::string line; 
  Eigen::Vector3d color;

  // skip .ply header
  for (int i=0; i<10; i++) std::getline(fin, line);
  
  while (std::getline(fin, line)){
    std::istringstream in(line);
    in >> p(0);
    in >> p(1);
    in >> p(2);
    in >> color(0);
    in >> color(1);
    in >> color(2);

    p(0) += 840.3;
    p(1) += 90;
    p(2) += -37;

    points.push_back(p);
    labels.push_back(40);
  }

  fin.close();
  computeSceneNormal();

}

void DataLoader_PLY::readBin(std::string ply_path) {
  points.clear();
  labels.clear();
  std::ifstream fin(ply_path, std::ios::binary);
  std::string line; 
  Eigen::Vector3d color;
  int c=0;
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    
         if (c==0) p(0) = f;
    else if (c==1) p(1) = f;
    else if (c==2) p(2) = f;
    else {
      if (!std::isnan(p(0)) && !std::isnan(p(1)) && !std::isnan(p(2))) {
        points.push_back(p);
        labels.push_back(40);
      }
    }
    c = (c + 1) %4;

  }

  fin.close();
  computeSceneNormal();

}



void DataLoader_SemKITTI::readData(int seq, int idx, YAML::Node &sample_data) {
  points.clear();
  labels.clear();
  std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  c = 0;
  std::ifstream fin(dataset_path+"sequences/"+new_seq_s+"/velodyne/"+new_idx_s+".bin", std::ios::binary);
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    
         if (c==0) p(0) = f;
    else if (c==1) p(1) = f;
    else if (c==2) p(2) = f;
    else points.push_back(p);

    c = (c + 1) %4;
  }

  std::ifstream lin(dataset_path+"sequences/"+new_seq_s+"/labels/"+new_idx_s+".label", std::ios::binary);
  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int)))
    labels.push_back(c & 0xFFFF);

  computeSceneNormal();
}

void DataLoader_PandaSet::readData(int seq, int idx, YAML::Node &sample_data) {
  points.clear();
  labels.clear();
  // std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  // auto new_seq_s = std::string(3 - MIN(3, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  c = 0;
  int l;
  float h;
  
  std::string lidar_path = dataset_path + "/lidar_bin/"+new_idx_s+".bin";
  std::string label_path = dataset_path + "/semseg/"+new_idx_s+".bin";

  std::ifstream fin(lidar_path, std::ios::binary);
  if (!fin) throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. cannot open file") + lidar_path);

  std::ifstream lin(label_path, std::ios::binary);
  if (!lin) throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. cannot open file") + label_path);
  
  while (fin.read(reinterpret_cast<char*>(&h), sizeof(float))) {
    
         if (c==0) p(0) = h;
    else if (c==1) p(1) = h;
    else if (c==2) {
      p(2) = h;
      if (!std::isnan(p(0)) && !std::isnan(p(1)) && !std::isnan(p(2))) {
        points.push_back(p);
        lin.read(reinterpret_cast<char*>(&l), sizeof(int64_t));
        labels.push_back( l);
      }
      else {
        lin.read(reinterpret_cast<char*>(&l), sizeof(int64_t));
      }
    }
    c = (c + 1) %4;

  }

  fin.close();
  lin.close();

  std::cout << "points: " << points.size() << " labels: " << labels.size() << "\n";

  computeSceneNormal();
}

void DataLoader_NuSc::readData(int , int idx, YAML::Node &) {
  points.clear();
  labels.clear();
  c = 0;

//   uint8_t c;
  std::ifstream fin(pts_s[idx], std::ios::binary);
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    
         if (c==0) p(0) = f;
    else if (c==1) p(1) = f;
    else if (c==2) p(2) = f;
    else if (c==4) points.push_back(p);
    c = (c + 1) %5;
  }
  // TOTEST: if it does not work, reintroduce uint8_t before (see its commented)
  std::ifstream lin(lab_s[idx], std::ios::binary);
  while (lin.read(reinterpret_cast<char*>(&c), sizeof(uint8_t)))
    labels.push_back((uint32_t) c);

  computeSceneNormal();

}


void DataLoader_SemKITTI::readPredicted(int seq, int idx, YAML::Node &sample_data) {
  std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  c = 0;

  pred_labels.clear();

  std::string base_path = sample_data["general"]["predicted_path"].as<std::string>();
  std::string lin_path = base_path + "/"+new_idx_s+".label";
  std::ifstream lin(lin_path, std::ios::binary);
  std::cout << lin_path << std::endl;

  if (!lin)
    throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. cannot open file") + lin_path);

  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int))) {
    if (c==2) {pred_labels.push_back(NOT_TRAV_CELL_LABEL);}
    else if (c==1) pred_labels.push_back(TRAV_CELL_LABEL);
    else if (c==3) pred_labels.push_back(3); // sidewalk
    else pred_labels.push_back(0);
  }

  // for (int i=0; i<20; i++) std::cout << labels[i] << std::endl;
}

// void DataLoader_PandaSet::readPredicted(int seq, int idx, YAML::Node &sample_data) {
//   pred_labels.clear();
//   std::string idx_s = std::to_string(idx);
//   auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

//   std::string base_path = sample_data["general"]["predicted_path"].as<std::string>();
//   std::string lin_path = base_path + "/"+new_idx_s+".bin";
//   std::ifstream lin(lin_path, std::ios::binary);

//   if (!lin)
//     throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. cannot open file") + lin_path);

//   while (lin.read(reinterpret_cast<char*>(&c), sizeof(uint32_t))) {
//     int32_t h = (int32_t) (c);
//     if (h==2) {pred_labels.push_back(NOT_TRAV_CELL_LABEL);}
//     else if (h==1) pred_labels.push_back(TRAV_CELL_LABEL);
//     else if (h==3) pred_labels.push_back(3); // sidewalk
//     else pred_labels.push_back(0);
//   }
// }

void DataLoader_PandaSet::readPredicted(int seq, int idx, YAML::Node &sample_data) {
  pred_labels.clear();
  std::string idx_s = std::to_string(idx);
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  std::string base_path = sample_data["general"]["predicted_path"].as<std::string>();
  std::string lin_path = base_path + "/"+new_idx_s+".bin";
  std::ifstream lin(lin_path, std::ios::binary);

  if (!lin)
    throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. cannot open file") + lin_path);

  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int))) {
    int32_t h = (int32_t) (c);
    if (h==2) {pred_labels.push_back(NOT_TRAV_CELL_LABEL);}
    else if (h==1) pred_labels.push_back(TRAV_CELL_LABEL);
    else if (h==3) pred_labels.push_back(3); // sidewalk
    else pred_labels.push_back(0);
  }
}

void DataLoader_NuSc::readPredicted(int seq, int idx, YAML::Node &sample_data) {
  pred_labels.clear();
  std::string idx_s = std::to_string(idx);
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;
  c = 0;

  std::string base_path = sample_data["general"]["predicted_path"].as<std::string>();
  std::string lin_path = base_path + "/"+new_idx_s+".label";
  std::ifstream lin(lin_path, std::ios::binary);
  std::cout << lin_path << std::endl;

  if (!lin)
    throw std::runtime_error(std::string("\033[1;31mERROR\033[0m. cannot open file") + lin_path);

  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int))) {
    if (c==2) {pred_labels.push_back(NOT_TRAV_CELL_LABEL);}
    else if (c==1) pred_labels.push_back(TRAV_CELL_LABEL);
    else if (c==3) pred_labels.push_back(3); // sidewalk
    else pred_labels.push_back(0);
  }

}

int DataLoader::count_files_in_folder(std::string folder_path) {
  int fileCount = 0;
  DIR *dp;
  struct dirent *ep;     
  dp = opendir (folder_path.c_str());
  if (dp != NULL) {
    while ((ep = readdir (dp))) fileCount++;
    (void) closedir (dp);
  }
  else {
    std::cout << folder_path << std::endl; 
    perror ("Couldn't open the directory");
  }
  return fileCount - 2;  
}

int DataLoader::count_samples(int seq) {return 0;}

int DataLoader_SemKITTI::count_samples(int seq) {
  auto seq_s = std::string(2 - MIN(2, std::to_string(seq).length()), '0') + std::to_string(seq);
  std::string path = dataset_path +"sequences/"+seq_s+"/labels/";

  return count_files_in_folder(path);
  
}

int DataLoader_PandaSet::count_samples(int seq) {
  std::string path = dataset_path + "lidar_bin/";

  return count_files_in_folder(path);

}

int DataLoader_NuSc::count_samples(int) {
  return pts_s.size();
}

void DataLoader::assertConsistency() {
  if (points.size() != labels.size())
    throw std::runtime_error(
      std::string("\033[1;31mERROR\033[0m. Size mismatch between points and labels\n") +
      std::string("lidar points: ") + std::to_string(points.size()) + " points\n" + 
      std::string("lidar labels: ") + std::to_string(labels.size()) + " labels\n" + 
      std::string("Exit.") );
}

void DataLoader::assertDLConsistency() {
  if (labels.size() != pred_labels.size())
    throw std::runtime_error(
      std::string("\033[1;31mERROR\033[0m. Size mismatch between labels and predicted labels\n") +
      std::string("lidar labels: ") + std::to_string(labels.size()) + " labels\n" + 
      std::string("predicted labels: ") + std::to_string(pred_labels.size()) + " labels\n" + 
      std::string("Exit.") );
}



#if OPEN3D == 1
std::shared_ptr<const open3d::geometry::PointCloud> DataLoader::getPaintedCloud()
{
  return std::shared_ptr<const open3d::geometry::PointCloud>();
}

std::shared_ptr<const open3d::geometry::PointCloud> DataLoader_SemKITTI::getPaintedCloud() {

  if (points.size() != labels.size()) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. cloud and labels differ in size.\n"));
  
  pc = open3d::geometry::PointCloud(points);
  pc.colors_.resize(points.size());
  for (int i=0; i<(int)pc.points_.size(); i++) {
    color_util.setColor(pc.colors_[i], labels[i]);
  }
  return std::make_shared<open3d::geometry::PointCloud>(pc);
}

std::shared_ptr<const open3d::geometry::PointCloud> DataLoader_PandaSet::getPaintedCloud() {

  if (points.size() != labels.size()) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. cloud and labels differ in size.\n"));
  
  pc = open3d::geometry::PointCloud(points);
  pc.colors_.resize(points.size());
  for (int i=0; i<(int)pc.points_.size(); i++) {
    color_util.setColor(pc.colors_[i], labels[i]);
  }
  return std::make_shared<open3d::geometry::PointCloud>(pc);
}

std::shared_ptr<const open3d::geometry::PointCloud> DataLoader_NuSc::getPaintedCloud() {
  if (points.size() != labels.size()) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. cloud and labels differ in size.\n"));
  
  pc = open3d::geometry::PointCloud(points);
  pc.colors_.resize(points.size());
  for (int i=0; i<(int)pc.points_.size(); i++) {
    color_util.setColor(pc.colors_[i], labels[i]);
  }
  return std::make_shared<open3d::geometry::PointCloud>(pc);
}

std::shared_ptr<const open3d::geometry::PointCloud> DataLoader_PLY::getPaintedCloud() {
  Eigen::Vector3d    white(0.8f, 0.5f, 0.5f);
  pc = open3d::geometry::PointCloud(points);
  pc.colors_.resize(points.size(), white);
  return std::make_shared<open3d::geometry::PointCloud>(pc);
}


std::shared_ptr<const open3d::geometry::VoxelGrid> DataLoader::getPaintedCloudVoxeled(float voxel_size)
{
  return std::shared_ptr<const open3d::geometry::VoxelGrid>();
}

std::shared_ptr<const open3d::geometry::VoxelGrid> DataLoader_SemKITTI::getPaintedCloudVoxeled(float voxel_size)
{
  pc = open3d::geometry::PointCloud(points);
  pc.colors_.resize(points.size());
  for (int i=0; i<(int)pc.points_.size(); i++) {
    color_util.setColor(pc.colors_[i], labels[i]);
  }
  voxel = open3d::geometry::VoxelGrid::CreateFromPointCloud(pc, voxel_size);  
  return voxel;
}

std::shared_ptr<const open3d::geometry::VoxelGrid> DataLoader_NuSc::getPaintedCloudVoxeled(float voxel_size)
{
  return std::shared_ptr<const open3d::geometry::VoxelGrid>();
}

std::shared_ptr<const open3d::geometry::PointCloud> DataLoader::getPaintedCloud_DL() {

  if (points.size() != labels.size()) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. cloud and labels differ in size.\n"));

  pc = open3d::geometry::PointCloud(points);

  pc.colors_.resize(points.size());
  for (int i=0; i<(int)points.size(); i++)
    color_util.setColor_DL(pc.colors_[i], labels[i]);

  // std::cout << "0 : " << l0 << std::endl;
  // std::cout << "1 : " << l1 << std::endl;
  // std::cout << "-1: " << lm1 << std::endl;
  // std::cout << "2 : " << lo << std::endl;
  return std::make_shared<open3d::geometry::PointCloud>(pc);

}

std::shared_ptr<const open3d::geometry::PointCloud> DataLoader::getUniformPaintedCloud() {
  Eigen::Vector3d    white(0.5f, 0.5f, 0.5f);
  pc = open3d::geometry::PointCloud(points);
  pc.colors_.resize(points.size(), white);
  return std::make_shared<open3d::geometry::PointCloud>(pc);
}
#endif


void DataLoader::computeSceneNormal() {
  float cx = 0, cy = 0, cz = 0;
  float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
  int numpoints = points.size();

  // compute eigenvalues and eigenvectors
  Eigen::MatrixXd matA1 = Eigen::MatrixXd::Zero(3, 3);

  for (auto point: points) {
      cx += point(0);
      cy += point(1);
      cz += point(2);
  }
  cx /= (float) numpoints; cy /= (float) numpoints;  cz /= (float) numpoints;

  float ax, ay, az;

  for (auto point: points) {
      ax = point(0) - cx;
      ay = point(1) - cy;
      az = point(2) - cz;

      a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
      a22 += ay * ay; a23 += ay * az;
      a33 += az * az;
  }

  a11 /= (float) numpoints; a12 /= (float) numpoints; a13 /= (float) numpoints; a22 /= (float) numpoints; a23 /= (float) numpoints; a33 /= (float) numpoints;

  matA1 << a11, a12, a13, a12, a22, a23, a13, a23, a33;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matA1);
  if (eigensolver.info() != Eigen::Success) scene_normal = Eigen::MatrixXd::Zero(3, 3);
  else scene_normal = eigensolver.eigenvectors().col(0); // because eigenvalues are sorted in increasing order
}