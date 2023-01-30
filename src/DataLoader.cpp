#include "DataLoader.h"

#define MIN(a, b) (a>b ? b : a)


DataLoader::DataLoader() {

}

DataLoader_SemKITTI::DataLoader_SemKITTI() : DataLoader() {}

DataLoader_NuSc::DataLoader_NuSc() : DataLoader() {
  pts_s.clear();
  lab_s.clear();
  std::ifstream file("recipe.txt");
  std::string str;
  while (std::getline(file, str)) {
    pts_s.push_back(str);
    std::getline(file, str);
    lab_s.push_back(str);
  }
  file.close();
}


void DataLoader::readData(int seq, int idx, YAML::Node &sample_data) {}
void DataLoader::readPredicted(int seq, int idx, YAML::Node &sample_data) {}


void DataLoader_SemKITTI::readData(int seq, int idx, YAML::Node &sample_data) {
  points.clear();
  labels.clear();
  std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  c = 0;
  std::ifstream fin(sample_data["general"]["dataset_path"].as<std::string>()+"sequences/"+new_seq_s+"/velodyne/"+new_idx_s+".bin", std::ios::binary);
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    
         if (c==0) p(0) = f;
    else if (c==1) p(1) = f;
    else if (c==2) p(2) = f;
    else points.push_back(p);

    c = (c + 1) %4;
  }

  std::ifstream lin(sample_data["general"]["dataset_path"].as<std::string>()+"sequences/"+new_seq_s+"/labels/"+new_idx_s+".label", std::ios::binary);
  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int)))
    labels.push_back(c & 0xFFFF);

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
  points.clear();
  labels.clear();
  std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  c = 0;
  std::ifstream fin(sample_data["general"]["dataset_path"].as<std::string>()+"sequences/"+new_seq_s+"/velodyne/"+new_idx_s+".bin", std::ios::binary);
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    
         if (c==0) p(0) = f;
    else if (c==1) p(1) = f;
    else if (c==2) p(2) = f;
    else points.push_back(p);

    c = (c + 1) %4;
  }

  std::string base_path = sample_data["general"]["predicted_path"].as<std::string>();
  std::ifstream lin(base_path + "/"+new_idx_s+".label", std::ios::binary);
  // std::cout << base_path+new_seq_s+"/"+new_idx_s+".label" << std::endl;

  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int))) {
    if (c==2) {labels.push_back(NOT_TRAV_CELL_LABEL);}
    else if (c==1) labels.push_back(TRAV_CELL_LABEL);
    else if (c==3) labels.push_back(3); // sidewalk
    else labels.push_back(0);
  }

  // for (int i=0; i<20; i++) std::cout << labels[i] << std::endl;
}

void DataLoader_NuSc::readPredicted(int seq, int idx, YAML::Node &sample_data) {
  points.clear();
  labels.clear();
  std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;
  c = 0;
  // uint8_t c;
  std::ifstream fin(pts_s[idx], std::ios::binary);
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
         if (c==0) p(0) = f;
    else if (c==1) p(1) = f;
    else if (c==2) p(2) = f;
    else if (c==4) points.push_back(p);
    c = (c + 1) %5;
  }

  std::string base_path = sample_data["general"]["predicted_path"].as<std::string>();
  std::ifstream lin(base_path + "/"+new_idx_s+".label", std::ios::binary);
  // std::cout << base_path+new_seq_s+"/"+new_idx_s+".label" << std::endl;

  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int))) {
    if (c==2) {labels.push_back(NOT_TRAV_CELL_LABEL);}
    else if (c==1) labels.push_back(TRAV_CELL_LABEL);
    else if (c==3) labels.push_back(3); // sidewalk
    else labels.push_back(0);
  }

}


int DataLoader::count_samples(YAML::Node &sample_data, int seq) {

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

std::shared_ptr<const open3d::geometry::PointCloud> DataLoader_NuSc::getPaintedCloud() {
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