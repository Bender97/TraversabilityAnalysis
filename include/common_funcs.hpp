#pragma once

#ifndef COMMON_FUNCS_HPP
#define COMMON_FUNCS_HPP

#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>
#include <stdexcept>
#include <Eigen/Dense>
#include "Cell.h"
#include "Cylinder.h"
#include "Synchro.h"


#include <sys/types.h>
#include <dirent.h>

#include "ColorUtil.h"
#include "common_macro.hpp"
#include "yaml-cpp/yaml.h"


void readData(int seq, int idx, std::vector<Eigen::Vector3d> &points, std::vector<int> &labels, YAML::Node &sample_data) {
  points.clear();
  labels.clear();
  std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  float f;
  int counter=0;
  Eigen::Vector3d p;

  int c;
  std::ifstream fin(sample_data["general"]["dataset_path"].as<std::string>()+"sequences/"+new_seq_s+"/velodyne/"+new_idx_s+".bin", std::ios::binary);
  //std::ifstream fin("/home/fusy/repos/trav_analysis_2/simulated_cloud/mixed.bin", std::ios::binary);
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    
         if (counter==0) p(0) = f;
    else if (counter==1) p(1) = f;
    else if (counter==2) p(2) = f;
    else      points.push_back(p);

    counter = (counter + 1) %4;
  }

  std::ifstream lin(sample_data["general"]["dataset_path"].as<std::string>()+"sequences/"+new_seq_s+"/labels/"+new_idx_s+".label", std::ios::binary);
  //std::ifstream lin("/home/fusy/repos/trav_analysis_2/simulated_cloud/labels.label", std::ios::binary);
  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int)))
    labels.push_back(c & 0xFFFF);

}

void readDataNu(std::string pts_s, std::string lab_s, std::vector<Eigen::Vector3d> &points, std::vector<int> &labels, YAML::Node &sample_data) {
  points.clear();
  labels.clear();

  float f;
  int counter=0;
  Eigen::Vector3d p;

  uint8_t c;
  std::ifstream fin(pts_s, std::ios::binary);
  //std::ifstream fin("/home/fusy/repos/trav_analysis_2/simulated_cloud/mixed.bin", std::ios::binary);
  
  while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    
         if (counter==0) p(0) = f;
    else if (counter==1) p(1) = f;
    else if (counter==2) p(2) = f;
    else if (counter==4)      points.push_back(p);

    counter = (counter + 1) %5;
    
  }

  std::ifstream lin(lab_s, std::ios::binary);
  //std::ifstream lin("/home/fusy/repos/trav_analysis_2/simulated_cloud/labels.label", std::ios::binary);
  while (lin.read(reinterpret_cast<char*>(&c), sizeof(uint8_t)))
    labels.push_back((uint32_t) c);

}

// void readPredicted(int seq, int idx, std::vector<int> &labels, std::string base_path) {
//   labels.clear();
//   std::string seq_s = std::to_string(seq);
//   std::string idx_s = std::to_string(idx);
//   auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
//   auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

//   int c;
//   std::ifstream lin(base_path+new_seq_s+"/"+new_idx_s+".label", std::ios::binary);

//   while (lin.read(reinterpret_cast<char*>(&c), sizeof(int)))
//     labels.push_back(c & 0xFFFF);
//   // for (int i=0; i<20; i++) std::cout << labels[i] << std::endl;
// }

void readPredicted(int seq, int idx, std::vector<int> &labels, std::string base_path) {
  labels.clear();
  std::string seq_s = std::to_string(seq);
  std::string idx_s = std::to_string(idx);
  auto new_seq_s = std::string(2 - MIN(2, seq_s.length()), '0') + seq_s;
  auto new_idx_s = std::string(6 - MIN(6, idx_s.length()), '0') + idx_s;

  int c;
  std::ifstream lin(base_path + "/"+new_idx_s+".label", std::ios::binary);
  // std::ifstream lin(base_path+new_idx_s+".label", std::ios::binary);

  std::cout << base_path+new_seq_s+"/"+new_idx_s+".label" << std::endl;

  //while (lin.read(reinterpret_cast<char*>(&c), sizeof(int)))
  //  labels.push_back(c & 0xFFFF);

  while (lin.read(reinterpret_cast<char*>(&c), sizeof(int))) {
    if (c==2) {labels.push_back(NOT_TRAV_CELL_LABEL);}
    else if (c==1) labels.push_back(TRAV_CELL_LABEL);
    else if (c==3) labels.push_back(3); // sidewalk
    else labels.push_back(0);
  }

  // for (int i=0; i<20; i++) std::cout << labels[i] << std::endl;
}

void loadCyls(std::vector<Cylinder> &cyls, YAML::Node &sample_data, bool produce_features=false) {
  cyls.clear();
  int tot_geom_features = sample_data["tot_geom_features"].as<int>();
  int level;
  for (level=0; ; level++) {
    auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(level).length()), '0') + std::to_string(level);
    YAML::Node node = sample_data["general"][cyl_s.c_str()];
    if (!node) break;
    
    Cylinder *back_cyl = (level>0) ? &(cyls[level-1]) : nullptr;
    
    Cylinder cyl = Cylinder(node, back_cyl, tot_geom_features, produce_features);

    cyls.push_back(cyl);
  }

  if (!level) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. please provide"
                  " at least a cylinder in yaml config file.\n"));
}


void loadCyls(std::vector<Cylinder> &cyls, Synchro *synchro_, YAML::Node &sample_data, bool produce_features=false) {
  cyls.clear();
  int tot_geom_features = sample_data["tot_geom_features"].as<int>();
  int level;
  for (level=0; ; level++) {
    auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(level).length()), '0') + std::to_string(level);
    YAML::Node node = sample_data["general"][cyl_s.c_str()];
    if (!node) break;

    node["dataset"] = sample_data["general"]["dataset"];
    
    Cylinder *back_cyl = (level>0) ? &(cyls[level-1]) : nullptr;
    
    Cylinder cyl = Cylinder(node, synchro_, back_cyl, tot_geom_features, produce_features);    
    cyls.push_back(cyl);
  }

  if (!level) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. please provide"
                  " at least a cylinder in yaml config file.\n"));
}


#ifndef TRAIN
Eigen::MatrixXd computeSceneNormal(std::vector<Eigen::Vector3d> &points) {
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
  if (eigensolver.info() != Eigen::Success) { /*std::cout << " error in computing scene normal" << std::endl;*/ return Eigen::MatrixXd::Zero(3, 3);}

  return eigensolver.eigenvectors().col(0); // because eigenvalues are sorted in increasing order
}

void computeCorrelationMatrix(float &cx, float &cy, float &cz, Eigen::MatrixXd &matA1, std::vector<int> &points_idx, std::vector<Eigen::Vector3d> &points) {
    
    cx=0; cy=0; cz=0;  
    float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
    
    int numpoints = points_idx.size();
    Eigen::Vector3d *p;

    // compute eigenvalues and eigenvectors
    matA1 = Eigen::MatrixXd::Zero(3, 3);

    for (auto p_idx: points_idx) {
        p = &(points[p_idx]);
        cx += (*p)(0);
        cy += (*p)(1);
        cz += (*p)(2);
    }
    cx /= (float) numpoints; cy /= (float) numpoints;  cz /= (float) numpoints;

    float ax, ay, az;

    for (auto p_idx: points_idx) {
        p = &(points[p_idx]);
        ax = (*p)(0) - cx;
        ay = (*p)(1) - cy;
        az = (*p)(2) - cz;

        a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
        a22 += ay * ay; a23 += ay * az;
        a33 += az * az;
    }
    a11 /= (float) numpoints; a12 /= (float) numpoints; a13 /= (float) numpoints; a22 /= (float) numpoints; a23 /= (float) numpoints; a33 /= (float) numpoints;

    matA1 << a11, a12, a13, a12, a22, a23, a13, a23, a33;
}

void computeGridGroundTruth(std::vector<Cell> &grid, std::vector<int> &labels) {
  int trav_cont, non_trav_cont;

  int road, sidewalk;

  for (auto &cell : grid) {
    if (cell.points_idx.size() < 2) {
      cell.label = UNKNOWN_CELL_LABEL;
      continue;
    }

    trav_cont = 0;
    non_trav_cont = 0;
    road=0;
    sidewalk=0;


    int label;

    for (auto &p_idx : cell.points_idx) {
      label = labels[p_idx];
      if (POINTBELONGSTOROAD(label)) trav_cont ++;
      else if (LABELED(label)) non_trav_cont ++;

      if (label==40) road++;
      else if (label==48) sidewalk++;

    }

    if (road>0 && sidewalk>0) cell.label=NOT_TRAV_CELL_LABEL;
    else {
        if (non_trav_cont > 3 )
          cell.label = NOT_TRAV_CELL_LABEL;
        else if (trav_cont > 1)
          cell.label = TRAV_CELL_LABEL;
        else {
          cell.label = UNKNOWN_CELL_LABEL;
        }
    }
  }
}
#endif

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


void paintCloud_cyl(open3d::geometry::PointCloud &cloud, std::vector<int> &labels) {

  if (cloud.points_.size() != labels.size()) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. cloud and labels differ in size.\n"));

  ColorUtil color_util;

  cloud.colors_.resize(cloud.points_.size());
  for (int i=0; i<(int)cloud.points_.size(); i++) {
    color_util.setColor(cloud.colors_[i], labels[i]);
  }
}

void paintCloud_cyl_NuDL(open3d::geometry::PointCloud &cloud, std::vector<int> &labels) {

  if (cloud.points_.size() != labels.size()) throw std::runtime_error(
                 std::string("\033[1;31mERROR\033[0m. cloud and labels differ in size.\n"));

  Eigen::Vector3d      red(1.0f, 0.0f, 0.0f);
  Eigen::Vector3d  darkred(0.545f, 0.0f, 0.0f);
  Eigen::Vector3d lightred(1.0f, 0.7f, 0.7f);
  Eigen::Vector3d    white(1.0f, 1.0f, 1.0f);
  Eigen::Vector3d darkgray(0.92f, 0.92f, 0.92f);
  Eigen::Vector3d road(0.7f, 0.7f, 0.5f);
  Eigen::Vector3d limegreen(0.19f, 0.8f, 0.19f);
  Eigen::Vector3d yellow(0.94f, 0.91f, 0.64f);
  Eigen::Vector3d darkorange(1.0f, 0.54f, 0.0f);
  Eigen::Vector3d blue(0.0f, 0.0f, 1.0f);

  int l0=0, l1=0, lm1=0, lo=0;

  cloud.colors_.resize(cloud.points_.size());
  for (int i=0; i<(int)cloud.points_.size(); i++) {
    int l = labels[i];
    //if (l==2) {cloud.colors_[i] = yellow; l0++;}
    if(l==1 || l==2) {cloud.colors_[i] = limegreen;l1++;}
    else if(l==-1) {cloud.colors_[i] = darkred;lm1++;}
    else {cloud.colors_[i] = darkgray;lo++;}
  }

  std::cout << "0 : " << l0 << std::endl;
  std::cout << "1 : " << l1 << std::endl;
  std::cout << "-1: " << lm1 << std::endl;
  std::cout << "2 : " << lo << std::endl;

}

void paintCloud_cylB(open3d::geometry::PointCloud &cloud) {

  Eigen::Vector3d    white(0.5f, 0.5f, 0.5f);
  cloud.colors_.resize(cloud.points_.size(), white);
  
}


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

#endif