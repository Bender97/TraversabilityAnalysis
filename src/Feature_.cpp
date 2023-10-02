#include "Feature.h"

void Feature::computeCorrelationMatrix(std::vector<int> &points_idx, std::vector<Eigen::Vector3d> &points) {
    
    cx=0; cy=0; cz=0;  
    a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
    
    // compute eigenvalues and eigenvectors
    matA1 = Eigen::MatrixXd::Zero(3, 3);

    for (auto p_idx: points_idx) {
        p = &(points[p_idx]);
        cx += (*p)(0);
        cy += (*p)(1);
        cz += (*p)(2);
    }
    cx *= numpoints_inverse; cy *= numpoints_inverse;  cz *= numpoints_inverse;

    for (auto p_idx: points_idx) {
        p = &(points[p_idx]);
        ax = (*p)(0) - cx;
        ay = (*p)(1) - cy;
        az = (*p)(2) - cz;

        a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
        a22 += ay * ay; a23 += ay * az;
        a33 += az * az;
    }
    a11 *= numpoints_inverse; a12 *= numpoints_inverse; a13 *= numpoints_inverse; 
    a22 *= numpoints_inverse; a23 *= numpoints_inverse; a33 *= numpoints_inverse;

    matA1 << a11, a12, a13, a12, a22, a23, a13, a23, a33;
}

std::string Feature::toString() {
  std::stringstream ss("");

  ss << std::to_string(linearity) << " ";
  ss << std::to_string(planarity) << " ";
  ss << std::to_string(sphericity) << " ";
  ss << std::to_string(omnivariance) << " ";
  ss << std::to_string(anisotropy) << " ";
  ss << std::to_string(eigenentropy) << " ";
  ss << std::to_string(sum_of_eigenvalues) << " ";
  ss << std::to_string(curvature) << " ";
  ss << std::to_string(angle) << " ";
  ss << std::to_string(goodness_of_fit) << " ";
  ss << std::to_string(roughness) << " ";
  ss << std::to_string(nvx) << " ";
  ss << std::to_string(nvy) << " ";
  ss << std::to_string(nvz) << " ";
  ss << std::to_string(unevenness) << " ";
  ss << std::to_string(surface_density) << " ";
  ss << std::to_string(z_diff) << " ";

  ss << " derived: ";
  for (int i=0; i<(int)derived_features.size(); i++)
    ss << " " << std::to_string(derived_features[i]);

  return ss.str();
}

std::vector<float> Feature::toVector() {
  std::vector<float> feature = std::vector<float>(TOT_GEOM_FEATURES + (int)derived_features.size());
  feature[0] = linearity;
  feature[1] = planarity;
  feature[2] = sphericity;
  feature[3] = omnivariance;
  feature[4] = anisotropy;
  feature[5] = eigenentropy;
  feature[6] = sum_of_eigenvalues;
  feature[7] = curvature;
  feature[8] = angle;
  feature[9] = goodness_of_fit;
  feature[10] = roughness;
  feature[11] = nvx;
  feature[12] = nvy;
  feature[13] = nvz;
  feature[14] = unevenness;
  feature[15] = surface_density;
  feature[16] = z_diff;

  for (int i=0; i<(int)derived_features.size(); i++)
    feature[TOT_GEOM_FEATURES+i] = derived_features[i];
  return feature;
}

std::vector<float> Feature::toVectorTransformed() {
  std::vector<float> feature = toVector();
  for (int i=0; i<TOT_GEOM_FEATURES; i++) // TOT_GEOM_FEATURES because derived feature are already transformed
    feature[i] = (double)log( fabs((double)feature[i]) + 1e-4 );
  return feature;
}

void Feature::toVectorTransformed(std::vector<float> &feature) {
  // feature[0] = linearity;
  // feature[1] = planarity;
  // feature[2] = sphericity;
  // feature[3] = omnivariance;
  // feature[4] = anisotropy;
  // feature[5] = eigenentropy;
  // feature[6] = sum_of_eigenvalues;
  // feature[7] = curvature;
  // feature[8] = angle;
  // feature[9] = goodness_of_fit;
  // feature[10] = roughness;
  // feature[11] = nvx;
  // feature[12] = nvy;
  // feature[13] = nvz;
  // feature[14] = unevenness;
  // feature[15] = surface_density;
  // feature[16] = z_diff;
  // for (int i=0; i<(int)derived_features.size(); i++)
  //   feature[TOT_GEOM_FEATURES+i] = derived_features[i];
  // for (int i=0; i<TOT_GEOM_FEATURES; i++) // TOT_GEOM_FEATURES because derived feature are already transformed
  //   feature[i] = log( abs(feature[i]) + 1e-4 );
}

void Feature::toFile(std::ofstream &out) {
  out.write( reinterpret_cast<const char*>( &(linearity) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(planarity) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(sphericity) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(omnivariance) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(anisotropy) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(eigenentropy) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(sum_of_eigenvalues) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(curvature) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(angle) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(goodness_of_fit) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(roughness) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(nvx) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(nvy) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(nvz) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(unevenness) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(surface_density) ), sizeof( float ));
  out.write( reinterpret_cast<const char*>( &(z_diff) ), sizeof( float ));
  
  for (int i=0; i<(int)derived_features.size(); i++)
    out.write( reinterpret_cast<const char*>( &(derived_features[i]) ), sizeof( float ));

}

int Feature::fromFileLine(std::ifstream &in, int derived_features_num) {
  if (!in.read( reinterpret_cast< char*>( &(linearity) ), sizeof( float ))) return 0;
    in.read( reinterpret_cast< char*>( &(planarity) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(sphericity) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(omnivariance) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(anisotropy) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(eigenentropy) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(sum_of_eigenvalues) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(curvature) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(angle) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(goodness_of_fit) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(roughness) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(nvx) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(nvy) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(nvz) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(unevenness) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(surface_density) ), sizeof( float ));
    in.read( reinterpret_cast< char*>( &(z_diff) ), sizeof( float ));

    if (derived_features_num<=0) return 1;
    derived_features.resize(derived_features_num); 
    for (int i=0; i<derived_features_num; i++)
      in.read( reinterpret_cast< char*>( &(derived_features[i]) ), sizeof( float ));


  return 1;
}

int Feature::ignoreFeatureFromFile(std::ifstream &in, int derived_features_num) {
  if (!in.read( reinterpret_cast< char*>( &(linearity) ), sizeof( float ))) return 0;
  
  in.ignore(sizeof(float)*16);
  if (derived_features_num<=0) return 1;
  
  in.ignore(sizeof(float)*derived_features_num);

  return 1;
}


int Feature::computeFeatures(Cell *cell, Eigen::MatrixXd &scene_normal, std::vector<Eigen::Vector3d> &points) {
  cx = 0; cy = 0; cz = 0;
  numpoints = cell->points_idx.size();
  numpoints_inverse = 1.0f/numpoints;

  float scene_normal_2_inverse = 1.0f / scene_normal(2);
  
  computeCorrelationMatrix(cell->points_idx, points);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matA1);
  if (eigensolver.info() != Eigen::Success) {
    // std::cout << "eigen solver info: not success\n";
    return 0;
  }
  // store eigenvalues in an easy way
  d1 = eigensolver.eigenvalues()(0);
  d2 = eigensolver.eigenvalues()(1);
  d3 = eigensolver.eigenvalues()(2);

  if (d1<1e-16)  {
    //  std::cout << "d1 " << cell->points_idx.size() << " " << d1 << " " << d2 << " " << d3 << "\n";
    //return 0;
    d1=1e-16;
  }

  if (d2<1e-16) d2=1e-16;

  if (d3<1e-16) d3=1e-16;

  nvx = eigensolver.eigenvectors()(0, 0);
  nvy = eigensolver.eigenvectors()(1, 0);
  nvz = eigensolver.eigenvectors()(2, 0);

  float d1_inverse = 1.0f / d1;           // @Optimized

  /// COVARIANCE-BASED
  linearity    = (d1 - d2)  * d1_inverse; // @Optimized
  planarity    = (d2 - d3)  * d1_inverse; // @Optimized
  sphericity   = d3  * d1_inverse;        // @Optimized
  omnivariance = std::cbrt(d1*d2*d3);
  anisotropy   = (d1 - d3) * d1_inverse;  // @Optimized
  eigenentropy = - ( d1*std::log(d1) + d2*std::log(d2) + d3*std::log(d3) );
  sum_of_eigenvalues = d1 + d2 + d3;
  curvature    = d3 / sum_of_eigenvalues;

  d = - ( nvx * cx + nvy * cy + nvz * cz );
  normal_magnitude = eigensolver.eigenvectors().col(0).norm();
  if (!normal_magnitude) {
    // std::cout << "normal_magnitude\n";
    return 0;
  }

  /// ROUGHNESS-BASED
  angle = std::acos(nvz);    // normal.z = normal.x*0 + normal.y*0 + normal.z*1, with z = (0, 0, 1)
  
  float min = (points[cell->points_idx[0]])(2), max = (points[cell->points_idx[0]])(2);

  float normal_magnitude_inverse = 1.0f / normal_magnitude;  // @Optimized
  
  for (auto p_idx : cell->points_idx) {
      p = &(points[p_idx]);
      goodness_of_fit +=
              std::abs((nvx*(*p)(0) + nvy*(*p)(1) + nvz*(*p)(2)) + d) * normal_magnitude_inverse;
      
      roughness += NUM2SQ((*p)(2) - cz);

      // float local_d = - SCALAR_PRODUCT_2p(scene_normal, (*p));
      // float z = - (scene_normal(0)*cx + scene_normal(1)*cy - NUM2SQ(scene_normal(0))/scene_normal(2)*cz
      //              - NUM2SQ(scene_normal(1))/scene_normal(2)*cz + local_d)
      //           / (NUM2SQ(scene_normal(0))/scene_normal(2) + NUM2SQ(scene_normal(1))/scene_normal(2) + scene_normal(2));
      float local_d = - SCALAR_PRODUCT_2p(scene_normal, (*p));
      float z = - (scene_normal(0)*cx + scene_normal(1)*cy - NUM2SQ(scene_normal(0))*scene_normal_2_inverse*cz
                   - NUM2SQ(scene_normal(1))*scene_normal_2_inverse*cz + local_d)
                / (NUM2SQ(scene_normal(0))*scene_normal_2_inverse + NUM2SQ(scene_normal(1))*scene_normal_2_inverse + scene_normal(2));
      if (z>max) max = z;
      else if (z<min) min = z;
  }
  roughness *= numpoints_inverse;
  unevenness = normal_magnitude * numpoints_inverse;
  surface_density = numpoints * cell->area_inverse;

  /// Z_DIFF feature
  z_diff = max-min;

  return 1;
}
