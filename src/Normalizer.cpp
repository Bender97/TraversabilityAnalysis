#include "Normalizer.h"

void Normalizer::init(int tot_geom_features_) {
    tot_geom_features = tot_geom_features_;
    avg.resize(tot_geom_features);
    var.resize(tot_geom_features);
    var_inv.resize(tot_geom_features);
}

Normalizer::Normalizer() {}

Normalizer::Normalizer(int tot_geom_features_) {
    init(tot_geom_features_);
}

Normalizer::Normalizer(int tot_geom_features_, std::string fileName) {
    init(tot_geom_features_);
    loadConfig(fileName);
}

bool Normalizer::empty() {return avg.empty();}

void Normalizer::computeStatistics(cv::Mat &X_train) {

    std::cout << "computing statistics over X_train with " << X_train.rows << " rows\n";

    std::vector<float> sqavg(tot_geom_features, .0f); // avg & var
    int i, row;
    float *row_ptr;
    float temp;
    
    // initialize values to zero, or for min/max
    for (i=0; i<tot_geom_features; i++)
        avg[i] = var[i] = .0;


    for ( row=0; row<X_train.rows; row++) {
        row_ptr = X_train.ptr<float>(row);
        for ( i=0; i<tot_geom_features; i++)
              avg[i] += (double) row_ptr[i];
    }

    for ( i=0; i<tot_geom_features; i++)
        avg[i]   = avg[i] / X_train.rows;

    for ( row=0; row<X_train.rows; row++) {
        row_ptr = X_train.ptr<float>(row);
        for ( i=0; i<tot_geom_features; i++) {
            temp  = row_ptr[i] - avg[i];
            var[i] += (temp * temp);
        }
    }

    for ( i=0; i<tot_geom_features; i++) {
        var[i] = std::sqrt(var[i] / X_train.rows );
        var_inv[i] = 1.0f / var[i];
    }

}

void Normalizer::normalize(cv::Mat &X_train) {
    float *row_ptr;
    for (int row = 0; row < X_train.rows; row++) {
        row_ptr = X_train.ptr<float>(row);
        for (int i=0; i < tot_geom_features; i++)
            row_ptr[i] = (row_ptr[i] - avg[i]) * var_inv[i];
    }
}

void Normalizer::normalize(cv::Mat &X_train, int valid_rows) {
    float *row_ptr;
    int i;
    
    for (int row = 0; row < valid_rows; row++) {
        row_ptr = X_train.ptr<float>(row);
        for (i=0; i < tot_geom_features; i++)
            row_ptr[i] = (row_ptr[i] - avg[i]) * var_inv[i];
    }
}

void Normalizer::normalize_train_and_store(cv::Mat &X_train, std::string fileName) {
    std::string n = fileName;
    computeStatistics(X_train);
    normalize(X_train);
    toFile(fileName);
}

void Normalizer::normalize_predict(cv::Mat &X_train, std::string fileName) {
    loadConfig(fileName);
    normalize(X_train);
}

void Normalizer::toFile(std::string fileName) {

    std::ofstream out(fileName.c_str(), std::ios::binary);

    uint32_t size= (uint32_t) avg.size();
    std::cout << "written size " << size << "\n";
    out.write( reinterpret_cast<const char*>( &(size) ), sizeof( uint32_t ));
    for (size_t i=0; i<(size_t)tot_geom_features; ++i) {
      double avg_ = avg[i];
      out.write( reinterpret_cast<const char*>( &(avg_) ), sizeof( double ));
    }
    for (size_t i=0; i<(size_t)tot_geom_features; ++i) {
      double var_ = var[i];
      out.write( reinterpret_cast<const char*>( &(var_) ), sizeof( double ));
    }
    out.close();
}

void Normalizer::loadConfig(std::string fileName) {

    std::ifstream in_(fileName.c_str(), std::ios::in | std::ios::binary);

    uint32_t size;
    in_.read( reinterpret_cast<char*>( &(size) ), sizeof( uint32_t ));
    size=tot_geom_features;

    avg = std::vector<double>(size);
    var = std::vector<double>(size);
    
    for (size_t i=0; i<size; ++i) {
      double avg_;
      in_.read( reinterpret_cast<char*>( &(avg_) ), sizeof( double ));
      avg[i] = avg_;
    }

    for (size_t i=0; i<size; ++i) {
      double var_;
      in_.read( reinterpret_cast<char*>( &(var_) ), sizeof( double ));
      var[i] = var_;
      var_inv[i] = 1.0f/var[i];
    }
    in_.close();

}

void Normalizer::print() {
    int i;
    std::cout << "avg: ";
    for (i=0; i<tot_geom_features; i++) std::cout << avg[i] << " ";
    std::cout << std::endl << "var: ";
    for (i=0; i<tot_geom_features; i++) std::cout << var[i] << " ";
}


int Normalizer::check4NormalizationErrors(cv::Mat &X_train) {
  int row, col;
  double mean[tot_geom_features], stds[tot_geom_features];

  double mean_radius = 0.01;
  double std_radius = 0.01;

  for ( col=0; col<tot_geom_features; col++) {
        mean[col] = 0;
        stds[col] = 0;
  }
  for ( row=0; row<X_train.rows; row++) {
    float* row_ptr = X_train.ptr<float>(row);
    for ( col=0; col<tot_geom_features; col++) {
          mean[col] += (double) row_ptr[col];
    }
  }

  for ( col=0; col<tot_geom_features; col++) {
      mean[col]   = mean[col] / (double) X_train.rows;
      if (std::abs(mean[col]) > mean_radius) return 0;
  }

  for ( row=0; row<X_train.rows; row++) {
    float* row_ptr = X_train.ptr<float>(row);
    for ( col=0; col<tot_geom_features; col++) {
        double temp  = (double) row_ptr[col] - (double) mean[col];
        stds[col] += (temp * temp);
    }
  }

  for ( col=0; col<tot_geom_features; col++) {
    stds[col] = std::sqrt(stds[col] / X_train.rows );
    if (std::abs(stds[col]) > std_radius) return 0;
  }
  
  return 1;
}