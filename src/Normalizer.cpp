#include "Normalizer.h"

void Normalizer::init(int tot_geom_features_) {
    tot_geom_features = tot_geom_features_;
    min.resize(tot_geom_features);
    max.resize(tot_geom_features);
    p2p.resize(tot_geom_features);
    avg.resize(tot_geom_features);
    var.resize(tot_geom_features);
}

Normalizer::Normalizer() {}

Normalizer::Normalizer(int tot_geom_features_) {
    init(tot_geom_features_);
}

Normalizer::Normalizer(int tot_geom_features_, std::string fileName) {
    init(tot_geom_features_);
    loadConfig(fileName);
}

bool Normalizer::empty() {return min.empty();}

void Normalizer::computeStatistics(cv::Mat &X_train) {

    std::vector<float> sqavg(tot_geom_features, .0f); // avg & var
    int maxint = 100000, i, row;
    float *row_ptr;
    float temp;
    
    // initialize values to zero, or for min/max
    for (i=0; i<tot_geom_features; i++) {
        max[i] = FLT_MIN;
        min[i] = FLT_MAX;
        p2p[i] = avg[i] = var[i] = .0f;
    }

    // find min and max of each features, and avg/sqavg pre-proc values
    for ( row=0; row<X_train.rows; row++) {
        row_ptr = X_train.ptr<float>(row);
        for ( i=0; i<tot_geom_features; i++) {
            if (row_ptr[i]>max[i]) max[i] = row_ptr[i];
            if (row_ptr[i]<min[i]) min[i] = row_ptr[i];

                temp  = row_ptr[i];
              avg[i] += temp;
            sqavg[i] += (temp * temp);
        }
    }

    for ( i=0; i<tot_geom_features; i++) {
        // compute the peek to peek distance for each column
        p2p[i] = max[i] - min[i];

        // sanitize
        min[i] = ( std::floor(min[i]*maxint))     / (float)maxint;
        max[i] = (  std::ceil(max[i]*maxint))     / (float)maxint;
        p2p[i] = (  std::ceil(p2p[i]*maxint) + 1) / (float)maxint;
        
        // actually average values
        avg[i]   /= X_train.rows;
        sqavg[i] /= X_train.rows;

        // compute variance
        if (sqavg[i] - (avg[i] * avg[i]) <= 0) var[i]=1e-8;
        else var[i] = std::sqrt(sqavg[i] - (avg[i] * avg[i]));
    }

}

void Normalizer::normalize(cv::Mat &X_train) {
    float *row_ptr;
    for (int row = 0; row < X_train.rows; row++) {
        row_ptr = X_train.ptr<float>(row);
        for (int i=0; i < tot_geom_features; i++)
            row_ptr[i] = (row_ptr[i] - avg[i]) / var[i];
    }
}

void Normalizer::normalize(cv::Mat &X_train, int valid_rows) {
    float *row_ptr;
    int i;
    
    for (int row = 0; row < valid_rows; row++) {
        row_ptr = X_train.ptr<float>(row);
        for (i=0; i < tot_geom_features; i++)
            row_ptr[i] = (row_ptr[i] - avg[i]) / var[i];
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

    std::ofstream out(fileName.c_str());

    YAML::Emitter outyaml(out);

    outyaml << YAML::BeginMap;

    outyaml << YAML::Key << "min" << YAML::Flow << min;
    outyaml << YAML::Key << "max" << YAML::Flow << max;
    outyaml << YAML::Key << "p2p" << YAML::Flow << p2p;
    outyaml << YAML::Key << "avg" << YAML::Flow << avg;
    outyaml << YAML::Key << "var" << YAML::Flow << var;

    outyaml << YAML::EndMap;

    out.close();
}

void Normalizer::loadConfig(std::string fileName) {

    YAML::Node sample_data = YAML::LoadFile(fileName);

    min = sample_data["min"].as<std::vector<float>>();
    max = sample_data["max"].as<std::vector<float>>();
    p2p = sample_data["p2p"].as<std::vector<float>>();
    avg = sample_data["avg"].as<std::vector<float>>();
    var = sample_data["var"].as<std::vector<float>>();

}

void Normalizer::print() {
    int i;
    std::cout << "min: ";
    for (i=0; i<tot_geom_features; i++) std::cout << min[i] << " ";
    std::cout << std::endl << "max: ";
    for (i=0; i<tot_geom_features; i++) std::cout << max[i] << " ";
    std::cout << std::endl << "p2p: ";
    for (i=0; i<tot_geom_features; i++) std::cout << p2p[i] << " ";
    std::cout << std::endl;
}


int Normalizer::check4NormalizationErrors(cv::Mat &X_train) {
  /*std::cout << " checking for correct normalization ..." << std::flush;
  std::vector<float> *row;
  float val;
  int cols = 19;//(int) normalized_data[0].size();
  for (int r=0; r<(int) normalized_data.size(); r++) {
    bool flag = false;
    row = &(normalized_data[r]);

    for (int c=0; c<cols; c++) {
      val = (*row)[c];
      if (val < -1.0f || val > 1.0f) {
        flag = true;
        std::cout << " ERROR: data (" << std::setprecision(20) << val << ") not normalized at row "  << r << std::endl;
        break;
      }
    }
    if (flag) {
      for (int c=0; c<cols; c++)
        std::cout << (*row)[c] << " ";
      std::cout << std::endl;
      return 0;
    }
  }
  std::cout << "done!" << std::endl;*/
  return 1;
}