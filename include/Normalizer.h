#pragma once

#include <iomanip>
#include <fstream>
#include <iostream>

#include <vector>
#include <cfloat>
#include "yaml-cpp/yaml.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

class Normalizer {

public:

    std::vector<float> min, max, p2p;
    std::vector<float> avg, var;
    int tot_geom_features;

    void init(int tot_geom_features_);

    Normalizer(int tot_geom_features_);

    Normalizer();

    Normalizer(int tot_geom_features_, std::string fileName);

    bool empty();

    void computeStatistics(cv::Mat &X_train);

    void normalize(cv::Mat &X_train);
    void normalize(cv::Mat &X_train, int valid_rows);

    void normalize_v(cv::Mat &X_train);

    void normalize_train_and_store(cv::Mat &X_train, std::string fileName);

    void normalize_predict(cv::Mat &X_train, std::string fileName);

    void toFile(std::string fileName);

    void loadConfig(std::string fileName);

    void print();

    int check4NormalizationErrors(cv::Mat &X_train);
};