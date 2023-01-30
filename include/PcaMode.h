#include <iostream>
#include <stdexcept>
#include "yaml-cpp/yaml.h"
#include "common_macro.hpp"

class CylinderConfiguration {
    enum class modes {geom, geom_label, geom_all, geom_pca, geom_pca_label, geom_pca_all_label};
    CylinderConfiguration();
    CylinderConfiguration(YAML::Node &node);
    int training_cols, tot_geom_features_across_all_levels;

    modes mode;
};