#include "PcaMode.h"

CylinderConfiguration::CylinderConfiguration() {}

CylinderConfiguration::CylinderConfiguration(YAML::Node &node) {

    if (!node["mode"])
        throw std::runtime_error(
            std::string("\033[1;31mERROR!\033[0m mode missing!") );
  
    mode_ = node["mode"].as<int>();

    switch(mode_) {
        case 0:
            mode = modes::geom;
            training_cols = TOT_GEOM_FEATURES;
            tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
            for (int i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
            break;
        case 1:
            mode = modes::geom_label;
            // if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
            training_cols = TOT_GEOM_FEATURES + level;
            tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
            for (int i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
            for (int l=1; l<=level; l++) re_idx.push_back(TOT_GEOM_FEATURES + (TOT_GEOM_FEATURES+1)*l -1);
            break;
        case 2:
            mode = modes::geom_all;
        case 3:
            mode = modes::geom_pca;
            training_cols = TOT_GEOM_FEATURES;
            tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
            for (int i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
            break;
        case 4:
            // if (level==0) throw std::runtime_error(std::string("\033[1;31mERROR!\033[0m mode at level 0 not allowed!"));
            training_cols = TOT_GEOM_FEATURES + level;
            tot_geom_features_across_all_levels = TOT_GEOM_FEATURES;
            for (int i=0; i<TOT_GEOM_FEATURES; i++) re_idx.push_back(i);
            for (int l=1; l<=level; l++) re_idx.push_back(TOT_GEOM_FEATURES + (TOT_GEOM_FEATURES+1)*l -1);
            break;
            mode = modes::geom_pca_label;
        case 5:
            mode = modes::geom_pca_all_label;
    }


    std::cout << "training in " << PcaMode[mode] << " mode  " << std::flush;
}

