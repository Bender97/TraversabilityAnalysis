#include "ColorUtil.h"

Eigen::Vector3d darkgray(0.92f, 0.92f, 0.92f);
Eigen::Vector3d  darkred(0.545f, 0.0f, 0.0f);
Eigen::Vector3d limegreen(0.19f, 0.8f, 0.19f);

ColorUtil::ColorUtil() {}
ColorUtil_SemKITTI::ColorUtil_SemKITTI() : ColorUtil() {load_colors();}
ColorUtil_NuSC::ColorUtil_NuSC() : ColorUtil() {load_colors();}
ColorUtil_PandaSet::ColorUtil_PandaSet() : ColorUtil() {load_colors();}

void ColorUtil::load_colors() {}

void ColorUtil_SemKITTI::load_colors() {
  colors = std::vector<Color>(260);
  colors[0].r =  0;  colors[0].g = 0; colors[0].b = 0;        // "unlabeled"
  colors[1].r =  0;  colors[1].g = 0; colors[1].b = 255;      // "outlier"
  colors[10].r = 245; colors[10].g = 150; colors[10].b = 100;  // "car"
  colors[11].r = 245; colors[11].g = 230; colors[11].b = 100;  // "bicycle"
  colors[13].r = 250; colors[13].g = 80; colors[13].b = 100;   // "bus"
  colors[15].r = 150; colors[15].g = 60; colors[15].b = 30;    // "motorcycle"
  colors[16].r = 255; colors[16].g = 0; colors[16].b = 0;      // "on-rails"
  colors[18].r = 180; colors[18].g = 30; colors[18].b = 80;    // "truck"
  colors[20].r = 255; colors[20].g = 0; colors[20].b = 0;      // "other-vehicle"
  colors[30].r = 30;  colors[30].g = 30; colors[30].b = 255;    // "person"
  colors[31].r = 200; colors[31].g = 40; colors[31].b = 255;   // "bicyclist"
  colors[32].r = 90;  colors[32].g = 30; colors[32].b = 150;    // "motorcyclist"
  colors[40].r = 255; colors[40].g = 0; colors[40].b = 255;    // "road"
  colors[44].r = 255; colors[44].g = 150; colors[44].b = 255;  // "parking"
  colors[48].r = 75;  colors[48].g = 0; colors[48].b = 75;      // "sidewalk"
  colors[49].r = 75;  colors[49].g = 0; colors[49].b = 175;     // "other-ground"
  colors[50].r = 0;   colors[50].g = 200; colors[50].b = 255;    // "building"
  colors[51].r = 50;  colors[51].g = 120; colors[51].b = 255;   // "fence"
  colors[52].r = 0;   colors[52].g = 150; colors[52].b = 255;    // "other-structure"
  colors[60].r = 170; colors[60].g = 255; colors[60].b = 150;  // "lane-marking"
  colors[70].r = 0;   colors[70].g = 175; colors[70].b = 0;      // "vegetation"
  colors[71].r = 0;   colors[71].g = 60; colors[71].b = 135;     // "trunk"
  colors[72].r = 80;  colors[72].g = 240; colors[72].b = 150;   // "terrain"
  colors[80].r = 150; colors[80].g = 240; colors[80].b = 255;  // "pole"
  colors[81].r = 0;   colors[81].g = 0; colors[81].b = 255;      // "traffic-sign"
  colors[99].r = 255; colors[99].g = 255; colors[99].b = 50;   // "other-object"
  colors[252].r = 245; colors[252].g = 150; colors[252].b = 100; // "moving-car"
  colors[253].r = 200; colors[253].g = 40; colors[253].b = 255;  // "moving-bicyclist"
  colors[254].r = 30;  colors[254].g = 30; colors[254].b = 255;   // "moving-person"
  colors[255].r = 90;  colors[255].g = 30; colors[255].b = 150;   // "moving-motorcyclist"
  colors[256].r = 255; colors[256].g = 0; colors[256].b = 0;     // "moving-on-rails"
  colors[257].r = 250; colors[257].g = 80; colors[257].b = 100;  // "moving-bus"
  colors[258].r = 180; colors[258].g = 30; colors[258].b = 80;   // "moving-truck"
  colors[259].r = 255; colors[259].g = 0; colors[259].b = 0;     // "moving-other-vehicle"
}

void ColorUtil_PandaSet::load_colors() {
  colors = std::vector<Color>(43);
  colors[0].r =  0;  colors[0].g = 0; colors[0].b = 0;        // "unlabeled"
  colors[1].r =  0;  colors[1].g = 0; colors[1].b = 255;      // Smoke -> "outlier"
  colors[2].r =  0;  colors[2].g = 0; colors[2].b = 255;      // Exhaust -> "outlier"
  colors[3].r =  0;  colors[3].g = 0; colors[3].b = 255;      // Spray or rain -> "outlier"
  colors[4].r =  0;  colors[4].g = 0; colors[4].b = 255;      // Reflection -> "outlier"
  colors[5].r = 0;   colors[5].g = 175; colors[5].b = 0;      // "vegetation"
  colors[6].r = 75;  colors[6].g = 0; colors[6].b = 175;     // Ground -> "other-ground"
  colors[7].r = 255; colors[7].g = 0; colors[7].b = 255;    // "road"                                          X
  colors[8].r = 170; colors[8].g = 255; colors[8].b = 150;  // "Lane Line Marking" -> lane-marking             X
  colors[9].r = 170; colors[9].g = 255; colors[9].b = 150;  // "Stop Line Marking" -> lane-marking             X
  colors[10].r = 170; colors[10].g = 255; colors[10].b = 150;  // "Other Road Marking" -> lane-marking         X
  colors[11].r = 75;  colors[11].g = 0; colors[11].b = 75;      // "sidewalk"                                  X
  colors[12].r = 75;  colors[12].g = 0; colors[12].b = 75;      // Driveway -> "sidewalk"                      X
  colors[13].r = 245; colors[13].g = 150; colors[13].b = 100;  // "car"
  colors[14].r = 245; colors[14].g = 150; colors[14].b = 100;  // Pickup Truck -> "car"
  colors[15].r = 180; colors[15].g = 30; colors[15].b = 80;    // Medium-sized Truck -> "truck"
  colors[16].r = 180; colors[16].g = 30; colors[16].b = 80;    // Semi-truck -> "truck"
  colors[17].r = 255; colors[17].g = 0; colors[17].b = 0;      // Towed Object -> "other-vehicle"
  colors[18].r = 150; colors[18].g = 60; colors[18].b = 30;    // "motorcycle"
  colors[19].r = 255; colors[19].g = 0; colors[19].b = 0;      // Other Vehicle - Construction Vehicle ->"other-vehicle"
  colors[20].r = 255; colors[20].g = 0; colors[20].b = 0;      // Other Vehicle - Uncommon -> "other-vehicle"
  colors[21].r = 255; colors[21].g = 0; colors[21].b = 0;      // Other Vehicle - Pedicab -> "other-vehicle"
  colors[22].r = 245; colors[22].g = 150; colors[22].b = 100;  // Emergency Vehicle -> "car"
  colors[23].r = 180; colors[23].g = 30; colors[23].b = 80;    // Bus -> "truck"
  colors[24].r = 245; colors[24].g = 230; colors[24].b = 100;  // Personal Mobility Device -> "bicycle"
  colors[25].r = 150; colors[25].g = 60; colors[25].b = 30;    // Motorized Scooter -> "motorcycle"
  colors[26].r = 245; colors[26].g = 230; colors[26].b = 100;  // "bicycle"
  colors[27].r = 255; colors[27].g = 0; colors[27].b = 0;      // Train -> "other-vehicle"
  colors[28].r = 255; colors[28].g = 0; colors[28].b = 0;      // Trolley -> "other-vehicle"
  colors[29].r = 255; colors[29].g = 0; colors[29].b = 0;      // Tram / Subway -> "other-vehicle"
  colors[30].r = 30;  colors[30].g = 30; colors[30].b = 255;    // Pedestrian -> "person"
  colors[31].r = 30;  colors[31].g = 30; colors[31].b = 255;    // Pedestrian with Object -> "person"
  colors[32].r = 30;  colors[32].g = 30; colors[32].b = 255;    // Animals - Bird -> "person"
  colors[33].r = 30;  colors[33].g = 30; colors[33].b = 255;    // Animals - Other -> "person"
  colors[34].r = 150; colors[34].g = 240; colors[34].b = 255;  // Pylons -> "pole"
  colors[35].r = 50;  colors[35].g = 120; colors[35].b = 255;   // Road Barriers -> "fence"
  colors[36].r = 0;   colors[36].g = 0; colors[36].b = 255;      // Signs -> "traffic-sign"
  colors[37].r = 0;   colors[37].g = 0; colors[37].b = 255;      // Cones -> "traffic-sign"
  colors[38].r = 0;   colors[38].g = 0; colors[38].b = 255;      // Construction Signs -> "traffic-sign"
  colors[39].r = 50;  colors[39].g = 120; colors[39].b = 255;   // Temporary Construction Barriers -> "fence"
  colors[40].r = 255; colors[40].g = 255; colors[40].b = 50;   // Rolling Containers -> "other-object"
  colors[41].r = 0;   colors[41].g = 200; colors[41].b = 255;    // "building"
  colors[42].r = 255; colors[42].g = 255; colors[42].b = 50;   // Other Static Object -> "other-object"
}

void ColorUtil_NuSC::load_colors() {
  colors = std::vector<Color>(32);
  colors[0].r = 0;    colors[0].g = 0;    colors[0].b = 0;
  colors[1].r = 70;   colors[1].g = 130;  colors[1].b = 180;
  colors[2].r = 0;    colors[2].g = 0;    colors[2].b = 230;
  colors[3].r = 135;  colors[3].g = 206;  colors[3].b = 235;
  colors[4].r = 100;  colors[4].g = 149;  colors[4].b = 237;
  colors[5].r = 219;  colors[5].g = 112;  colors[5].b = 147;
  colors[6].r = 0;    colors[6].g = 0;    colors[6].b = 128;
  colors[7].r = 240;  colors[7].g = 128;  colors[7].b = 128;
  colors[8].r = 138;  colors[8].g = 43;   colors[8].b = 226;
  colors[9].r = 112;  colors[9].g = 128;  colors[9].b = 144;
  colors[10].r = 210; colors[10].g = 105;  colors[10].b = 30;
  colors[11].r = 105; colors[11].g = 105;  colors[11].b = 105;
  colors[12].r = 47;  colors[12].g = 79;   colors[12].b = 79;
  colors[13].r = 188; colors[13].g = 143;  colors[13].b = 143;
  colors[14].r = 220; colors[14].g = 20;   colors[14].b = 60;
  colors[15].r = 255; colors[15].g = 127;  colors[15].b = 80;
  colors[16].r = 255; colors[16].g = 69;   colors[16].b = 0;
  colors[17].r = 255; colors[17].g = 158;  colors[17].b = 0;
  colors[18].r = 233; colors[18].g = 150;  colors[18].b = 70;
  colors[19].r = 255; colors[19].g = 83;   colors[19].b = 0;
  colors[20].r = 255; colors[20].g = 215;  colors[20].b = 0;
  colors[21].r = 255; colors[21].g = 61;   colors[21].b = 99;
  colors[22].r = 255; colors[22].g = 140;  colors[22].b = 0;
  colors[23].r = 255; colors[23].g = 99;   colors[23].b = 71;
  colors[24].r = 0;   colors[24].g = 207;  colors[24].b = 191;
  colors[25].r = 175; colors[25].g = 0;    colors[25].b = 75;
  colors[26].r = 75;  colors[26].g = 0;    colors[26].b = 75;
  colors[27].r = 112; colors[27].g = 180;  colors[27].b = 60;
  colors[28].r = 222; colors[28].g = 184;  colors[28].b = 135;
  colors[29].r = 255; colors[29].g = 228;  colors[29].b = 196;
  colors[30].r = 0;   colors[30].g = 175;  colors[30].b = 0;
  colors[31].r = 255; colors[31].g = 240;  colors[31].b = 245;
}

void ColorUtil::setColor(Eigen::Vector3d &color_, int &label) {
  Color color = colors[label];
  color_(0) = color.b/255.0f;
  color_(1) = color.g/255.0f;
  color_(2) = color.r/255.0f;
}
void ColorUtil::setColor_DL(Eigen::Vector3d &color_, int &label) {
       if(label==1 || label==2) color_ = limegreen;
  else if(label==-1) color_ = darkred;
  else color_ = darkgray;
}