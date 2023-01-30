#pragma once
#include <iostream>

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