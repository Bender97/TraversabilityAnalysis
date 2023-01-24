#pragma once

#include <iostream>
#include <iomanip>
#include "Metric.h"

class Results {
public:
  float nu, C, gamma;
  Metric tm, vm;
  Results(float nu_, float C_, float gamma_, Metric &tm_, Metric &vm_);
  bool operator> (Results& r) const;
  bool operator< (Results& r) const;
  friend std::ostream& operator<<(std::ostream& os, Results& r);
};