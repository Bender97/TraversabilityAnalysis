#include "Results.h"

Results::Results(float nu_, float C_, float gamma_, Metric &tm_, Metric &vm_) {
nu = nu_; C = C_; gamma = gamma_;
tm = tm_; vm = vm_;
}
bool Results::operator> (Results& r) const {
return (vm.acc() > r.vm.acc());
}
bool Results::operator< (Results& r) const {
return (vm.acc() > r.vm.acc());
}

std::ostream& operator<<(std::ostream& os, Results& r) {
  os << "nu: " << r.nu << " C: " << r.C << " gamma: " << r.gamma << std::endl;

  os << "\033[4;37mvalid\033[0m:  TP:" << std::setw(8) << std::setprecision(4) << r.vm.avgTP() 
     << " FN:" << std::setw(8) << std::setprecision(4) << r.vm.avgFN() 
     << "  \033[4;37mtrain\033[0m:  TP:" << std::setw(8) << std::setprecision(4) << r.tm.avgTP() 
     << " FN:" << std::setw(8) << std::setprecision(4) << r.tm.avgFN() << std::endl;

  os << "        FP:" << std::setw(8) << std::setprecision(4) << r.vm.avgFP()
     << " TN:" << std::setw(8) << std::setprecision(4) << r.vm.avgTN()
     << "          FP:" << std::setw(8) << std::setprecision(4) << r.tm.avgFP()
     << " TN:" << std::setw(8) << std::setprecision(4) << r.tm.avgTN() << std::endl;

  os << "        acc: \033[1;32m" << std::setw(8) << std::setprecision(4) << r.vm.acc() << "\033[0m" << std::setw(8) << " "
     << "          acc: \033[1;32m" << std::setw(8) << std::setprecision(4) << r.tm.acc() << "\033[0m" << std::endl;

  return os;
}