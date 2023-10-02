#include <Metric.h>

Metric::Metric() {tp=0;tn=0;fp=0;fn=0;tot=0;checkpointTime_=0;}
void Metric::update(float pred, float gt) {
  pred>0 ? (gt>0 ? tp++ : fp++) : (gt>0 ? fn ++ : tn ++); 
  tot ++;
}
Metric& Metric::operator+=(const Metric& rhs) {
  tp += rhs.tp; tn += rhs.tn; fp += rhs.fp; fn += rhs.fn; tot+= rhs.tot;
  checkpointTime_ += rhs.checkpointTime_;
  return *this;
}


Metric& Metric::operator=(const Metric& rhs) {
  tp = rhs.tp; tn = rhs.tn; fp = rhs.fp; fn = rhs.fn; tot= rhs.tot;
  checkpointTime_ = rhs.checkpointTime_;
  return *this;
}

bool Metric::operator>(const Metric& rhs) {
  return acc()>rhs.acc();
}
float Metric::avgTP() const {return (double)tp/(double)(tot);}
float Metric::avgTN() const {return (double)tn/(double)(tot);}
float Metric::avgFP() const {return (double)fp/(double)(tot);}
float Metric::avgFN() const {return (double)fn/(double)(tot);}
float Metric::acc() const {return tot==0 ? 0 : (double)(tp+tn)/(double)(tot);}
void Metric::compute() {
  iouT = (double)tp / (tp + fn + fp);
  iouF = (double)tn / (tn + fn + fp);
  f1 = 2.0f*avgTP() / (2.0f*avgTP()+avgFN()+avgFP());
  cohen = 2.0f * (avgTP() * avgTN() -avgFN()*avgFP()) / 
          ( (avgTP() + avgFP())*(avgFP() + avgTN()) + (avgTP() + avgFN())*(avgFN() + avgTN()) );
}

void Metric::print(std:: string msg, int tot_cells, int tot_workers) {
  compute();
  std::cout << std::setw(12) << "\033[1;31m" << msg << "\033[0m" << std::endl;
  std::cout << "         TP:" << std::setw(8) << std::setprecision(4) << avgTP();
  std::cout << " FN:" << std::setw(8) << std::setprecision(4) << avgFN() << std::endl;
  std::cout << "         FP:" << std::setw(8) << std::setprecision(4) << avgFP();
  std::cout << " TN:" << std::setw(8) << std::setprecision(4) << avgTN() << std::endl;
  std::cout << "        acc: \033[1;32m" << std::setw(8) << std::setprecision(4) << acc() << "\033[0m";
  std::cout << " iouT:" << std::setw(8) << std::setprecision(3) << iouT << std::endl;
  std::cout << " iouF:" << std::setw(8) << std::setprecision(3) << iouF << std::endl;
  std::cout << " f1:" << std::setw(8) << std::setprecision(3) << f1 << std::endl;
  std::cout << " cohen:" << std::setw(8) << std::setprecision(3) << cohen << std::endl;

  std::cout << " lat: \033[1;35m" << std::setw(5) << (checkpointTime_ ) << "\033[0m ms";
  std::cout << " lat: \033[1;35m" << std::setw(5) << (checkpointTime_  * tot_cells / tot) << "\033[0m ms";
  std::cout << std::endl;

  // latex
  // std::cout << std::endl << std::endl;
  // std::cout << "& mode & pca & "
  //           << std::setw(5)<< std::setprecision(4) << acc()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << iou*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << f1*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgFP()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgTP()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgFN()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgTN()*100.0f << std::endl;


  // std::cout << std::endl << std::endl;
}

void Metric::printV(const char *msg, int tot_cells, int tot_workers) const {
  // float iou, f1;
  // iou = (float)tp / (tp+fn+fp);
  // f1 = 2.0f*tp / (2*tp+fn+fp);

  std::cout << std::setw(12) << "\033[1;31m" << msg << "\033[0m" << std::endl;
  std::cout << "         TP:" << std::setw(8) << std::setprecision(4) << avgTP();
  std::cout << " FN:" << std::setw(8) << std::setprecision(4) << avgFN() << std::endl;
  std::cout << "         FP:" << std::setw(8) << std::setprecision(4) << avgFP();
  std::cout << " TN:" << std::setw(8) << std::setprecision(4) << avgTN() << std::endl;
  std::cout << "        acc: \033[1;33m" << std::setw(8) << std::setprecision(4) << acc() << "\033[0m";
  std::cout << " lat: \033[1;35m" << std::setw(5) << (checkpointTime_ ) << "\033[0m ms";
  std::cout << " lat: \033[1;35m" << std::setw(5) << (checkpointTime_  * tot_cells / tot) << "\033[0m ms";
  std::cout << std::endl;

  // latex
  // std::cout << std::endl << std::endl;
  // std::cout << "& mode & pca & "
  //           << std::setw(5)<< std::setprecision(4) << acc()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << iou*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << f1*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgFP()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgTP()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgFN()*100.0f << " & "
  //           << std::setw(5)<< std::setprecision(4) << avgTN()*100.0f << std::endl;


  // std::cout << std::endl << std::endl;
}

void Metric::printLight(const char *msg, int tot_cells, int tot_workers) const {
  std::cout << "\033[1;31m" << std::setw(12) << msg << "\033[0m";
  std::cout << "  acc: \033[1;32m" << std::setw(8) << std::setprecision(4) << acc() << "\033[0m";
  std::cout << " lat: \033[1;35m" << std::setw(5) << (checkpointTime_ ) << "\033[0m ms";
  std::cout << std::endl;// << std::endl;
}

std::string Metric::getresults() const {
  std::stringstream ss("");
  ss << "         TP:" << std::setw(8) << std::setprecision(4) << avgTP();
  ss << " FN:" << std::setw(8) << std::setprecision(4) << avgFN() << std::endl;
  ss << "         FP:" << std::setw(8) << std::setprecision(4) << avgFP();
  ss << " TN:" << std::setw(8) << std::setprecision(4) << avgTN() << std::endl;
  ss << "        acc:" << std::setw(8) << std::setprecision(4) << acc() << std::endl;
  ss << std::endl;
  return ss.str();
}

void Metric::log2YAML(float nu, float gamma, float C, int pca, int rows, int tot_cells, std::string filename) {
    
  std::ofstream out(filename.c_str());

  YAML::Emitter outyaml(out);

  outyaml << YAML::BeginMap;

  outyaml << YAML::Key << "nu" << YAML::Flow << std::to_string(nu);
  outyaml << YAML::Key << "gamma" << YAML::Flow << std::to_string(gamma);
  outyaml << YAML::Key << "valrows" << YAML::Flow << std::to_string(rows);
  outyaml << YAML::Key << "tn" << YAML::Flow << std::to_string(tn);
  outyaml << YAML::Key << "fp" << YAML::Flow << std::to_string(fp);
  outyaml << YAML::Key << "fn" << YAML::Flow << std::to_string(fn);
  outyaml << YAML::Key << "tp" << YAML::Flow << std::to_string(tp);
  outyaml << YAML::Key << "valid_acc" << YAML::Flow << std::to_string((tp+tn)/(float)tot);
  outyaml << YAML::Key << "valid_latency" << YAML::Flow << std::to_string(checkpointTime_  * tot_cells / tot);
  outyaml << YAML::Key << "seed" << YAML::Flow << std::to_string(seed);

  outyaml << YAML::EndMap;
  out.close();
}



void Metric::resetTime() { checkpointTime_=0; bt.reset(); }
void Metric::resetAcc() { tp=0;tn=0;fp=0;fn=0;tot=0; }
void Metric::resetAll() { resetAcc(); resetTime(); }
void Metric::checkpointTime() { checkpointTime_ = bt.elapsedTimeMs(); }