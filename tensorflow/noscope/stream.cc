#include "stream.h"

namespace noscope {

Stream::Stream(const int num_skip,
               const std::vector<Filter*> filters,
               std::string& out_file) :
    kSkip_(num_skip),
    kFilterList_(filters),
    out_file_(out_file),
    skip_counter_(0),
    last_labeled_(-1) {

}//Stream()

Stream::~Stream() {

}//~Stream()

int RunFilters(cv::Mat frame) {
  return -1;
}//RunFilters()

int SetOutFile(std::string& fname) {
  return -1;
}//SetOutFile()

int WriteResult(int result) {
  return -1;
}//WriteResult()

}//namespace noscope
