#include "global_diff_detector.h"

namespace noscope {

GlobalDifferenceDetector::GlobalDifferenceDetector(
    const size_t resolution,
    cv::Mat ref,
    const float threshold,
    const bool r_type,
    const int ref_offset) :
    DifferenceDetector(resolution, ref, threshold, r_type, ref_offset) {
  //if necessary, load weights here
}//GlobalDifferenceDetector()

GlobalDifferenceDetector::~GlobalDifferenceDetector() {

}//~GlobalDifferenceDetector()

int GlobalDifferenceDetector::CheckFrame(cv::Mat frame) {
  return -1;
}//CheckFrame()

} //namespace noscope
