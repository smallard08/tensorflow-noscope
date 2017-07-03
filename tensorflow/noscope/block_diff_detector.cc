#include "block_diff_detector.h"

namespace noscope {

BlockDifferenceDetector::BlockDifferenceDetector(
    const size_t resolution,
    cv::Mat ref,
    const float threshold,
    const bool r_type,
    const int ref_offset,
    const std::string& weight_file,
    const size_t num_blocks) :
    DifferenceDetector(resolution, ref, threshold, r_type, ref_offset),
    kWeights_(NULL),
    kNumBlocks_(num_blocks) {
  //if necessary, load weights here
}//DifferenceDetector()

BlockDifferenceDetector::~BlockDifferenceDetector() {

}//~DifferenceDetector()

int BlockDifferenceDetector::CheckFrame(cv::Mat frame) {
  return -1;
}//CheckFrame()

void BlockDifferenceDetector::LoadWeights(std::string& fname) {
  //load weights into kWeights
}//LoadWeights()

} //namespace noscope
