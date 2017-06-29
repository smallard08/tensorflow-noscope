#include "diff_detector.h"

namespace noscope {

DiffDetector::DiffDetector(const size_t resolution,
                           uint8_t *ref,
                           const float threshold,
                           const DiffType d_type,
                           const std::string& weight_file,
                           const size_t num_blocks = 10,
                           const bool r_type = true,
                           const int ref_offset = 0) :
    Filter(resolution),
    ref_img_(ref),
    kThreshold_(threshold),
    kDType_(d_type),
    kWeights_(NULL),
    kNumBlocks_(num_blocks),
    kFixedRef_(r_type),
    kRefOffset_(ref_offset) {
  //if necessary, load weights here
}//DiffDetector()

DiffDetector::~DiffDetector() {

}//~DiffDetector()

int DiffDetector::CheckFrame(uint8_t *frame) {
  return -1;
}//CheckFrame()

void UpdateRefImage(uint8_t *ref) {
  //reset the reference image to the parameter
}//UpdateRefImage()

void LoadWeights(std::string& fname) {
  //load the weights into kWeights_
}//LoadWeights()

} //namespace noscope
