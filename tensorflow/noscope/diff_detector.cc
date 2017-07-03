#include "diff_detector.h"

namespace noscope {

DifferenceDetector::DifferenceDetector(const size_t resolution,
                                       cv::Mat ref,
                                       const float threshold,
                                       const bool r_type,
                                       const int ref_offset) :
    Filter(resolution),
    ref_img_(ref),
    kThreshold_(threshold),
    kFixedRef_(r_type),
    kRefOffset_(ref_offset) {
  //if necessary, load weights here
}//DifferenceDetector()

DifferenceDetector::~DifferenceDetector() {

}//~DifferenceDetector()

void UpdateRefImage(cv::Mat ref) {
  //reset the reference image to the parameter
}//UpdateRefImage()

} //namespace noscope
