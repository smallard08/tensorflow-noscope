#include "diff_detector.h"

namespace noscope {

class GlobalDifferenceDetector : public DifferenceDetector {

public:
 GlobalDifferenceDetector(const size_t resolution,
                          cv::Mat ref,
                          const float threshold,
                          const bool r_type,
                          const int ref_offset);
 ~GlobalDifferenceDetector();

 int CheckFrame(cv::Mat frame) override;

}; //class GlobalDifferenceDetector

} //namespace noscope
