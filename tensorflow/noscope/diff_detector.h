#include "filter.h"

namespace noscope {

class DifferenceDetector : public Filter {

public:
 DifferenceDetector(const size_t resolution,
                    cv::Mat ref,
                    const float threshold,
                    const bool r_type,
                    const int ref_offset);
 ~DifferenceDetector();

 //pointer to reference image
 cv::Mat ref_img_;

 //difference threshold to make decision
 const float kThreshold_;

 //Reference image specific variables
 const bool kFixedRef_;
 const int kRefOffset_;
 int offset_counter_ = 0;

 virtual int CheckFrame(cv::Mat frame);

 //Reset reference image (if not fixed)
 void UpdateRefImage(cv::Mat ref);

}; //class DifferenceDetector

} //namespace noscope
