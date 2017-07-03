#include "diff_detector.h"

namespace noscope {

class BlockDifferenceDetector : public DifferenceDetector {

public:
 BlockDifferenceDetector(const size_t resolution,
                         cv::Mat ref,
                         const float threshold,
                         const bool r_type,
                         const int ref_offset,
                         const std::string& weight_file,
                         const size_t num_blocks);
 ~BlockDifferenceDetector();
 int CheckFrame(cv::Mat frame) override;

private:
 //Block specific variables
 float* kWeights_;
 const size_t kNumBlocks_;

 //Load the weights into kWeights_ array from file
 void LoadWeights(std::string& fname);

}; //class BlockDifferenceDetector

} //namespace noscope
