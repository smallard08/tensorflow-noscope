#include "filter.h"

namespace noscope {

class DiffDetector : public Filter {

public:
  enum DiffType {
    kBlocked,
    kGlobal,
    kNone
  }; //enum DiffType
  DiffDetector(const size_t resolution,
               uint8_t *ref,
               const float threshold,
               const DiffType d_type,
               const std::string& weight_file,
               const size_t num_blocks,
               const bool r_type,
               const int ref_offset);
  ~DiffDetector();
  int CheckFrame(uint8_t *frame);

private:
  //pointer to reference image
  uint8_t *ref_img_;

  //difference threshold to make decision
  const float kThreshold_;

  //Difference type (blocked, global, none..)
  const DiffType kDType_;

  //Block specific variables
  float* kWeights_;
  const size_t kNumBlocks_;

  //Reference image specific variables
  const bool kFixedRef_;
  const int kRefOffset_;
  int offset_counter_ = 0;

  //Reset reference image (if not fixed)
  void UpdateRefImage(uint8_t *ref);

  //Load the weights into kWeights_ array from file
  void LoadWeights(std::string& fname);

}; //class DiffDetector

} //namespace noscope
