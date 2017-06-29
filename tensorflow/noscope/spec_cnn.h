#include "tensorflow/core/public/session.h"

#include "filter.h"

namespace noscope {

class SpecCNN : public Filter {

public:
  SpecCNN(const size_t resolution,
          tensorflow::Session *session,
          const float u_thresh,
          const float l_thresh);
  ~SpecCNN();
  int CheckFrame(uint8_t *frame);

private:
  //pointer to tensorflow session to run
  tensorflow::Session *kSession_;

  //upper confidence threshold (something is there)
  const float kUThreshold_;

  //lower confidence threshold (nothing is there)
  const float kLThreshold_;
};

} //namespace noscope
