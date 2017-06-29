#include "filter.h"

class SpecCNN : public Filter {

public:
  SpecCNN(int session, float u_thresh, float l_thresh);
  ~SpecCNN();
  int CheckFrame(int frame);

private:
  const int kSession_;
  const float kUThreshold_;
  const float kLThreshold_;
};
