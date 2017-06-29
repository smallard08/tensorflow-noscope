#include "filter.h"

class DiffDetector : public Filter {

public:
  DiffDetector(int ref, float threshold);
  ~DiffDetector();
  int CheckFrame(int frame);

private:
  int ref_img_;
  const float kThreshold_;
};
