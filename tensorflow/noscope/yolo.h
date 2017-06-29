#include "filter.h"

class YOLO : public Filter {

public:
  YOLO(int model);
  ~YOLO();
  int CheckFrame(int frame);

private:
  const int kModel_;
};
