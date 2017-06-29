#include "spec_cnn.h"

SpecCNN::SpecCNN(int session, float u_thresh, float l_thresh) :
  kSession_(session),
  kUThreshold_(u_thresh),
  kLThreshold_(l_thresh) {
}

SpecCNN::~SpecCNN() {

}

int SpecCNN::CheckFrame(int frame) {
  return -1;
}
