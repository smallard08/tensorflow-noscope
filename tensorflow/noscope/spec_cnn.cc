#include "spec_cnn.h"

namespace noscope {

SpecCNN::SpecCNN(const size_t resolution,
                 tensorflow::Session *session,
                 const float u_thresh,
                 const float l_thresh) :
  Filter(resolution),
  kSession_(session),
  kUThreshold_(u_thresh),
  kLThreshold_(l_thresh) {
}//SpecCNN()

SpecCNN::~SpecCNN() {

}//~SpecCNN

int SpecCNN::CheckFrame(uint8_t *frame) {
  return -1;
}//CheckFrame()

}//namespace noscope
