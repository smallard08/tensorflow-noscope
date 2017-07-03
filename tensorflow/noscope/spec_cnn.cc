#include "spec_cnn.h"

namespace noscope {

SpecializedCNN::SpecializedCNN(const size_t resolution,
                               tensorflow::Session *session,
                               const float u_thresh,
                               const float l_thresh) :
    Filter(resolution),
    kSession_(session),
    kUThreshold_(u_thresh),
    kLThreshold_(l_thresh) {
}//SpecializedCNN()

SpecializedCNN::~SpecializedCNN() {

}//~SpecializedCNN

int SpecializedCNN::CheckFrame(cv::Mat frame) {
  return -1;
}//CheckFrame()

}//namespace noscope
