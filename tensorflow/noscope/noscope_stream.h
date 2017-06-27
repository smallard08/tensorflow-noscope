#ifndef TENSORFLOW_VUSE_VUSEDATA_H_
#define TENSORFLOW_VUSE_VUSEDATA_H_

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"

#include "tensorflow/noscope/filters.h"

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

namespace noscope {

class NoscopeStream {

 public:  
  //how many frames to skip
  const size_t kSkip_;
  
  //largest the backlog queue should grow to
  const size_t kQueueSize_;
  
  //if we should skip input frames (0 - kSkip_)
  size_t skip_counter_;
  
  //backlog of unlabeled frames
  std::vector<uint8_t> frame_queue_;
  
  //filters for this particular stream
  tensorflow::Session *small_cnn_;
  noscope::filters::DifferenceFilter diff_filt_;
  
  NoscopeStream(const size_t kSkip, 
                const size_t kQueueSize,
                const tensorflow::Session *small_cnn,
                const noscope::filters::DifferenceFilter diff_filt);
  
  //Assume next little piece has been saved into a file
  bool AddFrames(std::string& fname);

  bool DequeueFrames(int num);

};

} // namespace noscope

#endif
