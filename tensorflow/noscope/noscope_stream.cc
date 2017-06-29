#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"

#include "tensorflow/noscope/filters.h"
#include "tensorflow/noscope/noscope_stream.h"

namespace noscope {

bool NoscopeStream::AddFrames(std::string& fname) {
  return false;
}//AddFrames()

bool NoscopeStream::DequeueFrames(int num) {
  return false;
}//DequeueFrames()

NoscopeStream::NoscopeStream(const size_t kSkip, 
                             const size_t kQueueSize,
                             const tensorflow::Session *small_cnn,
                             const noscope::filters::DifferenceFilter diff_filt) :
				  kSkip_(kSkip),
				  kQueueSize_(kQueueSize),
				  small_cnn_(small_cnn),
				  diff_filt_(diff_filt),
				  skip_counter_(0),
				  frame_queue_(kQueueSize) {

}//NoscopeStream()
  
}//namespace noscope
