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
	//Assume next little piece has been saved into a file
	bool AddFrames(std::string& fname);

	bool DequeueFrames(int num);

  NoscopeStream(const size_t kSkip, 
                const size_t kQueueSize,
                const tensorflow::Session *small_cnn,
                const noscope::filters::DifferenceFilter diff_filt);

	//how many frames to skip
  const size_t kSkip_;

  //largest the backlog queue should grow to
  const size_t kQueueSize_;

  //filters for this particular stream
  const tensorflow::Session *small_cnn_;
  noscope::filters::DifferenceFilter diff_filt_;

 private:
	//if we should skip input frames (0 - kSkip_)
	size_t skip_counter_;

	//backlog of unlabeled frames
	std::vector<uint8_t> frame_queue_;

};

} // namespace noscope

#endif
