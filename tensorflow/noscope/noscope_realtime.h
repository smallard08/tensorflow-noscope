#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include <iterator>
#include <memory>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/noscope/darknet/src/yolo.h"
#include "tensorflow/noscope/noscope_stream.h"

namespace noscope {

class NoscopeRealtime {

public:
	void LabelVideo(int stream_id, std::string& video_fname, std::string& output_fname);

	void Kill();

	void Init();

private:
	NoscopeStream *streams_;
	yolo::YOLO *yolo_;
}

}
