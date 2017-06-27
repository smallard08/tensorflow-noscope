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

#include "tensorflow/noscope/mse.h"
#include "tensorflow/noscope/filters.h"
#include "tensorflow/noscope/MemoryTests.h"
#include "tensorflow/noscope/noscope_labeler.h"
#include "tensorflow/noscope/noscope_data.h"
#include "tensorflow/noscope/darknet/src/yolo.h"

namespace noscope {

class NoscopeRealtime {

public:
	void LabelVideo(int stream_id, std::string& video_fname, std::string& output_fname);

	void KillLabeler();

	void Init();

private:
	NoscopeStream *streams_;
	yolo::YOLO *yolo_;
}

}
