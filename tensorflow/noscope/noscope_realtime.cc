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

#include "tensorflow/noscope/filters.h"
#include "tensorflow/noscope/darknet/src/yolo.h"

void LabelVideo(int stream_id, std::string& video_fname, std::string& output_fname) {
	//decode video
	//add frames to correct NoscopeStream
}

void Kill() {
  //kill all waiting threads
	//clean all remaining memory
}

void Init() {
	//initialize YOLO classifier
	//initialize queue into YOLO classifier
	//start threads listening to queue into YOLO
	//initialize queue out of YOLO classifier
	//start threads listening to queue out of YOLO
	//parse config file
	//initialize array of streams
	//for number of streams:
		//initialize difference filter
		//initialize tf session
		//initialize NoscopeStream
}

