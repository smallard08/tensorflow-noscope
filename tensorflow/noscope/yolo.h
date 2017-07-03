#include <mutex>

#include "filter.h"

#include "tensorflow/noscope/darknet/src/yolo.h"
#include "tensorflow/noscope/darknet/src/image.h"

namespace noscope {

class YOLOClassify : Filter {

public:
 YOLOClassify(size_t resolution,
              yolo::YOLO *model);
 ~YOLOClassify();

 //can only be called if resource has been locked
 int CheckFrame(cv::Mat frame, int yolo_class);

private:
 //holds yolo model object
 const yolo::YOLO *kModel_;

 //which stream has lock on yolo resource (-1 if none)
 std::mutex stream_lock_;
};

} //namespace noscope
