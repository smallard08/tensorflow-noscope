#include <string>
#include <atomic>

#include "tensorflow/noscope/darknet/src/yolo.h"
#include "tensorflow/noscope/darknet/src/image.h"

namespace noscope {

class YOLOClassify{

public:
  YOLOClassify(size_t resolution,
               yolo::YOLO *model);
  ~YOLOClassify();

  //lock the yolo nn, return success
  bool Lock(int stream_id);

  //unlock the yolo nn, return success
  bool Unlock();

  //can only be called if resource has been locked
  int CheckFrame(uint8_t *frame, int yolo_class);

private:
  //what resolution incoming image should be
  const size_t kResolution_;

  //holds yolo model object
  const yolo::YOLO *kModel_;

  //which stream has lock on yolo resource (-1 if none)
  std::atomic<int> stream_lock_;
};

} //namespace noscope
