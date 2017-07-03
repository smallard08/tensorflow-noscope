#include "yolo.h"

namespace noscope {

YOLOClassify::YOLOClassify(size_t resolution,
                           yolo::YOLO *model) :
    Filter(resolution),
    kModel_(model) {
}//YOLOClassify

YOLOClassify::~YOLOClassify() {

}//~YOLOClassify

bool Lock(int stream_id) {
  return false;
}//Lock()

bool Unlock() {
  return false;
}//Unlock()

int YOLOClassify::CheckFrame(cv::Mat frame, int yolo_class) {
  return -1;
}//CheckFrame()

}//namespace noscope
