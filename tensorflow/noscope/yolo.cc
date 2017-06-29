#include "yolo.h"

namespace noscope {

YOLOClassify::YOLOClassify(size_t resolution,
                           yolo::YOLO *model) :
  kResolution_(resolution),
  kModel_(model),
  stream_lock_(-1) {
}//YOLOClassify

YOLOClassify::~YOLOClassify() {

}//~YOLOClassify

bool Lock(int stream_id) {
  return false;
}//Lock()

bool Unlock() {
  return false;
}//Unlock

int YOLOClassify::CheckFrame(uint8_t *frame, int yolo_class) {
  return -1;
}//CheckFrame()

}//namespace noscope
