#include "realtime.h"

namespace noscope {

RealtimeLabeler::RealtimeLabeler(std::vector<Stream*> stream_list) :
    kStreamList_(stream_list) {
  //start all listener threads
}//RealtimeLabeler()

RealtimeLabeler::~RealtimeLabeler() {
  //kill all hanging threads
}//~RealtimeLabeler()

std::vector<int> RealtimeLabeler::LabelFile(std::string& fname,
                                            int stream_id) {
  //enqueue frame into appropriate stream
  //when that returns, dequeue from filter queue and enqueue into yolo queue if necessary
  return std::vector<int>();
}//LabelFile()

} //namespace noscope
