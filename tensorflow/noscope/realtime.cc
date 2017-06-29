#include "realtime.h"
#include "stream.h"

RealtimeLabeler::RealtimeLabeler(std::vector<Stream*> stream_list) :
  kStreamList_(stream_list) {
  //start all listener threads
}

RealtimeLabeler::~RealtimeLabeler() {
  //kill all hanging threads
}

int RealtimeLabeler::LabelFrame(int frame, int which_stream) {
  //enqueue frame into appropriate stream
  //when that returns, dequeue from filter queue and enqueue into yolo queue if necessary
  return -1;
}
