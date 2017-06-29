#include <queue>
#include <thread>

#include "filter.h"
#include "stream.h"
#include "yolo.h"

namespace noscope {

struct YoloQueueTriple {
  uint8_t *frame;
  int stream_id;
  int yolo_class;
}; //YoloQueueTuple

class RealtimeLabeler{

public:
  RealtimeLabeler(std::vector<Stream*> stream_list);

  ~RealtimeLabeler();

  std::vector<int> LabelFile(std::string& fname, int stream_id);

private:
  //list of all streams for this session
  const std::vector<Stream*> kStreamList_;

  //vector of all queued frames for each stream
  std::vector<std::queue<uint8_t*>> filter_queues_;

  //queue to access yolo labeler
  std::queue<YoloQueueTriple> yolo_queue_;

  //vector of all threads running each stream
  std::vector<std::thread> thread_list_;
}; //RealtimeLabeler

} //namespace noscope
