#include <queue>
#include <vector>

#include "filter.h"
#include "stream.h"
#include "yolo.h"

struct YoloQueueTuple {
  int frame;
  int which_stream;
};

class RealtimeLabeler{

public:
  RealtimeLabeler(std::vector<Stream*> stream_list);
  ~RealtimeLabeler();
  int LabelFrame(int frame, int which_stream);

private:
  const std::vector<Stream*> kStreamList_;
  std::vector<std::queue<int>> filter_queues_;
  std::queue<YoloQueueTuple> yolo_queue_;
};
