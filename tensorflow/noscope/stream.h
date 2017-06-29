#ifndef STREAM_H_
#define STREAM_H_

#include <vector>

#include "filter.h"

class Stream {

public:
   Stream(int skip_num, std::vector<Filter*> filters);

   ~Stream();

   int CheckFilters(int frame);

private:
   const int kSkip_;
   const std::vector<Filter*> kFilterList_;
   int skip_counter_;
   int last_labeled_;
};

#endif
