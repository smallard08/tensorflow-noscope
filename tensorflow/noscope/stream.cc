#include <vector>

#include "stream.h"
#include "filter.h"

Stream::Stream(int skip, std::vector<Filter*> filters) :
  kSkip_(skip),
  kFilterList_(filters),
  skip_counter_(0),
  last_labeled_(-1) {

}

Stream::~Stream() {

}

int CheckFilters(int frame) {
  return -1;
}
