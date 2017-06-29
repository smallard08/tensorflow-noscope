#ifndef STREAM_H_
#define STREAM_H_

#include "filter.h"

namespace noscope {

class Stream {

public:
   Stream(const int num_skip,
          const std::vector<Filter*> filters,
          std::string& out_file);

   ~Stream();

   //run a frame through all filters
   int RunFilters(uint8_t *frame);

   //set the outfile path, return success status
   int SetOutFile(std::string& fname);

   //Write a decision to file, return success status
   int WriteResults(int result);

private:
   //number of frames to skip
   const int kSkip_;

   //filters to run through (in order)
   const std::vector<Filter*> kFilterList_;

   //output file
   std::string& out_file_;

   //indicator when to skip frames
   int skip_counter_;

   //last frame labeled (relevant for skipped frames)
   int last_labeled_;
}; //Stream

} //namespace noscope

#endif
