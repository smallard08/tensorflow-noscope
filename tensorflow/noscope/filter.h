#ifndef FILTER_H_
#define FILTER_H_

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace noscope {

class Filter {

public:
   Filter(const size_t resolution);
   virtual ~Filter();

   //Run frame through filter and return 1, 0, -1 (yes, uncertain, no)
   virtual int CheckFrame(uint8_t *frame);

protected:
   //Number of channels (seems constant throughout noscope?)
   const size_t kNbChannels_ = 3;

   //Expected resolution of image passed into filter
   const size_t kResolution_;
}; //class Filter

} //namespace noscope

#endif
