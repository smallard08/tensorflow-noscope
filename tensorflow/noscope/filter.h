#ifndef NOSCOPE_TENSORFLOW_NOSCOPE_FILTER_H_
#define NOSCOPE_TENSORFLOW_NOSCOPE_FILTER_H_

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

namespace noscope {

class Filter {

public:
 Filter(const size_t resolution);
 virtual ~Filter();

protected:
 //Number of channels
 const size_t kNbChannels_ = 3;

 //Expected resolution of image passed into filter
 const size_t kResolution_;
}; //class Filter

} //namespace noscope

#endif //NOSCOPE_TENSORFLOW_NOSCOPE_FILTER_H_
