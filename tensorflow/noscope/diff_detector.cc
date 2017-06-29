#include "diff_detector.h"

DiffDetector::DiffDetector(int ref, float threshold) :
  ref_img_(ref),
  kThreshold_(threshold) {
}

DiffDetector::~DiffDetector() {

}

int DiffDetector::CheckFrame(int frame) {
  return -1;
}
