#include "yolo.h"

YOLO::YOLO(int model) :
  kModel_(model) {
}

YOLO::~YOLO() {

}

int YOLO::CheckFrame(int frame) {
  return -1;
}
