#include <sys/mman.h>

#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include <iterator>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/platform/cuda.h"

#include "opencv2/opencv.hpp"

#include "tensorflow/noscope/noscope_labeler.h"
#include "tensorflow/noscope/darknet/src/yolo.h"
#include "tensorflow/noscope/darknet/src/image.h"

namespace noscope {

NoscopeLabeler::NoscopeLabeler(tensorflow::Session *session, 
                         yolo::YOLO *yolo_classifier,
                         noscope::filters::DifferenceFilter diff_filt,
                         const std::string& avg_fname,
                         const noscope::NoscopeData& data) :
    kNbFrames_(data.kNbFrames_),
    all_data_(data),
    frame_status_(kNbFrames_, kUnprocessed), labels_(kNbFrames_),
    kDifferenceFilter_(diff_filt),
    diff_confidence_(kNbFrames_),
    cnn_confidence_(kNbFrames_),
    yolo_(yolo_classifier),
    yolo_confidence_(kNbFrames_),
    avg_(NoscopeData::kDistResol_, CV_32FC3),
    session_(session) {
  std::ifstream is(avg_fname);
  std::istream_iterator<float> start(is), end;
  std::vector<float> nums(start, end);
  if (nums.size() != NoscopeData::kDistFrameSize_) {
    throw std::runtime_error("nums not right size");
  }
  memcpy(avg_.data, &nums[0], NoscopeData::kDistFrameSize_ * sizeof(float));
}

/*
*diff_confidence -> actually storing how similar one frame is to the previous frame
*cnn_confidence -> actually storing the probability that whatever object you are looking for is in the frame
*/

void NoscopeLabeler::RunDifferenceFilter(const float lower_thresh, //threshold to pass on to next level
                                      const float upper_thresh,
                                      const bool const_ref, 
                                      const size_t kRef) {
  const std::vector<uint8_t>& kFrameData = all_data_.diff_data_; //load the stuff the difference filter should be handling
  const int kFrameSize = NoscopeData::kDiffFrameSize_; //Getting how big each frame is so you can fastforward to stuff
  #pragma omp parallel for num_threads(kNumThreads_) schedule(static)
  for (size_t i = kDiffDelay_; i < kNbFrames_; i++) { //Run over every single frame in the batch
    const uint8_t *kRefImg = const_ref ?
        &kFrameData[kRef * kFrameSize] :
        &kFrameData[(i - kDiffDelay_) * kFrameSize]; //retrieve the reference image
    float tmp = kDifferenceFilter_.fp(&kFrameData[i * kFrameSize], kRefImg); //actual run of the difference filter
    diff_confidence_[i] = tmp; //save the difference value back as a confidence value
    if (tmp < lower_thresh) { //If there is very low difference
      labels_[i] = false; //nothing there - always???
      frame_status_[i] = kDiffFiltered; //set this frame as filtered
    } else {
      frame_status_[i] = kDiffUnfiltered; //otherwise - pass onto the specialized CNN to eyeball
    }
  }
  for (size_t i = kDiffDelay_; i < kNbFrames_; i++) //iterate over every frame
    if (frame_status_[i] == kDiffUnfiltered)
      cnn_frame_ind_.push_back(i); //find every frame that is problematic and add it to an array for the CNN to look at
}

void NoscopeLabeler::PopulateCNNFrames() {
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < kDiffDelay_; i++) cnn_frame_ind_.push_back(i); //Add every frame before the difference delay

  const std::vector<float>& kDistData = all_data_.dist_data_;
  const std::vector<float>& kDistData = all_data_.dist_data_;
  const int kFrameSize = NoscopeData::kDistFrameSize_; //Need this to jump to random places in the array?


  using namespace tensorflow;
  const size_t kNbCNNFrames = cnn_frame_ind_.size(); //How many frames need to look at
  const size_t kNbLoops = (kNbCNNFrames + kMaxCNNImages_ - 1) / kMaxCNNImages_; //Figure the number of passes you need
  const float* avg = (float *) avg_.data;
  for (size_t i = 0; i < kNbLoops; i++) { //Creating tensors of the correct size
    const size_t kImagesToRun =
        std::min(kMaxCNNImages_, cnn_frame_ind_.size() - i * kMaxCNNImages_); //number of frames in this batch
    Tensor input(DT_FLOAT,
                 TensorShape({kImagesToRun,
                             NoscopeData::kDistResol_.height,
                             NoscopeData::kDistResol_.width,
                             kNbChannels_})); //create the input tensor with the correct number of frames
    auto input_mapped = input.tensor<float, 4>(); //size everything and cast to correct data type?
    float *tensor_start = &input_mapped(0, 0, 0, 0); //get a handle on the first input
    #pragma omp parallel for
    for (size_t j = 0; j < kImagesToRun; j++) { //iterate through every frame to load into the tensor
      const size_t kImgInd = i * kMaxCNNImages_ + j; //get the frame index
      float *output = tensor_start + j * kFrameSize; //find the pointer to the piece of tensor to modify
      const float *input = &kDistData[cnn_frame_ind_[kImgInd] * kFrameSize]; //find the pointer to the frame needed
      for (size_t k = 0; k < kFrameSize; k++) //loop through all the floats? representing the frame
        output[k] = input[k] / 255. - avg[k]; //load the tensor with the normalized and mean-subtracted frame data
    }
    dist_tensors_.push_back(input); //add the normed/mean-subbed data to the dist_tensor_ array
  }


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // std::cout << "PopulateCNNFrames time: " << diff.count() << " s" << std::endl;
}

void NoscopeLabeler::RunSmallCNN(const float lower_thresh, const float upper_thresh) {
  using namespace tensorflow;

  // Round up
  const size_t kNbCNNFrames = cnn_frame_ind_.size(); //the number of frames to run through the CNN
  const size_t kNbLoops = (kNbCNNFrames + kMaxCNNImages_ - 1) / kMaxCNNImages_; //Calculate the number of loops needed

  for (size_t i = 0; i < kNbLoops; i++) { //loop over number of tensors?
    const size_t kImagesToRun =
        std::min(kMaxCNNImages_, cnn_frame_ind_.size() - i * kMaxCNNImages_); //Figure number of images in this batch?
    auto input = dist_tensors_[i]; //Grab input tensor
    /*cudaHostRegister(&(input.tensor<float, 4>()(0, 0, 0, 0)),
                     kImagesToRun * kFrameSize * sizeof(float),
                     cudaHostRegisterPortable);*/


    std::vector<tensorflow::Tensor> outputs; //create output tensor
    std::vector<std::pair<string, tensorflow::Tensor> > inputs = {
      {"input_img", input},
      // {"keras_learning_phase", learning_phase},
    }; //Name and create the input tensor

    tensorflow::Status status = session_->Run(inputs, {"output_prob"}, {}, &outputs); //Update tensor status and run the
    // TF_CHECK_OK(status);
    // FIXME: should probably check the tensor output size here.

    {
      auto output_mapped = outputs[0].tensor<float, 2>(); //resize and map the output tensor
      for (size_t j = 0; j < kImagesToRun; j++) { //loop through all frames needed
        Status s; //Create a new status
        const int kInd = cnn_frame_ind_[i * kMaxCNNImages_ + j]; //find which frame to look at
        cnn_confidence_[kInd] = output_mapped(j, 1); //store the confidence level
        if (output_mapped(j, 1) < lower_thresh) { //Definitely a no
          labels_[kInd] = false;
          s = kDistillFiltered;
        } else if (output_mapped(j, 1) > upper_thresh) { //Definitely a yes
          labels_[kInd] = true;
          s = kDistillFiltered;
        } else { //Pass to YOLO
          s = kDistillUnfiltered;
        }
        frame_status_[kInd] = s; //store the frame status in larger array
      }
    }
  }
}

static image ipl_to_image(IplImage* src) {
  unsigned char *data = (unsigned char *)src->imageData;
  // float *data = (float *) src->imageData;
  int h = src->height;
  int w = src->width;
  int c = src->nChannels;
  int step = src->widthStep;// / sizeof(float);
  image out = make_image(w, h, c);
  int count = 0;

  for (int k = 0; k < c; ++k) {
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        out.data[count++] = data[i*step + j*c + (2 - k)] / 255.;
        // out.data[count++] = data[i*step + j*c + k] / 255.;
      }
    }
  }
  return out;
}
static void noscope_rgbgr_image(image im) {
  int i;
  for (i = 0; i < im.w*im.h; ++i) {
    float swap = im.data[i];
    im.data[i] = im.data[i+im.w*im.h*2];
    im.data[i+im.w*im.h*2] = swap;
  }
}

void NoscopeLabeler::RunYOLO(const bool actually_run) {
  if (actually_run) {
    for (size_t i = 0; i < kNbFrames_; i++) { //Run through every frame
      // Run YOLO on every unprocessed frame
      if (frame_status_[i] == kDistillFiltered) //check frame hasn't been filtered by small CNN
        continue;
      if (frame_status_[i] == kDiffFiltered) //check frame hasn't been filtered by difference
        continue;
      cv::Mat cpp_frame(NoscopeData::kYOLOResol_, CV_8UC3,
                        const_cast<uint8_t *>(&all_data_.yolo_data_[i * all_data_.kYOLOFrameSize_])); //create an opencv matrix

      IplImage frame = cpp_frame; //cast to IplImage
      image yolo_frame = ipl_to_image(&frame); //convert to tensorflow(yolo) image
      // noscope_rgbgr_image(yolo_frame);
      yolo_confidence_[i] = yolo_->LabelFrame(yolo_frame); //actually run frame through yolo and retrieve label
      free_image(yolo_frame);  //from image.h -> frees image from memory?
      /*if (yolo_confidence_[i] != 0)
        std::cerr << "frame " << i << ": " << yolo_confidence_[i] << "\n";*/
      labels_[i] = yolo_confidence_[i] > 0; //save labeling back
      frame_status_[i] = kYoloLabeled; //set frame status
    }
  } else { //if you don't want to actually run through YOLO
    for (size_t i = 0; i < kNbFrames_; i++)
      if (frame_status_[i] == kDistillUnfiltered)
        frame_status_[i] = kYoloLabeled;
  }
}

void NoscopeLabeler::DumpConfidences(const std::string& fname,
                                  const std::string& model_name,
                                  const size_t kSkip,
                                  const bool kSkipSmallCNN,
                                  const float diff_thresh,
                                  const float distill_thresh_lower,
                                  const float distill_thresh_upper,
                                  const std::vector<double>& runtimes) {
  std::ofstream csv_file;
  csv_file.open(fname);

  std::stringstream rt;
  std::copy(runtimes.begin(), runtimes.end(), std::ostream_iterator<double>(rt, " "));

  csv_file << "# diff_thresh: "  << diff_thresh <<
      ", distill_thresh_lower: " << distill_thresh_lower <<
      ", distill_thresh_upper: " << distill_thresh_upper <<
      ", skip: " << kSkip <<
      ", skip_cnn: " << kSkipSmallCNN <<
      ", runtime: " << rt.str() << "\n";
  csv_file << "# model: " << model_name <<
      ", diff_detection: " << kDifferenceFilter_.name << "\n";

  csv_file << "# frame,status,diff_confidence,cnn_confidence,yolo_confidence,label\n";
  for (size_t i = 0; i < kNbFrames_; i++) {
    csv_file << (kSkip*i + 1) << ",";
    csv_file << frame_status_[i] << ",";
    csv_file << diff_confidence_[i] << ",";
    csv_file << cnn_confidence_[i] << ",";
    csv_file << yolo_confidence_[i] << ",";
    csv_file << labels_[i] << "\n";

    // repeat the previous label for skipped frames
    for(size_t j = 0; j < kSkip-1; j++){
      csv_file << (kSkip*i + j + 1) << ",";
      csv_file << kSkipped << ",";
      csv_file << 0 << ",";
      csv_file << 0 << ",";
      csv_file << 0 << ",";
      csv_file << labels_[i] << "\n";
    }
  }
}

} // namespace noscope
