#ifndef FILTER_H_
#define FILTER_H_

class Filter {

public:
   Filter();
   virtual ~Filter();
   virtual int CheckFrame(int frame)=0;
};

#endif
