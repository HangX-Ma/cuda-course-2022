#include <cuda_runtime.h>
#include "real.hpp"

#define QUICK_TRANS (1)
#define HEAT_TRANSFER_SPEED (0.05f)

extern int* adjInfo;
extern REAL* gIntensityIn_d_;
extern REAL* gIntensityOut_d_;
extern int* gAdjTriNums_d_ptr;
extern int* gPrefix_sum_d_ptr;

void doPropagateKernel(int flag, int num);
void doGPUInit(int num, int flag);