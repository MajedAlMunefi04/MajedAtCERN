#ifndef TIMING_H
#define TIMING_H

struct Timing {
    bool noError;
    float timeUploadAvg, timeDownloadAvg, timeCalcAvg, timeCalcCpuAvg;
    float timeUploadstd, timeDownloadstd, timeCalcstd, timeCalcCpustd =0.0;
};

#endif // TIMING_H