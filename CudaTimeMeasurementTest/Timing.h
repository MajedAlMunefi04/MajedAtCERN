#ifndef TIMING_H
#define TIMING_H

struct Timing {
    bool noError;
    
    float timeUploadAvg = 0.0;
    float timeDownloadAvg = 0.0;
    float timeCalcAvg = 0.0;
    float timeCalcCpuAvg = 0.0;

    float timeUploadstd = 0.0;
    float timeDownloadstd = 0.0;
    float timeCalcstd = 0.0;
    float timeCalcCpustd = 0.0;
};

#endif // TIMING_H
