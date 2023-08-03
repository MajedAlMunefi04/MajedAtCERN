//initializing the libraries needed
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <cuda.h>
#include <unistd.h>
#include <cmath>  // for abs() from <cmath>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "Timing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

__global__ void addVectorsGpu(float *vect1, float *vect2, float *vect3, int size, int taskN) {
        int first = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
	for (int j = first; j < size; j += stride) {
	   vect3[j] = vect2[j] + vect1[j];    
	}
       }

class calculate{	
public:
       	calculate(int iteration, int vectorSize){ //int comType, int vectorSize, int avg){
        vecSize = vectorSize;
        avg = iteration * vectorSize;
    	outGPU.resize(vecSize);
    	generateRandomData(vectorSize);
    }

    Timing calculateAverage(int comMethod){
            switch (comMethod) {
                case 1:
                    cudaMethod0();
                    break;
                case 2:
                   cudaMethod1(); 
                    break;
                case 3:
                    cudaMethod2(); 
                    break;
                case 4:
                    cudaMethod3(); 
                    break;
                case 5:
                    cudaMethod4();  
                    break;
		case 6:
                    cudaMethod5();  
                    break;
                default:
                    std::cerr << "\n\n\tError: Invalid function number!\n";
                    abort();
            }

/*	        std::cout << "error: " << timing.noError <<
                "IN CLASS Average upload: " << timing.timeUploadAvg <<
                " std upload: " << timing.timeUploadstd <<
                " Average download: " << timing.timeDownloadAvg <<
                " std download: " << timing.timeDownloadstd <<
                " Average calc dev: " << timing.timeCalcAvg <<
                " std calc dev: " << timing.timeCalcstd <<
                " average calc host: " << timing.timeCalcCpuAvg <<
                " std calc host: " << timing.timeCalcCpustd ;
*/
	   return timing;
    }

private:
    int vecSize, avg;
    Timing timing;
    std::vector<float> v1_, v2_, outCPU, outGPU;
    std::vector<float> timeUpload, timeCalc, timeDownload, timeCalcCpu;
    float timeCPUCalc;
    std::chrono::duration<double> duration;

    cudaEvent_t start, stop;
//    float startTime, stopTime;
    std::chrono::steady_clock::time_point endTime;
    std::chrono::steady_clock::time_point startTime;

    float *devVec1, *devVec2, *devVecOut;
    float *devVec1Ex, *devVec2Ex, *devVecOutEx;

    void generateRandomData(int vectorSize) {
        std::random_device rand;  // Random device used to seed the random engine.
        std::default_random_engine gener(rand());  // Default random engine.
        std::uniform_real_distribution<> dis(0., 1.);  // Uniform distribution from 0 to 1.
        for (int i = 0; i < vectorSize; i++) {
                v1_.push_back(dis(gener));
                v2_.push_back(dis(gener));
                outCPU.push_back(v1_[i]+v2_[i]);
        }
        timeUpload.resize(vectorSize);
        timeCalc.resize(vectorSize);
        timeDownload.resize(vectorSize);
        timeCalcCpu.resize(vectorSize);
    }

       void calculateAverageDeviation(){
        std::cout << "0 " << timing.timeUploadstd << "\n";

        for (int i = 0; i < vecSize ; i++){
            timing.timeUploadAvg += timeUpload[i];
            timing.timeDownloadAvg += timeDownload[i];
            timing.timeCalcAvg += timeCalc[i];
            timing.timeCalcCpuAvg += timeCalcCpu[i];
        }
        std::cout << "1 " << timing.timeUploadstd << "\n";

        
        timing.timeUploadAvg = timing.timeUploadAvg/vecSize;
        timing.timeDownloadAvg = timing.timeDownloadAvg/vecSize;
        timing.timeCalcAvg = timing.timeCalcAvg/vecSize;
        timing.timeCalcCpuAvg = timing.timeCalcCpuAvg/vecSize;
        
        std::cout << "2 " << timing.timeUploadstd << "\n";

        for (int i = 0; i < vecSize; ++i) {
            timing.timeUploadstd += pow(timeUpload[i] - timing.timeUploadAvg, 2);
            timing.timeDownloadstd += pow(timeDownload[i] - timing.timeDownloadAvg, 2);
            timing.timeCalcstd += pow(timeCalc[i] - timing.timeCalcAvg, 2);
            timing.timeCalcCpustd += pow(timeCalcCpu[i] - timing.timeCalcCpuAvg, 2);
        }
        std::cout << "4 " << timing.timeUploadstd << "\n";

        if(!timing.timeUploadAvg){
            timing.timeUploadstd = 0.0000;
        }else{
            timing.timeUploadstd = sqrt(timing.timeUploadstd / vecSize);
        }
        std::cout << "5 " << timing.timeUploadstd << "\n";

        if(!timing.timeDownloadAvg){
            timing.timeDownloadstd = 0.0000;
        }else{
            timing.timeDownloadstd = sqrt(timing.timeDownloadstd / vecSize);
        }
        std::cout << "6 " << timing.timeUploadstd << "\n";

        timing.timeCalcstd = sqrt(timing.timeCalcstd / vecSize);
        timing.timeCalcCpustd = sqrt(timing.timeCalcCpustd / vecSize);

        std::cout << "7 " << timing.timeUploadstd << "\n";

        timing.noError = checkingResultsPrintout();
  std::cout << "error: " << timing.noError <<
                "IN CLASS Average upload: " << timing.timeUploadAvg <<
                " std upload: " << timing.timeUploadstd <<
                " Average download: " << timing.timeDownloadAvg <<
                " std download: " << timing.timeDownloadstd <<
                " Average calc dev: " << timing.timeCalcAvg <<
                " std calc dev: " << timing.timeCalcstd <<
                " average calc host: " << timing.timeCalcCpuAvg <<
                " std calc host: " << timing.timeCalcCpustd ;
    }

    bool checkingResultsPrintout() {
        float percent{0.0};
        float totalError{0.0};
        int size = outCPU.size();
        for (int j = 0; j < size; j++) {
            percent = ((outCPU[j] - outGPU[j]) / outCPU[j]) * 100;
            totalError += percent;
        }
        if (totalError) {
            std::cout << "\n------------------------------------\n";
            std::cout << "| CpuSum | GpuSum | Error  | Error %| ";
            std::cout << "\n------------------------------------\n";
            //std::cout.precision(4);
            for (int j = 0; j < size; j++) {
            std::cout.flags(std::ios::fixed | std::ios::showpoint);
            std::cout.precision(4);
            std::cout << "| " << outCPU[j] << " | " << outGPU[j] << " | " << outCPU[j] - outGPU[j] << " | " << percent
                        << " |\n";
            }
            std::cout << "-------------------------------------\n";
            std::cout << "-Total Error is " << totalError << std::endl;
            return false;
        }
        return true;
    }
    void cudaMethod0() {
	    std::cout << "Method 0 \n";
	for (int i = 0; i < avg; i++) {
            std::fill(outGPU.begin(), outGPU.end(), 0);  //clear each value of vector's elements
            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            duration = endTime - startTime;
            timeUpload[i] = duration.count();

            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

            cudaCheck(cudaEventRecord(start));
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
	    addVectorsGpu<<<blocks, threads>>>(v1_.data(),
                                            v2_.data(),
                                            outGPU.data(),
                                            vecSize,
					                        100
                                           );  //call device function to add two vectors and save into vect3Gpu.
	    cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            cudaCheck(cudaEventRecord(stop));
            duration = endTime - startTime;
            timeCalc[i] = duration.count();
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            duration = endTime - startTime;
            timeDownload[i] = duration.count();
            cudaEventElapsedTime(&timeCPUCalc, start, stop); 
            timeCalcCpu[i] = timeCPUCalc * 1000;

            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));
        }

        calculateAverageDeviation();
    }


    void cudaMethod1() {
            std::cout << "Method 1 \n";
        cudaCheck(cudaMalloc((void **)&devVec1, vecSize * sizeof(float)));  //allocate memory space for vector in the global memory of the Device.
        cudaCheck(cudaMalloc((void **)&devVec2, vecSize * sizeof(float)));
        cudaCheck(cudaMalloc((void **)&devVecOut, vecSize * sizeof(float)));

        for (int i = 0; i < avg; i++) {
            std::fill(outGPU.begin(), outGPU.end(), 0);  //clear each value of vector's elements
            
            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaMemcpy(devVec1, v1_.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));  //copy random vector from host to device.
            cudaCheck(cudaMemcpy(devVec2, v2_.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeUpload[i] = duration.count();


            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

            cudaCheck(cudaEventRecord(start));
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            addVectorsGpu<<<blocks, threads>>>(devVec1,
                                            devVec2,
                                            devVecOut,
                                            vecSize,
                                            100);  //call device function to add two vectors and save into vect3Gpu.
            cudaCheck(cudaGetLastError());

            cudaCheck(cudaDeviceSynchronize());
         
            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            cudaCheck(cudaEventRecord(stop));
            duration = endTime - startTime;
            timeCalc[i] = duration.count();
	        
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            cudaCheck(cudaMemcpy(
                outGPU.data(),
                devVecOut,
                vecSize * sizeof(float),
                cudaMemcpyDeviceToHost));  //copy summing result vector from Device to Host.// Try_Regist(3) delete this
            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeDownload[i] = duration.count();

            cudaEventElapsedTime(&timeCPUCalc, start, stop); 
            timeCalcCpu[i] = timeCPUCalc * 1000;

            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));
        }
        calculateAverageDeviation();

        cudaCheck(cudaFree(devVec1));
        cudaCheck(cudaFree(devVec2));
        cudaCheck(cudaFree(devVecOut));
    }


    void cudaMethod2() {
	  std::cout << "Method 2 \n";

        for (int i = 0; i < avg; i++) {
            std::fill(outGPU.begin(), outGPU.end(), 0);

            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaHostRegister(v1_.data(),  vecSize * sizeof(float), cudaHostRegisterDefault));
            cudaCheck(cudaHostRegister(v2_.data(),  vecSize * sizeof(float), cudaHostRegisterDefault));
            cudaCheck(cudaHostRegister(outGPU.data(),  vecSize * sizeof(float), cudaHostRegisterDefault));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeUpload[i] = duration.count();

            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            cudaCheck(cudaEventRecord(start));
            cudaCheck(cudaEventSynchronize(start));  //If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

            addVectorsGpu<<<blocks, threads>>>(v1_.data(),
                                            v2_.data(),
                                            outGPU.data(),
                                            vecSize,
                                            100);  //call device function to add two vectors and save.

            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaEventRecord(stop));
            cudaCheck(cudaEventSynchronize(stop));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeCalc[i] = duration.count();

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            cudaCheck(cudaHostUnregister(v1_.data()));
            cudaCheck(cudaHostUnregister(v2_.data()));
            cudaCheck(cudaHostUnregister(outGPU.data()));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeDownload[i] = duration.count();

            cudaEventElapsedTime(&timeCPUCalc, start, stop); 
            timeCalcCpu[i] = timeCPUCalc * 1000;

            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));
        }
        
        calculateAverageDeviation();
    }

    void cudaMethod3() {
	  std::cout << "Method 3 \n";

        cudaCheck(cudaMallocHost((void **)&devVec1, vecSize * sizeof(float)));  //allocate memory space for vector in the host memory.
        cudaCheck(cudaMallocHost((void **)&devVec2, vecSize * sizeof(float)));
        cudaCheck(cudaMallocHost((void **)&devVecOut, vecSize * sizeof(float)));

        //////////// Start Average From Here /////////////////////
        for (int i = 0; i < avg; i++) {
            std::fill(outGPU.begin(), outGPU.end(), 0);

            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaMemcpy(devVec1, v1_.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(devVec2, v2_.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeUpload[i] = duration.count();
            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            cudaCheck(cudaEventRecord(start));
            cudaCheck(cudaEventSynchronize(start));  //Waits for an event to complete.If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

            addVectorsGpu<<<blocks, threads>>>(devVec1,
                                            devVec2,
                                            devVecOut,
                                            vecSize,
                                            100);  //call device function to add two vectors and save into vect3Gpu.
            cudaCheck(cudaGetLastError());

            cudaCheck(cudaDeviceSynchronize());

            cudaCheck(cudaEventRecord(stop));
            cudaCheck(cudaEventSynchronize(stop));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeCalc[i] = duration.count();
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaMemcpy(outGPU.data(), devVecOut, vecSize * sizeof(float), cudaMemcpyDeviceToHost));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point

            duration = endTime - startTime;
            timeDownload[i] = duration.count();

            cudaEventElapsedTime(&timeCPUCalc, start, stop); 
            timeCalcCpu[i] = timeCPUCalc * 1000;

            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));
        }
        calculateAverageDeviation();
    
        cudaCheck(cudaFreeHost(devVec1));
        cudaCheck(cudaFreeHost(devVec2));
        cudaCheck(cudaFreeHost(devVecOut));
    }

    void cudaMethod4() {
        std::cout << "Method 4 \n";

        //Using cudaMallocHost for pinning Vector Memory.
        cudaCheck(cudaMallocHost((void **)&devVec1, vecSize * sizeof(float)));  //allocate memory space for vector in the host memory.
        cudaCheck(cudaMallocHost((void **)&devVec2, vecSize * sizeof(float)));
        cudaCheck(cudaMallocHost((void **)&devVecOut, vecSize * sizeof(float)));

        cudaCheck(cudaMalloc((void **)&devVec1Ex, vecSize * sizeof(float)));  //Allocate memory inside the device.
        cudaCheck(cudaMalloc((void **)&devVec2Ex, vecSize * sizeof(float)));
        cudaCheck(cudaMalloc((void **)&devVecOutEx, vecSize * sizeof(float)));

        //////////// Start Average From Here /////////////////////
        for (int i = 0; i < avg; i++) {
            std::fill(outGPU.begin(), outGPU.end(), 0);

            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            memcpy(devVec1, v1_.data(), vecSize * sizeof(float));  //Copy from vector host to pinned buffer Host.
            memcpy(devVec2, v2_.data(), vecSize * sizeof(float));

            cudaCheck(cudaMemcpy(devVec1Ex, devVec1, vecSize * sizeof(float), cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(devVec2Ex, devVec2, vecSize * sizeof(float), cudaMemcpyHostToDevice));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeUpload[i] = duration.count();
            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaEventRecord(start));
            cudaCheck(cudaEventSynchronize(start));  //Waits for an event to complete.If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

            addVectorsGpu<<<blocks, threads>>>(devVec1Ex,
                                            devVec2Ex,
                                            devVecOutEx,
                                            vecSize,
                                            100);  //call device function to add two vectors and save into vect3Gpu.
            cudaCheck(cudaGetLastError());

            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaEventRecord(stop));
            cudaCheck(cudaEventSynchronize(stop));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeCalc[i] = duration.count();
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaMemcpy(devVecOut, devVecOutEx, vecSize * sizeof(float), cudaMemcpyDeviceToHost));
            memcpy(outGPU.data(), devVecOut, vecSize * sizeof(float));  //copy pinned host buffer to vector host.

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            duration = endTime - startTime;
            timeDownload[i] = duration.count();

            cudaEventElapsedTime(&timeCPUCalc, start, stop); 
            timeCalcCpu[i] = timeCPUCalc * 1000;

            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));
        }
        calculateAverageDeviation();

        cudaCheck(cudaFreeHost(devVec1));
        cudaCheck(cudaFreeHost(devVec2));
        cudaCheck(cudaFreeHost(devVecOut));
        cudaCheck(cudaFree(devVec1Ex));
        cudaCheck(cudaFree(devVec2Ex));
        cudaCheck(cudaFree(devVecOutEx));
    }

    void cudaMethod5() {
        cudaCheck(cudaMalloc((void **)&devVec1, vecSize * sizeof(float)));  //allocate memory space for vector in the host memory.
        cudaCheck(cudaMalloc((void **)&devVec2, vecSize * sizeof(float)));
        cudaCheck(cudaMalloc((void **)&devVecOut, vecSize * sizeof(float)));
        std::cout << "Method 5 \n";

        for (int i = 0; i < avg; i++) {
            std::fill(outGPU.begin(), outGPU.end(), 0);

            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            cudaCheck(cudaHostRegister(v1_.data(), vecSize * sizeof(float), cudaHostRegisterDefault));
            cudaCheck(cudaHostRegister(v2_.data(), vecSize * sizeof(float), cudaHostRegisterDefault));
            cudaCheck(cudaHostRegister(outGPU.data(), vecSize * sizeof(float), cudaHostRegisterDefault));

            cudaCheck(cudaMemcpy(devVec1, v1_.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));  //copy pinned vector in the host to buffer in the device.
            cudaCheck(cudaMemcpy(devVec2, v2_.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeUpload[i] = duration.count();
            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.

            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaEventRecord(start));
            cudaCheck(cudaEventSynchronize(start));  //If the cudaEventBlockingSync flag has not been set, then the CPU thread will busy-wait until the event has been completed by the GPU.

            addVectorsGpu<<<blocks, threads>>>(devVec1,
                                            devVec2,
                                            devVecOut,
                                            vecSize,
                                            100);  //call device function to add two vectors and save into vect3Gpu.
            cudaCheck(cudaGetLastError());

            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaEventRecord(stop));
            cudaCheck(cudaEventSynchronize(stop));

            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.
            duration = endTime - startTime;
            timeCalc[i] = duration.count();
            startTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            cudaCheck(cudaMemcpy(outGPU.data(), devVecOut, vecSize * sizeof(float), cudaMemcpyDeviceToHost));  //copy  buffer in the device to pinned vector in the host.


            endTime = std::chrono::steady_clock::now();  //get current tick time  in monotonic point.

            duration = endTime - startTime;
            timeDownload[i] = duration.count();

            cudaCheck(cudaHostUnregister(v2_.data()));
            cudaCheck(cudaHostUnregister(v1_.data()));
            cudaCheck(cudaHostUnregister(outGPU.data()));

            cudaEventElapsedTime(&timeCPUCalc, start, stop); 
            timeCalcCpu[i] = timeCPUCalc * 1000;

            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));
        }
        calculateAverageDeviation();

        cudaCheck(cudaFree(devVec1));
        cudaCheck(cudaFree(devVec2));
        cudaCheck(cudaFree(devVecOut));
    } 
};
