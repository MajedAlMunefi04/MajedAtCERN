//initializing the libraries needed
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <mpi.h>
#include <unistd.h>
#include <cmath>  // for abs() from <cmath>
#include "processInterface.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

__global__ void addVectorsGpu(float *vect1, float *vect2, float *vect3, int size, int taskN) {
        //blockDim.x gives the number of threads in a block, in the x direction.
        //gridDim.x gives the number of blocks in a grid, in the x direction.
        //blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case).
        int first = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = 0; i < taskN; ++i) {
            for (int j = first; j < size; j += stride) {
            vect3[j] = vect2[j] + vect1[j];
            }
        }
    }
class WorkerProcess : public MPIBase {
public:
    WorkerProcess(){
            cms::cudatest::requireDevices();
            MPI_Comm_size(MPI_COMM_WORLD, &size_);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            //assert(rank_ != 0  && size_ >= 2); 

    }

   

    std::vector<float> part1() override {
            //ROOT RANK IS ALWAYS ZERO 
            std::fill(vOut.begin(), vOut.end(), 0);

            MPI_Status status;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_FLOAT, &messageSize);

            // Resize vectors
            v1.resize(messageSize);
            v2.resize(messageSize);
            vecSize = v1.size();

            cudaCheck(cudaMalloc((void **)&dVect1, vecSize * sizeof(float)));  //allocate memory space for vector in the global memory of the Device.
            cudaCheck(cudaMalloc((void **)&dVect2, vecSize * sizeof(float)));
            cudaCheck(cudaMalloc((void **)&dVect3, vecSize * sizeof(float)));   

            startTime = MPI_Wtime();
            MPI_Recv(&v1[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&v2[0], messageSize, MPI_FLOAT, 0 ,0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            cudaCheck(cudaMemcpy(dVect1, v1.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));  //copy random vector from host to device.
            cudaCheck(cudaMemcpy(dVect2, v2.data(), vecSize * sizeof(float), cudaMemcpyHostToDevice));
            endTime = MPI_Wtime(); //timeinputprep root 
            
            timeRoot[0] = (endTime - startTime) * 1000; 

            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));

            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.
            
            cudaCheck(cudaEventRecord(start));
            startTime = MPI_Wtime();
            addVectorsGpu<<<blocks, threads>>>(dVect1,
                                            dVect2,
                                            dVect3,
                                            vecSize,
                                            100);  //call device function to add two vectors and save into vect3Gpu.
            
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaEventRecord(stop));
            endTime = MPI_Wtime(); //timekernel host  
            
            timeRoot[2] = (endTime - startTime) * 1000; 

	        vOut.resize(vecSize);
            cudaEventElapsedTime(&operationOnDevice,
                                start,
                                stop);  //get the time elapse in Device operation with device perspective.
            timeRoot[3] = operationOnDevice;

            startTime = MPI_Wtime();
            cudaCheck(cudaMemcpy(vOut.data(), dVect3, vecSize * sizeof(float), cudaMemcpyDeviceToHost));  //copy summing result vector from Device to Host.// Try_Regist(3) delete this
            MPI_Send(&vOut[0], vOut.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            endTime = MPI_Wtime(); //timekernel host 
            timeRoot[1] = (endTime - startTime) * 1000; 

            cudaCheck(cudaFree(dVect1));
            cudaCheck(cudaFree(dVect2));
            cudaCheck(cudaFree(dVect3));
            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));
            
            MPI_Send(&timeRoot[0], timeRoot.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    	    return std::vector<float>{0.0 ,0.0};
    }

    std::vector<float> part2() override {
            //ROOT RANK IS ALWAYS ZERO 
            float *dVect1Host, *dVect2Host, *dVect3Host;

            MPI_Status status;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_FLOAT, &messageSize);

            // Resize vectors
            v1.resize(messageSize);
            v2.resize(messageSize);
            vecSize = v1.size();

            cudaCheck(cudaMalloc((void **)&dVect1, vecSize * sizeof(float)));  //allocate memory space for vector in the global memory of the Device.
            cudaCheck(cudaMalloc((void **)&dVect2, vecSize * sizeof(float)));
            cudaCheck(cudaMalloc((void **)&dVect3, vecSize * sizeof(float)));
            cudaCheck(cudaMallocHost((void **)&dVect1Host, vecSize * sizeof(float)));  //allocate memory space for vector in the global memory of the Device.
            cudaCheck(cudaMallocHost((void **)&dVect2Host, vecSize * sizeof(float)));
            cudaCheck(cudaMallocHost((void **)&dVect3Host, vecSize * sizeof(float)));
            std::fill(vOut.begin(), vOut.end(), 0);

            startTime = MPI_Wtime();
            MPI_Recv(&dVect1[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&dVect2[0], messageSize, MPI_FLOAT, 0 ,0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            cudaCheck(cudaMemcpy(dVect1Host, dVect1, vecSize * sizeof(float), cudaMemcpyHostToDevice));  //copy random vector from host to device.
            cudaCheck(cudaMemcpy(dVect2Host, dVect2, vecSize * sizeof(float), cudaMemcpyHostToDevice));
            
            endTime = MPI_Wtime(); //timeinputprep root 
            timeRoot[0] = (endTime - startTime) * 1000; 

            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));       
            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.
            
            cudaCheck(cudaEventRecord(start));
            startTime = MPI_Wtime();
            addVectorsGpu<<<blocks, threads>>>(dVect1Host,
                                            dVect2Host,
                                            dVect3Host,
                                            vecSize,
                                            100);  //call device function to add two vectors and save into vect3Gpu.
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaEventRecord(stop));
            endTime = MPI_Wtime(); //timekernel host  

            timeRoot[2] = (endTime - startTime) * 1000; 

	        vOut.resize(vecSize);
            cudaEventElapsedTime(&operationOnDevice,
                                start,
                                stop);  //get the time elapse in Device operation with device perspective.
            timeRoot[3] = operationOnDevice;
            
            startTime = MPI_Wtime();
            cudaCheck(cudaMemcpy(dVect3, dVect3Host, vecSize * sizeof(float), cudaMemcpyDeviceToHost));  //copy summing result vector from Device to Host.// Try_Regist(3) delete this
            MPI_Send(&dVect3[0], vecSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            endTime = MPI_Wtime(); //timekernel host 
            timeRoot[1] = (endTime - startTime) * 1000; 

            cudaCheck(cudaFree(dVect1));
            cudaCheck(cudaFree(dVect2));
            cudaCheck(cudaFree(dVect3));
            cudaCheck(cudaFreeHost(dVect1Host));
            cudaCheck(cudaFreeHost(dVect2Host));
            cudaCheck(cudaFreeHost(dVect3Host));
            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));

            MPI_Send(&timeRoot[0], timeRoot.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

	    return std::vector<float>{0.0 ,0.0};
    }

    std::vector<float> part3() override {
            //ROOT RANK IS ALWAYS ZERO 

            MPI_Status status;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_FLOAT, &messageSize);

            // Resize vectors
            v1.resize(messageSize);
            v2.resize(messageSize);
            vecSize = v1.size();

            
            cudaCheck(cudaMalloc((void **)&dVect1, vecSize * sizeof(float)));  //allocate memory space for vector in the global memory of the Device.
            cudaCheck(cudaMalloc((void **)&dVect2, vecSize * sizeof(float)));
            cudaCheck(cudaMalloc((void **)&dVect3, vecSize * sizeof(float)));
            
            startTime = MPI_Wtime();
            MPI_Recv(&dVect1[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&dVect2[0], messageSize, MPI_FLOAT, 0 ,0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            endTime = MPI_Wtime(); //timeinputprep root 
            timeRoot[0] = (endTime - startTime) * 1000; 

            std::fill(vOut.begin(), vOut.end(), 0);

            cudaCheck(cudaEventCreate(&start));  //inialize Event.
            cudaCheck(cudaEventCreate(&stop));   
            int threads = 512;                                    //arbitrary number.
            int blocks = (vecSize + threads - 1) / threads;  //get ceiling number of blocks.
            blocks = std::min(blocks, 8);  // Number 8 is least number can be got from lowest Nevedia GPUs.
            
            cudaCheck(cudaEventRecord(start));
            startTime = MPI_Wtime();
            addVectorsGpu<<<blocks, threads>>>(dVect1,
                                            dVect2,
                                            dVect3,
                                            vecSize,
                                            100);  //call device function to add two vectors and save into vect3Gpu.
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaEventRecord(stop));

            endTime = MPI_Wtime(); //timekernel host  
            timeRoot[2] = (endTime - startTime) * 1000; 

	        vOut.resize(vecSize);
            cudaEventElapsedTime(&operationOnDevice,
                                start,
                                stop);  //get the time elapse in Device operation with device perspective.
            timeRoot[3] = operationOnDevice;
        
            startTime = MPI_Wtime();
            MPI_Send(&dVect3[0], vecSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            endTime = MPI_Wtime(); //timekernel host 
            timeRoot[1] = (endTime - startTime) * 1000; 

            cudaCheck(cudaFree(dVect1));
            cudaCheck(cudaFree(dVect2));
            cudaCheck(cudaFree(dVect3));
            cudaCheck(cudaEventDestroy(start));
            cudaCheck(cudaEventDestroy(stop));

            MPI_Send(&timeRoot[0], timeRoot.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    	    return std::vector<float>{0.0 ,0.0};
    }
private:
    std::vector<float> timeRoot {4,0};// the timeSend vector is like this [inputPrepHost, outputPrepHost, calcTimeHost, calcTimeDevice]
    float *dVect1, *dVect2, *dVect3;
    int size_;
    int rank_;
    int messageSize;           //size for the worker vects
    int vecSize;
    float startTime, endTime, operationOnDevice;
    std::vector<float> v1;
    std::vector<float> v2;
    std::vector<float> vOut;
    cudaEvent_t start, stop;

};
