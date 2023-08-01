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

class RootProcess: public MPIBase {
public:

    RootProcess(int vectorSize){ //int comType, int vectorSize, int avg
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        returnRes.resize(7);
        generateRandomData(vectorSize);
    }

    std::vector<float> part1() override {
            // Send input data from root process to worker processes.
        batchSize = v1_.size() / (size_ - 1);  //execluding root
        extraBatches = v1_.size() % (size_ - 1); //execluding root

        startTime = MPI_Wtime();
        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Send(&v1_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&v2_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                curIdx += curSize;
        }
        endTimeInputRoot = MPI_Wtime(); //timeinputprep root 

        returnRes[0] = (endTimeInputRoot - startTime) * 1000;

        result.resize(v1_.size()) ;

        startTime = MPI_Wtime();
        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Recv(&result[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                curIdx += curSize;
        }
        endTime = MPI_Wtime();
        returnRes[1] = (endTime - startTime)*1000 ; //timeoutputprep root 
        returnRes[5] = (startTime - endTimeInputRoot)*1000 ; // calc on root POV

        MPI_Recv(&timeFromHost[0], 5, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        returnRes[2] = timeFromHost[0];
        returnRes[3] = timeFromHost[1];
        returnRes[4] = timeFromHost[2];
        returnRes[5] = timeFromHost[3];
        returnRes[6] = timeFromHost[4];

        checkResult(result);
        return returnRes;
    }
    
    std::vector<float> part2() override {
            // Send input data from root process to worker processes.
        batchSize = v1_.size() / (size_ - 1);  //execluding root
        extraBatches = v1_.size() % (size_ - 1); //execluding root

        startTime = MPI_Wtime();

        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Send(  &v1_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(  &v2_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                curIdx += curSize;
        }

        endTimeInputRoot = MPI_Wtime();

        returnRes[0] = (endTimeInputRoot - startTime) * 1000;

        result.resize(v1_.size()) ;

        startTime = MPI_Wtime();

        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Recv(&result[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                curIdx += curSize;
	}


        endTime = MPI_Wtime();
        returnRes[1] = (endTime - startTime)*1000 ; //timeoutputprep root 
        returnRes[5] = (startTime - endTimeInputRoot)*1000 ; // calc on root POV

        MPI_Recv(&timeFromHost[0], 5, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        returnRes[2] = timeFromHost[0];
        returnRes[3] = timeFromHost[1];
        returnRes[4] = timeFromHost[2];
        returnRes[5] = timeFromHost[3];
        returnRes[6] = timeFromHost[4];

        checkResult(result);
        return returnRes;
    }

    std::vector<float> part3() override {


        // Send input data from root process to worker processes.
        batchSize = v1_.size() / (size_ - 1);  //execluding root
        extraBatches = v1_.size() % (size_ - 1); //execluding root

        startTime = MPI_Wtime();

        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Send(  &v1_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(  &v2_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                curIdx += curSize;
        }

        endTimeInputRoot = MPI_Wtime();

        returnRes[0] = (endTimeInputRoot - startTime) * 1000;

        result.resize(v1_.size()) ;

        startTime = MPI_Wtime();

        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Recv(&result[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                curIdx += curSize;
        }

        endTime = MPI_Wtime();
        returnRes[1] = (endTime - startTime)*1000 ; //timeoutputprep root 
        returnRes[5] = (startTime - endTimeInputRoot)*1000 ; // calc on root POV

        MPI_Recv(&timeFromHost[0], 5, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        returnRes[2] = timeFromHost[0];
        returnRes[3] = timeFromHost[1];
        returnRes[4] = timeFromHost[2];
        returnRes[5] = timeFromHost[3];
        returnRes[6] = timeFromHost[4];

        checkResult(result);
        return returnRes;
    }
private:
    const int precisionFactor = 4;
    std::vector<float> v1_, v2_;
    std::vector<float> timeFromHost;
    std::vector<float> result;
    std::vector<float> returnRes; 
// the result vector is like this [inputPrepRoot, outputPrepRoot, inputPrepHost, outputPrepHost, calcTimeRoot, calcTimeHost, calcTimeDevice]
    int batchSize;      //the size for each process
    int extraBatches;   //the size for the batches thatll get the extra (%)
    int curIdx;         // used for iterating over
    int size_;          //num of process
    int rank_;          //rank of process - HERE its always zero
    float endTime, endTimeInputRoot;  
    float startTime;
    

    void generateRandomData(int vectorSize) {
            std::random_device rand;  // Random device used to seed the random engine.
            std::default_random_engine gener(rand());  // Default random engine.
            std::uniform_real_distribution<> dis(0., 1.);  // Uniform distribution from 0 to 1.
            // Generate a random number and assign it to the vector element.
            for (int i = 0; i < vectorSize; i++) {
                    v1_.push_back(dis(gener));
                    v2_.push_back(dis(gener));
                //validationReference_.push_back(mainInput1_[i] + mainInput2_[i]);
            }
        }

    void checkResult(std::vector<float> resultVect){

            float totalError{0.0};  // Variable to store the total error.


            // Calculate the percentage difference and accumulate the total error.
            for (size_t i = 0; i < resultVect.size(); i++) {
                    float s = v1_[i] + v2_[i];
                    totalError += ((s - resultVect[i]) / s) * 100.0;
            }

            // If there is a non-zero total error, print the results and error analysis.    
            if ( totalError == 0.0) {
                    return ; // No error Found;
            }

            std::cout << "\n-------------------------------------------------------\n";
            std::cout << "| RootSum | WorksSum | Error   | Error %  | Process # |";
            std::cout << "\n-------------------------------------------------------\n";
            std::cout.precision(precisionFactor);



            int batchSize = v1_.size() / (size_ - 1);  //execluding root
            int extraBatches = v1_.size() % (size_ - 1); //execluding root
            int curBatchSize = batchSize + (extraBatches > 0)? 1 : 0;
            int workerRank = 0;
            for (size_t i = 0; i < resultVect.size(); i++) {
                    float correct = v1_[i] + v2_[i];
                    float error = correct-resultVect[i] ;
                    if(error != 0.0) {
                            float errorPercent = (error/correct)*100.0 ;
                            std::cout << "| " << correct << "  | " << resultVect[i] << "  |"
                                    << std::setw(9) << error << " |"
                                    << std::setw(9) << errorPercent << " |"
                                    << std::setw(9) << workerRank << " |\n";
                    }
                    if(i > static_cast<size_t>(curBatchSize)){
                            workerRank += 1;
                            curBatchSize = batchSize + (extraBatches - workerRank > 0)? 1 : 0;
                    }

            }

            std::cout << "-------------------------------------------------------\n";
            std::cout << "Total Error = " << totalError << std::endl;
    }
};