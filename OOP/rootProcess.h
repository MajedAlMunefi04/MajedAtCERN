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

    RootProcess(int vectorSize){ //int comType, int vectorSize, int avg){
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        generateRandomData(vectorSize);
    }

    std::pair<float, float> blockingSend() override {
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

        endTime = MPI_Wtime();

        sendDuration = (endTime - startTime) * 1000;

        result.resize(v1_.size()) ;

        startTime = MPI_Wtime();

        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Recv(&result[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                curIdx += curSize;
        }

        endTime = MPI_Wtime();
        recvDuration = (endTime - startTime)*1000 ;

        checkResult(result);
        return std::pair<float, float>(sendDuration, recvDuration);
    }

    std::pair<float, float> nonBlockingSend() override {

        MPI_Request requestSend[2*(size_ -1)]; //two for each process (one for sending v1 and the other for sending v2)
        MPI_Request requestRecv;

        batchSize = v1_.size() / (size_ - 1);  //execluding root
        extraBatches = v1_.size() % (size_ - 1); //execluding root

        startTime = MPI_Wtime();

        curIdx = 0;

        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Issend(  &v1_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestSend[(i-1)*2]);
                MPI_Issend(  &v2_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestSend[(i-1)*2 + 1]);
                curIdx += curSize;
        }
	MPI_Waitall( 2*(size_ - 1), requestSend, MPI_STATUS_IGNORE);
        endTime = MPI_Wtime();
        sendDuration = (endTime - startTime)*1000;

        result.resize(v1_.size()) ;

        startTime = MPI_Wtime();

        curIdx = 0;
        for (int i = 1; i < size_; i++) {
                int curSize = batchSize + ((extraBatches >= i)? 1 : 0) ; //size of cur Batch
                MPI_Irecv(&result[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestRecv);
                MPI_Wait(&requestRecv, MPI_STATUS_IGNORE);
                curIdx += curSize;
        }

        endTime = MPI_Wtime();
        recvDuration = (endTime - startTime)*1000 ;

        checkResult(result);
        return std::pair<float, float>(sendDuration, recvDuration);
    }


    std::pair<float, float> blockingScatter() override {
        std::vector<int> numDataPerProcess_ (size_, 0);
        std::vector<int> displacementIndices_ (size_, 0);
        
        batchSize = v1_.size() / (size_ - 1);  //execluding root
        extraBatches = v1_.size() % (size_ - 1); //execluding root                   

        curIdx = 0;
        for(int i = 1; i < size_; i++){
            numDataPerProcess_[i] = batchSize + ((extraBatches >= i)? 1 : 0) ;
            displacementIndices_[i] = curIdx;
            curIdx += batchSize + ((extraBatches >= i)? 1 : 0) ;
        }

        for (int i = 1; i < size_; i++) {
            int sizeToSend = batchSize + ((extraBatches >= i) ? 1 : 0);
            MPI_Send(&sizeToSend, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        startTime = MPI_Wtime();
        // Start scattering.
        MPI_Scatterv(
            &v1_[0],
            &numDataPerProcess_[0],
            &displacementIndices_[0],
            MPI_FLOAT,
            NULL,
            0,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
        );
        MPI_Scatterv(
            &v2_[0],
            &numDataPerProcess_[0],
            &displacementIndices_[0],
            MPI_FLOAT,
            NULL,
            0,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
        );

        endTime = MPI_Wtime();
        result.resize(v1_.size()) ;

        sendDuration = (endTime - startTime) * 1000;

        startTime = MPI_Wtime();

        MPI_Gatherv(
            NULL, 
            0, 
            MPI_FLOAT, 
            &result[0], 
            &numDataPerProcess_[0], 
            &displacementIndices_[0], 
            MPI_FLOAT, 
            0, 
            MPI_COMM_WORLD
        );
        endTime = MPI_Wtime();
        recvDuration = (endTime - startTime)*1000 ;

        checkResult(result);

        return std::pair<float, float>(sendDuration, recvDuration);
    }


    std::pair<float, float> nonBlockingScatter() override{
        
        std::vector<int> numDataPerProcess_ (size_, 0);
        std::vector<int> displacementIndices_ (size_, 0);

        batchSize = v1_.size() / (size_ - 1);  //execluding root
        extraBatches = v1_.size() % (size_ - 1); //execluding root

        MPI_Request requestScatter[2];
        MPI_Request requestGather;

        curIdx = 0;
        for(int i = 1; i < size_; i++){
            numDataPerProcess_[i] = batchSize + ((extraBatches >= i)? 1 : 0) ;
            displacementIndices_[i] = curIdx;
            curIdx += batchSize + ((extraBatches >= i)? 1 : 0) ;
        }

        for (int i = 1; i < size_; i++) {
            int sizeToSend = batchSize + ((extraBatches >= i) ? 1 : 0);
            MPI_Send(&sizeToSend, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        

        startTime = MPI_Wtime();  // Get the start time before scattering.

        MPI_Iscatterv(
            &v1_[0],
            &numDataPerProcess_[0],
            &displacementIndices_[0],
            MPI_FLOAT,
            NULL,
            0,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD,
            &requestScatter[0]
        );

        MPI_Iscatterv(
            &v2_[0],
            &numDataPerProcess_[0],
            &displacementIndices_[0],
            MPI_FLOAT,
            NULL,
            0,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD,
            &requestScatter[1]
        );

        endTime = MPI_Wtime();
        result.resize(v1_.size()) ;

        sendDuration = (endTime - startTime) * 1000;

        startTime = MPI_Wtime();
        MPI_Igatherv(
            NULL, 
            0, 
            MPI_FLOAT, 
            &result[0], 
            &numDataPerProcess_[0], 
            &displacementIndices_[0], 
            MPI_FLOAT, 
            0, 
            MPI_COMM_WORLD,
            &requestGather
        );

        MPI_Wait(&requestGather, MPI_STATUS_IGNORE);  // Wait for gather operation to complete.

        endTime = MPI_Wtime();

        recvDuration = (endTime - startTime)*1000 ;

        checkResult(result);

        return std::pair<float, float>(sendDuration, recvDuration);
    }

private:
    const int precisionFactor = 4;
    std::vector<float> v1_;
    std::vector<float> v2_;
    std::vector<float> result ;
    int batchSize;      //the size for each process
    int extraBatches;   //the size for the batches thatll get the extra (%)
    int curIdx;         // used for iterating over
    int size_;          //num of process
    int rank_;          //rank of process - HERE its always zero
    float endTime;  
    float startTime;
    float sendDuration;
    float recvDuration;

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
            for (int i = 0; i < resultVect.size(); i++) {
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
            for (int i = 0; i < resultVect.size(); i++) {
                    float correct = v1_[i] + v2_[i];
                    float error = correct-resultVect[i] ;
                    if(error != 0.0) {
                            float errorPercent = (error/correct)*100.0 ;
                            std::cout << "| " << correct << "  | " << resultVect[i] << "  |"
                                    << std::setw(9) << error << " |"
                                    << std::setw(9) << errorPercent << " |"
                                    << std::setw(9) << workerRank << " |\n";
                    }
                    if(i > curBatchSize){
                            workerRank += 1;
                            curBatchSize = batchSize + (extraBatches - workerRank > 0)? 1 : 0;
                    }

            }

            std::cout << "-------------------------------------------------------\n";
            std::cout << "Total Error = " << totalError << std::endl;

    }
};
