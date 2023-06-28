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


class MPISetup {
public:

    int numProcesses_{0};                          // Total number of processes.
    int processRank_{0};                           // Rank of the current process.
    std::pair<int, int> taskDistribution_{0, 0};   // Pair representing the distribution of tasks among processes.
    std::vector<float> mainInput1_;                // First input vector.
    std::vector<float> mainInput2_;                // Second input vector.
    std::vector<float> collectedOutput_;           // Vector storing the output collected from worker processes.
    std::vector<float> validationReference_;       // Vector storing reference output to verify results from each process.
    std::vector<float> workerInput1_;              // Subset of 'mainInput1' for worker processes only.
    std::vector<float> workerInput2_;              // Subset of 'mainInput2' for worker processes only.
    std::vector<int> displacementIndices_;         // Vector storing the starting index for each process's data.
    std::vector<int> numDataPerProcess_;           // Number of data elements to be sent for each process.
    std::vector<int> userSelectedFunctions_;       // Vector to store user's function selections.

    MPISetup(int userChoice_, int vectorSize_, int avgRunCount_){
        numProcesses_ = MPI::COMM_WORLD.Get_size();               // Rank of the current process.
        processRank_ = MPI::COMM_WORLD.Get_rank();                // Rank of the current process.
        
        distributeTasks();                                         // SETUP for taskDistribution_ || Pair representing the distribution of tasks among processes.
        calculateNumDataToSend();                                  // SETUP for numDataPerProcess_ || Calculate the number of data elements to be sent for each process.
        calculateDisplacementIndices();                            // SETUP for displacementIndices_ || Vector storing the starting index for each process's data.
        convertIntToVector();                                      // SETUP for userSelectedFunctions_ || Vector to store user's function selections.

        collectedOutput_.resize(vectorSize_);                      // Vector storing the output collected from worker processes.
        workerInput1_.resize(numDataPerProcess_[processRank_]);    // Subset of 'mainInput1' for worker processes only.
        workerInput2_.resize(numDataPerProcess_[processRank_]);    // Subset of 'mainInput2' for worker processes only.

        if (processRank_ == rootProcess_) {
            generateRandomNumbers();                               // SETUP for mainInput 1 and 2Generate random floating-point numbers between 0 and 1 in the root process.
            std::cout << "\n\tNumber of Processes: " << numProcesses_ << std::endl;
            std::cout << "\tTask Distribution First: " << taskDistribution_.first << std::endl;
            std::cout << "\tTask Distribution Second: " << taskDistribution_.second << std::endl;
            for (int j = 0; j < vectorSize_; j++) {
                validationReference_.push_back(mainInput1_[j] + mainInput2_[j]);  // Calculate the sum for verification.
            }
        }
    } // Constructor - parse arguments, initialize variables and data, etc.
   
private:

    //methods used 
    void convertIntToVector() {
        int digit = 1;  // Variable to store each digit.

        while (userChoice_ > 0) {
            digit = userChoice_ % 10;  // Extract the rightmost digit.
            if (digit > mpiFunctionCount_) {  // Check if the digit is within the valid range.
                std::cerr << "\n\tError: Argument must be an integer <= " << mpiFunctionCount_ << std::endl;
                break;  // Return an empty vector to indicate an error.
            }
            userSelectedFunctions_.push_back(digit);  // Store the digit in the vector.
            userChoice_ /= 10;                        // Remove the rightmost digit from the input integer.
        }
        std::reverse(userSelectedFunctions_.begin(), userSelectedFunctions_.end());  // Reverse the order of digits to match the original input.
    }
            
    void distributeTasks() {
        if (numProcesses_ > 1 && numProcesses_ <= vectorSize_) {
            taskDistribution_.first = vectorSize_ / (numProcesses_ - 1);   // Number of tasks for each process.
            taskDistribution_.second = vectorSize_ % (numProcesses_ - 1);  // Extra tasks for the process.
        } else {
            std::cerr << "\tError: Either no workers are found or number of processes is larger than the task length!\n";
        }

        if (!taskDistribution_.first) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                        // Abort the program if the task distribution is not valid.
        }
    }

    void calculateNumDataToSend() {
        numDataPerProcess_.assign(numProcesses_, taskDistribution_.first);
        numDataPerProcess_[0] = 0;

        for (int i = 1; i < taskDistribution_.second + 1; i++) {
            numDataPerProcess_[i] += 1;  // Extra work for each first process.
        }
    }

    void calculateDisplacementIndices(){
        displacementIndices_.assign(numProcesses_, taskDistribution_.first);

        displacementIndices_[0] = 0;
        displacementIndices_[1] = 0;  // Start Here.

        for (int i = 2; i < numProcesses_; i++) {  // neglect root
            displacementIndices_[i] = numDataPerProcess_[i] + displacementIndices_[i];
            // Calculate the starting index for each process's data based on the number of data elements to be sent.
        }
    }

    void generateRandomNumbers() {
        std::random_device rand;  // Random device used to seed the random engine.
        std::default_random_engine gener(rand());  // Default random engine.
        std::uniform_real_distribution<> dis(0., 1.);  // Uniform distribution from 0 to 1.
        // Generate a random number and assign it to the vector element.
        for (int i = 0; i < vectorSize_; i++) {
            mainInput1_.push_back(dis(gener));
            mainInput2_.push_back(dis(gener));    
        }
    }
};
