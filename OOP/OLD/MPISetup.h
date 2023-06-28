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

    int vectSize;
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
    std::vector<int> userSelectedFunctions_;   // Vector to store user's function selections.

    MPISetup(int userChoice_, int vectorSize_){
        numProcesses_ = MPI::COMM_WORLD.Get_size();                                                                                   // Rank of the current process.
        vectSize = vectorSize_;
        processRank_ = MPI::COMM_WORLD.Get_rank();                                                                                    // Rank of the current process.
        taskDistribution_ = distributeTasks(vectorSize_, numProcesses_);                                              // Pair representing the distribution of tasks among processes.

        if (!taskDistribution_.first) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                                                                      // Abort the program if the task distribution is not valid.
        }

        mainInput1_.resize(vectorSize_);                                                                                      // First input vector.
        mainInput2_.resize(vectorSize_);                                                                                     // Second input vector.
        collectedOutput_.resize(vectorSize_);                                                                                // Vector storing the output collected from worker processes.
        validationReference_.resize(vectorSize_);                                                                            // Vector storing reference output to verify results from each process.
        numDataPerProcess_ = calculateNumDataToSend(numProcesses_, taskDistribution_);                                   // Calculate the number of data elements to be sent for each process.
        displacementIndices_ = calculateDisplacementIndices(numProcesses_, taskDistribution_, numDataPerProcess_);       // Vector storing the starting index for each process's data.
        workerInput1_.resize(numDataPerProcess_[processRank_]);                                                              // Subset of 'mainInput1' for worker processes only.
        workerInput2_.resize(numDataPerProcess_[processRank_]);                                                              // Subset of 'mainInput2' for worker processes only.


        userSelectedFunctions_ = convertIntToVector(userChoice_);   // Vector to store user's function selections.

        if (!processRank_) {
            generateRandomNumbers(mainInput1_);  // Generate random floating-point numbers between 0 and 1 in the root process.
            generateRandomNumbers(mainInput2_);
            std::cout << "\n\tNumber of Processes: " << numProcesses_ << std::endl;
            std::cout << "\tTask Distribution First: " << taskDistribution_.first << std::endl;
            std::cout << "\tTask Distribution Second: " << taskDistribution_.second << std::endl;
            for (int j = 0; j < vectorSize_; j++) {
            validationReference_[j] = mainInput1_[j] + mainInput2_[j];  // Calculate the sum for verification.
            }
        }


    } // Constructor - parse arguments, initialize variables and data, etc.
   
private:
    int mpiFunctionCount_ = 5;
    //methods used 

    const std::vector<int> convertIntToVector(int number) {
        std::vector<int> digits;  // Vector to store the individual digits of the input integer.

        int digit = 1;  // Variable to store each digit.

        while (number > 0) {
            digit = number % 10;  // Extract the rightmost digit.
            if (digit > mpiFunctionCount_) {  // Check if the digit is within the valid range.
            std::cout << "\n\tError: Argument must be an integer <= " << mpiFunctionCount_ << std::endl;
            return std::vector<int>();  // Return an empty vector to indicate an error.
            }
            digits.push_back(digit);  // Store the digit in the vector.
            number /= 10;  // Remove the rightmost digit from the input integer.
        }
        std::reverse(digits.begin(), digits.end());  // Reverse the order of digits to match the original input.
        return digits;  // Return the vector containing the individual digits in the correct order.
    }
            
    std::pair<int, int> distributeTasks(int totalTasks, int numProcs) {
        std::pair<int, int> taskDistribution{0, 0};  // Pair to store the task distribution.

        if (numProcs > 1 && numProcs <= totalTasks) {
            taskDistribution.first = totalTasks / (numProcs - 1);   // Number of tasks for each process.
            taskDistribution.second = totalTasks % (numProcs - 1);  // Extra tasks for the process.
        } else {
            std::cout << "\tError: Either no workers are found or number of processes is larger than the task length!\n";
        }

        return taskDistribution;
    }

    const std::vector<int> calculateNumDataToSend(
        int numProcs, std::pair<int, int> taskDist) {
        std::vector<int> numDataToSend(numProcs, taskDist.first);
        numDataToSend[0] = 0;

        for (int i = 1; i < taskDist.second + 1; i++) {
            numDataToSend[i] += 1;  // Extra work for each first process.
        }

        return numDataToSend;
    }


    const std::vector<int> calculateDisplacementIndices(
        int numProcs,
        std::pair<int, int> taskDist,
        const std::vector<int>& numDataToSend){
        std::vector<int> displacementIndices(numProcs, taskDist.first);

        displacementIndices[0] = 0;
        displacementIndices[1] = 0;  // Start Here.

        for (int i = 2; i < numProcs; i++) {  // neglect root
            displacementIndices[i] = numDataToSend[i - 1] + displacementIndices[i - 1];
            // Calculate the starting index for each process's data based on the number of data elements to be sent.
        }

        return displacementIndices;
    }


    void generateRandomNumbers(std::vector<float>& vect) {
        std::random_device rand;  // Random device used to seed the random engine.
        std::default_random_engine gener(rand());  // Default random engine.
        std::uniform_real_distribution<> dis(0., 1.);  // Uniform distribution from 0 to 1.
        int size = vect.size();  // Size of the vector.

        for (int i = 0; i < size; i++) {
            vect.at(i) = dis(gener);  // Generate a random number and assign it to the vector element.
        }
    }

};