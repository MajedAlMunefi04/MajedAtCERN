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
#include "WorkerProcess.h"
#include "RootProcess.h"

int vectorSize_ = 21;                     // Default size of vectors.
int avgRunCount_ = 5;                   // Default number of runs for each function to calculate average time.
int rank;
int mpiFunctionCount_ = 5;
std::vector<int> userSelectedFunctions_;   // Vector to store user's function selections.

std::vector<std::pair<float, float>> executionTiming_(
    mpiFunctionCount_, std::make_pair(0, 0));    // Vector to store execution time of scatter/send and gather/receive for each function.

void parseCommands(int argc, char* argv[]);

const std::vector<int> convertIntToVector(int number);

void compareExecutionTimes(const std::vector<std::pair<float, float>>& executionTimes,
                            const std::vector<int>& decimalPrecisions,
                            int runCount);


std::pair<float, float> calculateAverageTime(int funcNum, WorkerProcess& workerObject, RootProcess& rootObject, int rank);


int main(int argc, char* argv[]) {

    parseCommands(argc,argv);

    MPI_Init(&argc, &argv);                            // Initialize the MPI communicator environment.

    MPI_Comm_size(MPI_COMM_WORLD, &rank);

    WorkerProcess worker(vectorSize_);
    RootProcess root(vectorSize_);

    for (size_t i = 0; i < userSelectedFunctions_.size(); ++i) {
                    executionTiming_[userSelectedFunctions_[i] - 1] = calculateAverageTime(userSelectedFunctions_[i], worker, root, rank);
    }

    compareExecutionTimes(executionTiming_,userSelectedFunctions_, avgRunCount_);

    MPI::Finalize();  // Finalize the MPI environment.

    return 0;
}

std::pair<float, float> calculateAverageTime(int funcNum, WorkerProcess& workerObject, RootProcess& rootObject, int rank) {
        std::pair<float, float> averageTime;
        
        for (unsigned int i = 0; i < avgRunCount_; ++i){
            std::pair<float, float> timing;
            switch (funcNum) {
            case 1:
                if(rank != 0){
                    workerObject.nonBlockingScatter(vectorSize_);
                }else{
                    timing = rootObject.nonBlockingScatter(vectorSize_); // Adjust this according to your implementation
                }
                break;
            case 2:
                if(rank != 0){
                    workerObject.blockingScatter(vectorSize_);
                }else{
                    timing = rootObject.blockingScatter(vectorSize_); // Adjust this according to your implementation
                }
                break;
            case 3:
                if(rank != 0){
                    workerObject.nonBlockingSend(vectorSize_);
                }else{
                    timing = rootObject.nonBlockingSend(vectorSize_); // Adjust this according to your implementation
                }               
                break;
            case 4:
                if(rank != 0){
                    workerObject.blockingSend(vectorSize_);
                }else{
                    timing = rootObject.blockingSend(vectorSize_); // Adjust this according to your implementation
                }                
                break;
            case 5:
                if(rank != 0){
                    workerObject.multiNonBlockingSend(vectorSize_);
                }else{
                    timing = rootObject.multiNonBlockingSend(vectorSize_); // Adjust this according to your implementation
                }                       
                break;
            default:
                std::cout << "\n\n\tError: Invalid function number!\n";
                return averageTime;
            }
            averageTime.first += timing.first;
            averageTime.second += timing.second;
        }

        // Calculate the average timings by dividing the accumulated values
        averageTime.first /= avgRunCount_;
        averageTime.second /= avgRunCount_;

        return averageTime;
 }

void parseCommands(int argc, char* argv[]){
    int c;   // Parsing command-line arguments

    while ((c = getopt(argc, argv, "s:r:n:")) != -1) {
        switch (c) {
        case 's':
            try {
            vectorSize_ = std::stoll(optarg, nullptr, 0);  // Set the vector size based on the command-line argument.
            } catch (std::exception& err) {
            std::cout << "\n\tError: Argument must be an integer!";
            std::cout << "\n\t" << err.what() << std::endl;
            }
            break;
        case 'r': // change to functions
            try {
            userSelectedFunctions_ = convertIntToVector(std::stoll(optarg, nullptr, 0)); // whats in the function input is an int that the user wriote
            } catch (std::exception& err) {
            std::cout << "\n\tError: Argument must be an integer!";
            std::cout << "\n\t" << err.what() << std::endl;
            }
            break;
        case 'n': // change to iterations 
            try {
            avgRunCount_ = std::stoll(optarg, nullptr, 0);  // Set the average run count based on the command-line argument.
            } catch (std::exception& err) {
            std::cout << "\n\tError: Argument must be an integer!";
            std::cout << "\n\t" << err.what() << std::endl;
            }
            break;
        default:
            abort();
        }
    }
}

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

void compareExecutionTimes(const std::vector<std::pair<float, float>>& executionTimes,
                            const std::vector<int>& decimalPrecisions,
                            int runCount) {
        std::cout.setf(std::ios::fixed, std::ios::floatfield);

        int scatterIndex = 0;
        int decimalIndex = 0;

        // Print the method names for non-zero execution times
        for (long unsigned int i = 0; i < executionTimes.size(); ++i) {
            if (executionTimes[i].first) {
            switch (i) {
                case 0:
                std::cout << "\n\t\t(1) Non-Blocking Scatter" << std::endl;
                break;
                case 1:
                std::cout << "\n\t\t(2) Blocking Scatter" << std::endl;
                break;
                case 2:
                std::cout << "\n\t\t(3) Non-Blocking Send and Receive" << std::endl;
                break;
                case 3:
                std::cout << "\n\t\t(4) Blocking Send and Receive" << std::endl;
                break;
                case 4:
                std::cout << "\n\t\t(5) Non-Blocking Send and Receive with Multiple Tasks" << std::endl;
                break;
                default:
                std::cout << "\nSomething went wrong!\n";
            }
            }
        }

        std::cout << "\n\n\t=============================================================";
        std::cout << "\n\t|| Func ||  Scatter/Send ||   Gather/Receive  || Number Run||";
        std::cout << "\n\t=============================================================";

        // Print the execution times and related information
        for (long unsigned int i = 0; i < executionTimes.size(); ++i) {
            if (executionTimes[i].first) {
            if (decimalIndex < scatterIndex) {
                std::cout << "\n\t------------------------------------------------------------";
            }
            std::cout.flags(std::ios::fixed | std::ios::showpoint);
            std::cout.precision(decimalPrecisions[decimalIndex]);
            std::cout << "\n\t||  " << std::setw(1) << decimalPrecisions[decimalIndex] << "   ||     " << std::setw(5) << executionTimes[i].first
                        << "    ||        " << std::setw(5) << executionTimes[i].second << "     ||    " << std::setw(3) << runCount
                        << "    ||";
            scatterIndex += 2;
            ++decimalIndex;
            }
        }

        std::cout << "\n\t=============================================================\n\n";
    }