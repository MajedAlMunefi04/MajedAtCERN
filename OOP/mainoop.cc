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
#include "workerProcess.h"
#include "rootProcess.h"
#include "processInterface.h"

int vectorSize_ = 21;                     // Default size of vectors.
int avgRunCount_ = 5;                   // Default number of runs for each function to calculate average time.
int mpiFunctionCount_ = 5;
std::vector<int> userSelectedFunctions_;   // Vector to store user's function selections.

void parseCommands(int argc, char* argv[]);
const std::vector<int> convertIntToVector(int number);
void compareExecutionTimes(const std::vector<std::pair<float, float>>& executionTimes,
                const std::vector<int>& userFuncs,
                int runCount);

std::pair<float, float> calculateAverageTime(int funcNum, MPIBase* p, int rank);

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);                            // Initialize the MPI communicator environment.

    int rank;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    parseCommands(argc,argv);

    MPIBase* MPIObject ;
    if(rank != 0){
        MPIObject = new WorkerProcess();
    }else{
        MPIObject = new RootProcess(vectorSize_);
    }

    std::vector<std::pair<float, float>> executionTiming_(mpiFunctionCount_, std::make_pair(0, 0));

    for (size_t i = 0; i < userSelectedFunctions_.size(); ++i) {
        executionTiming_[userSelectedFunctions_[i] - 1] = MPIObject->calculateAverageTime(userSelectedFunctions_[i], avgRunCount_);
    }

    if (rank == 0){
        compareExecutionTimes(executionTiming_,userSelectedFunctions_, avgRunCount_);
    }

    MPI_Finalize();  // Finalize the MPI environment.

    delete MPIObject;
    MPIObject = nullptr;
    
    return 0;
}


void parseCommands(int argc, char* argv[]){

    int c;   // Parsing command-line arguments

    while ((c = getopt(argc, argv, "s:r:n:")) != -1) {              
        switch (c) {
            case 's':
                try {
                    vectorSize_ = std::stoll(optarg, nullptr, 0);  // Set the vector size based on the command-line argument.
                } catch (std::exception& err) {
                    std::cout << "\n\tError: Argument s must be an integer!";
                    std::cout << "\n\t" << err.what() << std::endl;
                    abort();
                }
                break;
            case 'f':
                try {
                    userSelectedFunctions_ = convertIntToVector(std::stoll(optarg, nullptr, 0)); // whats in the function input is an int that the user wriote
                } catch (std::exception& err) {
                    std::cout << "\n\tError: Argument r must be an integer!";
                    std::cout << "\n\t" << err.what() << std::endl;
                    abort();
                }
                break;
            case 'i':
                try {
                    avgRunCount_ = std::stoll(optarg, nullptr, 0);  // Set the average run count based on the command-line argument.
                } catch (std::exception& err) {
                    std::cout << "\n\tError: Argument n must be an integer!";
                    std::cout << "\n\t" << err.what() << std::endl;
                    abort();
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
                const std::vector<int>& userFuncs,
                int runCount) {

    std::cout.setf(std::ios::fixed, std::ios::floatfield);

    int outputLinesCount = 0;
    int functionIndex = 0;

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
    for (int i = 0; i < executionTimes.size(); ++i) {
        if (executionTimes[i].first) {
            if (functionIndex < outputLinesCount) {
                std::cout << "\n\t------------------------------------------------------------";
            }
            std::cout.flags(std::ios::fixed | std::ios::showpoint);
            std::cout.precision(4);
            std::cout << "\n\t||  " << std::setw(1) << userFuncs[functionIndex] << "   ||     " << std::setw(5) << executionTimes[i].first
                    << "    ||        " << std::setw(5) << executionTimes[i].second << "     ||    " << std::setw(3) << runCount
                    << "    ||";
            outputLinesCount += 2;
            ++functionIndex;
        }
    }

    std::cout << "\n\t=============================================================\n\n";
}
