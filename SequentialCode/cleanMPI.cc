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


//Building a Struct for the data needed to be used for the MPI distributed program
struct MPIData {
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
};


//Global Variables Intializing
int mpiFunctionCount_ = 5;                       // Number of MPI function options.
int vectorSize_ = 21;                     // Default size of vectors.
unsigned int avgRunCount_ = 5;                   // Default number of runs for each function to calculate average time.
int decimalPrecision_ = 4;                       // Default decimal precision for printing.
int rootProcess_ = 0;                            // Rank of the root process.
int userChoice_ = 0;                             // User's choice for which function to run.

std::vector<int> userSelectedFunctions_(1, 1);   // Vector to store user's function selections.

std::vector<std::pair<float, float>> executionTiming_(
    mpiFunctionCount_, std::make_pair(0, 0));    // Vector to store execution time of scatter/send and gather/receive for each function.


//Functions Intialized
const std::vector<int> convertIntToVector(int number);             // Convert an integer to a vector. User inputs an int of the MPI functions he want and this makes it a vect

void generateRandomNumbers(std::vector<float>& vect);              // Generate uniform random numbers - 0 to 1 - in a vector.

std::pair<int, int> distributeTasks(int totalTasks, int numProcs); // Calculate the number of tasks for each process.

const std::vector<int> calculateNumDataToSend(
    int numProcs, std::pair<int, int> taskDist);                    // Find out the number of data elements to be sent for each process.

const std::vector<int> calculateDisplacementIndices(
    int numProcs,
    std::pair<int, int> taskDist,
    const std::vector<int>& numDataToSend);                          // Find out the starting index for each process's data.

void printResultsAndCheck(
    std::vector<float>& referenceOutput,
    std::vector<float>& collectedOutput,
    std::pair<int, int> taskDist,
    const std::vector<int>& displacementIndices,
    const std::vector<int>& numDataToSend);                          // Function to print the results and check if they match the reference output.


// Different MPI function prototypes. These are the ones the user chooses from
const std::pair<float, float> nonBlockingScatter(MPIData& mpiData);
const std::pair<float, float> blockingScatter(MPIData& mpiData);
const std::pair<float, float> nonBlockingSend(MPIData& mpiData);
const std::pair<float, float> blockingSend(MPIData& mpiData);
const std::pair<float, float> multiNonBlockingSend(MPIData& mpiData);

// Print execution times for selected MPI functions.
void compareExecutionTimes(
    const std::vector<std::pair<float, float>>& executionTimes,
    int numChoices,
    const std::vector<int>& decimalPrecisions,
    int runCount);  

// Return average execution time for given MPI function.
const std::pair<float, float> calculateAverageTime(
    const std::pair<float, float> (*mpiFunction)(MPIData&),
    MPIData& mpiData,
    unsigned int runCount);



//Main Starts Here



int main(int argc, char* argv[]) {
  int c;

  // Parsing command-line arguments
  while ((c = getopt(argc, argv, "s:r:n:")) != -1) {
    switch (c) {
      case 's':
        try {
          vectorSize_ = std::stoll(optarg, nullptr, 0);  // Set the vector size based on the command-line argument.
        } catch (std::exception& err) {
          std::cout << "\n\tError: Argument must be an integer!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'r':
        try {
          userChoice_ = std::stoll(optarg, nullptr, 0);  // Set the user's choice based on the command-line argument.
          userSelectedFunctions_ = convertIntToVector(userChoice_);  // Convert the user's choice into a vector of function numbers.
        } catch (std::exception& err) {
          std::cout << "\n\tError: Argument must be an integer!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'n':
        try {
          avgRunCount_ = std::stoll(optarg, nullptr, 0);  // Set the average run count based on the command-line argument.
        } catch (std::exception& err) {
          std::cout << "\n\tError: Argument must be an integer!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      default:
        abort();
    }
  }

  MPIData mpiData;  // Create an object from the MPIData structure to pass into MPI functions.

  MPI_Init(&argc, &argv);                            // Initialize the MPI communicator environment.
  mpiData.numProcesses_ = MPI::COMM_WORLD.Get_size();  // Get the total number of processes.
  mpiData.processRank_ = MPI::COMM_WORLD.Get_rank();       // Get the rank of the current process.

  mpiData.mainInput1_.resize(vectorSize_);                        // Initialize the input and output vectors.
  mpiData.mainInput2_.resize(vectorSize_);
  mpiData.collectedOutput_.resize(vectorSize_);
  mpiData.validationReference_.resize(vectorSize_);

  mpiData.taskDistribution_ = distributeTasks(vectorSize_, mpiData.numProcesses_);  // Calculate the number of tasks for each process.

  if (!mpiData.taskDistribution_.first) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);  // Abort the program if the task distribution is not valid.
    return 0;
  }

  mpiData.numDataPerProcess_ = calculateNumDataToSend(mpiData.numProcesses_, mpiData.taskDistribution_);     // Calculate the number of data elements to be sent for each process.
  mpiData.displacementIndices_ = calculateDisplacementIndices(mpiData.numProcesses_, mpiData.taskDistribution_, mpiData.numDataPerProcess_);  // Calculate the starting index for each process's data.

  mpiData.workerInput1_.resize(mpiData.numDataPerProcess_[mpiData.processRank_]);  // Resize each process's vector to hold the appropriate received data.
  mpiData.workerInput2_.resize(mpiData.numDataPerProcess_[mpiData.processRank_]);

  if (!mpiData.processRank_) {
    generateRandomNumbers(mpiData.mainInput1_);  // Generate random floating-point numbers between 0 and 1 in the root process.
    generateRandomNumbers(mpiData.mainInput2_);
    std::cout << "\n\tNumber of Processes: " << mpiData.numProcesses_ << std::endl;
    std::cout << "\tTask Distribution First: " << mpiData.taskDistribution_.first << std::endl;
    std::cout << "\tTask Distribution Second: " << mpiData.taskDistribution_.second << std::endl;
    for (int j = 0; j < vectorSize_; j++) {
      mpiData.validationReference_[j] = mpiData.mainInput1_[j] + mpiData.mainInput2_[j];  // Calculate the sum for verification.
    }
  }

  // Execute the selected MPI functions and calculate the average execution times
  for (size_t i = 0; i < userSelectedFunctions_.size(); ++i) {
    if (userSelectedFunctions_[i] == 1) {
      executionTiming_[0] = calculateAverageTime(nonBlockingScatter, mpiData, avgRunCount_);
    } else if (userSelectedFunctions_[i] == 2) {
      executionTiming_[1] = calculateAverageTime(blockingScatter, mpiData, avgRunCount_);
    } else if (userSelectedFunctions_[i] == 3) {
      executionTiming_[2] = calculateAverageTime(nonBlockingSend, mpiData, avgRunCount_);
    } else if (userSelectedFunctions_[i] == 4) {
      executionTiming_[3] = calculateAverageTime(blockingSend, mpiData, avgRunCount_);
    } else if (userSelectedFunctions_[i] == 5) {
      executionTiming_[4] = calculateAverageTime(multiNonBlockingSend, mpiData, avgRunCount_);
    } else {
      std::cout << "\n\n\tError: The user has not chosen any function number!\n";
      break;
    }
  }

  if (!mpiData.processRank_) {
    compareExecutionTimes(executionTiming_, mpiFunctionCount_, userSelectedFunctions_, avgRunCount_);  // Compare and print the execution times.
  }

  MPI::Finalize();  // Finalize the MPI environment.

  return 0;
}



//Function Start Here



// Convert an integer to a vector. User inputs an integer representing the MPI functions they want,
// and this function converts it into a vector.

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


// Generate uniform random numbers between 0 and 1 and store them in a vector.

void generateRandomNumbers(std::vector<float>& vect) {
  std::random_device rand;  // Random device used to seed the random engine.
  std::default_random_engine gener(rand());  // Default random engine.
  std::uniform_real_distribution<> dis(0., 1.);  // Uniform distribution from 0 to 1.
  int size = vect.size();  // Size of the vector.

  for (int i = 0; i < size; i++) {
    vect.at(i) = dis(gener);  // Generate a random number and assign it to the vector element.
  }
}


// Calculate the number of tasks for each process based on the total number of tasks and the number
// of processes.

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


// Determine the number of data elements to be sent to each process based on the number of processes
// and the task distribution.

const std::vector<int> calculateNumDataToSend(
    int numProcs, std::pair<int, int> taskDist) {
  std::vector<int> numDataToSend(numProcs, taskDist.first);
  numDataToSend[0] = 0;

  for (int i = 1; i < taskDist.second + 1; i++) {
    numDataToSend[i] += 1;  // Extra work for each first process.
  }

  return numDataToSend;
}


// Determine the starting index for each process's data based on the number of processes, task distribution,
// and the number of data elements to be sent.

const std::vector<int> calculateDisplacementIndices(
    int numProcs,
    std::pair<int, int> taskDist,
    const std::vector<int>& numDataToSend) {
  std::vector<int> displacementIndices(numProcs, taskDist.first);

  displacementIndices[0] = 0;
  displacementIndices[1] = 0;  // Start Here.

  for (int i = 2; i < numProcs; i++) {  // neglect root
    displacementIndices[i] = numDataToSend[i - 1] + displacementIndices[i - 1];
    // Calculate the starting index for each process's data based on the number of data elements to be sent.
  }

  return displacementIndices;
}


// Print the results and check if they match the reference output. It takes the reference output, 
// collected output, task distribution, displacement indices, and the number of data elements sent.

void printResultsAndCheck(
    std::vector<float>& referenceOutput,
    std::vector<float>& collectedOutput,
    std::pair<int, int> taskDist,
    const std::vector<int>& displacementIndices,
    const std::vector<int>& numDataToSend) {

  float percent{0.0};  // Variable to store the percentage difference between reference and output.
  float totalError{0.0};  // Variable to store the total error.
  int processIndex{1};  // Current process index.

  // Calculate the percentage difference and accumulate the total error.
  for (int j = 0; j < vectorSize_; j++) {
    percent = ((referenceOutput[j] - collectedOutput[j]) / referenceOutput[j]) * 100;
    totalError += percent;
  }

  // If there is a non-zero total error, print the results and error analysis.
  if (totalError) {
    std::cout << "\n-------------------------------------------------------\n";
    std::cout << "| RootSum | WorksSum | Error   | Error %  | Process # |";
    std::cout << "\n-------------------------------------------------------\n";
    std::cout.precision(decimalPrecision_);

    // Print the root sum, work sum, error, error percentage, and process number for each data element.
    for (int j = 0; j < vectorSize_; j++) {
      std::cout << "| " << referenceOutput[j] << "  | " << collectedOutput[j] << "  |"
                << std::setw(9) << referenceOutput[j] - collectedOutput[j] << " |"
                << std::setw(9) << percent << " |"
                << std::setw(9) << processIndex << " |\n";

      // Update the current process index if the next displacement index is reached.
      if (j + 1 == displacementIndices[processIndex + 1]) {
        ++processIndex;
      }
    }

    std::cout << "-------------------------------------------------------\n";
    std::cout << "-Total Error is " << totalError << std::endl;

    // Print the number of data elements processed by each process.
    for (long unsigned int j = 1; j < displacementIndices.size(); j++) {
      std::cout << "Process [" << j << "] Worked On " << numDataToSend[j] << " Data\n";
    }
  }
}


// Print the results and check if they match the reference output. It takes the reference output,
// collected output, task distribution, displacement indices, and the number of data elements sent.

const std::pair<float, float> nonBlockingScatter(MPIData& mpiData) {
  std::pair<float, float> returnValue;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  double startTimeScatter = 0;
  double endTimeScatter = 0;
  double startTimeGather = 0;
  double endTimeGather = 0;

  MPI_Request requestRootScatter[2];
  MPI_Request requestRootGather;

  startTimeScatter = MPI_Wtime();  // Get the start time before scattering.

  // Non-Blocking Scatter.
  MPI_Iscatterv(&mpiData.mainInput1_[0],
                &mpiData.numDataPerProcess_[0],
                &mpiData.displacementIndices_[0],
                MPI_FLOAT,
                &mpiData.workerInput1_[0],
                mpiData.numDataPerProcess_[mpiData.processRank_],
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD,
                &requestRootScatter[0]);

  MPI_Iscatterv(&mpiData.mainInput2_[0],
                &mpiData.numDataPerProcess_[0],
                &mpiData.displacementIndices_[0],
                MPI_FLOAT,
                &mpiData.workerInput2_[0],
                mpiData.numDataPerProcess_[mpiData.processRank_],
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD,
                &requestRootScatter[1]);

  MPI_Waitall(2, requestRootScatter, MPI_STATUS_IGNORE);  // Wait for scatter operations to complete.

  endTimeScatter = MPI_Wtime();  // Get the end time after scattering.

  if (mpiData.processRank_ != rootProcess_) {  // Only for worker processes.
    for (long unsigned int i = 0; i < mpiData.workerInput1_.size(); i++) {
      mpiData.workerInput1_[i] += mpiData.workerInput2_[i];
    }
  }

  startTimeGather = MPI_Wtime();  // Get the start time before gathering.

  // Non-Blocking Gathering.
  MPI_Igatherv(&mpiData.workerInput1_[0],
               mpiData.numDataPerProcess_[mpiData.processRank_],
               MPI_FLOAT,
               &mpiData.collectedOutput_[0],
               &mpiData.numDataPerProcess_[0],
               &mpiData.displacementIndices_[0],
               MPI_FLOAT,
               rootProcess_,
               MPI_COMM_WORLD,
               &requestRootGather);

  MPI_Wait(&requestRootGather, MPI_STATUS_IGNORE);  // Wait for gather operation to complete.

  endTimeGather = MPI_Wtime();  // Get the end time after gathering.

  if (mpiData.processRank_ == rootProcess_) {  // Only the root process prints the results.
    printResultsAndCheck(mpiData.validationReference_,
                         mpiData.collectedOutput_,
                         mpiData.taskDistribution_,
                         mpiData.displacementIndices_,
                         mpiData.numDataPerProcess_);

    returnValue.first = (endTimeScatter - startTimeScatter) * 1000;  // Calculate scatter time in milliseconds.
    returnValue.second = (endTimeGather - startTimeGather) * 1000;  // Calculate gather time in milliseconds.
  }

  return returnValue;
}


// Perform blocking scatter operation using MPI. Returns the time taken for scatter and gather.
const std::pair<float, float> blockingScatter(MPIData& mpiData) {
  std::pair<float, float> returnValue;
  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  double startTimeScatter = 0;
  double endTimeScatter = 0;
  double startTimeGather = 0;
  double endTimeGather = 0;

  // Blocking Scattering.
  mpiData.workerInput1_.resize(mpiData.numDataPerProcess_[mpiData.processRank_]);  // Resize each process to receive appropriate data.
  mpiData.workerInput2_.resize(mpiData.numDataPerProcess_[mpiData.processRank_]);

  startTimeScatter = MPI_Wtime();  // Get the start time before scattering.
  MPI_Scatterv(&mpiData.mainInput1_[0],
               &mpiData.numDataPerProcess_[0],
               &mpiData.displacementIndices_[0],
               MPI_FLOAT,
               &mpiData.workerInput1_[0],
               mpiData.numDataPerProcess_[mpiData.processRank_],
               MPI_FLOAT,
               rootProcess_,
               MPI_COMM_WORLD);

  MPI_Scatterv(&mpiData.mainInput2_[0],
               &mpiData.numDataPerProcess_[0],
               &mpiData.displacementIndices_[0],
               MPI_FLOAT,
               &mpiData.workerInput2_[0],
               mpiData.numDataPerProcess_[mpiData.processRank_],
               MPI_FLOAT,
               rootProcess_,
               MPI_COMM_WORLD);

  endTimeScatter = MPI_Wtime();  // Get the end time after scattering.

  if (mpiData.processRank_ != rootProcess_) {  // Only for worker processes.
    for (size_t i = 0; i < mpiData.workerInput1_.size(); i++) {
      mpiData.workerInput1_[i] += mpiData.workerInput2_[i];
    }
  }

  startTimeGather = MPI_Wtime();  // Get the start time before gathering.

  // Blocking Gathering.
  MPI_Gatherv(&mpiData.workerInput1_[0],
              mpiData.numDataPerProcess_[mpiData.processRank_],
              MPI_FLOAT,
              &mpiData.collectedOutput_[0],
              &mpiData.numDataPerProcess_[0],
              &mpiData.displacementIndices_[0],
              MPI_FLOAT,
              rootProcess_,
              MPI_COMM_WORLD);

  endTimeGather = MPI_Wtime();  // Get the end time after gathering.

  if (mpiData.processRank_ == rootProcess_) {  // Only the root process prints the results.
    printResultsAndCheck(mpiData.validationReference_,
                         mpiData.collectedOutput_,
                         mpiData.taskDistribution_,
                         mpiData.displacementIndices_,
                         mpiData.numDataPerProcess_);

    returnValue.first = (endTimeScatter - startTimeScatter) * 1000;  // Calculate scatter time in milliseconds.
    returnValue.second = (endTimeGather - startTimeGather) * 1000;  // Calculate gather time in milliseconds.
  }

  return returnValue;
}


// Perform non-blocking send and receive operation using MPI. Returns the time taken for send and receive.

const std::pair<float, float> nonBlockingSend(MPIData& mpiData) {
  std::pair<float, float> returnValue;

  double startTimeRootSend = 0;
  double endTimeRootSend = 0;
  double startTimeRootRecv = 0;
  double endTimeRootRecv = 0;

  MPI_Request requestRootSend[2];
  MPI_Request requestRootRecv;
  MPI_Request requestWorkerSend;
  MPI_Request requestWorkerRecv[1];

  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  if (mpiData.processRank_ == rootProcess_) {  // Only for the root process.
    startTimeRootSend = MPI_Wtime();  // Get the start time before sending.

    // Send input data from root process to worker processes.
    for (int i = 1; i < mpiData.numProcesses_; i++) {
      MPI_Issend(&mpiData.mainInput1_[mpiData.displacementIndices_[i]],
                 mpiData.numDataPerProcess_[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[0]);  // Tag is 0

      MPI_Issend(&mpiData.mainInput2_[mpiData.displacementIndices_[i]],
                 mpiData.numDataPerProcess_[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[1]);

      MPI_Waitall(2, requestRootSend, MPI_STATUS_IGNORE);
    }

    endTimeRootSend = MPI_Wtime();  // Get the end time after sending.
  }

  if (mpiData.processRank_ != rootProcess_) {  // Only for worker processes.
    // Receive input data from the root process.
    MPI_Irecv(&mpiData.workerInput1_[0],
              mpiData.numDataPerProcess_[mpiData.processRank_],
              MPI_FLOAT,
              rootProcess_,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[0]);

    MPI_Irecv(&mpiData.workerInput2_[0],
              mpiData.numDataPerProcess_[mpiData.processRank_],
              MPI_FLOAT,
              rootProcess_,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[1]);

    MPI_Waitall(2, requestWorkerRecv, MPI_STATUS_IGNORE);

    // Perform computation on the received input data.
    for (size_t i = 0; i < mpiData.workerInput1_.size(); i++) {
      mpiData.workerInput1_[i] += mpiData.workerInput2_[i];
    }

    // Send the computed data back to the root process.
    MPI_Issend(&mpiData.workerInput1_[0],
               mpiData.numDataPerProcess_[mpiData.processRank_],
               MPI_FLOAT,
               rootProcess_,
               0,
               MPI_COMM_WORLD,
               &requestWorkerSend);  // Tag is 0

    MPI_Wait(&requestWorkerSend, MPI_STATUS_IGNORE);
  }

  if (mpiData.processRank_ == rootProcess_) {  // Only for the root process.
    startTimeRootRecv = MPI_Wtime();  // Get the start time before receiving.

    // Receive computed data from worker processes.
    for (int i = 1; i < mpiData.numProcesses_; i++) {
      MPI_Irecv(&mpiData.collectedOutput_[mpiData.displacementIndices_[i]],
                mpiData.numDataPerProcess_[i],
                MPI_FLOAT,
                i,
                0,
                MPI_COMM_WORLD,
                &requestRootRecv);

      MPI_Wait(&requestRootRecv, MPI_STATUS_IGNORE);
    }

    endTimeRootRecv = MPI_Wtime();  // Get the end time after receiving.

    // Print results and check for correctness.
    printResultsAndCheck(mpiData.validationReference_,
                         mpiData.collectedOutput_,
                         mpiData.taskDistribution_,
                         mpiData.displacementIndices_,
                         mpiData.numDataPerProcess_);

    // Calculate the time durations in milliseconds.
    returnValue.first = (endTimeRootSend - startTimeRootSend) * 1000;
    returnValue.second = (endTimeRootRecv - startTimeRootRecv) * 1000;
  }

  return returnValue;
}


// Perform blocking send and receive operation using MPI. Returns the time taken for send and receive.

const std::pair<float, float> blockingSend(MPIData& mpiData) {
  std::pair<float, float> returnValue;

  double startTimeRootSend = 0;
  double endTimeRootSend = 0;
  double startTimeRootRecv = 0;
  double endTimeRootRecv = 0;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  if (mpiData.processRank_ == rootProcess_) {  // Only for the root process.
    startTimeRootSend = MPI_Wtime();  // Get the start time before sending.

    // Send input data from root process to worker processes.
    for (int i = 1; i < mpiData.numProcesses_; i++) {
      MPI_Send(&mpiData.mainInput1_[mpiData.displacementIndices_[i]],
               mpiData.numDataPerProcess_[i],
               MPI_FLOAT,
               i,
               0,
               MPI_COMM_WORLD);  // Tag is 0

      MPI_Send(&mpiData.mainInput2_[mpiData.displacementIndices_[i]],
               mpiData.numDataPerProcess_[i],
               MPI_FLOAT,
               i,
               0,
               MPI_COMM_WORLD);
    }

    endTimeRootSend = MPI_Wtime();  // Get the end time after sending.
  }

  if (mpiData.processRank_ != rootProcess_) {  // Only for worker processes.
    // Receive input data from the root process.
    MPI_Recv(&mpiData.workerInput1_[0],
             mpiData.numDataPerProcess_[mpiData.processRank_],
             MPI_FLOAT,
             rootProcess_,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    MPI_Recv(&mpiData.workerInput2_[0],
             mpiData.numDataPerProcess_[mpiData.processRank_],
             MPI_FLOAT,
             rootProcess_,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    // Perform computations using the received input data.
    for (size_t i = 0; i < mpiData.workerInput1_.size(); i++) {
      mpiData.workerInput1_[i] += mpiData.workerInput2_[i];
    }

    // Send the computed data back to the root process.
    MPI_Send(&mpiData.workerInput1_[0],
             mpiData.numDataPerProcess_[mpiData.processRank_],
             MPI_FLOAT,
             rootProcess_,
             0,
             MPI_COMM_WORLD);  // Tag is 0
  }

  if (mpiData.processRank_ == rootProcess_) {  // Only for the root process.
    startTimeRootRecv = MPI_Wtime();  // Get the start time before receiving.

    // Receive computed data from worker processes.
    for (int i = 1; i < mpiData.numProcesses_; i++) {
      MPI_Recv(&mpiData.collectedOutput_[mpiData.displacementIndices_[i]],
               mpiData.numDataPerProcess_[i],
               MPI_FLOAT,
               i,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    endTimeRootRecv = MPI_Wtime();  // Get the end time after receiving.

    // Print out the results and check if they match the reference output.
    printResultsAndCheck(mpiData.validationReference_,
                         mpiData.collectedOutput_,
                         mpiData.taskDistribution_,
                         mpiData.displacementIndices_,
                         mpiData.numDataPerProcess_);

    returnValue.first = (endTimeRootSend - startTimeRootSend) * 1000;  // Convert to milliseconds.
    returnValue.second = (endTimeRootRecv - startTimeRootRecv) * 1000;  // Convert to milliseconds.
  }

  return returnValue;
}


// Perform non-blocking send and receive operation with multiple tasks using MPI. 
//Returns the time taken for send and receive.

const std::pair<float, float> multiNonBlockingSend(MPIData& mpiData) {
  std::pair<float, float> returnValue;
  int lastPointCount = 0;
  double startTimeRootSend = 0;
  double endTimeRootSend = 0;
  double startTimeRootRecv = 0;
  double endTimeRootRecv = 0;

  MPI_Request requestRootSend[2];
  MPI_Request requestRootRecv;
  MPI_Request requestWorkerSend;
  MPI_Request requestWorkerRecv[2];

  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  if (mpiData.processRank_ == rootProcess_) {  // Only for the root process.
    startTimeRootSend = MPI_Wtime();  // Get the start time before sending.

    int flage = 0;  // Set the operation flag to processed.
    for (int i = 1; i < mpiData.numProcesses_; i++) {
      MPI_Issend(&mpiData.mainInput1_[mpiData.displacementIndices_[i]],
                 mpiData.numDataPerProcess_[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[0]);  // Tag is 0

      MPI_Issend(&mpiData.mainInput2_[mpiData.displacementIndices_[i]],
                 mpiData.numDataPerProcess_[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[1]);

      do {
        MPI_Testall(2, requestRootSend, &flage, MPI_STATUS_IGNORE);  // Check the flag for completion.
        for (; lastPointCount < vectorSize_ && !flage; lastPointCount++) {
          // Perform the summing while waiting for the sending request to complete.
          mpiData.validationReference_[lastPointCount] = mpiData.mainInput1_[lastPointCount] + mpiData.mainInput2_[lastPointCount];
          MPI_Testall(2, requestRootSend, &flage, MPI_STATUS_IGNORE);  // Check the flag for completion.
        }
      } while (!flage);
    }

    endTimeRootSend = MPI_Wtime();  // Get the end time after sending.
  }

  if (mpiData.processRank_ != rootProcess_) {  // Only for worker processes.
    // Receive input data from the root process.
    MPI_Irecv(&mpiData.workerInput1_[0],
              mpiData.numDataPerProcess_[mpiData.processRank_],
              MPI_FLOAT,
              rootProcess_,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[0]);

    MPI_Irecv(&mpiData.workerInput2_[0],
              mpiData.numDataPerProcess_[mpiData.processRank_],
              MPI_FLOAT,
              rootProcess_,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[1]);

    MPI_Waitall(2, requestWorkerRecv, MPI_STATUS_IGNORE);  // Wait for the receive operations to complete.

    // Perform computations using the received input data.
    for (long unsigned int i = 0; i < mpiData.workerInput1_.size(); i++) {
      mpiData.workerInput1_[i] += mpiData.workerInput2_[i];
    }

    // Send the computed data back to the root process.
    MPI_Issend(&mpiData.workerInput1_[0],
               mpiData.numDataPerProcess_[mpiData.processRank_],
               MPI_FLOAT,
               rootProcess_,
               0,
               MPI_COMM_WORLD,
               &requestWorkerSend);  // Tag is 0

    MPI_Wait(&requestWorkerSend, MPI_STATUS_IGNORE);  // Wait for the send operation to complete.
  }

  if (mpiData.processRank_ == rootProcess_) {  // Only for the root process.
    int flage2 = 0;  // Set the operation flag to processed.
    startTimeRootRecv = MPI_Wtime();  // Get the start time before receiving.

    for (int i = 1; i < mpiData.numProcesses_; i++) {
      MPI_Irecv(&mpiData.collectedOutput_[mpiData.displacementIndices_[i]],
                mpiData.numDataPerProcess_[i],
                MPI_FLOAT,
                i,
                0,
                MPI_COMM_WORLD,
                &requestRootRecv);

      do {
        MPI_Test(&requestRootRecv, &flage2, MPI_STATUS_IGNORE);  // Check the flag for completion.
        for (; lastPointCount < vectorSize_ && !flage2; lastPointCount++) {
          // Perform the summing while waiting for the receiving request to complete.
          mpiData.validationReference_[lastPointCount] = mpiData.mainInput1_[lastPointCount] + mpiData.mainInput2_[lastPointCount];
          MPI_Test(&requestRootRecv, &flage2, MPI_STATUS_IGNORE);  // Check the flag for completion.
        }
      } while (!flage2);
    }

    endTimeRootRecv = MPI_Wtime();  // Get the end time after receiving.

    for (; lastPointCount < vectorSize_; lastPointCount++) {
      // Perform the summing for the remaining data points.
      mpiData.validationReference_[lastPointCount] = mpiData.mainInput1_[lastPointCount] + mpiData.mainInput2_[lastPointCount];
    }

    printResultsAndCheck(mpiData.validationReference_,
                         mpiData.collectedOutput_,
                         mpiData.taskDistribution_,
                         mpiData.displacementIndices_,
                         mpiData.numDataPerProcess_); // Print the results and check if they match.

    returnValue.first = (endTimeRootSend - startTimeRootSend) * 1000;  // Convert to milliseconds.
    returnValue.second = (endTimeRootRecv - startTimeRootRecv) * 1000;  // Convert to milliseconds.
  }

  return returnValue;
}


// Compare and print the execution times for selected MPI functions. It takes the 
// execution times, the number of choices, decimal precisions, and the run count.

void compareExecutionTimes(const std::vector<std::pair<float, float>>& executionTimes,
                           int numChoices,
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


// Calculate the average execution time for a given MPI function.
// It takes the MPI function, MPI data, and the run count.

const std::pair<float, float> calculateAverageTime(const std::pair<float, float> (*mpiFunction)(MPIData&),
                                                   MPIData& mpiData,
                                                   unsigned int runCount) {
  std::pair<float, float> averageTime;

  // Perform multiple runs of the MPI function and accumulate the timings
  for (long unsigned int i = 0; i < runCount; ++i) {
    auto timing = mpiFunction(mpiData);
    averageTime.first += timing.first;
    averageTime.second += timing.second;
  }

  // Calculate the average timings by dividing the accumulated values
  averageTime.first /= runCount;
  averageTime.second /= runCount;

  return averageTime;
}
