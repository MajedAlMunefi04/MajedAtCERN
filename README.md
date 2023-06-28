### Majed @ CERN

# MPI Function Timing Tool 



This tool is designed to measure the performance of different MPI function calls. In particular, it compares the execution times of scatter/send and gather/receive operations. It does this through 2 vector addition.

## Data Structure

MPIData is a struct that encapsulates the various data needed for MPI processing. The struct fields include:

* numProcesses_: The total number of MPI processes.
* processRank_: The rank of the current MPI process.
* taskDistribution_: A pair object that defines the distribution of work among the processes.
* mainInput1_, mainInput2_: The input data vectors.
* collectedOutput_: Output data vector that captures results from worker processes.
* validationReference_: A vector used to verify the results from each process.
* workerInput1_, workerInput2_: Data vectors designated specifically for worker processes.
* displacementIndices_: A vector specifying the starting index for each process's chunk of data.
* numDataPerProcess_: A vector defining the number of data elements to be sent to each process.

## Functions

* convertIntToVector(): Accepts an integer and returns a vector of integers representing the functions selected by the user.
* generateRandomNumbers(): Generates and assigns random floating-point numbers to a given vector.
* distributeTasks(): Determines the quantity of tasks each process should handle, given the total number of tasks and processes.
* calculateNumDataToSend(): Computes how many data elements will be sent to each process.
* calculateDisplacementIndices(): Identifies the index at which each process should begin handling data.
* printResultsAndCheck(): Compares and prints the output results with reference data to ensure correctness.
* The timing functions (nonBlockingScatter(), blockingScatter(), nonBlockingSend(), blockingSend(), multiNonBlockingSend()) each return the average time taken for the scatter/send and gather/receive operations over multiple runs for non-blocking scatter, blocking scatter, non-blocking send, blocking send, and non-blocking send with multiple tasks, respectively.
* The compareExecutionTimes() function prints out the time taken for each function selected by the user.
* The calculateAverageTime() function computes the average time taken over multiple runs for each function.

## Usage

* mpiFunctionCount_: Number of available MPI functions.
* vectorSize_: Default size of data vectors.
* avgRunCount_: Default number of iterations each function will run to calculate the average time.
* decimalPrecision_: Default number of digits after the decimal point in output.
* rootProcess_: Total number of functions in the program.
* userChoice_:User's choice for which function to run.
* userSelectedFunctions_: Vector to store user's function selections.

## Command line arguments:

* -s: Specify the size of data vectors.
* -r: Specify the number of functions to run.
* -i: Specify the number of iterations for each function to calculate the average time.

Example of command line execution:

mpirun -n 3 cleanMPI1 -s 10 -r 12345 -i 2

Please adapt the implementation according to your needs for optimal utility.
