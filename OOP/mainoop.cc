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
#include <cassert>
#include<tuple>


//Implement MPI_sendrecv

const int MPI_METHODS_COUNT = 4; 



std::tuple<int, std::vector<int>, int> parseCommands(int argc, char* argv[]){

    enum INPUT_OPTIONS { VECTOR_SIZE = 's', COMMUNICATION_METHOD = 'f', ITERATIONS = 'i'};
    
    //default values
    int vecSize = 100; 
    int iterations = 10; 
    std::vector<int> commMethods; 


    int input;   // Parsing command-line arguments
    
    while ((input = getopt(argc, argv, "s:f:i:")) != -1) {              
        switch (input) {
		case VECTOR_SIZE:
                try {
                    vecSize = std::stoll(optarg, nullptr, 0);
                } catch (std::exception& err) {
                    std::cout << "\n\tError: Argument s must be an integer!";
                    std::cout << "\n\t" << err.what() << std::endl;
                    abort();
                }
                break;
		case COMMUNICATION_METHOD:
                try {
                    int inputNum = std::stoll(optarg, nullptr, 0); // Sending Methods selected by user (e.g. 34 user selected methods blocking and nonblocking)
		    while(inputNum > 0 ){
			    int digit = inputNum % 10; 
			    if (digit > MPI_METHODS_COUNT) {
				    //FIXME: Raise an exception here
				    std::cout << "\n\tError: Argument must be an integer <= " << MPI_METHODS_COUNT << std::endl;
				    abort(); 
			    }
			    commMethods.push_back(digit); 
			    inputNum = inputNum / 10; 
		    }
                } catch (std::exception& err) {
                    std::cout << "\n\tError: Argument r must be an integer!";
                    std::cout << "\n\t" << err.what() << std::endl;
                    abort();
                }
                break;
		case ITERATIONS:
                try {
                    iterations = std::stoll(optarg, nullptr, 0);  // Set the average run count based on the command-line argument.
                } catch (std::exception& err) {
                    std::cout << "\n\tError: Argument n must be an integer!";
                    std::cout << "\n\t" << err.what() << std::endl;
                    abort();
                }
                break;
            default:
		std::cerr<<"\n\t WRONGE INPUT ****** ABORT! "<<input<<"\n"; 
                abort();
        }
    }
    return std::make_tuple(vecSize, commMethods, iterations); 
}

void printResults(const std::vector<std::tuple<int, float, float>> executionTimes, int iterations) {


    const std::string  COMM_METHOD_NAMES[] = {"NONBLOCKING SCATTER", "BLOCKING SCATTER", "NONBLOCKING SEND/RECV", "BLOCKING SEND/RECV"};
    const auto COL1 = 25, COL2 = 15, COL3 = 15, COL4 = 11;
    std::string ROW    = "================================================================================";
    std::string DASHES = "--------------------------------------------------------------------------------";
    std::cout.flags(std::ios::fixed | std::ios::showpoint);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(4);


    std::cout << "\n\n\t"<<ROW;
    std::cout << "\n\t|| "<<std::left
              <<std::setw(COL1)<<"Communication Method"<<"|| "
              <<std::setw(COL2)<<"Scatter/Send"<<"|| "
              <<std::setw(COL3)<<"Gather/Receive"<<"|| "
              <<std::setw(COL4)<<"Iterations"<<"||"
              << "\n\t"<<ROW;



    // Print the execution times and related information
    for (int i = 0; i < executionTimes.size(); ++i) {
	if(i > 0) std::cout << "\n\t"<<DASHES;    
	
	auto [commMethod, avgSendTime, avgRecvTime] = executionTimes[i]; 
           
        std::cout << "\n\t|| " <<std::left
                  << std::setw(COL1) << COMM_METHOD_NAMES[commMethod-1] << "|| "
                  << std::setw(COL2) << avgSendTime << "|| "
                  << std::setw(COL3) << avgRecvTime << "|| "
                  << std::setw(COL4) << iterations<< "||";
	     
    }

    std::cout << "\n\t"<<ROW<<"\n\n";
}


int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv); 

    int rank;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    assert(size >= 2);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto [vecSize, commMethods, iterations] = parseCommands(argc,argv);
    
    std::unique_ptr<MPIBase> MPIObject ;

    if(rank != 0){
        MPIObject = std::make_unique<WorkerProcess>();
    }else{
        MPIObject = std::make_unique<RootProcess>(vecSize);
    }
	
    //tuple<commMethod, avgSendTime, avgRecvTime> 
    std::vector<std::tuple<int, float, float>> results; 
    
    
    for (auto i = 0; i < commMethods.size(); ++i) {
	auto [avgSendTime, avgRecvTime] =  MPIObject->calculateAverageTime(commMethods[i], iterations);
        results.push_back(std::make_tuple(commMethods[i], avgSendTime, avgRecvTime));
    }

    if (rank == 0){
        printResults(results, iterations);
    }

    MPI_Finalize(); 

    return 0;
}

