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
//#include "Timing.h"
#include <cassert>
#include<tuple>
#include <fstream>


//Implement MPI_sendrecv

const int MPI_METHODS_COUNT = 3; 

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
void saveToFile(const std::string &name, std::vector<float> time, int vecSize, int iterations) {
  std::ofstream file("MPICUDA_" + name + ".txt", std::ios::out | std::ios::app);

  if (!file.is_open()) {
    std::cout << "\nCannot open File nor Create File!" << std::endl;
  }
  file << "| Vector Size: " << vecSize << " |" << "\n| Number of iterations : " << vecSize << " |" << std::endl;
  file << "inputPrepRoot \n" << time[0] << " " << time[7] << std::endl;
  file << "outputPrepRoot \n" << time[1] << " " << time[8] << std::endl;
  file << "inputPrepHost \n" << time[2] << " " << time[9] << std::endl;
  file << "outputPrepHost \n" << time[3] << " " << time[10] << std::endl;
  file << "calcTimeRoot \n" << time[4] << " " << time[11] << std::endl;
  file << "calcTimeHost \n" << time[5] << " " << time[12] << std::endl;
  file << "calcTimeDevice \n" << time[6] << " " << time[13] << std::endl;

  file << "----------------------------------------------------------------" << std::endl;
  file.close();
  if (!file.good()) {
    std::cout << "\n*ERROR While Writing The " + name + " file!!" << std::endl;
  }
}
void printHeader(){

    const auto COL1 = 25, COL2to10 = 15, COL11 = 11;
    std::string ROW    = "=====================================================================================================";
    std::string DASHES = "-----------------------------------------------------------------------------------------------------";
    std::cout.flags(std::ios::fixed | std::ios::showpoint);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    std::cout << "\n\n\t"<<ROW<<ROW<<ROW;
    std::cout << "\n\t|| "<<std::left
              <<std::setw(COL1)<<"Communication Method"<<"|| "
              <<std::setw(COL2to10)<<"inputPrepRoot Avg"<<"|| "
              <<std::setw(COL2to10)<<"inputPrepRoot Std"<<"|| "
              <<std::setw(COL2to10)<<"outputPrepRoot Avg"<<"|| "
              <<std::setw(COL2to10)<<"outputPrepRoot Std"<<"|| "
              <<std::setw(COL2to10)<<"inputPrepHost Avg"<<"|| "
              <<std::setw(COL2to10)<<"inputPrepHost Std"<<"|| "
              <<std::setw(COL2to10)<<"outputPrepHost Avg"<<"|| "
              <<std::setw(COL2to10)<<"outputPrepHost Std"<<"|| "
              <<std::setw(COL2to10)<<"calcTimeRoot Avg"<<"|| "
              <<std::setw(COL2to10)<<"calcTimeRoot Std"<<"|| "
              <<std::setw(COL2to10)<<"calcTimeHost Avg"<<"|| "
              <<std::setw(COL2to10)<<"calcTimeHost Std"<<"|| "
              <<std::setw(COL2to10)<<"calcTimeDevice Avg"<<"|| "
              <<std::setw(COL2to10)<<"calcTimeDevice Std"<<"|| "
              <<std::setw(COL11)<<"Iterations"<<"||"
              << "\n\t"<<ROW <<ROW <<ROW;
}
void printResults(int vecSize, int iterations, std::vector<float> time, int commMethods) {
    const std::string COMM_METHOD_NAMES[] = {"Method1","Methods2","Method3"};
 
    const auto COL1 = 25, COL2to10 = 15, COL11 = 11;
    std::string ROW    = "=====================================================================================================";
    std::string DASHES = "-----------------------------------------------------------------------------------------------------";
    std::cout.flags(std::ios::fixed | std::ios::showpoint);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    //std::cout << "\n\t"<<DASHES<<DASHES;   
    std::string error;
        
    std::cout << "\n\t|| " <<std::left
                <<std::setw(COL1) << COMM_METHOD_NAMES[commMethods - 1] << "|| "
                <<std::setw(COL2to10)<<time[0]   <<"|| "
                <<std::setw(COL2to10)<<time[7]   <<"|| "
                <<std::setw(COL2to10)<<time[1]   <<"|| "
                <<std::setw(COL2to10)<<time[8]   <<"|| "
                <<std::setw(COL2to10)<<time[2]   <<"|| "
                <<std::setw(COL2to10)<<time[9]   <<"|| "
                <<std::setw(COL2to10)<<time[3]   <<"|| "
                <<std::setw(COL2to10)<<time[10]  <<"|| "
                <<std::setw(COL2to10)<< time[4]  << "||"
                <<std::setw(COL2to10)<< time[11] << "||"
                <<std::setw(COL2to10)<<time[5]   <<"|| "
                <<std::setw(COL2to10)<<time[12]  <<"|| "
                <<std::setw(COL2to10)<< time[6]  << "||"
                <<std::setw(COL2to10)<< time[13] << "||"
                <<std::setw(COL11)<< iterations  << "||";
	     
    std::cout << "1\n\t"<<ROW<<ROW<<ROW;
    saveToFile(COMM_METHOD_NAMES[commMethods - 1], time , vecSize, iterations); 
}





int main(int argc, char* argv[]) {
    std::vector<std::vector<float>> results;
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    assert(size >= 1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto [vecSize, commMethods, iterations] = parseCommands(argc,argv);
    int numMeths = commMethods.size();
  
    std::unique_ptr<MPIBase> MPIObject ;

    if(rank != 0){
        MPIObject = std::make_unique<WorkerProcess>();
    }else{
        MPIObject = std::make_unique<RootProcess>(vecSize);
    }
	
    results.resize(numMeths);    
    for (int i = 0; i < numMeths; ++i) {
	    results[i] = MPIObject->calculateAverageTime(commMethods[i], iterations);
    } 
    if (rank == 0){
        std::cout << "results \n";
        for(int i = 0; i < numMeths; ++i){
           printHeader();
           printResults(vecSize, iterations, results[i], commMethods[i]);
        }
    }

    MPI_Finalize(); 
    return 0;
}

