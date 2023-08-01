#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <utility>
#include <chrono>  //Time
#include<tuple>
#include <cassert>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <unistd.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "newTimeCalc.h"
#include "Timing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

std::tuple<int, std::vector<int>, int> parseCommands(int argc, char* argv[]);
void printResults(int vecSize, int iterations, Timing time, int commMethods);
void saveToFile(const std::string &name, const Timing &timing, int vecSize, int iterations);
void printHeader();

int main(int argc, char* argv[]) {
    cms::cudatest::requireDevices(); //pre-built test 

    auto [vecSize, commMethods, iterations] = parseCommands(argc,argv);
    
    std::vector<Timing> results; 
    std::cout << "\n";
    calculate calc(iterations, vecSize);

    for (size_t i = 0; i < commMethods.size(); i++) {
            std::cout << "comm methods: " << commMethods[i] << std::endl;
    }
    for (size_t i = 0; i < commMethods.size(); i++) {
    std::cout << "comm methods: " << commMethods[i] << std::endl;
	results.push_back(calc.calculateAverage(commMethods[i]));
  }
    printHeader();
    for (size_t i = 0; i < commMethods.size(); i++){
        printResults(vecSize, iterations ,results[i], commMethods[i]);
    }
    return 0;
}

void printHeader(){
/*    std::cout << "error: " << time.noError <<
                "IN MAIN Average upload: " << time.timeUploadAvg <<
                " std upload: " << time.timeUploadstd <<
                " Average download: " << time.timeDownloadAvg <<
                " std download: " << time.timeDownloadstd <<
                " Average calc dev: " << time.timeCalcAvg <<
                " std calc dev: " << time.timeCalcstd <<
                " average calc host: " << time.timeCalcCpuAvg <<
                " std calc host: " << time.timeCalcCpustd ;
*/
    const auto COL1 = 25, COL2to10 = 15, COL11 = 11;
    std::string ROW    = "=====================================================================================================";
    std::string DASHES = "-----------------------------------------------------------------------------------------------------";
    std::cout.flags(std::ios::fixed | std::ios::showpoint);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    std::cout << "\n\n\t"<<ROW<<ROW;
    std::cout << "\n\t|| "<<std::left
              <<std::setw(COL1)<<"Communication Method"<<"|| "
              <<std::setw(COL2to10)<<"Upload Avg"<<"|| "
              <<std::setw(COL2to10)<<"Upload Std"<<"|| "
              <<std::setw(COL2to10)<<"Calc Avg"<<"|| "
              <<std::setw(COL2to10)<<"Calc Std"<<"|| "
              <<std::setw(COL2to10)<<"Download Avg"<<"|| "
              <<std::setw(COL2to10)<<"Download Std"<<"|| "
              <<std::setw(COL2to10)<<"Cpu pov Avg"<<"|| "
              <<std::setw(COL2to10)<<"cpu pov Std"<<"|| "
              <<std::setw(COL11)<<"Iterations"<<"||"
              <<std::setw(COL11)<<"Error?"<<"||"
              << "\n\t"<<ROW <<ROW;
}

void printResults(int vecSize, int iterations, Timing time, int commMethods) {
    const std::string COMM_METHOD_NAMES[] = {"Method0","Method1","Method2" ,"Method3", "Method4", "Method5"};
 
    const auto COL1 = 25, COL2to10 = 15, COL11 = 11;
    std::string ROW    = "=====================================================================================================";
    std::string DASHES = "-----------------------------------------------------------------------------------------------------";
    std::cout.flags(std::ios::fixed | std::ios::showpoint);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    //std::cout << "\n\t"<<DASHES<<DASHES;   
    std::string error;
    if(!time.noError){
       error = "Error";
    }else{
       error = "no Error";
        }
              
    std::cout << "\n\t|| " <<std::left
                <<std::setw(COL1) << COMM_METHOD_NAMES[commMethods - 1] << "|| "
                <<std::setw(COL2to10)<<time.timeUploadAvg<<"|| "
                <<std::setw(COL2to10)<<time.timeUploadstd<<"|| "
                <<std::setw(COL2to10)<<time.timeCalcAvg<<"|| "
                <<std::setw(COL2to10)<<time.timeCalcstd<<"|| "
                <<std::setw(COL2to10)<<time.timeDownloadAvg<<"|| "
                <<std::setw(COL2to10)<<time.timeDownloadstd<<"|| "
                <<std::setw(COL2to10)<<time.timeCalcCpuAvg<<"|| "
                <<std::setw(COL2to10)<<time.timeCalcCpustd<<"|| "
                <<std::setw(COL11) << iterations<< "||"
                <<std::setw(COL11) << error << "||";
	     
    std::cout << "1\n\t"<<ROW<<ROW;
    saveToFile(COMM_METHOD_NAMES[commMethods - 1], time , vecSize, iterations);
    
}

void saveToFile(const std::string &name, const Timing &timing, int vecSize, int iterations) {
  std::ofstream file(name + ".txt", std::ios::out | std::ios::app);

  if (!file.is_open()) {
    std::cout << "\nCannot open File nor Create File!" << std::endl;
  }
  file << "| Vector Size: " << vecSize << " |" << "\n| Number of iterations : " << vecSize << " |" << std::endl;
  file << "upload to device: \n" << timing.timeUploadAvg << " " << timing.timeUploadstd << std::endl;
  file << "calculate in kernel: \n" << timing.timeCalcAvg << " " << timing.timeCalcstd << std::endl;
  file << "upload to host: \n" <<  timing.timeDownloadAvg << " " << timing.timeDownloadstd << std::endl;
  file << "calculate in kernel - CPU POV: \n" <<  timing.timeCalcCpuAvg << " " << timing.timeCalcCpustd << std::endl;
  file << "----------------------------------------------------------------" << std::endl;
  file.close();
  if (!file.good()) {
    std::cout << "\n*ERROR While Writing The " + name + " file!!" << std::endl;
  }
}

std::tuple<int, std::vector<int>, int> parseCommands(int argc, char* argv[]){
    const int METHODS_COUNT = 6; 
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
                  std::cout << "this is the input: " << inputNum;
	    	    while(inputNum > 0 ){
			    int digit = inputNum % 10; 
			    if (digit > METHODS_COUNT) {
				    //FIXME: Raise an exception here
				    std::cout << "\n\tError: Argument must be an integer <= " << METHODS_COUNT << std::endl;
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

