#include <cassert>
#include <cstdio>
#include <random>
#include <alpaka/alpaka.hpp>
#include "config.h"
#include "workdivision.h"
#include <cassert>
#include<tuple>
#include <cmath>
#include <chrono>


const int partNum = 1; 
const int MPI_METHODS_COUNT = 4; 

std::vector<float> vect1;
std::vector<float> vect2;
std::vector<float> vect3Cpu;  //this is only for Host to verify.
std::vector<float> vect3Gpu;  //this is only for Device.

struct Statistics {
    double stdDeviation1;
    double stdDeviation2;
    double stdDeviation3;
    double average1;
    double average2;
    double average3;
};



bool checkingResultsPrintout(std::vector<float> &vectCpu, std::vector<float> &vectGpu); 
std::tuple<int, std::vector<int>, int> parseCommands(int argc, char* argv[]);
void generateRandomData(int vectorSize);
void addVectorsCpu();
void partT(Host host, Device device, int vecSize, int iterations);
bool saveToFile(const std::string &name, const Statistics &stats, int vecSize, int iterations);
Statistics calcStatistics(const std::vector<std::chrono::duration<double, std::micro>>& durations1,
                               const std::vector<std::chrono::duration<double, std::micro>>& durations2,
                               const std::vector<std::chrono::duration<double, std::micro>>& durations3);

struct VectorAddKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ in1,
                                T const* __restrict__ in2,
                                T* __restrict__ out,
                                Vec1D size) const {
    int reps = 1;
    for (int i = 0; i < reps; ++i) { // this is not needed for functionality but its a choice for stress testing
      for (auto ndindex : elements_with_stride_nd(acc, size)) {
        auto index = ndindex[0];
        out[index] = in1[index] + in2[index];
      }
    }
  }
};


int main(int argc, char* argv[]) {
    auto [vecSize, commMethods, iterations] = parseCommands(argc,argv);

    generateRandomData(vecSize);

    std::size_t n = alpaka::getDevCount<Platform>();
    if (n == 0) {
        exit(EXIT_FAILURE);
    }

    // use the single host device
    Host host = alpaka::getDevByIdx<HostPlatform>(0u);
    std::cout << "Host:   " << alpaka::getName(host) << '\n';

    // use the first device
    Device device = alpaka::getDevByIdx<Platform>(0u);
    std::cout << "Device: " << alpaka::getName(device) << '\n';

    partT(host, device, vecSize, iterations);
}

void generateRandomData(int vectorSize) {
    std::random_device rand;  // Random device used to seed the random engine.
    std::default_random_engine gener(rand());  // Default random engine.
    std::uniform_real_distribution<> dis(0., 1.);  // Uniform distribution from 0 to 1.
    // Generate a random number and assign it to the vector element.
    for (int i = 0; i < vectorSize; i++) {
        vect1.push_back(dis(gener));
        vect2.push_back(dis(gener));
    }
}
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
void partT(Host host, Device device, int vecSize, int iterations) {
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    std::vector<std::chrono::duration<double, std::micro>> timingUpload(iterations);
    std::vector<std::chrono::duration<double, std::micro>> timingCalc(iterations);
    std::vector<std::chrono::duration<double, std::micro>> timiningDownload(iterations);

    // tolerance
    constexpr float epsilon = 0.000001;

    // equiv to the regular std vec vector
    auto v1_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});
    auto v2_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});
    auto out_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});


    for (size_t i = 0; i < vecSize; ++i) {
        v1_[i] = dist(rand);
        v2_[i] = dist(rand);
        out_[i] = 0.0;
    }

  // this is equivelent to the regular cudamalloc
  auto v1_D = alpaka::allocBuf<float, uint32_t>(device, Vec1D{vecSize});
  auto v2_D = alpaka::allocBuf<float, uint32_t>(device, Vec1D{vecSize});
  auto out_D = alpaka::allocBuf<float, uint32_t>(device, Vec1D{vecSize});


  // this is equivalent to a mallocHost
  auto v1_h = alpaka::allocMappedBuf<Platform, float, uint32_t>(host, Vec1D{vecSize});
  auto v2_h = alpaka::allocMappedBuf<Platform, float, uint32_t>(host, Vec1D{vecSize});
  auto out_h = alpaka::allocMappedBuf<Platform, float, uint32_t>(host, Vec1D{vecSize});

  // run the test the given device
  auto queue = Queue{device};
    for (int  i = 0; i < iterations; ++i) {
        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
        alpaka::memcpy(queue, v1_h, v1_);
        alpaka::memcpy(queue, v2_h, v2_);
        alpaka::memcpy(queue, v1_D, v1_h);
        alpaka::memcpy(queue, v2_D, v2_h);    
        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();

        timingUpload[i] = (endTime - startTime) ;

        // fill the output buffer with zeros; the vecSize is known from the buffer objects

        // launch the 1-dimensional kernel with scalar vecSize
        auto div = make_workdiv<Acc1D>(32, 32);
        startTime = std::chrono::steady_clock::now();

        alpaka::exec<Acc1D>(
            queue, div, VectorAddKernel{}, v1_D.data(), v2_D.data(), out_D.data(), vecSize);
        endTime = std::chrono::steady_clock::now();

        timingCalc[i] = (endTime - startTime) ;

            std::cout << "VectorAddKernel with scalar indices with a grid of " << i << "\n";
        startTime = std::chrono::steady_clock::now();
        alpaka::memcpy(queue, out_h, out_D);
        alpaka::memcpy(queue, out_, out_h);
        // copy the results from the device to the host
        endTime = std::chrono::steady_clock::now();

        timiningDownload[i] = (endTime - startTime) ;
       // alpaka::memcpy(queue, vect3Gpu.data(), out_h, vecSize * sizeof(float));

        // wait for all the operations to complete
        alpaka::wait(queue);

    }
   

    Statistics stats = calcStatistics(timingUpload, timingCalc, timiningDownload);

    bool test = 0;

    for (size_t i = 0; i < vecSize; ++i) {
        vect3Cpu.push_back(v1_[i] + v2_[i]);
        std::cout << "out host: " << vect3Cpu[i];
        vect3Gpu.push_back(out_[i]);
        std::cout << " out gpu: " << vect3Gpu[i] << std::endl;
    
    }
    
    test = checkingResultsPrintout(vect3Cpu, vect3Gpu);

           
    if(test) {
        std::cout << "success\n";
        test =saveToFile ("partT", stats, vecSize, iterations);
    }else{
         std::cout << "test: fail" << std::endl;
     }  

}


Statistics calcStatistics(const std::vector<std::chrono::duration<double, std::micro>>& durations1,
                               const std::vector<std::chrono::duration<double, std::micro>>& durations2,
                               const std::vector<std::chrono::duration<double, std::micro>>& durations3) {
    Statistics stats;

    double sum1 = 0.0;
    double sumSquaredDiff1 = 0.0;

    for (const auto& duration : durations1) {
        sum1 += duration.count();
    }

    stats.average1 = sum1 / durations1.size();

    for (const auto& duration : durations1) {
        double diff = duration.count() - stats.average1;
        sumSquaredDiff1 += diff * diff;
    }

    stats.stdDeviation1 = std::sqrt(sumSquaredDiff1 / durations1.size());

    double sum2 = 0.0;
    double sumSquaredDiff2 = 0.0;

    for (const auto& duration : durations2) {
        sum2 += duration.count();
    }

    stats.average2 = sum2 / durations2.size();

    for (const auto& duration : durations2) {
        double diff = duration.count() - stats.average2;
        sumSquaredDiff2 += diff * diff;
    }

    stats.stdDeviation2 = std::sqrt(sumSquaredDiff2 / durations2.size());

    double sum3 = 0.0;
    double sumSquaredDiff3 = 0.0;

    for (const auto& duration : durations3) {
        sum3 += duration.count();
    }

    stats.average3 = sum3 / durations3.size();

    for (const auto& duration : durations3) {
        double diff = duration.count() - stats.average3;
        sumSquaredDiff3 += diff * diff;
    }

    stats.stdDeviation3 = std::sqrt(sumSquaredDiff3 / durations3.size());

    return stats;
}



bool checkingResultsPrintout(std::vector<float> &vectCpu, std::vector<float> &vectGpu) {
  float percent{0.0};
  float totalError{0.0};
  int size = vectCpu.size();
  for (int j = 0; j < size; j++) {
    percent = ((vectCpu[j] - vectGpu[j]) / vectCpu[j]) * 100;
    totalError += percent;
  }
  if (totalError) {
    std::cout << "\n------------------------------------\n";
    std::cout << "| CpuSum | GpuSum | Error  | Error %| ";
    std::cout << "\n------------------------------------\n";
    //std::cout.precision(4);
    for (int j = 0; j < size; j++) {
      std::cout.flags(std::ios::fixed | std::ios::showpoint);
      std::cout.precision(4);
      std::cout << "| " << vectCpu[j] << " | " << vectGpu[j] << " | " << vectCpu[j] - vectGpu[j] << " | " << percent
                << " |\n";
    }
    std::cout << "-------------------------------------\n";
    std::cout << "-Total Error is " << totalError << std::endl;
    return false;
  }
  return true;
}


bool saveToFile(const std::string &name, const Statistics &timing, int vecSize, int iterations) {
  std::ofstream file(name + ".txt", std::ios::out | std::ios::app);

  if (!file.is_open()) {
    std::cout << "\nCannot open File nor Create File!" << std::endl;
    return 0;
  }
  file << "| Vector Size: " << vecSize << " |" << "\n| Number of iterations : " << vecSize << " |" << std::endl;
  file << "upload to device: \n" << timing.average1 << " " << timing.stdDeviation1 << std::endl;
  file << "calculate in kernel: \n" << timing.average2 << " "
       << timing.stdDeviation2<< std::endl;
  file << "upload to host: \n" <<  timing.average3 << " " << timing.stdDeviation3
       << std::endl;
  file << "----------------------------------------------------------------" << std::endl;


  file.close();
  if (!file.good()) {
    std::cout << "\n*ERROR While Writing The " + name + " file!!" << std::endl;
    return 0;
  }
  return 1;
}
