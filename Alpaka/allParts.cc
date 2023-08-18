// This is the first part
// this is equivlent to part 3 of cmssw/HeterogeneousCore/CUDACore/test/cudaTimeMeasurement.cu
// it is taking the vector from the host equv
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


    // this is equivalent to a std vector
    auto v1_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});
    auto v2_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});
    auto out_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});


    // this is equivalent to a mallocHost
    auto v1_h = alpaka::allocMappedBuf<Platform, float, uint32_t>(host, Vec1D{vecSize});
    auto v2_h = alpaka::allocMappedBuf<Platform, float, uint32_t>(host, Vec1D{vecSize});
    auto out_h = alpaka::allocMappedBuf<Platform, float, uint32_t>(host, Vec1D{vecSize});


    for (size_t i = 0; i < vecSize; ++i) {
        v1_[i] = dist(rand);
        v2_[i] = dist(rand);
        out_h[i] = 0.;
    }

  // run the test the given device
  auto queue = Queue{device};
    for (int  i = 0; i < iterations; ++i) {
        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
        alpaka::memcpy(queue, v1_h, v1_);
        alpaka::memcpy(queue, v2_h, v2_);
        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();

        timingUpload[i] = (endTime - startTime) ;

        // fill the output buffer with zeros; the vecSize is known from the buffer objects

        // launch the 1-dimensional kernel with scalar vecSize
        auto div = make_workdiv<Acc1D>(32, 32);
        startTime = std::chrono::steady_clock::now();

        alpaka::exec<Acc1D>(
            queue, div, VectorAddKernel{}, v1_h.data(), v2_h.data(), out_h.data(), vecSize);
        endTime = std::chrono::steady_clock::now();

        timingCalc[i] = (endTime - startTime) ;

            std::cout << "VectorAddKernel with scalar indices with a grid of " << i << "\n";
        startTime = std::chrono::steady_clock::now();
        alpaka::memcpy(queue, out_, out_h);
        // copy the results from the device to the host
        endTime = std::chrono::steady_clock::now();

        timiningDownload[i] = (endTime - startTime) ;
       // alpaka::memcpy(queue, vect3Gpu.data(), out_h, vecSize * sizeof(float));

        // wait for all the operations to complete
        alpaka::wait(queue);

    }
   

    Statistics stats = calcStatistics(timingUpload, timingCalc, timiningDownload);

    // Print the results
    std::cout << "Standard Deviation: " << stats.stdDeviation1 << std::endl;
    std::cout << "Average: " << stats.average1 << std::endl;

    bool test = 0;

    for (size_t i = 0; i < vecSize; ++i) {
        vect3Cpu.push_back(v1_h[i] + v2_h[i]);
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





/// This part is equal to method 0
/// PLEASE NOTE!!
/// This was not tested since it can only be called for a specific gpu that i dont have access to at the moment




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

    auto v1_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});
    auto v2_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});
    auto out_ = alpaka::allocBuf<float, uint32_t>(host, Vec1D{vecSize});


    for (size_t i = 0; i < vecSize; ++i) {
        v1_[i] = dist(rand);
        v2_[i] = dist(rand);
        out_[i] = 0.0;
    }

  // run the test the given device
  auto queue = Queue{device};
    for (int  i = 0; i < iterations; ++i) {
        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();

        timingUpload[i] = (endTime - startTime) ;

        // fill the output buffer with zeros; the vecSize is known from the buffer objects

        // launch the 1-dimensional kernel with scalar vecSize
        auto div = make_workdiv<Acc1D>(32, 32);
        startTime = std::chrono::steady_clock::now();

        alpaka::exec<Acc1D>(
            queue, div, VectorAddKernel{}, v1_.data(), v2_.data(), out_.data(), vecSize);
        endTime = std::chrono::steady_clock::now();

        timingCalc[i] = (endTime - startTime) ;

            std::cout << "VectorAddKernel with scalar indices with a grid of " << i << "\n";
        startTime = std::chrono::steady_clock::now();

        // copy the results from the device to the host
        endTime = std::chrono::steady_clock::now();

        timiningDownload[i] = (endTime - startTime) ;
       // alpaka::memcpy(queue, vect3Gpu.data(), out_h, vecSize * sizeof(float));

        // wait for all the operations to complete
        alpaka::wait(queue);

    }
   

    Statistics stats = calcStatistics(timingUpload, timingCalc, timiningDownload);

    // Print the results
    std::cout << "Standard Deviation: " << stats.stdDeviation1 << std::endl;
    std::cout << "Average: " << stats.average1 << std::endl;

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




///
/// this is equivlent to method 1 
/// it takes a std vector and copy to a cuda malloced memory buffer then push to kernel
///



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

  // run the test the given device
  auto queue = Queue{device};
    for (int  i = 0; i < iterations; ++i) {
        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
        alpaka::memcpy(queue, v1_D, v1_);
        alpaka::memcpy(queue, v2_D, v2_);    
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
        alpaka::memcpy(queue, out_, out_D);   
        // copy the results from the device to the host
        endTime = std::chrono::steady_clock::now();

        timiningDownload[i] = (endTime - startTime) ;
       // alpaka::memcpy(queue, vect3Gpu.data(), out_h, vecSize * sizeof(float));

        // wait for all the operations to complete
        alpaka::wait(queue);

    }
   

    Statistics stats = calcStatistics(timingUpload, timingCalc, timiningDownload);

    // Print the results
    std::cout << "Standard Deviation: " << stats.stdDeviation1 << std::endl;
    std::cout << "Average: " << stats.average1 << std::endl;

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






/// This is equivlant to method 4 
/// it memcpys to the cudamalloced memory buffer then memcpy to the caudeHostMalloced Memory
/// in the original code it used memcpy for the first then cudamemcpy for the second for my case I had to used alpaka::memcpy tw
/// so check if we can do it t



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
