#ifndef PROCESSINTERFACE_H
#define PROCESSINTERFACE_H

class MPIBase {
public:
    virtual std::pair<float, float> blockingSend() = 0;
    virtual std::pair<float, float> nonBlockingSend() = 0;
    virtual std::pair<float, float> blockingScatter() = 0;
    virtual std::pair<float, float> nonBlockingScatter() = 0;

    std::pair<float, float> calculateAverageTime(int funcNum, int iterations) {
        std::pair<float, float> averageTime;

        for (unsigned int i = 0; i < iterations; ++i){
            std::pair<float, float> timeDuration;
            switch (funcNum) {
                case 1:
                    timeDuration = nonBlockingScatter();
                    break;
                case 2:
                    timeDuration = blockingScatter(); 
                    break;
                case 3:
                    timeDuration = nonBlockingSend(); 
                    break;
                case 4:
                    timeDuration = blockingSend(); 
                    break;
                case 5:
                    //timeDuration = multiNonBlockingSend(); 
                    break;
                default:
                    std::cerr << "\n\n\tError: Invalid function number!\n";
                    abort() ;
            }

            averageTime.first += timeDuration.first; //sendDuration from root
            averageTime.second += timeDuration.second; //recvDuration to root
        }

        // Calculate the average timings by dividing the accumulated values
        averageTime.first /= iterations;
        averageTime.second /= iterations;

        return averageTime;
    }

};

#endif  // PROCESSINTERFACE_H