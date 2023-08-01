#ifndef PROCESSINTERFACE_H
#define PROCESSINTERFACE_H

class MPIBase {
public:
    virtual std::vector<float> part1() = 0;
    virtual std::vector<float> part2() = 0;
    virtual std::vector<float> part3() = 0;

    std::vector<float> calculateAverageTime(int funcNum, int iterations) {
        std::vector<float> resultsRecieved;
        std::vector<float> inputPrepRoot, outputPrepRoot, inputPrepHost, 
                    outputPrepHost, calcTimeRoot, calcTimeHost, calcTimeDevice;
        
        std::vector<float> resultsSent; 


        for (int i = 0; i < iterations; ++i){
            std::pair<float, float> timeDuration;
            switch (funcNum) {
                case 1:
                    resultsRecieved = part1();
                    break;
                case 2:
                    resultsRecieved = part2(); 
                    break;
                case 3:
                    resultsRecieved = part3(); 
                    break;
                default:
                    std::cerr << "\n\n\tError: Invalid function number!\n";
                    abort();
            }

            inputPrepRoot.push_back(resultsRecieved[0]);
            outputPrepRoot.push_back(resultsRecieved[1]);
            inputPrepHost.push_back(resultsRecieved[2]);
            outputPrepHost.push_back(resultsRecieved[3]);
            calcTimeRoot.push_back(resultsRecieved[4]);
            calcTimeHost.push_back(resultsRecieved[5]);
            calcTimeDevice.push_back(resultsRecieved[6]);
        }

        //results will be in this order [avg, std] in avg and std itll be in this order :
        //inputPrepRoot, outputPrepRoot, inputPrepHost, outputPrepHost, calcTimeRoot, calcTimeHost, calcTimeDevice
        resultsSent.push_back(calculateAverage(inputPrepRoot));
        resultsSent.push_back(calculateAverage(outputPrepRoot));
        resultsSent.push_back(calculateAverage(inputPrepHost));
        resultsSent.push_back(calculateAverage(outputPrepHost));
        resultsSent.push_back(calculateAverage(calcTimeRoot));
        resultsSent.push_back(calculateAverage(calcTimeHost));
        resultsSent.push_back(calculateAverage(calcTimeDevice));

        // Calculate standard deviation for each vector
        resultsSent.push_back(calculateStdDev(inputPrepRoot, resultsSent[0]));
        resultsSent.push_back(calculateStdDev(outputPrepRoot, resultsSent[1]));
        resultsSent.push_back(calculateStdDev(inputPrepHost, resultsSent[2]));
        resultsSent.push_back(calculateStdDev(outputPrepHost, resultsSent[3]));
        resultsSent.push_back(calculateStdDev(calcTimeRoot, resultsSent[4]));
        resultsSent.push_back(calculateStdDev(calcTimeHost, resultsSent[5]));
        resultsSent.push_back(calculateStdDev(calcTimeDevice, resultsSent[6]));

        return resultsSent;
    }

    float calculateAverage(const std::vector<float>& data) {
        if (data.empty()) {
            return 0.0f;
        }

        float sum = 0.0f;
        for (const float& value : data) {
            sum += value;
        }

        return sum / static_cast<float>(data.size());
    }
    float calculateStdDev(const std::vector<float>& data, float average) {
        if (data.size() <= 1) {
            return 0.0f;
        }

        float variance = 0.0f;
        for (const float& value : data) {
            variance += (value - average) * (value - average);
        }
        variance /= static_cast<float>(data.size() - 1);

        return std::sqrt(variance);
    }
};
#endif  // PROCESSINTERFACE_H