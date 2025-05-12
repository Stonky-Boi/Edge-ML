#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Arduino.h>
#include <vector>

class NeuralNetwork {
private:
    const float* dense1_weights; // 784 x 128, stored in PROGMEM
    const float* dense1_bias; // 128, stored in PROGMEM
    const float* dense2_weights; // 128 x 10, stored in PROGMEM
    const float* dense2_bias; // 10, stored in PROGMEM

    // Helper functions
    float relu(float x) const;
    std::vector<float> matmul(const std::vector<float>& a, const float* b, size_t rows, size_t cols) const;
    std::vector<float> add(const std::vector<float>& a, const float* b, size_t size) const;

public:
    NeuralNetwork(const float* w1, const float* b1, const float* w2, const float* b2);
    std::vector<float> predict(const std::vector<float>& input) const;
};

#endif // NEURAL_NETWORK_H