#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork( const float* w1,  const float* b1,  const float* w2, const float* b2) {
    dense1_weights = w1;
    dense1_bias = b1;
    dense2_weights = w2;
    dense2_bias = b2;
}

float NeuralNetwork::relu(float x) const {
    return x > 0 ? x : 0;
}

std::vector<float> NeuralNetwork::matmul(const std::vector<float>& a, const float* b, size_t rows, size_t cols) const {
    if (a.size() != rows) {
        Serial.println("Invalid matrix dimensions");
        return {};
    }
    std::vector<float> result(cols, 0.0f);
    for (size_t j = 0; j < cols; ++j) {
        for (size_t k = 0; k < rows; ++k) {
            result[j] += a[k] * pgm_read_float(&b[k * cols + j]);
        }
    }
    return result;
}

std::vector<float> NeuralNetwork::add(const std::vector<float>& a, const float* b, size_t size) const {
    if (a.size() != size) {
        Serial.println("Vector size mismatch");
        return {};
    }
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + pgm_read_float(&b[i]);
    }
    return result;
}

std::vector<float> NeuralNetwork::predict(const std::vector<float>& input) const {
    if (input.size() != 784) {
        Serial.println("Input size must be 784");
        return {};
    }

    // First dense layer
    std::vector<float> layer1 = matmul(input, dense1_weights, 784, 128);
    if (layer1.empty()) return {};
    layer1 = add(layer1, dense1_bias, 128);
    if (layer1.empty()) return {};

    // ReLU activation
    for (float& x : layer1) {
        x = relu(x);
    }

    // Second dense layer
    std::vector<float> layer2 = matmul(layer1, dense2_weights, 128, 10);
    if (layer2.empty()) return {};
    layer2 = add(layer2, dense2_bias, 10);

    return layer2; // Logits
}