# Edge ML: TensorFlow Lite Micro Implementation for ESP8266

[![PlatformIO](https://img.shields.io/badge/PlatformIO-Compatible-brightgreen)](https://platformio.org)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)
![Contributors](https://img.shields.io/github/contributors/Stonky-Boi/Edge_ML)

A lightweight neural network implementation optimized for ESP8266 microcontrollers, originally developed for traffic management and crowd control applications at IIT Indore's Electronics Club.

## Overview
This repository contains a quantized TensorFlow Lite model converted for edge deployment, implementing a two-layer neural network (784-128-10 architecture) trained on the Fashion MNIST dataset. Developed as a fallback solution after AM62A-LP board issues, this implementation demonstrates resource-constrained ML deployment.

## Key Features
- **Memory-efficient design**: 1.2MB flash consumption with PROGMEM storage for model parameters
- **TensorFlow Lite Micro integration**: Quantized model deployment for 8-bit microcontrollers
- **PlatformIO compatibility**: Pre-configured build environment for seamless deployment
- **Real-time inference**: 120ms average prediction time on ESP8266 (80MHz)

## Repository Structure
```
Edge_ML/
├── CONTRIBUTORS.md
├── README.md
├── platformio.ini
└── src/
    ├── NeuralNetwork.cpp
    ├── NeuralNetwork.h
    ├── dense1_bias.h
    ├── dense1_weights.h
    ├── dense2_bias.h
    ├── dense2_weights.h
    ├── main.cpp
    ├── model_data.cc
    ├── model_data.h
    └── test_images.h
```

## Installation
1. **Prerequisites**:
   - PlatformIO Core (VSCode extension recommended)
   - ESP8266 board support

2. **Clone repository**:
```
git clone https://github.com/Stonky-Boi/Edge_ML.git
cd Edge_ML
```

3. **Build & Upload**:
```
pio run -t upload
```

## Usage
```
#include "NeuralNetwork.h"

void setup() {
  Serial.begin(115200);
  NeuralNetwork nn(dense1_weights, dense1_bias, dense2_weights, dense2_bias);
  
  // Load test image from model_data.cc
  std::vector input(test_image, test_image + 784);
  
  auto prediction = nn.predict(input);
  Serial.print("Predicted class: ");
  Serial.println(std::distance(prediction.begin(), 
                 std::max_element(prediction.begin(), prediction.end())));
}
```

## Contributors
See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for the complete list of developers and mentors.

## Training Reference
For model training and quantization:
[Fashion MNIST Training Notebook](https://colab.research.google.com/github/bhattbhavesh91/freecodecamp-tflite/blob/main/tflite-notebook.ipynb)
