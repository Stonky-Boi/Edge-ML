#include <Arduino.h>
#include <NeuralNetwork.h>
#include <pgmspace.h>
#include <dense1_bias.h>
#include <dense2_bias.h>
#include <dense1_weights.h>
#include <dense2_weights.h>
#include <test_images.h>

void setup()
{
  Serial.begin(115200);
  while (!Serial)
    ;
  Serial.println("Starting");

  NeuralNetwork nn(dense1_weights, dense1_bias, dense2_weights, dense2_bias);

  float count = 0;
  float num = 0;
  for (int img = 1; img < 101; ++img)
  {
    std::vector<float> input(784);
    for (int i = 0; i < 784; ++i)
    {
      input[i] = pgm_read_float(&test_images[img * 784 + i]);
    }

    Serial.print("Checking image ");
    Serial.print(img);
    Serial.print(":  ");

    std::vector<float> logits = nn.predict(input);
    if (!logits.empty())
    {

      int predicted = 0;
      float best = logits[0];
      for (int j = 1; j < 10; ++j)
      {
        if (logits[j] > best)
        {
          best = logits[j];
          predicted = j;
        }
      }

      const char *classNames[] = {
          "T-shirt/top",
          "Trouser",
          "Pullover",
          "Dress",
          "Coat",
          "Sandal",
          "Shirt",
          "Sneaker",
          "Bag",
          "Ankle boot"};

      Serial.print("Predicted class: ");
      Serial.print(classNames[predicted]);
      Serial.print(",  ");
      Serial.print("Actual class: ");
      Serial.println(classNames[int(test_labels[img])]);
      num++;
      if (predicted == int(test_labels[img]))
      {
        count++;
      }
      delay(200);
    }
  
    else
    {
      Serial.println("Oops, image " + String(img) + " failed");
    }
  }
  Serial.print("Accuracy: ");
  Serial.print((count / num) * 100);
  Serial.println(" %");
}

void loop()
{
}