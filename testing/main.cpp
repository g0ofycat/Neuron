#include "../src/Neuron.hpp"
#include "../mnist-master/include/mnist/mnist_reader.hpp"
#include <iostream>

int main() {
    size_t num_samples = 1000;

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
        "../mnist-master/dataset"
    );

    if (dataset.training_images.size() < num_samples)
        num_samples = dataset.training_images.size();

    Tensor input_tensor(std::vector<size_t>{num_samples, 784});
    Tensor target_tensor(std::vector<size_t>{num_samples, 10});

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < 784; ++j)
            input_tensor(i, j) = static_cast<float>(dataset.training_images[i][j]) / 255.0f;

        target_tensor(i, dataset.training_labels[i]) = 1.0f;
    }

    Neuron nn(784, 128, 2, 10, 0.05);
    
    nn.train(input_tensor, target_tensor, 2, 0.01, 64);

    nn.save_model("../training/model/model_data.bin");

    Tensor test_input(std::vector<size_t>{1, 784});
    for (size_t j = 0; j < 784; ++j)
        test_input(0, j) = static_cast<double>(dataset.test_images[0][j]) / 255.0;

    Tensor out_tensor = nn.predict_classes(test_input);

    std::cout << "Prediction: ";
    for (float val : out_tensor.data)
        std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}