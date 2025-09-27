#include "../src/Neuron.hpp"

int main() {
    // inputNeurons, hiddenNeurons, hiddenLayers, outputNeurons, dropout_rate
    Neuron nn(3, 5, 2, 2, 0.01);

    Tensor input = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };

    Tensor target = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0}
    };

    // input, target, epochs, learning_rate, batch_size
    nn.train(input, target, 10000, 0.1, 64);

    // nn.save_model("model_data.bin");

    // nn.load_model("model_data.bin");

    Tensor test_input = {0.1, 0.2, 0.3};

    Tensor out_tensor = nn.predict(test_input);

    std::cout << "Output after training:" << std::endl;

    for (double val : out_tensor.data) {
        std::cout << val << " ";
    }

    std::cout << std::endl;

    return 0;
}