# Neuron

fast lightweight general purposed library for supervised training of deep neural nets

## basic implementation

```cpp
#include "path/to/Neuron.hpp"

int main() {
    // inputNeurons, hiddenNeurons, hiddenLayers, outputNeurons, dropout_rate
    Neuron nn(3, 5, 2, 2, 0.01);

    std::vector<Tensor> input = {
        Tensor({3}).assign_data({0.1, 0.2, 0.3}),
        Tensor({3}).assign_data({0.4, 0.5, 0.6}),
        Tensor({3}).assign_data({0.7, 0.8, 0.9})
    };

    std::vector<Tensor> target = {
        Tensor({2}).assign_data({1.0, 0.0}),
        Tensor({2}).assign_data({0.0, 1.0}),
        Tensor({2}).assign_data({1.0, 0.0})
    };

    // input, target, epochs, learning_rate, batch_size
    nn.train(input, target, 10000, 0.1, 64);

    Tensor test_input({3});
    test_input.data = {0.1, 0.2, 0.3};

    Tensor out_tensor = nn.predict(test_input);

    std::cout << "Output after training:" << std::endl;

    for (double val : out_tensor.data) {
        std::cout << val << " ";
    }

    std::cout << std::endl;

    return 0;
}
```