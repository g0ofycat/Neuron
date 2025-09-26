#include "../src/Neuron.hpp"

int main() {
    Neuron nn(3, 5, 2, 2, 0.01);

    const std::vector<std::vector<double>>& input = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };

    const std::vector<std::vector<double>>& target = {
        {1.0, 0.0}, 
        {0.0, 1.0}, 
        {1.0, 0.0}
      };

    nn.train(input, target, 1000, 0.01, 64);
    
    // nn.save_model("../training/model_data.bin");

    // nn.load_model("../training/model_data.bin");

    std::vector<double> output = nn.predict({0.1, 0.2, 0.3});

    std::cout << "Output after training:" << std::endl;

    for (double val : output) {
        std::cout << val << " ";
    }

    std::cout << std::endl;

    return 0;
}