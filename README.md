# Neuron

fast lightweight general purposed library for supervised training of deep neural nets. handles any dimension tensor along with the ability to save and load models (.bin files)

## basic implementation

```cpp
#include "path/to/Neuron.hpp"

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
    nn.train(input, target, 100, 0.1, 64);

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
```

## benchmarks

**benchmarking will used a medium sized neural network to replicate general purpose data you would see in a normal project**

*neural network settings:*

```cpp
const int INPUT_SIZE = 64;
const int OUTPUT_SIZE = 10;
const int HIDDEN_NEURONS = 256;
const int HIDDEN_LAYERS = 3;
const int SAMPLES = 1024;
const int BATCH_SIZE = 64;
const int EPOCHS = 1;
```

*benchmarking code:*

```cpp
Neuron nn(INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_LAYERS, OUTPUT_SIZE, 0);

Tensor input  = Tensor::random_tensor({SAMPLES, INPUT_SIZE});
Tensor target = Tensor::random_tensor({SAMPLES, OUTPUT_SIZE});

nn.train(input, target, 1, 0.01, BATCH_SIZE);

double total_ms = 0.0;

for (int t = 0; t < TRIALS; t++) {
    auto start = std::chrono::high_resolution_clock::now();

    nn.train(input, target, EPOCHS, 0.01, BATCH_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Trial " << t << ": " << elapsed_ms << " ms\n";
    total_ms += elapsed_ms;
}

std::cout << "Average: " << (total_ms / TRIALS) << " ms\n";

return 0;
```

### results (average over 100 loops):

<table>
  <thead>
    <tr>
      <th>Setup</th>
      <th>Average Time (ms)</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>No Flags (PS)</td>
      <td>1,566.14</td>
      <td>Default compilation, unoptimized</td>
    </tr>
    <tr>
      <td>Flags (PS)</td>
      <td>356.24</td>
      <td>-O3 -march=native -ffast-math -DNDEBUG</td>
    </tr>
    <tr>
      <td>Flags (WSL)</td>
      <td>117.27</td>
      <td>Optimized on WSL, faster CPU execution</td>
    </tr>
  </tbody>
</table>

### specs

**CPU:** AMD Ryzen 5 8640HS (6 cores / 12 threads) w/ Radeon 760M Graphics
**RAM:** 8 GB DDR5
**GPU:** AMD Radeon 760M Graphics
**OS:** Windows 11
**Compiler:** g++ 15.2.0

### flags used for fastest training (in training dir)

#### PS:

```
g++ -O3 -march=native -DNDEBUG -funroll-loops -ffast-math main.cpp -o main
```

#### WSL:

```
nvcc -O3 -use_fast_math -Xcompiler "-fopenmp -march=native -DNDEBUG" main.cpp -o main -lcublas
```