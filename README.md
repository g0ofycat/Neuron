# Neuron

fast lightweight general purposed library for supervised training of deep neural nets. handles 1d-3d tensors along with the ability to save and load models (.bin files). also allows for multithreading

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
    nn.train(input, target, 100, 0.1, 64, TrainType::Classification);

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

## multithreading

*basic implentation of multithreading:*

```cpp
#include "../src/Neuron.hpp"
#include <vector>
#include <thread>

const int THREADS_AMOUT = 5;
const int INPUT_SIZE = 64;
const int OUTPUT_SIZE = 10;
const int HIDDEN_NEURONS = 256;
const int HIDDEN_LAYERS = 3;
const int SAMPLES = 1024;
const int BATCH_SIZE = 64;
const int EPOCHS = 10;

void start_benchmark(int id) {
    Neuron nn(INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_LAYERS, OUTPUT_SIZE, 0);

    Tensor input = Tensor::random_tensor({SAMPLES, INPUT_SIZE});
    Tensor target = Tensor::random_tensor({SAMPLES, OUTPUT_SIZE});

    nn.train(input, target, EPOCHS, 0.01, BATCH_SIZE, TrainType::Classification);

    nn.save_model("../training/model/ID_"+std::to_string(id)+"-model_data.bin");
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < THREADS_AMOUT; ++i) {
        threads.emplace_back([i](){ start_benchmark(i); });
    }

    for (std::thread& t : threads) {
        t.join();
    }
    
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

const int EPOCHS = 100;

const int THREADS_AMOUT = 1;
```

*benchmarking code:*

```cpp
void start_benchmark(int id) {
    Neuron nn(INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_LAYERS, OUTPUT_SIZE, 0);

    Tensor input  = Tensor::random_tensor({SAMPLES, INPUT_SIZE});
    Tensor target = Tensor::random_tensor({SAMPLES, OUTPUT_SIZE});

    nn.train(input, target, 1, 0.01, BATCH_SIZE, TrainType::Classification);

    auto start = std::chrono::high_resolution_clock::now();

    nn.train(input, target, EPOCHS, 0.01, BATCH_SIZE, TrainType::Classification);

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Average: " << elapsed_ms / EPOCHS << " ms\n";
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < THREADS_AMOUT; ++i) {
        threads.emplace_back([i](){ start_benchmark(i); });
    }

    for (std::thread& t : threads) {
        t.join();
    }

    return 0;
}
```

### results:

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
      <td>1853.26</td>
      <td>Default compilation, unoptimized</td>
    </tr>
    <tr>
      <td>Flags (PS)</td>
      <td>512.315</td>
      <td>-O3 -march=native -ffast-math -DNDEBUG</td>
    </tr>
    <tr>
      <td>Flags (WSL)</td>
      <td>114.599</td>
      <td>Optimized on WSL, faster CPU execution</td>
    </tr>
  </tbody>
</table>

### specs

**CPU:** AMD Ryzen 5 8640HS (6 cores / 12 threads) w/ Radeon 760M Graphics

**RAM:** 8 GB DDR5

**GPU:** AMD Radeon 760M Graphics

**OS:** Windows 11

**C++ Version:** C++ 17

**Compiler:** g++ 15.2.0

### flags used for fastest training

#### PS:

```
g++ -O3 -march=native -DNDEBUG -funroll-loops -ffast-math main.cpp -o main
```

#### WSL:

```
hipcc -O3 -ffast-math -funroll-loops -march=native -std=c++17 -x c++ -fopenmp -o main main.cpp -lhipblas
```