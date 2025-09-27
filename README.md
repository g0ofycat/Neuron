# Neuron

fast lightweight general purposed library for supervised training of deep neural nets. handles 1d-3d tensors along with the ability to save and load models (.bin files)

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
```

## pybind11 bindings if u wanna use it in python

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "path/to/Neuron.hpp"

namespace py = pybind11;

PYBIND11_MODULE(Neuron, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<std::initializer_list<double>>())
        .def(py::init<std::initializer_list<std::initializer_list<double>>>())
        .def(py::init<const std::vector<size_t>&>())
        .def_static("zeros", [](const std::vector<size_t>& dims){ return Tensor(dims); })
        .def_static("random_normal", &Tensor::random_normal, 
                    py::arg("dims"), py::arg("mean")=0.0, py::arg("stddev")=1.0)
        .def_static("extract_row", &Tensor::extract_row)
        .def("ndim", &Tensor::ndim)
        .def("numel", &Tensor::numel)
        .def("reshape", &Tensor::reshape)
        .def("sum", &Tensor::sum)
        .def("matmul2D", &Tensor::matmul2D)
        .def("fill", &Tensor::fill)
        .def("__getitem__", [](const Tensor &t, std::vector<size_t> idx){ return t.at(idx); })
        .def("__setitem__", [](Tensor &t, std::vector<size_t> idx, double v){ t.at(idx)=v; })
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("shape", &Tensor::shape);

    py::class_<Neuron>(m, "Neuron")
        .def(py::init<int,int,int,int,double>(),
             py::arg("inputNeurons"), py::arg("hiddenNeurons"), py::arg("hiddenLayers"),
             py::arg("outputNeurons"), py::arg("dropout_rate"))
        .def("forward_propagate", &Neuron::forward_propagate, py::arg("input"), py::arg("apply_dropout")=false)
        .def("back_propagate", &Neuron::back_propagate)
        .def("train", &Neuron::train)
        .def("predict", &Neuron::predict)
        .def("predict_classes", &Neuron::predict_classes)
        .def("save_model", &Neuron::save_model)
        .def("load_model", &Neuron::load_model)
        .def("print_network_info", &Neuron::print_network_info);
}
```