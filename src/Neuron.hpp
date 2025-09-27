#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <random>
#include <cmath>
#include <functional>
#include <algorithm>

// ====== TENSOR CLASS ======

class Tensor {
    public:
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        std::vector<double> data;

        Tensor() = default;

        // 1d tensor support (vector)
        Tensor(std::initializer_list<double> list) {
            data.assign(list.begin(), list.end());
            shape = { list.size() };
            compute_strides();
        }

        // 2d tensor support (matrix)
        Tensor(std::initializer_list<std::initializer_list<double>> list2d) {
            size_t rows = list2d.size();
            if (rows == 0) { shape = {0,0}; return; }
            size_t cols = list2d.begin()->size();
            for (auto& row : list2d) {
                if (row.size() != cols) throw std::invalid_argument("All rows must have same length");
                data.insert(data.end(), row.begin(), row.end());
            }
            shape = {rows, cols};
            compute_strides();
        }

        // creating tensors with specific dimensions (filled with zeros); Tensor({3, 4, 5})
        explicit Tensor(std::initializer_list<size_t> dims): shape(dims) {
            compute_strides();
            data.assign(numel(), 0.0);
        }
        
        // used for dynamic shape creation; Tensor({3, 4, 5})
        Tensor(const std::vector<size_t>& dims): shape(dims) {
            compute_strides();
            data.assign(numel(), 0.0);
        }

        static Tensor zeros(const std::vector<size_t>& dims) { return Tensor(dims); }

        static Tensor random_normal(const std::vector<size_t>& dims, double mean = 0.0, double stddev = 1.0) {
            Tensor t(dims);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(mean, stddev);
            for (auto &v : t.data) v = d(gen);
            return t;
        }

        static Tensor extract_row(const Tensor& tensor, size_t row_idx) {
            if (tensor.shape.size() != 2) {
                throw std::invalid_argument("extract_row: Tensor must be 2D");
            }
            
            size_t row_size = tensor.shape[1];
            size_t start_idx = row_idx * row_size;
            
            std::vector<double> row_data(tensor.data.begin() + start_idx, tensor.data.begin() + start_idx + row_size);
            
            Tensor result;
            result.data = row_data;
            result.shape = {row_data.size()};
            result.compute_strides();
            
            return result;
        }

        size_t ndim() const { return shape.size(); }
        size_t numel() const { return shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>()); }

        Tensor& assign_data(const std::vector<double>& src) { 
            data = src; return *this; 
        }
        
        void compute_strides() {
            strides.assign(shape.size(), 1);
            if (!shape.empty()) {
                for (int i = (int)shape.size() - 2; i >= 0; --i)
                    strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        size_t index_of(const std::vector<size_t>& idx) const {
            if (idx.size() != shape.size()) throw std::out_of_range("index_of(): Index rank mismatch");
            size_t offs = 0;
            for (size_t i = 0; i < idx.size(); ++i) {
                if (idx[i] >= shape[i]) throw std::out_of_range("index_of(): Index out of range");
                offs += idx[i] * strides[i];
            }
            return offs;
        }

        template<typename... Idx>

        double& operator()(Idx... idxs) {
            std::vector<size_t> v{ static_cast<size_t>(idxs)... };
            return data[index_of(v)];
        }

        template<typename... Idx>

        double operator()(Idx... idxs) const {
            std::vector<size_t> v{ static_cast<size_t>(idxs)... };
            return data[index_of(v)];
        }

        double& at(const std::vector<size_t>& idx) { return data[index_of(idx)]; }
        double  at(const std::vector<size_t>& idx) const { return data[index_of(idx)]; }

        void reshape(const std::vector<size_t>& new_shape) {
            size_t new_num = std::accumulate(new_shape.begin(), new_shape.end(), (size_t)1, std::multiplies<size_t>());
            if (new_num != numel()) throw std::invalid_argument("reshape(): Reshape size mismatch");
            shape = new_shape;
            compute_strides();
        }

        Tensor sum(size_t axis) const {
            if (axis >= ndim()) throw std::out_of_range("sum(): Axis out of range");
            std::vector<size_t> out_shape = shape;
            out_shape.erase(out_shape.begin() + axis);
            Tensor out(out_shape.empty() ? std::vector<size_t>{1} : out_shape);
            std::fill(out.data.begin(), out.data.end(), 0.0);

            std::vector<size_t> idx(shape.size(), 0);
            for (size_t linear = 0; linear < numel(); ++linear) {
                size_t tmp = linear;
                for (size_t d = 0; d < shape.size(); ++d) {
                    idx[d] = tmp / strides[d];
                    tmp %= strides[d];
                }
                std::vector<size_t> out_idx;
                out_idx.reserve(idx.size()-1);
                for (size_t d=0; d<idx.size(); ++d) if (d!=axis) out_idx.push_back(idx[d]);
                size_t out_off = out.index_of(out_idx.empty() ? std::vector<size_t>{0} : out_idx);
                out.data[out_off] += data[linear];
            }
            return out;
        }

        static Tensor matmul2D(const Tensor& A, const Tensor& B) {
            if (A.ndim() != 2 || B.ndim() != 2) throw std::invalid_argument("matmul2D(): matmul2D requires 2D tensors");
            size_t m = A.shape[0], k = A.shape[1];
            size_t k2 = B.shape[0], n = B.shape[1];
            if (k != k2) throw std::invalid_argument("matmul2D(): matmul shape mismatch");
            Tensor C({m,n});
            for (size_t i=0;i<m;++i) {
                for (size_t j=0;j<n;++j) {
                    double sum = 0.0;
                    for (size_t t=0;t<k;++t) sum += A(i,t) * B(t,j);
                    C(i,j) = sum;
                }
            }
            return C;
        }

        void fill(double v) { std::fill(data.begin(), data.end(), v); }
};

// ====== NEURON CLASS ======

class Neuron {
    private:
        int inputNeurons, hiddenNeurons, hiddenLayers, outputNeurons;
        double dropout_rate;
        std::vector<Tensor> weights;
        std::vector<Tensor> biases;
        std::vector<Tensor> activations;
        std::vector<Tensor> z_values;

        // ====== ACTIVATION FUNCTIONS ======

        double ReLU(double x) {
            return (x > 0) ? x : 0;
        }
        
        double ReLU_derivative(double x) {
            return (x > 0) ? 1.0 : 0.0;
        }

        double Sigmoid(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

        double Sigmoid_derivative(double a) {
            return a * (1.0 - a);
        }

        // ====== MISC ======
        
        Tensor Softmax(const Tensor& logits) {
            double max_logit = *std::max_element(logits.data.begin(), logits.data.end());
            Tensor exp_z(logits.shape);
            double sum_exp = 0.0;

            for (size_t i = 0; i < logits.data.size(); ++i) {
                exp_z.data[i] = std::exp(logits.data[i] - max_logit);
                sum_exp += exp_z.data[i];
            }

            for (auto &v : exp_z.data)
                v /= sum_exp;

            return exp_z;
        }

        double mean_squared_error(const std::vector<double>& output, const std::vector<double>& target) {
            if (output.size() != target.size()) {
                throw std::invalid_argument("Output and target sizes do not match for MSE calculation.");
            }

            double sum = 0.0;
            for (size_t i = 0; i < output.size(); ++i) {
                double diff = output[i] - target[i];
                sum += diff * diff;
            }
            return sum / static_cast<double>(output.size());
        }

        double cross_entropy_loss(const std::vector<double>& output, const std::vector<double>& target) {
            if (output.size() != target.size()) {
                throw std::invalid_argument("Output and target sizes do not match for Cross-Entropy calculation.");
            }

            double loss = 0.0;

            for (size_t i = 0; i < output.size(); ++i) {
                loss -= target[i] * std::log(output[i] + 1e-15);
            }

            return loss / static_cast<double>(output.size());
        }
        
    public:
        // ====== CONSTRUCTOR ======

        /*
        Neuron(): Neuron constructor

        @param inputNeurons: Number of input neurons
        @param hiddenNeurons: Number of hidden neurons per layer
        @param hiddenLayers: Number of hidden layers
        @param outputNeurons: Number of output neurons
        @param dropout_rate: Dropout rate during training
        */
        explicit Neuron(int inputNeurons, int hiddenNeurons, int hiddenLayers, int outputNeurons, double dropout_rate): inputNeurons(inputNeurons), hiddenNeurons(hiddenNeurons), hiddenLayers(hiddenLayers), outputNeurons(outputNeurons), dropout_rate(dropout_rate){
            weights.resize(hiddenLayers + 1);
            biases.resize(hiddenLayers + 1);
            activations.resize(hiddenLayers + 2);
            z_values.resize(hiddenLayers + 1);

            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t in  = (i == 0) ? inputNeurons : hiddenNeurons;
                size_t out = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;

                double stddev = std::sqrt(1.0 / double(in)); // xavier (sigmoid): std::sqrt(1.0 / double(in)); he (relu): std::sqrt(2.0 / double(in));
                weights[i] = Tensor::random_normal({out, in}, 0.0, stddev);
                biases[i]  = Tensor(std::vector<size_t>{out});
                biases[i].fill(0.0);

                activations[i+1] = Tensor(std::vector<size_t>{out});
                z_values[i]      = Tensor(std::vector<size_t>{out});
            }

            activations[0] = Tensor(std::vector<size_t>{(size_t)inputNeurons});
        }

        // ====== DE-CONSTRUCTOR ======

        /*
        ~Neuron(): Neuron de-constructor
        */
        ~Neuron() = default;
        
        // ====== TRAINING ======

        /*
        forward_propagate(): Performs forward propagation through the network

        @param input: Input tensor to the network
        @param apply_dropout: Whether to apply dropout during forward propagation
        @return: Output tensor from the network
        */
        Tensor forward_propagate(const Tensor& input, bool apply_dropout=false) {
            Tensor flat = input;
            flat.reshape({flat.numel()});
            
            if (input.numel() != inputNeurons) 
                throw std::invalid_argument("forward_propagate(): Input size does not match number of input neurons.");

            for (size_t i=0; i<inputNeurons; ++i) 
                activations[0](i) = input.data[i];

            static thread_local std::mt19937 gen{std::random_device{}()};
            std::bernoulli_distribution drop_dist(1.0 - dropout_rate);

            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t in_size  = (layer == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (layer == hiddenLayers) ? outputNeurons : hiddenNeurons;

                for (size_t j = 0; j < out_size; ++j) {
                    double z = biases[layer](j);
                    for (size_t i = 0; i < in_size; ++i)
                        z += weights[layer](j, i) * activations[layer](i);
                    z_values[layer](j) = z;

                    if (layer == hiddenLayers) {
                        activations[layer + 1](j) = z;
                    } else {
                        double a = Sigmoid(z);
                        if (apply_dropout) {
                            if (!drop_dist(gen)) a = 0.0;
                            else a /= (1.0 - dropout_rate);
                        }
                        activations[layer + 1](j) = a;
                    }
                }
            }

            Tensor out({static_cast<size_t>(outputNeurons)});

            for (size_t i = 0; i < out.numel(); ++i)
                out(i) = activations[hiddenLayers + 1](i);

            return out;
        }
        
        /*
        back_propagate(): Performs backpropagation and updates weights and biases

        @param input: Input tensor to the network
        @param input_target: Target output tensor for the input
        @param learning_rate: Learning rate for weight updates
        */
        void back_propagate(const Tensor& input, const Tensor& target, double learning_rate) {
            forward_propagate(input, true);

            std::vector<std::vector<double>> deltas(hiddenLayers + 1);
            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t out_size = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;
                deltas[i].resize(out_size);
            }

            for (size_t i = 0; i < static_cast<size_t>(outputNeurons); ++i) {
                double error = activations[hiddenLayers + 1](i) - target(i);
                deltas[hiddenLayers][i] = error;
            }

            for (int layer = hiddenLayers - 1; layer >= 0; --layer) {
                for (size_t i = 0; i < static_cast<size_t>(hiddenNeurons); ++i) {
                    double error = 0.0;
                    size_t next_size = (layer == hiddenLayers - 1) ? outputNeurons : hiddenNeurons;
                    for (size_t j = 0; j < next_size; ++j)
                        error += deltas[layer + 1][j] * weights[layer + 1](j, i);
                    deltas[layer][i] = error * Sigmoid_derivative(activations[layer + 1](i));
                }
            }

            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t in_size  = (layer == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (layer == hiddenLayers) ? outputNeurons : hiddenNeurons;
                for (size_t j = 0; j < out_size; ++j) {
                    biases[layer](j) -= learning_rate * deltas[layer][j];
                    for (size_t i = 0; i < in_size; ++i)
                        weights[layer](j, i) -= learning_rate * deltas[layer][j] * activations[layer](i);
                }
            }
        }

        /*
        train(): Trains the neural network using the provided dataset

        @param input: Tensor of input vectors for training
        @param input_target: Tensor of target output vectors for training
        @param epochs: Number of training epochs
        @param learning_rate: Learning rate for weight updates
        @param batch_size: Size of each training batch
        */
        void train(const Tensor& inputs, const Tensor& targets, int epochs, double learning_rate, int batch_size) {            
            if (inputs.shape.size() != 2 || targets.shape.size() != 2) {
                std::cerr << "train(): Inputs and targets must be 2D tensors (batch_size x features)\n";
                return;
            }
            
            size_t num_samples = inputs.shape[0];
            if (num_samples != targets.shape[0]) {
                std::cerr << "train(): Number of input samples (" << num_samples 
                        << ") doesn't match target samples (" << targets.shape[0] << ")\n";
                return;
            }
            
            size_t input_features = inputs.shape[1];
            size_t output_features = targets.shape[1];
            
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 gen{std::random_device{}()};
            
            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), gen);
                double total_loss = 0.0;
                
                for (size_t i = 0; i < num_samples; i += batch_size) {
                    size_t end = std::min(i + batch_size, num_samples);
                    
                    for (size_t j = i; j < end; ++j) {
                        size_t sample_idx = indices[j];

                        Tensor input_sample = Tensor::extract_row(inputs, sample_idx);
                        Tensor target_sample = Tensor::extract_row(targets, sample_idx);
                        
                        back_propagate(input_sample, target_sample, learning_rate);
                        Tensor out = forward_propagate(input_sample);
                        
                        Tensor flat_target = target_sample;
                        flat_target.reshape({flat_target.numel()});
                        total_loss += mean_squared_error(out.data, flat_target.data);
                    }
                }
                
                if (epoch % 100 == 0) {
                    std::cout << "Epoch " << epoch 
                            << " | Loss: " << total_loss / num_samples << std::endl;
                }
            }
        }

        // ====== PREDICTION METHODS ======
        
        /*
        predict(): Makes a prediction for a tensor

        @param input: Input tensor for prediction
        */
        Tensor predict(const Tensor& input) {
            return forward_propagate(input);
        }

        /*
        predict_classes(): Makes a class prediction for a tensor (Softmaxxed output)

        @param input: Input vector for prediction
        */
        Tensor predict_classes(const Tensor& input) {
            return Softmax(forward_propagate(input));
        }

        // ====== MODEL ======

        /*
        save_model(): Saves the model data to a file

        @param filepath: Path of the file to save the model to
        */
        void save_model(const std::string &filepath) {
            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "save_model(): Could not open file for writing: " << filepath << std::endl;
                return;
            }
            
            file.write(reinterpret_cast<const char*>(&inputNeurons), sizeof(int));
            file.write(reinterpret_cast<const char*>(&hiddenNeurons), sizeof(int));
            file.write(reinterpret_cast<const char*>(&hiddenLayers), sizeof(int));
            file.write(reinterpret_cast<const char*>(&outputNeurons), sizeof(int));
            
            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t in_size  = (i == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;
                
                file.write(reinterpret_cast<const char*>(weights[i].data.data()),
                weights[i].data.size() * sizeof(double));
                
                file.write(reinterpret_cast<const char*>(biases[i].data.data()),
                biases[i].data.size() * sizeof(double));
            }
            
            file.close();
            std::cout << "save_model(): Model saved to: " << filepath << std::endl;
        }

        /*
        load_model(): Loads the model data from a file

        @param filepath: Path of the file to load the model from
        */
        void load_model(const std::string &filepath) {
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "load_model(): Could not open file for reading: " << filepath << std::endl;
                return;
            }
            
            int loaded_input, loaded_hidden, loaded_layers, loaded_output;
            file.read(reinterpret_cast<char*>(&loaded_input), sizeof(int));
            file.read(reinterpret_cast<char*>(&loaded_hidden), sizeof(int));
            file.read(reinterpret_cast<char*>(&loaded_layers), sizeof(int));
            file.read(reinterpret_cast<char*>(&loaded_output), sizeof(int));
            
            if (loaded_input != inputNeurons || loaded_hidden != hiddenNeurons || 
                loaded_layers != hiddenLayers || loaded_output != outputNeurons) {
                std::cerr << "load_model(): Model architecture doesn't match current network!" << std::endl;
                file.close();
                return;
            }
            
            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t in_size  = (i == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;
                
                for (size_t j = 0; j < out_size; ++j) {
                    file.read(reinterpret_cast<char*>(
                    &weights[i].data[j * in_size]),
                    sizeof(double) * in_size);
                }
                
                file.read(reinterpret_cast<char*>(biases[i].data.data()),
                sizeof(double) * out_size);
            }
            
            file.close();
            std::cout << "load_model(): Model loaded from: " << filepath << std::endl;
        }

        // ====== UTILITY METHODS ======
        
        /*
        print_network_info(): Prints information about the neural network architecture and parameters
        */
        void print_network_info() {
            std::cout << "Neural Network Architecture:" << std::endl;
            std::cout << "Input neurons: " << inputNeurons << std::endl;
            std::cout << "Hidden neurons: " << hiddenNeurons << std::endl;
            std::cout << "Hidden layers: " << hiddenLayers << std::endl;
            std::cout << "Output neurons: " << outputNeurons << std::endl;
            
            long long total_weights = 0;
            long long total_biases = 0;
            
            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t in_size  = (layer == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (layer == hiddenLayers) ? outputNeurons : hiddenNeurons;
                
                long long layer_weights = in_size * out_size;

                total_weights += layer_weights;
                
                total_biases += out_size;
                
                std::cout << "Layer " << layer << ": " << in_size << " -> " << out_size 
                          << " (Weights: " << layer_weights << ", Biases: " << out_size << ")" << std::endl;
            }
            
            long long total_params = total_weights + total_biases;

            std::cout << "\nParameter Summary:" << std::endl;
            std::cout << "Total Weights: " << total_weights << std::endl;
            std::cout << "Total Biases: " << total_biases << std::endl;
            std::cout << "Total Parameters: " << total_params << std::endl;
        }
};