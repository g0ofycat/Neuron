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
#include <omp.h>

// ====== TENSOR CLASS ======

class Tensor {
    public:
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        std::vector<double> data;

        // ====== CONSTRUCTORS ======

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

        // ====== STATIC FUNCTIONS ======

        static Tensor zeros(const std::vector<size_t>& dims) { return Tensor(dims); }

        static Tensor random(const std::vector<size_t>& dims, double min = 0.0, double max = 1.0) {
            Tensor t(dims);
            static thread_local std::mt19937 gen{std::random_device{}()};
            std::uniform_real_distribution<double> dist(min, max);

            for (auto &v : t.data)
                v = dist(gen);

            return t;
        }

        static Tensor random_normal(const std::vector<size_t>& dims, double mean = 0.0, double stddev = 1.0) {
            Tensor t(dims);
            static thread_local std::mt19937 gen{std::random_device{}()};
            std::normal_distribution<> d(mean, stddev);
            for (auto &v : t.data) v = d(gen);
            return t;
        }

        static Tensor extract_row(const Tensor& tensor, size_t row_idx) {
            if (tensor.shape.size() != 2) {
                throw std::invalid_argument("extract_row(): Tensor must be 2D");
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

        static Tensor matmul2D(const Tensor& A, const Tensor& B) {
            if (A.ndim() != 2 || B.ndim() != 2) 
                throw std::invalid_argument("matmul2D(): requires 2D tensors");
            
            size_t m = A.shape[0], k = A.shape[1];
            size_t k2 = B.shape[0], n = B.shape[1];
            
            if (k != k2) 
                throw std::invalid_argument("matmul2D(): shape mismatch");
            
            Tensor C({m, n});
            C.fill(0.0);
            
            const double* A_data = A.data.data();
            const double* B_data = B.data.data();
            double* C_data = C.data.data();
            
            for (size_t i = 0; i < m; ++i) {
                for (size_t t = 0; t < k; ++t) {
                    double a_val = A_data[i * k + t];
                    size_t b_offset = t * n;
                    size_t c_offset = i * n;
                    for (size_t j = 0; j < n; ++j) {
                        C_data[c_offset + j] += a_val * B_data[b_offset + j];
                    }
                }
            }
            
            return C;
        }

        // ====== UTILITY ======

        size_t ndim() const { return shape.size(); }
        size_t numel() const { return shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>()); }

        size_t index_of(const std::vector<size_t>& idx) const {
            if (idx.size() != shape.size()) throw std::out_of_range("index_of(): Index rank mismatch");
            size_t offs = 0;
            for (size_t i = 0; i < idx.size(); ++i) {
                if (idx[i] >= shape[i]) throw std::out_of_range("index_of(): Index out of range");
                offs += idx[i] * strides[i];
            }
            return offs;
        }

        Tensor& assign_data(const std::vector<double>& src) {
            data = src; return *this; 
        }

        Tensor sum(size_t axis) const {
            if (axis >= ndim()) throw std::out_of_range("sum(): Axis out of range");
            std::vector<size_t> out_shape = shape;
            out_shape.erase(out_shape.begin() + axis);
            Tensor out(out_shape.empty() ? std::vector<size_t>{1} : out_shape);
            out.fill(0.0);

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

        void reshape(const std::vector<size_t>& new_shape) {
            size_t new_num = std::accumulate(new_shape.begin(), new_shape.end(), (size_t)1, std::multiplies<size_t>());
            if (new_num != numel()) throw std::invalid_argument("reshape(): Reshape size mismatch");
            shape = new_shape;
            compute_strides();
        }
        
        void compute_strides() {
            strides.assign(shape.size(), 1);
            if (!shape.empty()) {
                for (int i = (int)shape.size() - 2; i >= 0; --i)
                    strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        void fill(double v) { std::fill(data.begin(), data.end(), v); }

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

        Tensor temp_input;
        Tensor temp_target;
        Tensor temp_output;
        std::vector<Tensor> weight_gradients;
        std::vector<Tensor> bias_gradients;
        std::vector<std::vector<double>> deltas;

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

        // ====== TRAINING ======

        void Softmax_inplace(const Tensor& logits, Tensor& output) {
            double max_logit = *std::max_element(logits.data.begin(), logits.data.end());
            double sum_exp = 0.0;

            for (size_t i = 0; i < logits.data.size(); ++i) {
                output.data[i] = std::exp(logits.data[i] - max_logit);
                sum_exp += output.data[i];
            }

            for (auto &v : output.data)
                v /= sum_exp;
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
        
        // ====== HELPERS ======

        size_t get_layer_input_size(int layer) const {
            return (layer == 0) ? inputNeurons : hiddenNeurons;
        }

        size_t get_layer_output_size(int layer) const {
            return (layer == hiddenLayers) ? outputNeurons : hiddenNeurons;
        }

        static std::mt19937& get_rng() {
            static thread_local std::mt19937 gen{std::random_device{}()};
            return gen;
        }

        void zero_gradients() {
            for (int i = 0; i <= hiddenLayers; ++i) {
                weight_gradients[i].fill(0.0);
                bias_gradients[i].fill(0.0);
            }
        }

        void allocate_training_buffers() {
            temp_input = Tensor({static_cast<size_t>(inputNeurons)});
            temp_target = Tensor({static_cast<size_t>(outputNeurons)});
            temp_output = Tensor({static_cast<size_t>(outputNeurons)});
            
            weight_gradients.resize(hiddenLayers + 1);
            bias_gradients.resize(hiddenLayers + 1);
            deltas.resize(hiddenLayers + 1);
            
            for (int i = 0; i <= hiddenLayers; ++i) {
                weight_gradients[i] = Tensor::zeros(weights[i].shape);
                bias_gradients[i] = Tensor::zeros(biases[i].shape);
                deltas[i].resize(get_layer_output_size(i));
            }
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

                double stddev = std::sqrt(2.0 / double(in)); // xavier (sigmoid): std::sqrt(1.0 / double(in)); he (relu): std::sqrt(2.0 / double(in));
                weights[i] = Tensor::random_normal({out, in}, 0.0, stddev);
                biases[i]  = Tensor(std::vector<size_t>{out});
                biases[i].fill(0.0);

                activations[i+1] = Tensor(std::vector<size_t>{out});
                z_values[i] = Tensor(std::vector<size_t>{out});
            }

            activations[0] = Tensor(std::vector<size_t>{(size_t)inputNeurons});

            allocate_training_buffers();
        }

        // ====== DE-CONSTRUCTOR ======

        /*
        ~Neuron(): Neuron de-constructor
        */
        ~Neuron() = default;
        
        // ====== HELPERS ======

        /*
        compute_gradients(): Computes gradients for all weights and biases

        @param target: Target output tensor
        */
        void compute_gradients(const Tensor& target) {
            const size_t out_size = outputNeurons;
            double* delta_out = deltas[hiddenLayers].data();
            const double* act_out = activations[hiddenLayers + 1].data.data();
            const double* tgt = target.data.data();
            
            #pragma omp parallel for simd
            for (size_t i = 0; i < out_size; ++i) {
                delta_out[i] = act_out[i] - tgt[i];
            }

            for (int layer = hiddenLayers - 1; layer >= 0; --layer) {
                size_t curr_size = hiddenNeurons;
                size_t next_size = get_layer_output_size(layer + 1);
                
                const double* w_next = weights[layer + 1].data.data();
                const double* delta_next = deltas[layer + 1].data();
                const double* z_curr = z_values[layer].data.data();
                double* delta_curr = deltas[layer].data();
                
                #pragma omp parallel for
                for (size_t i = 0; i < curr_size; ++i) {
                    double error = 0.0;
                    #pragma omp simd reduction(+:error)
                    for (size_t j = 0; j < next_size; ++j) {
                        error += delta_next[j] * w_next[j * curr_size + i];
                    }
                    delta_curr[i] = error * ReLU_derivative(z_curr[i]);
                }
            }

            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t in_size = get_layer_input_size(layer);
                size_t out_size = get_layer_output_size(layer);
                
                const double* act_in = activations[layer].data.data();
                const double* delta_layer = deltas[layer].data();
                double* w_grad = weight_gradients[layer].data.data();
                double* b_grad = bias_gradients[layer].data.data();

                #pragma omp parallel for
                for (size_t j = 0; j < out_size; ++j) {
                    double delta_j = delta_layer[j];
                    b_grad[j] += delta_j;
                    
                    size_t w_offset = j * in_size;
                    #pragma omp simd
                    for (size_t i = 0; i < in_size; ++i) {
                        w_grad[w_offset + i] += delta_j * act_in[i];
                    }
                }
            }
        }

        /*
        apply_gradients(): Applies computed gradients to weights and biases

        @param learning_rate: Learning rate for updates
        @param batch_size: Batch size for gradient averaging
        */
        void apply_gradients(double learning_rate, size_t batch_size) {
            double scale = learning_rate / batch_size;
            
            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t w_size = weights[layer].data.size();
                size_t b_size = biases[layer].data.size();
                
                double* w = weights[layer].data.data();
                double* b = biases[layer].data.data();
                const double* w_grad = weight_gradients[layer].data.data();
                const double* b_grad = bias_gradients[layer].data.data();
                
                #pragma omp parallel for simd
                for (size_t i = 0; i < w_size; ++i) {
                    w[i] -= scale * w_grad[i];
                }
                
                #pragma omp parallel for simd
                for (size_t i = 0; i < b_size; ++i) {
                    b[i] -= scale * b_grad[i];
                }
            }
        }

        // ====== TRAINING ======

        /*
        forward_propagate(): Unified forward propagation with optional dropout

        @param input: Input tensor to the network
        @param apply_dropout: Whether to apply dropout during forward propagation
        @return: Output tensor from the network (raw logits)
        */
        Tensor forward_propagate(const Tensor& input, bool apply_dropout = false) {
            if (input.numel() != static_cast<size_t>(inputNeurons))
                throw std::invalid_argument("Input size mismatch");

            #pragma omp parallel for
            for (size_t i = 0; i < input.numel(); ++i)
                activations[0].data[i] = input.data[i];

            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t in_size = get_layer_input_size(layer);
                size_t out_size = get_layer_output_size(layer);

                double* act_out = activations[layer + 1].data.data();
                double* z_out = z_values[layer].data.data();
                const double* act_in = activations[layer].data.data();
                const double* w = weights[layer].data.data();
                const double* b = biases[layer].data.data();

                bool is_output_layer = (layer == hiddenLayers);

                #pragma omp parallel
                {
                    static thread_local std::mt19937 local_gen{std::random_device{}()};
                    std::bernoulli_distribution local_drop(1.0 - dropout_rate);

                    #pragma omp for
                    for (size_t j = 0; j < out_size; ++j) {
                        double z = b[j];
                        size_t w_offset = j * in_size;
                        
                        #pragma omp simd reduction(+:z)
                        for (size_t i = 0; i < in_size; ++i) {
                            z += w[w_offset + i] * act_in[i];
                        }
                        z_out[j] = z;

                        if (is_output_layer) {
                            act_out[j] = z;
                        } else {
                            double a = ReLU(z);
                            if (apply_dropout) {
                                if (!local_drop(local_gen))
                                    a = 0.0;
                                else
                                    a /= (1.0 - dropout_rate);
                            }
                            act_out[j] = a;
                        }
                    }
                }
            }

            Tensor out({static_cast<size_t>(outputNeurons)});
            
            #pragma omp parallel for
            for (size_t i = 0; i < out.numel(); ++i)
                out.data[i] = activations[hiddenLayers + 1].data[i];

            return out;
        }

        /*
        back_propagate(): Performs backpropagation and updates weights and biases

        @param input: Input tensor to the network
        @param target: Target output tensor for the input
        @param learning_rate: Learning rate for weight updates
        */
        void back_propagate(const Tensor& input, const Tensor& target, double learning_rate) {
            forward_propagate(input, true);
    
            zero_gradients();
            
            compute_gradients(target);

            apply_gradients(learning_rate, 1);
        }

        /*
        train(): Trains the neural network using mini-batch gradient descent

        @param inputs: Tensor of input vectors for training
        @param targets: Tensor of target output vectors for training
        @param epochs: Number of training epochs
        @param learning_rate: Learning rate for weight updates
        @param batch_size: Size of each training batch
        */
        void train(const Tensor& inputs, const Tensor& targets, int epochs, double learning_rate, int batch_size) {
            if (inputs.shape.size() != 2 || targets.shape.size() != 2) {
                std::cerr << "train(): Inputs and targets must be 2D tensors\n";
                return;
            }
            
            size_t num_samples = inputs.shape[0];

            if (num_samples != targets.shape[0]) {
                std::cerr << "train(): Number of samples mismatch\n";
                return;
            }
            if (inputs.shape[1] != static_cast<size_t>(inputNeurons)) {
                std::cerr << "train(): Input feature size mismatch\n";
                return;
            }
            if (targets.shape[1] != static_cast<size_t>(outputNeurons)) {
                std::cerr << "train(): Target size mismatch\n";
                return;
            }
            
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), get_rng());
                double total_loss = 0.0;
                
                for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
                    size_t batch_end = std::min(batch_start + batch_size, num_samples);
                    size_t current_batch_size = batch_end - batch_start;
                    
                    zero_gradients();
                    
                    for (size_t b = batch_start; b < batch_end; ++b) {
                        size_t sample_idx = indices[b];

                        size_t input_offset = sample_idx * inputs.shape[1];
                        size_t target_offset = sample_idx * targets.shape[1];

                        double* inp_ptr = temp_input.data.data();
                        double* tgt_ptr = temp_target.data.data();
                        const double* src_inp = inputs.data.data() + input_offset;
                        const double* src_tgt = targets.data.data() + target_offset;
                        
                        std::copy(src_inp, src_inp + inputNeurons, inp_ptr);
                        std::copy(src_tgt, src_tgt + outputNeurons, tgt_ptr);
                        
                        forward_propagate(temp_input, true);

                        Softmax_inplace(activations[hiddenLayers + 1], temp_output);
                        
                        total_loss += cross_entropy_loss(temp_output.data, temp_target.data);
                        
                        compute_gradients(temp_target);
                    }
                    
                    apply_gradients(learning_rate, current_batch_size);
                }

                /*
                if (epoch % 1000 == 0) {
                    std::cout << "Epoch " << epoch
                            << " | Loss: " << (total_loss / num_samples) << std::endl;
                }
                */
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
            Tensor out = forward_propagate(input);
            Tensor result({static_cast<size_t>(outputNeurons)});
            Softmax_inplace(out, result);
            return result;
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