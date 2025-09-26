#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>

class Neuron {
    private:
        int inputNeurons, hiddenNeurons, hiddenLayers, outputNeurons;
        double dropout_rate;
        double*** weights;
        double**  biases;
        std::vector<std::vector<double>> activations;
        std::vector<std::vector<double>> z_values;

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

        double Sigmoid_derivative(double x) {
            double s = Sigmoid(x);
            return s * (1.0 - s);
        }

        // ====== MISC ======
        
        std::vector<double> Softmax(const std::vector<double>& logits) {
            double max_logit = *std::max_element(logits.begin(), logits.end());
            std::vector<double> exp_z(logits.size());
            double sum_exp = 0.0;

            for (size_t i = 0; i < logits.size(); ++i) {
                exp_z[i] = std::exp(logits[i] - max_logit);
                sum_exp += exp_z[i];
            }
            
            for (size_t i = 0; i < logits.size(); ++i)
                exp_z[i] /= sum_exp;
            
            return exp_z;
        }

        // ====== INIT ======

        void initialize_weights(double** layerWeights, size_t out_neurons, size_t in_neurons) {
            double std_dev = std::sqrt(1.0 / static_cast<double>(in_neurons)); // xavier (sigmoid): std::sqrt(1.0 / static_cast<double>(in_neurons)); he (relu): std::sqrt(2.0 / static_cast<double>(in_neurons));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0.0, std_dev);

            for (size_t i = 0; i < out_neurons; ++i)
                for (size_t j = 0; j < in_neurons; ++j)
                    layerWeights[i][j] = d(gen);
        }
        
    public:
        // ====== CONSTRUCTOR ======

        /*
        Neuron(): Neuron constructor

        @param inputNeurons: number of input neurons
        @param hiddenNeurons: number of hidden neurons per layer
        @param hiddenLayers: number of hidden layers
        @param outputNeurons: number of output neurons
        @param dropout_rate: Dropout rate during training
        */
        Neuron(int inputNeurons, int hiddenNeurons, int hiddenLayers, int outputNeurons, double dropout_rate): inputNeurons(inputNeurons), hiddenNeurons(hiddenNeurons), hiddenLayers(hiddenLayers), outputNeurons(outputNeurons), dropout_rate(dropout_rate){
            weights = new double**[hiddenLayers + 1];
            biases  = new double*[hiddenLayers + 1];
            
            activations.resize(hiddenLayers + 2);
            z_values.resize(hiddenLayers + 1);

            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t in  = (i == 0) ? inputNeurons : hiddenNeurons;
                size_t out = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;

                weights[i] = new double*[out];
                biases[i]  = new double[out];

                for (size_t j = 0; j < out; ++j)
                    weights[i][j] = new double[in];

                initialize_weights(weights[i], out, in);

                for (size_t j = 0; j < out; ++j)
                    biases[i][j] = 0.0;

                activations[i + 1].resize(out);
                z_values[i].resize(out);
            }
            activations[0].resize(inputNeurons);
        }

        // ====== DE-CONSTRUCTOR ======

        /*
        ~Neuron(): Neuron de-constructor
        */
        ~Neuron() {
            for (int i = 0; i <= hiddenLayers; i++) {
                size_t out = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;
                for (size_t j = 0; j < out; j++)
                    delete[] weights[i][j];
                delete[] weights[i];
                delete[] biases[i];
            }
            delete[] weights;
            delete[] biases;
        }
        
        // ====== TRAINING ======

        /*
        forward_propagate(): Performs forward propagation through the network

        @param input: Input vector to the network
        @param apply_dropout: Whether to apply dropout during forward propagation
        @return: Output vector from the network
        */
        std::vector<double> forward_propagate(const std::vector<double>& input, bool apply_dropout = false) {
            if (input.size() != static_cast<size_t>(inputNeurons)) {
                throw std::invalid_argument("Input size doesn't match network inputNeurons size");
            }

            activations[0] = input;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::bernoulli_distribution drop_dist(1.0 - dropout_rate);

            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t in_size  = (layer == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (layer == hiddenLayers) ? outputNeurons : hiddenNeurons;

                for (size_t j = 0; j < out_size; ++j) {
                    double z = biases[layer][j];
                    for (size_t i = 0; i < in_size; ++i) {
                        z += weights[layer][j][i] * activations[layer][i];
                    }
                    z_values[layer][j] = z;

                    if (layer == hiddenLayers) {
                        activations[layer + 1][j] = z;
                    } else {
                        double a = Sigmoid(z);
                        if (apply_dropout) {
                            if (!drop_dist(gen))
                                a = 0.0;
                            else
                                a /= (1.0 - dropout_rate);
                        }

                        activations[layer + 1][j] = a;
                    }
                }
            }

            return activations[hiddenLayers + 1];
        }
        
        /*
        back_propagate(): Performs backpropagation and updates weights and biases

        @param input: Input vector to the network
        @param input_target: Target output vector for the input
        @param learning_rate: Learning rate for weight updates
        */
        void back_propagate(const std::vector<double>& input, const std::vector<double>& input_target, double learning_rate) {
            forward_propagate(input, true);

            std::vector<std::vector<double>> deltas(hiddenLayers + 1);

            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t out_size = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;
                deltas[i].resize(out_size);
            }

            for (size_t i = 0; i < outputNeurons; ++i) {
                double error = activations[hiddenLayers + 1][i] - input_target[i];
                deltas[hiddenLayers][i] = error * 1.0;
            }

            for (int layer = hiddenLayers - 1; layer >= 0; --layer) {
                size_t current_size = hiddenNeurons;
                size_t next_size = (layer == hiddenLayers - 1) ? outputNeurons : hiddenNeurons;
                
                for (size_t i = 0; i < current_size; ++i) {
                    double error = 0.0;
                    for (size_t j = 0; j < next_size; ++j) {
                        error += deltas[layer + 1][j] * weights[layer + 1][j][i];
                    }
                    deltas[layer][i] = error * Sigmoid_derivative(z_values[layer][i]);
                }
            }

            for (int layer = 0; layer <= hiddenLayers; ++layer) {
                size_t in_size  = (layer == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (layer == hiddenLayers) ? outputNeurons : hiddenNeurons;
                
                for (size_t j = 0; j < out_size; ++j) {
                    biases[layer][j] -= learning_rate * deltas[layer][j];

                    for (size_t i = 0; i < in_size; ++i) {
                        weights[layer][j][i] -= learning_rate * deltas[layer][j] * activations[layer][i];
                    }
                }
            }
        }

        /*
        train(): Trains the neural network using the provided dataset

        @param input: Vector of input vectors for training
        @param input_target: Vector of target output vectors for training
        @param epochs: Number of training epochs
        @param learning_rate: Learning rate for weight updates
        @param batch_size: Size of each training batch
        */
        void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& input_target, int epochs, double learning_rate, int batch_size) {
            
            if (input.size() != input_target.size()) {
                std::cerr << "Error: Input and target sizes don't match!" << std::endl;
                return;
            }
            
            std::vector<size_t> indices(input.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            std::random_device rd;
            std::mt19937 gen(rd());
            
            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), gen);
                
                double total_loss = 0.0;

                for (size_t i = 0; i < input.size(); i += batch_size) {
                    size_t batch_end = std::min(i + batch_size, input.size());
                    
                    for (size_t j = i; j < batch_end; ++j) {
                        size_t idx = indices[j];
                        back_propagate(input[idx], input_target[idx], learning_rate);
                        
                        auto output = forward_propagate(input[idx]);
                        for (size_t k = 0; k < output.size(); ++k) {
                            double diff = output[k] - input_target[idx][k];
                            total_loss += diff * diff;
                        }
                    }
                }
                
                if (epoch % 100 == 0) {
                    std::cout << "Epoch " << epoch << ", Loss: " << total_loss / input.size() << std::endl;
                }
            }
        }

        // ====== PREDICTION METHODS ======
        
        /*
        predict(): Makes a prediction for a single input vector

        @param input: Input vector for prediction
        */
        std::vector<double> predict(const std::vector<double>& input) {
            return forward_propagate(input);
        }

        /*
        predict_classes(): Makes a class prediction for a single input vector (Softmax output)

        @param input: Input vector for prediction
        */
        std::vector<double> predict_classes(const std::vector<double>& input) {
            return Softmax(forward_propagate(input));
        }

        // ====== MODEL ======

        /*
        save_model(): Saves the model data to a file

        @param filename: Name of the file to save the model to
        */
        void save_model(const std::string &filename) {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
                return;
            }
            
            file.write(reinterpret_cast<const char*>(&inputNeurons), sizeof(int));
            file.write(reinterpret_cast<const char*>(&hiddenNeurons), sizeof(int));
            file.write(reinterpret_cast<const char*>(&hiddenLayers), sizeof(int));
            file.write(reinterpret_cast<const char*>(&outputNeurons), sizeof(int));
            
            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t in_size  = (i == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;
                
                for (size_t j = 0; j < out_size; ++j) {
                    file.write(reinterpret_cast<const char*>(weights[i][j]), sizeof(double) * in_size);
                }
                
                file.write(reinterpret_cast<const char*>(biases[i]), sizeof(double) * out_size);
            }
            
            file.close();
            std::cout << "Model saved to: " << filename << std::endl;
        }

        /*
        load_model(): Loads the model data from a file

        @param filename: Name of the file to load the model from
        */
        void load_model(const std::string &filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
                return;
            }
            
            int loaded_input, loaded_hidden, loaded_layers, loaded_output;
            file.read(reinterpret_cast<char*>(&loaded_input), sizeof(int));
            file.read(reinterpret_cast<char*>(&loaded_hidden), sizeof(int));
            file.read(reinterpret_cast<char*>(&loaded_layers), sizeof(int));
            file.read(reinterpret_cast<char*>(&loaded_output), sizeof(int));
            
            if (loaded_input != inputNeurons || loaded_hidden != hiddenNeurons || 
                loaded_layers != hiddenLayers || loaded_output != outputNeurons) {
                std::cerr << "Error: Model architecture doesn't match current network!" << std::endl;
                file.close();
                return;
            }
            
            for (int i = 0; i <= hiddenLayers; ++i) {
                size_t in_size  = (i == 0) ? inputNeurons : hiddenNeurons;
                size_t out_size = (i == hiddenLayers) ? outputNeurons : hiddenNeurons;
                
                for (size_t j = 0; j < out_size; ++j) {
                    file.read(reinterpret_cast<char*>(weights[i][j]), sizeof(double) * in_size);
                }
                
                file.read(reinterpret_cast<char*>(biases[i]), sizeof(double) * out_size);
            }
            
            file.close();
            std::cout << "Model loaded from: " << filename << std::endl;
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