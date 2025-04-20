#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// CUDA Error checking
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Host memory allocation functions
double** allocateMatrixHost(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

void freeMatrixHost(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// CUDA kernel for ReLU activation
__global__ void reluKernel(double* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

// CUDA kernel for forward pass (hidden layer computation)
__global__ void forwardHiddenKernel(double* input, double* W1, double* b1, double* hidden, 
                                  int inputSize, int hiddenSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        hidden[i] = b1[i];
        for (int j = 0; j < inputSize; j++) {
            hidden[i] += W1[i * inputSize + j] * input[j];
        }
    }
}

// CUDA kernel for forward pass (output layer computation)
__global__ void forwardOutputKernel(double* hidden, double* W2, double* b2, double* output, 
                                  int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputSize) {
        output[i] = b2[i];
        for (int j = 0; j < hiddenSize; j++) {
            output[i] += W2[i * hiddenSize + j] * hidden[j];
        }
    }
}

// CUDA kernel for softmax activation
__global__ void softmaxKernel(double* x, int size) {
    // First find max for numerical stability
    double max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Apply exp and sum
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] /= sum;
    }
}

// CUDA kernel for output layer gradient computation
__global__ void outputGradientKernel(double* output, double* target, double* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = output[i] - target[i];
    }
}

// CUDA kernel for hidden layer gradient computation
__global__ void hiddenGradientKernel(double* d_output, double* W2, double* hidden, double* d_hidden,
                                   int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        d_hidden[i] = 0;
        for (int j = 0; j < outputSize; j++) {
            d_hidden[i] += W2[j * hiddenSize + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0);
    }
}

// CUDA kernel for updating output layer weights
__global__ void updateOutputWeightsKernel(double* W2, double* d_output, double* hidden, 
                                       double learningRate, int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < outputSize && j < hiddenSize) {
        W2[i * hiddenSize + j] -= learningRate * d_output[i] * hidden[j];
    }
}

// CUDA kernel for updating hidden layer weights
__global__ void updateHiddenWeightsKernel(double* W1, double* d_hidden, double* input,
                                       double learningRate, int inputSize, int hiddenSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < hiddenSize && j < inputSize) {
        W1[i * inputSize + j] -= learningRate * d_hidden[i] * input[j];
    }
}

// CUDA kernel for updating biases
__global__ void updateBiasesKernel(double* bias, double* gradient, double learningRate, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        bias[i] -= learningRate * gradient[i];
    }
}

// Neural network structure
typedef struct {
    // Host arrays
    double** h_W1;
    double** h_W2;
    double* h_b1;
    double* h_b2;
    
    // Device arrays (flattened for better memory access)
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_target;
    double* d_d_hidden;
    double* d_d_output;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->h_W1 = allocateMatrixHost(HIDDEN_SIZE, INPUT_SIZE);
    net->h_W2 = allocateMatrixHost(OUTPUT_SIZE, HIDDEN_SIZE);
    net->h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Initialize weights with random values
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->h_W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->h_W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_target, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_output, OUTPUT_SIZE * sizeof(double)));

    // Copy weights to device
    double* h_W1_flat = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2_flat = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            h_W1_flat[i * INPUT_SIZE + j] = net->h_W1[i][j];
    
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            h_W2_flat[i * HIDDEN_SIZE + j] = net->h_W2[i][j];
    
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, h_W1_flat, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, h_W2_flat, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->h_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->h_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(h_W1_flat);
    free(h_W2_flat);
    
    return net;
}

// Forward pass (GPU)
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Hidden layer computation
    int blockSize = 128;
    int numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    forwardHiddenKernel<<<numBlocks, blockSize>>>(net->d_input, net->d_W1, net->d_b1, net->d_hidden, 
                                              INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // ReLU activation
    reluKernel<<<numBlocks, blockSize>>>(net->d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Output layer computation
    numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    forwardOutputKernel<<<numBlocks, blockSize>>>(net->d_hidden, net->d_W2, net->d_b2, net->d_output, 
                                             HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Softmax activation
    softmaxKernel<<<numBlocks, blockSize>>>(net->d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
}

// Backpropagation (GPU)
void backward(NeuralNetwork* net, double* input, double* target) {
    // Copy target to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    int blockSize = 128;
    
    // Compute output layer gradient
    int numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    outputGradientKernel<<<numBlocks, blockSize>>>(net->d_output, net->d_target, net->d_d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Compute hidden layer gradient
    numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    hiddenGradientKernel<<<numBlocks, blockSize>>>(net->d_d_output, net->d_W2, net->d_hidden, net->d_d_hidden, 
                                               HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update weights using 2D block configuration
    dim3 blockDim(16, 16);
    dim3 gridDimW2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                 (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    updateOutputWeightsKernel<<<gridDimW2, blockDim>>>(net->d_W2, net->d_d_output, net->d_hidden, 
                                                   LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 gridDimW1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                 (INPUT_SIZE + blockDim.y - 1) / blockDim.y);
    updateHiddenWeightsKernel<<<gridDimW1, blockDim>>>(net->d_W1, net->d_d_hidden, net->d_input, 
                                                   LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update biases
    numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    updateBiasesKernel<<<numBlocks, blockSize>>>(net->d_b2, net->d_d_output, LEARNING_RATE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    updateBiasesKernel<<<numBlocks, blockSize>>>(net->d_b1, net->d_d_hidden, LEARNING_RATE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    // Allocate host memory for temporary results
    double* hidden = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // Forward pass
            forward(net, images[i], hidden, output);
            
            // Backward pass
            backward(net, images[i], labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k] + 1e-10);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    
    free(hidden);
    free(output);
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double* hidden = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    free(hidden);
    free(output);
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrixHost(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrixHost(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    // Free device memory
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    cudaFree(net->d_input);
    cudaFree(net->d_hidden);
    cudaFree(net->d_output);
    cudaFree(net->d_target);
    cudaFree(net->d_d_hidden);
    cudaFree(net->d_d_output);
    
    // Free host memory
    freeMatrixHost(net->h_W1, HIDDEN_SIZE);
    freeMatrixHost(net->h_W2, OUTPUT_SIZE);
    free(net->h_b1);
    free(net->h_b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network with CUDA\n\n");

    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    
    // Free dataset memory
    freeMatrixHost(train_images, 60000);
    freeMatrixHost(train_labels, 60000);
    freeMatrixHost(test_images, 10000);
    freeMatrixHost(test_labels, 10000);
    
    return 0;
}