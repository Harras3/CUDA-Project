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
#define BATCH_SIZE 64  // Reduced from 10000 to a more optimal size for GPU
#define NUM_CLASSES 10  // Digits 0-9
#define NUM_STREAMS 10   // Number of CUDA streams for concurrent operations
#define TILE_SIZE 16 
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
                                  int inputSize, int hiddenSize, int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h_idx < hiddenSize && b_idx < batch_size) {
        double sum = b1[h_idx];
        // Use shared memory for W1 to reduce global memory access
        for (int j = 0; j < inputSize; j++) {
            sum += W1[h_idx * inputSize + j] * input[b_idx * inputSize + j];
        }
        hidden[b_idx * hiddenSize + h_idx] = sum;
    }
}

// CUDA kernel for forward pass (output layer computation)
__global__ void forwardOutputKernel(double* hidden, double* W2, double* b2, double* output, 
                                  int hiddenSize, int outputSize, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (o_idx < outputSize && b_idx < batch_size) {
        double sum = b2[o_idx];
        for (int j = 0; j < hiddenSize; j++) {
            sum += W2[o_idx * hiddenSize + j] * hidden[b_idx * hiddenSize + j];
        }
        output[b_idx * outputSize + o_idx] = sum;
    }
}

// CUDA kernel for softmax activation - modified to handle batches
__global__ void softmaxKernel(double* x, int size, int batch_size) {
    int b_idx = blockIdx.x;
    
    if (b_idx < batch_size) {
        // Compute base index for this sample in batch
        double* sample = x + b_idx * size;
        
        // Find max for numerical stability
        double max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (sample[i] > max_val) {
                max_val = sample[i];
            }
        }
        
        // Apply exp and compute sum
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sample[i] = exp(sample[i] - max_val);
            sum += sample[i];
        }
        
        // Normalize
        int t_idx = threadIdx.x;
        if (t_idx < size) {
            sample[t_idx] /= sum;
        }
    }
}

// CUDA kernel for output layer gradient computation - modified for batches
__global__ void outputGradientKernel(double* output, double* target, double* d_output, 
                                  int size, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < size && b < batch_size) {
        int idx = b * size + i;
        d_output[idx] = output[idx] - target[idx];
    }
}

// CUDA kernel for hidden layer gradient computation - modified for batches
__global__ void hiddenGradientKernel(double* d_output, double* W2, double* hidden, double* d_hidden,
                                   int hiddenSize, int outputSize, int batch_size) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h < hiddenSize && b < batch_size) {
        int h_idx = b * hiddenSize + h;
        double sum = 0.0;
        for (int j = 0; j < outputSize; j++) {
            sum += W2[j * hiddenSize + h] * d_output[b * outputSize + j];
        }
        d_hidden[h_idx] = sum * (hidden[h_idx] > 0);
    }
}

// CUDA kernel for updating output layer weights - accumulates gradients from batch
__global__ void updateOutputWeightsKernel(double* W2, double* d_output, double* hidden, 
                                       double learningRate, int hiddenSize, int outputSize, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < outputSize && j < hiddenSize) {
        double gradient_sum = 0.0;
        // Accumulate gradients across the batch
        for (int b = 0; b < batch_size; b++) {
            gradient_sum += d_output[b * outputSize + i] * hidden[b * hiddenSize + j];
        }
        // Update weight with average gradient
        W2[i * hiddenSize + j] -= (learningRate * gradient_sum) / batch_size;
    }
}

// CUDA kernel for updating hidden layer weights - accumulates gradients from batch
__global__ void updateHiddenWeightsKernel(double* W1, double* d_hidden, double* input,
                                       double learningRate, int inputSize, int hiddenSize, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < hiddenSize && j < inputSize) {
        double gradient_sum = 0.0;
        // Accumulate gradients across the batch
        for (int b = 0; b < batch_size; b++) {
            gradient_sum += d_hidden[b * hiddenSize + i] * input[b * inputSize + j];
        }
        // Update weight with average gradient
        W1[i * inputSize + j] -= (learningRate * gradient_sum) / batch_size;
    }
}

// CUDA kernel for updating biases - accumulates gradients from batch
__global__ void updateBiasesKernel(double* bias, double* gradient, 
                                 double learningRate, int size, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double gradient_sum = 0.0;
        // Accumulate gradients across the batch
        for (int b = 0; b < batch_size; b++) {
            gradient_sum += gradient[b * size + i];
        }
        // Update bias with average gradient
        bias[i] -= (learningRate * gradient_sum) / batch_size;
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
    
    // Batch processing arrays
    double* d_input_batch;
    double* d_target_batch;
    double* d_hidden_batch;
    double* d_output_batch;
    double* d_d_hidden_batch;
    double* d_d_output_batch;
    
    // CUDA streams
    cudaStream_t streams[NUM_STREAMS];
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->h_W1 = allocateMatrixHost(HIDDEN_SIZE, INPUT_SIZE);
    net->h_W2 = allocateMatrixHost(OUTPUT_SIZE, HIDDEN_SIZE);
    net->h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Initialize weights with Xavier initialization
    srand(time(NULL));
    double scale_w1 = sqrt(2.0 / INPUT_SIZE);
    double scale_w2 = sqrt(2.0 / HIDDEN_SIZE);
    
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->h_W1[i][j] = ((double)rand() / RAND_MAX * 2 - 1) * scale_w1;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->h_W2[i][j] = ((double)rand() / RAND_MAX * 2 - 1) * scale_w2;

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

    // Allocate batch memory on device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_target_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden_batch, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_hidden_batch, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_output_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));

    // Create CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&net->streams[i]));
    }

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

// Forward pass (GPU) with batch processing
void forwardBatch(NeuralNetwork* net, double* input_batch, double* hidden_batch, double* output_batch, 
                 int batch_size, cudaStream_t stream) {
    // Copy input batch to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_input_batch, input_batch, 
                                    batch_size * INPUT_SIZE * sizeof(double), 
                                    cudaMemcpyHostToDevice, stream));
    
    // Hidden layer computation - use 2D grid for batch processing
    dim3 hiddenBlockSize(16, 16);
    dim3 hiddenGridSize((HIDDEN_SIZE + hiddenBlockSize.x - 1) / hiddenBlockSize.x,
                      (batch_size + hiddenBlockSize.y - 1) / hiddenBlockSize.y);
    forwardHiddenKernel<<<hiddenGridSize, hiddenBlockSize, 0, stream>>>(
        net->d_input_batch, net->d_W1, net->d_b1, net->d_hidden_batch, 
        INPUT_SIZE, HIDDEN_SIZE, batch_size);
    
    // ReLU activation
    int totalHiddenElements = batch_size * HIDDEN_SIZE;
    int blockSize = 256;
    int numBlocks = (totalHiddenElements + blockSize - 1) / blockSize;
    reluKernel<<<numBlocks, blockSize, 0, stream>>>(net->d_hidden_batch, totalHiddenElements);
    
    // Output layer computation - use 2D grid for batch processing
    dim3 outputBlockSize(16, 16);
    dim3 outputGridSize((OUTPUT_SIZE + outputBlockSize.x - 1) / outputBlockSize.x,
                      (batch_size + outputBlockSize.y - 1) / outputBlockSize.y);
    forwardOutputKernel<<<outputGridSize, outputBlockSize, 0, stream>>>(
        net->d_hidden_batch, net->d_W2, net->d_b2, net->d_output_batch, 
        HIDDEN_SIZE, OUTPUT_SIZE, batch_size);
    
    // Softmax activation - process each batch sample
    softmaxKernel<<<batch_size, OUTPUT_SIZE, 0, stream>>>(net->d_output_batch, OUTPUT_SIZE, batch_size);
    
    // Copy results back to host asynchronously only if needed for reporting
    if (hidden_batch != NULL) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(hidden_batch, net->d_hidden_batch, 
                                        batch_size * HIDDEN_SIZE * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
    }
    if (output_batch != NULL) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output_batch, net->d_output_batch, 
                                        batch_size * OUTPUT_SIZE * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
    }
}

// Backpropagation (GPU) with batch processing
void backwardBatch(NeuralNetwork* net, double* input_batch, double* target_batch, 
                  int batch_size, cudaStream_t stream) {
    // Copy target batch to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_target_batch, target_batch, 
                                    batch_size * OUTPUT_SIZE * sizeof(double), 
                                    cudaMemcpyHostToDevice, stream));
    
    // Compute output layer gradient with 2D grid
    dim3 outputBlockSize(16, 16);
    dim3 outputGridSize((OUTPUT_SIZE + outputBlockSize.x - 1) / outputBlockSize.x,
                      (batch_size + outputBlockSize.y - 1) / outputBlockSize.y);
    outputGradientKernel<<<outputGridSize, outputBlockSize, 0, stream>>>(
        net->d_output_batch, net->d_target_batch, net->d_d_output_batch, 
        OUTPUT_SIZE, batch_size);
    
    // Compute hidden layer gradient with 2D grid
    dim3 hiddenBlockSize(16, 16);
    dim3 hiddenGridSize((HIDDEN_SIZE + hiddenBlockSize.x - 1) / hiddenBlockSize.x,
                      (batch_size + hiddenBlockSize.y - 1) / hiddenBlockSize.y);
    hiddenGradientKernel<<<hiddenGridSize, hiddenBlockSize, 0, stream>>>(
        net->d_d_output_batch, net->d_W2, net->d_hidden_batch, net->d_d_hidden_batch, 
        HIDDEN_SIZE, OUTPUT_SIZE, batch_size);
    
    // Update weights using 2D block configuration
    dim3 blockDim(16, 16);
    dim3 gridDimW2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                 (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    updateOutputWeightsKernel<<<gridDimW2, blockDim, 0, stream>>>(
        net->d_W2, net->d_d_output_batch, net->d_hidden_batch, 
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);
    
    dim3 gridDimW1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                 (INPUT_SIZE + blockDim.y - 1) / blockDim.y);
    updateHiddenWeightsKernel<<<gridDimW1, blockDim, 0, stream>>>(
        net->d_W1, net->d_d_hidden_batch, net->d_input_batch, 
        LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE, batch_size);
    
    // Update biases
    int blockSize = 256;
    int numBlocksOutput = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    updateBiasesKernel<<<numBlocksOutput, blockSize, 0, stream>>>(
        net->d_b2, net->d_d_output_batch, LEARNING_RATE, OUTPUT_SIZE, batch_size);
    
    int numBlocksHidden = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    updateBiasesKernel<<<numBlocksHidden, blockSize, 0, stream>>>(
        net->d_b1, net->d_d_hidden_batch, LEARNING_RATE, HIDDEN_SIZE, batch_size);
}

// Train network with batch processing
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    
    // Create pinned memory for faster host-device transfers
    double *input_batch, *target_batch, *hidden_batch, *output_batch;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&target_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&hidden_batch, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&output_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        // Shuffle training data
        int* indices = (int*)malloc(numImages * sizeof(int));
        for (int i = 0; i < numImages; i++) {
            indices[i] = i;
        }
        for (int i = numImages - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Process in batches
        for (int batch_start = 0; batch_start < numImages; batch_start += BATCH_SIZE) {
            int batch_size = (batch_start + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - batch_start);
            int stream_idx = (batch_start / BATCH_SIZE) % NUM_STREAMS;
            
            // Prepare batch data - use shuffled indices
            for (int i = 0; i < batch_size; i++) {
                int idx = indices[batch_start + i];
                memcpy(input_batch + i * INPUT_SIZE, images[idx], INPUT_SIZE * sizeof(double));
                memcpy(target_batch + i * OUTPUT_SIZE, labels[idx], OUTPUT_SIZE * sizeof(double));
            }
            
            // Forward and backward pass for this batch
            forwardBatch(net, input_batch, hidden_batch, output_batch, batch_size, net->streams[stream_idx]);
            backwardBatch(net, input_batch, target_batch, batch_size, net->streams[stream_idx]);
            
            // Synchronize stream to ensure batch is complete before next batch in same stream
            cudaStreamSynchronize(net->streams[stream_idx]);
            
            // Compute loss & accuracy for this batch
            for (int i = 0; i < batch_size; i++) {
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    loss -= target_batch[i * OUTPUT_SIZE + k] * 
                            log(output_batch[i * OUTPUT_SIZE + k] + 1e-10);
                }
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output_batch[i * OUTPUT_SIZE + j] > output_batch[i * OUTPUT_SIZE + pred]) 
                        pred = j;
                    if (target_batch[i * OUTPUT_SIZE + j] > target_batch[i * OUTPUT_SIZE + actual]) 
                        actual = j;
                }
                if (pred == actual) correct++;
            }
            
            // Print progress periodically
            if ((batch_start / BATCH_SIZE) % 50 == 0) {
                printf("Epoch %d - Batch %d/%d processed\n", 
                       epoch + 1, batch_start / BATCH_SIZE, (numImages + BATCH_SIZE - 1) / BATCH_SIZE);
            }
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
        
        free(indices);
    }
    
    // Free pinned memory
    cudaFreeHost(input_batch);
    cudaFreeHost(target_batch);
    cudaFreeHost(hidden_batch);
    cudaFreeHost(output_batch);
    
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    // Allocate pinned memory for batches
    double *input_batch, *output_batch;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&output_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    
    int correct = 0;
    for (int batch_start = 0; batch_start < numImages; batch_start += BATCH_SIZE) {
        int batch_size = (batch_start + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - batch_start);
        
        // Prepare batch data
        for (int i = 0; i < batch_size; i++) {
            memcpy(input_batch + i * INPUT_SIZE, images[batch_start + i], INPUT_SIZE * sizeof(double));
        }
        
        // Forward pass for this batch - don't need hidden layer values for evaluation
        forwardBatch(net, input_batch, NULL, output_batch, batch_size, net->streams[0]);
        cudaStreamSynchronize(net->streams[0]);
        
        // Count correct predictions
        for (int i = 0; i < batch_size; i++) {
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output_batch[i * OUTPUT_SIZE + j] > output_batch[i * OUTPUT_SIZE + pred]) 
                    pred = j;
                if (labels[batch_start + i][j] > labels[batch_start + i][actual]) 
                    actual = j;
            }
            if (pred == actual) correct++;
        }
    }
    
    // Free pinned memory
    cudaFreeHost(input_batch);
    cudaFreeHost(output_batch);
    
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
    
   // Free batch memory
    cudaFree(net->d_input_batch);
    cudaFree(net->d_target_batch);
    cudaFree(net->d_hidden_batch);
    cudaFree(net->d_output_batch);
    cudaFree(net->d_d_hidden_batch);
    cudaFree(net->d_d_output_batch);
    
    // Destroy CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(net->streams[i]);
    }
    
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

    // Use a smaller fraction of the CUDA heap for better memory management
    size_t heapSize = 512 * 1024 * 1024;  // 512 MB
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);

    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    // Set device to higher performance mode if available
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    
    // Free dataset memory
    freeMatrixHost(train_images, 60000);
    freeMatrixHost(train_labels, 60000);
    freeMatrixHost(test_images, 10000);
    freeMatrixHost(test_labels, 10000);
    
    // Clean up any remaining CUDA resources
    cudaDeviceReset();
    
    return 0;
}