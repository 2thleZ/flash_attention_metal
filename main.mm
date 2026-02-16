#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Constants
const int N = 1024; // Sequence length
const int D = 64;   // Head dimension
const float SCALE = 1.0f / sqrt(D);

// Helper for Metal errors
void checkError(NSError *error) {
  if (error) {
    std::cerr << "Metal Error: " << [error.localizedDescription UTF8String]
              << std::endl;
    exit(1);
  }
}

void initRandom(float *data, int size) {
  std::mt19937 gen(42); // fixed seed for reproducibility
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  for (int i = 0; i < size; i++) {
    data[i] = dis(gen);
  }
}

int main() {
  @autoreleasepool {
    // Metal Capture Setup
    MTLCaptureDescriptor *captureDescriptor =
        [[MTLCaptureDescriptor alloc] init];
    captureDescriptor.captureObject = MTLCreateSystemDefaultDevice();
    captureDescriptor.destination = MTLCaptureDestinationGPUTraceDocument;

    // setup metal
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::cerr << "Error: No Metal device found." << std::endl;
      return -1;
    }
    std::cout << "Using device: " << [device.name UTF8String] << std::endl;

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // compile kernels
    NSError *error = nil;
    NSString *libraryFile = @"kernels.metal";
    NSString *librarySource =
        [NSString stringWithContentsOfFile:libraryFile
                                  encoding:NSUTF8StringEncoding
                                     error:&error];
    checkError(error);

    checkError(error);

    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> library = [device newLibraryWithSource:librarySource
                                                  options:options
                                                    error:&error];
    checkError(error);

    id<MTLFunction> naiveFunc =
        [library newFunctionWithName:@"naive_attention_kernel"];
    id<MTLComputePipelineState> naivePSO =
        [device newComputePipelineStateWithFunction:naiveFunc error:&error];
    checkError(error);

    // load flash kernel
    id<MTLFunction> flashFunc =
        [library newFunctionWithName:@"flash_attention_kernel"];
    id<MTLComputePipelineState> flashPSO =
        [device newComputePipelineStateWithFunction:flashFunc error:&error];
    checkError(error);

    // allocate memory
    size_t matrixSize = N * D * sizeof(float);
    size_t outputSize = N * D * sizeof(float); // Output is also N x D

    id<MTLBuffer> Q = [device newBufferWithLength:matrixSize
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> K = [device newBufferWithLength:matrixSize
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> V = [device newBufferWithLength:matrixSize
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> O_naive =
        [device newBufferWithLength:outputSize
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> O_flash =
        [device newBufferWithLength:outputSize
                            options:MTLResourceStorageModeShared];

    initRandom((float *)Q.contents, N * D);
    initRandom((float *)K.contents, N * D);
    initRandom((float *)V.contents, N * D);

    // dispatch naive
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:naivePSO];
    [computeEncoder setBuffer:Q offset:0 atIndex:0];
    [computeEncoder setBuffer:K offset:0 atIndex:1];
    [computeEncoder setBuffer:V offset:0 atIndex:2];
    [computeEncoder setBuffer:O_naive offset:0 atIndex:3];

    // pass constants
    [computeEncoder setBytes:&N length:sizeof(int) atIndex:4];
    [computeEncoder setBytes:&D length:sizeof(int) atIndex:5];
    [computeEncoder setBytes:&SCALE length:sizeof(float) atIndex:6];

    MTLSize gridSize =
        MTLSizeMake(N, D, 1); // One thread per output element (naive)
    MTLSize threadGroupSize = MTLSizeMake(32, 1, 1); // 1d block

    // Adjust grid size to cover N*D
    // one thread per query (row of q)
    gridSize = MTLSizeMake(N, 1, 1);
    threadGroupSize = MTLSizeMake(MIN(N, 256), 1, 1);

    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // dispatch flash
    commandBuffer = [commandQueue commandBuffer];
    computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:flashPSO];
    [computeEncoder setBuffer:Q offset:0 atIndex:0];
    [computeEncoder setBuffer:K offset:0 atIndex:1];
    [computeEncoder setBuffer:V offset:0 atIndex:2];
    [computeEncoder setBuffer:O_flash offset:0 atIndex:3];

    [computeEncoder setBytes:&N length:sizeof(int) atIndex:4];
    [computeEncoder setBytes:&D length:sizeof(int) atIndex:5];
    [computeEncoder setBytes:&SCALE length:sizeof(float) atIndex:6];

    // calculating grid dimensions (N total threads, 32 per group)

    MTLSize flashGridSize = MTLSizeMake(N, 1, 1);
    MTLSize flashGroupSize = MTLSizeMake(32, 1, 1); // Must match Br

    // dispatching threads to match grid size
    // ensuring threadgroups align with kernel logic (N/32 groups)

    [computeEncoder dispatchThreads:flashGridSize
              threadsPerThreadgroup:flashGroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Note: To capture a specific kernel, wrap dispatch calls with
    // MTLCaptureManager start/stop.

    std::cout << "FlashAttention Completed." << std::endl;

    // verify results
    float *naivePtr = (float *)O_naive.contents;
    float *flashPtr = (float *)O_flash.contents;

    float maxDiff = 0.0f;
    for (int i = 0; i < N * D; i++) {
      float diff = std::abs(naivePtr[i] - flashPtr[i]);
      if (diff > maxDiff)
        maxDiff = diff;
    }

    std::cout << "Max Difference: " << maxDiff << std::endl;
    if (maxDiff < 1e-3) {
      std::cout << "PASSED" << std::endl;
    } else {
      std::cout << "FAILED" << std::endl;
    }

    // verify causal masking
    std::cout << "Verifying Causal Masking..." << std::endl;
    // We use N=128
    int N_causal = 128;

    // using float buffers and casting for V4 input
    size_t sz_c = N_causal * D * sizeof(float);
    size_t sz_c_h = N_causal * D * sizeof(uint16_t);

    id<MTLBuffer> Q_c =
        [device newBufferWithLength:sz_c options:MTLResourceStorageModeShared];
    id<MTLBuffer> K_c =
        [device newBufferWithLength:sz_c options:MTLResourceStorageModeShared];
    id<MTLBuffer> V_c =
        [device newBufferWithLength:sz_c options:MTLResourceStorageModeShared];

    id<MTLBuffer> Q_ch =
        [device newBufferWithLength:sz_c_h
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> K_ch =
        [device newBufferWithLength:sz_c_h
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> V_ch =
        [device newBufferWithLength:sz_c_h
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> O_ch = [device
        newBufferWithLength:sz_c_h
                    options:MTLResourceStorageModeShared]; // Output Half
    id<MTLBuffer> L_ch =
        [device newBufferWithLength:N_causal * sizeof(float)
                            options:MTLResourceStorageModeShared];

    initRandom((float *)Q_c.contents, N_causal * D);
    initRandom((float *)K_c.contents, N_causal * D);
    initRandom((float *)V_c.contents, N_causal * D);

    // Convert to Half
    float *qc_f = (float *)Q_c.contents;
    float *kc_f = (float *)K_c.contents;
    float *vc_f = (float *)V_c.contents;
    uint16_t *qc_h = (uint16_t *)Q_ch.contents;
    uint16_t *kc_h = (uint16_t *)K_ch.contents;
    uint16_t *vc_h = (uint16_t *)V_ch.contents;

    for (int i = 0; i < N_causal * D; ++i) {
      __fp16 vq = (__fp16)qc_f[i];
      qc_h[i] = *(uint16_t *)&vq;
      __fp16 vk = (__fp16)kc_f[i];
      kc_h[i] = *(uint16_t *)&vk;
      __fp16 vv = (__fp16)vc_f[i];
      vc_h[i] = *(uint16_t *)&vv;
    }

    // dispatch v4 causal
    {
      id<MTLComputePipelineState> causalPSO = nil;
      // loading V4 kernel locally
      id<MTLFunction> fn =
          [library newFunctionWithName:@"flash_attention_v4_half_kernel"];
      causalPSO = [device newComputePipelineStateWithFunction:fn error:&error];
      checkError(error);

      id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
      [enc setComputePipelineState:causalPSO];
      [enc setBuffer:Q_ch offset:0 atIndex:0];
      [enc setBuffer:K_ch offset:0 atIndex:1];
      [enc setBuffer:V_ch offset:0 atIndex:2];
      [enc setBuffer:O_ch offset:0 atIndex:3];
      [enc setBytes:&N_causal length:sizeof(int) atIndex:4];
      [enc setBytes:&D length:sizeof(int) atIndex:5];
      [enc setBytes:&SCALE length:sizeof(float) atIndex:6];

      int b_stride = N_causal * D;
      int h_stride = N_causal * D;
      [enc setBytes:&b_stride length:sizeof(int) atIndex:7];
      [enc setBytes:&h_stride length:sizeof(int) atIndex:8];
      [enc setBuffer:L_ch offset:0 atIndex:9];

      bool causal_true = true;
      [enc setBytes:&causal_true length:sizeof(bool) atIndex:10];

      int Br = 16;
      int nb = (N_causal + Br - 1) / Br;
      [enc dispatchThreadgroups:MTLSizeMake(nb, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc endEncoding];
      [cmdbuf commit];
      [cmdbuf waitUntilCompleted];
    }

    // CPU reference for causal attention
    std::vector<float> O_ref(N_causal * D);
    for (int i = 0; i < N_causal; ++i) {
      float max_s = -INFINITY;
      std::vector<float> scores(i + 1);

      for (int j = 0; j <= i; ++j) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d)
          score += qc_f[i * D + d] * kc_f[j * D + d];
        score *= SCALE;
        scores[j] = score;
        if (score > max_s)
          max_s = score;
      }

      float sum_exp = 0.0f;
      for (int j = 0; j <= i; ++j) {
        scores[j] = exp(scores[j] - max_s);
        sum_exp += scores[j];
      }

      for (int d = 0; d < D; ++d) {
        float val = 0.0f;
        for (int j = 0; j <= i; ++j) {
          val += scores[j] * vc_f[j * D + d];
        }
        O_ref[i * D + d] = val / sum_exp;
      }
    }

    // Compare
    float maxDiffC = 0.0f;
    uint16_t *out_ptr = (uint16_t *)O_ch.contents;
    for (int i = 0; i < N_causal * D; ++i) {
      __fp16 val_h = *(__fp16 *)&out_ptr[i];
      float val_f = (float)val_h;
      float diff = std::abs(val_f - O_ref[i]);
      if (diff > maxDiffC)
        maxDiffC = diff;
    }
    std::cout << "Causal Max Diff: " << maxDiffC << std::endl;
    if (maxDiffC < 1e-2)
      std::cout << "CAUSAL PASSED" << std::endl;
    else
      std::cout << "CAUSAL FAILED" << std::endl;

    // loading FlashAttention V2 kernel
    id<MTLFunction> flashV2Func =
        [library newFunctionWithName:@"flash_attention_v2_kernel"];
    id<MTLComputePipelineState> flashV2PSO =
        [device newComputePipelineStateWithFunction:flashV2Func error:&error];
    checkError(error);

    // loading FlashAttention V3 kernel (Matrix Intrinsics)
    id<MTLFunction> flashV3Func =
        [library newFunctionWithName:@"flash_attention_simd_kernel"];
    id<MTLComputePipelineState> flashV3PSO =
        [device newComputePipelineStateWithFunction:flashV3Func error:&error];
    checkError(error);

    // benchmark
    std::cout << "\n--- Benchmarking ---\n";
    std::cout << "N,Naive(ms),Flash(ms),FlashV2(ms),FlashV3(ms),FlashV4(ms),"
                 "SpeedupV2,SpeedupV4"
              << std::endl;

    std::ofstream csvFile("benchmark_results.csv");
    if (csvFile.is_open()) {
      csvFile << "N,Naive(ms),Flash(ms),FlashV2(ms),FlashV3(ms),FlashV4(ms),"
                 "SpeedupV2,SpeedupV4\n";
    }

    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    // loading v4 kernel
    id<MTLFunction> flashV4Func =
        [library newFunctionWithName:@"flash_attention_v4_half_kernel"];
    id<MTLComputePipelineState> flashV4PSO =
        [device newComputePipelineStateWithFunction:flashV4Func error:&error];
    checkError(error);

    for (int curr_n : sizes) {
      size_t currSize = curr_n * D * sizeof(float);
      size_t currSizeHalf =
          curr_n * D * sizeof(uint16_t); // half precision size

      id<MTLBuffer> Q_curr =
          [device newBufferWithLength:currSize
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> K_curr =
          [device newBufferWithLength:currSize
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> V_curr =
          [device newBufferWithLength:currSize
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> O_curr =
          [device newBufferWithLength:currSize
                              options:MTLResourceStorageModeShared];

      // allocating buffers for half precision
      id<MTLBuffer> Q_half =
          [device newBufferWithLength:currSizeHalf
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> K_half =
          [device newBufferWithLength:currSizeHalf
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> V_half =
          [device newBufferWithLength:currSizeHalf
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> O_half =
          [device newBufferWithLength:currSizeHalf
                              options:MTLResourceStorageModeShared];

      // l buffer for v4
      size_t size_l_bytes = 1 * 1 * curr_n * sizeof(float);
      id<MTLBuffer> L_buf =
          [device newBufferWithLength:size_l_bytes
                              options:MTLResourceStorageModeShared];

      initRandom((float *)Q_curr.contents, curr_n * D);
      initRandom((float *)K_curr.contents, curr_n * D);
      initRandom((float *)V_curr.contents, curr_n * D);

      // convert to half
      float *q_f = (float *)Q_curr.contents;
      float *k_f = (float *)K_curr.contents;
      float *v_f = (float *)V_curr.contents;

      uint16_t *q_h = (uint16_t *)Q_half.contents;
      uint16_t *k_h = (uint16_t *)K_half.contents;
      uint16_t *v_h = (uint16_t *)V_half.contents;

      for (int i = 0; i < curr_n * D; ++i) {
        __fp16 val_q = (__fp16)q_f[i];
        __fp16 val_k = (__fp16)k_f[i];
        __fp16 val_v = (__fp16)v_f[i];
        q_h[i] = *(uint16_t *)&val_q;
        k_h[i] = *(uint16_t *)&val_k;
        v_h[i] = *(uint16_t *)&val_v;
      }

      // time naive
      double naiveTime = 0.0;
      if (curr_n <= 8192) {
        // clear output buffer
        memset(O_curr.contents, 0, currSize);
        auto start = std::chrono::high_resolution_clock::now();

        id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:naivePSO];
        [enc setBuffer:Q_curr offset:0 atIndex:0];
        [enc setBuffer:K_curr offset:0 atIndex:1];
        [enc setBuffer:V_curr offset:0 atIndex:2];
        [enc setBuffer:O_curr offset:0 atIndex:3];
        [enc setBytes:&curr_n length:sizeof(int) atIndex:4];
        [enc setBytes:&D length:sizeof(int) atIndex:5];
        [enc setBytes:&SCALE length:sizeof(float) atIndex:6];

        MTLSize grid = MTLSizeMake(curr_n, 1, 1);
        MTLSize group = MTLSizeMake(MIN(curr_n, 256), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        naiveTime =
            std::chrono::duration<double, std::milli>(end - start).count();
      }

      // time flash v1
      double flashTime = 0.0;
      {
        // clear output buffer
        memset(O_curr.contents, 0, currSize);
        auto start = std::chrono::high_resolution_clock::now();

        id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:flashPSO];
        [enc setBuffer:Q_curr offset:0 atIndex:0];
        [enc setBuffer:K_curr offset:0 atIndex:1];
        [enc setBuffer:V_curr offset:0 atIndex:2];
        [enc setBuffer:O_curr offset:0 atIndex:3];
        [enc setBytes:&curr_n length:sizeof(int) atIndex:4];
        [enc setBytes:&D length:sizeof(int) atIndex:5];
        [enc setBytes:&SCALE length:sizeof(float) atIndex:6];

        MTLSize grid = MTLSizeMake(curr_n, 1, 1);
        MTLSize group = MTLSizeMake(32, 1, 1); // Br=32
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        flashTime =
            std::chrono::duration<double, std::milli>(end - start).count();
      }

      // time v2
      double flashV2Time = 0.0;
      {
        // clear output buffer
        memset(O_curr.contents, 0, currSize);
        auto start = std::chrono::high_resolution_clock::now();

        id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:flashV2PSO];
        [enc setBuffer:Q_curr offset:0 atIndex:0];
        [enc setBuffer:K_curr offset:0 atIndex:1];
        [enc setBuffer:V_curr offset:0 atIndex:2];
        [enc setBuffer:O_curr offset:0 atIndex:3];
        [enc setBytes:&curr_n length:sizeof(int) atIndex:4];
        [enc setBytes:&D length:sizeof(int) atIndex:5];
        [enc setBytes:&SCALE length:sizeof(float) atIndex:6];

        MTLSize grid = MTLSizeMake(curr_n, 1, 1);
        MTLSize group = MTLSizeMake(32, 1, 1); // Br=32
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        flashV2Time =
            std::chrono::duration<double, std::milli>(end - start).count();
      }

      // time v3 (matrix)
      double flashV3Time = 0.0;
      {
        // clear output buffer
        memset(O_curr.contents, 0, currSize);
        auto start = std::chrono::high_resolution_clock::now();

        id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:flashV3PSO];
        [enc setBuffer:Q_curr offset:0 atIndex:0];
        [enc setBuffer:K_curr offset:0 atIndex:1];
        [enc setBuffer:V_curr offset:0 atIndex:2];
        [enc setBuffer:O_curr offset:0 atIndex:3];
        [enc setBytes:&curr_n length:sizeof(int) atIndex:4];
        [enc setBytes:&D length:sizeof(int) atIndex:5];
        [enc setBytes:&SCALE length:sizeof(float) atIndex:6];

        int Br = 16;
        int num_groups = (curr_n + Br - 1) / Br;
        MTLSize gridSize = MTLSizeMake(num_groups * 32, 1, 1);
        MTLSize groupSize = MTLSizeMake(32, 1, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        flashV3Time =
            std::chrono::duration<double, std::milli>(end - start).count();
      }

      // time v4 (fp16)
      double flashV4Time = 0.0;
      {
        // clear output buffers
        memset(O_half.contents, 0, currSizeHalf);
        memset(L_buf.contents, 0, size_l_bytes);
        auto start = std::chrono::high_resolution_clock::now();

        id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:flashV4PSO];
        [enc setBuffer:Q_half offset:0 atIndex:0];
        [enc setBuffer:K_half offset:0 atIndex:1];
        [enc setBuffer:V_half offset:0 atIndex:2];
        [enc setBuffer:O_half offset:0 atIndex:3];
        [enc setBytes:&curr_n length:sizeof(int) atIndex:4];
        [enc setBytes:&D length:sizeof(int) atIndex:5];
        [enc setBytes:&SCALE length:sizeof(float) atIndex:6];

        // If B=1, H=1.
        // Batch Stride = H * N * D? Yes.
        // Head Stride = N * D.
        int H = 1;
        int B = 1;
        int b_stride = H * curr_n * D;
        int h_stride = curr_n * D;

        [enc setBytes:&b_stride length:sizeof(int) atIndex:7];
        [enc setBytes:&h_stride length:sizeof(int) atIndex:8];
        [enc setBuffer:L_buf offset:0 atIndex:9]; // L output

        bool is_causal = false;
        [enc setBytes:&is_causal length:sizeof(bool) atIndex:10];

        // configuring grid dimensions: Z=Batch, Y=Head, X=Blocks
        int Br = 16;
        int num_blocks = (curr_n + Br - 1) / Br;
        MTLSize groupsGrid = MTLSizeMake(num_blocks, H, B);
        MTLSize threadsGroup = MTLSizeMake(32, 1, 1);

        [enc dispatchThreadgroups:groupsGrid
            threadsPerThreadgroup:threadsGroup];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        flashV4Time =
            std::chrono::duration<double, std::milli>(end - start).count();
      }

      double speedupV2 = (naiveTime > 0) ? naiveTime / flashV2Time : 0;
      double speedupV4 = (naiveTime > 0) ? naiveTime / flashV4Time : 0;

      std::cout << curr_n << "," << naiveTime << "," << flashTime << ","
                << flashV2Time << "," << flashV3Time << "," << flashV4Time
                << "," << speedupV2 << "," << speedupV4 << std::endl;

      if (csvFile.is_open()) {
        csvFile << curr_n << "," << naiveTime << "," << flashTime << ","
                << flashV2Time << "," << flashV3Time << "," << flashV4Time
                << "," << speedupV2 << "," << speedupV4 << "\n";
        csvFile.flush();
      }
    }

    // high occupancy benchmark (b=16, h=8)
    std::cout << "\n--- High Occupancy Benchmark (B=16, H=8) ---\n";
    std::cout << "N,FlashV2(ms),FlashV4(ms),Backward(ms),SpeedupV4vsV2"
              << std::endl;

    int B = 16;
    int H = 8;

    // load backward kernel
    id<MTLFunction> flashBwdFunc =
        [library newFunctionWithName:@"flash_attention_backward_kernel"];
    id<MTLComputePipelineState> flashBwdPSO =
        [device newComputePipelineStateWithFunction:flashBwdFunc error:&error];
    checkError(error);

    for (int curr_n : sizes) {
      // Allocate buffers (cap at 1GB)

      size_t total_elems = (size_t)B * H * curr_n * D;
      size_t size_bytes_f = total_elems * sizeof(float);
      size_t size_bytes_h = total_elems * sizeof(uint16_t);

      if (size_bytes_f > 1024 * 1024 * 1024) { // Cap at 1GB for safety
        break;
      }

      // Reuse buffers if possible? No, need new size.
      id<MTLBuffer> Q_f =
          [device newBufferWithLength:size_bytes_f
                              options:MTLResourceStorageModeShared];

      id<MTLBuffer> Q_h =
          [device newBufferWithLength:size_bytes_h
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> K_h =
          [device newBufferWithLength:size_bytes_h
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> V_h =
          [device newBufferWithLength:size_bytes_h
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> O_h =
          [device newBufferWithLength:size_bytes_h
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> dO_h =
          [device newBufferWithLength:size_bytes_h
                              options:MTLResourceStorageModeShared];

      // logsumexp buffer l: [b, h, n]
      size_t size_L = (size_t)B * H * curr_n * sizeof(float);
      id<MTLBuffer> L_buf =
          [device newBufferWithLength:size_L
                              options:MTLResourceStorageModeShared];

      // gradients: dq, dk, dv (float for atomics)
      id<MTLBuffer> dQ =
          [device newBufferWithLength:size_bytes_f
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> dK =
          [device newBufferWithLength:size_bytes_f
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> dV =
          [device newBufferWithLength:size_bytes_f
                              options:MTLResourceStorageModeShared];

      // Init Q_f -> Q_h
      initRandom((float *)Q_f.contents, curr_n * D);

      // Convert Q_f -> Q_h
      float *q_f_ptr = (float *)Q_f.contents;
      uint16_t *q_h_ptr = (uint16_t *)Q_h.contents;
      for (int i = 0; i < curr_n * D; ++i) {
        __fp16 val =
            (__fp16)(q_f_ptr[i] * 0.01f); // Scale down to avoid overflow
        q_h_ptr[i] = *(uint16_t *)&val;
      }

      // Init dO (random) -> dO_h
      initRandom((float *)Q_f.contents, curr_n * D);
      uint16_t *do_h_ptr = (uint16_t *)dO_h.contents;
      for (int i = 0; i < curr_n * D; ++i) {
        __fp16 val = (__fp16)(q_f_ptr[i] * 0.01f);
        do_h_ptr[i] = *(uint16_t *)&val;
      }

      // Initialize K and V (copy from Q for simplicity)
      memcpy(K_h.contents, Q_h.contents, size_bytes_h);
      memcpy(V_h.contents, Q_h.contents, size_bytes_h);

      // benchmarking V4 Forward (with L write)
      double flashV4Time = 0.0;
      {
        // clear output buffers
        memset(O_h.contents, 0, size_bytes_h);
        memset(L_buf.contents, 0, size_L);
        auto start = std::chrono::high_resolution_clock::now();

        id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:flashV4PSO];
        [enc setBuffer:Q_h offset:0 atIndex:0];
        [enc setBuffer:K_h offset:0 atIndex:1];
        [enc setBuffer:V_h offset:0 atIndex:2];
        [enc setBuffer:O_h offset:0 atIndex:3];
        [enc setBytes:&curr_n length:sizeof(int) atIndex:4];
        [enc setBytes:&D length:sizeof(int) atIndex:5];
        [enc setBytes:&SCALE length:sizeof(float) atIndex:6];

        int b_stride = H * curr_n * D;
        int h_stride = curr_n * D;

        [enc setBytes:&b_stride length:sizeof(int) atIndex:7];
        [enc setBytes:&h_stride length:sizeof(int) atIndex:8];
        [enc setBuffer:L_buf offset:0 atIndex:9]; // L output

        bool is_causal = false;
        [enc setBytes:&is_causal length:sizeof(bool) atIndex:10];

        int Br = 16;
        int num_blocks = (curr_n + Br - 1) / Br;

        MTLSize groupsGrid = MTLSizeMake(num_blocks, H, B);
        MTLSize threadsGroup = MTLSizeMake(32, 1, 1);

        [enc dispatchThreadgroups:groupsGrid
            threadsPerThreadgroup:threadsGroup];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        flashV4Time =
            std::chrono::duration<double, std::milli>(end - start).count();
      }

      // time backward
      double backwardTime = 0.0;
      {
        // zero gradients
        memset(dQ.contents, 0, size_bytes_f);
        memset(dK.contents, 0, size_bytes_f);
        memset(dV.contents, 0, size_bytes_f);
        // clear backward outputs if reused (dO is input, but dQ/dK/dV are
        // outputs)

        auto start = std::chrono::high_resolution_clock::now();

        id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:flashBwdPSO];
        [enc setBuffer:Q_h offset:0 atIndex:0];
        [enc setBuffer:K_h offset:0 atIndex:1];
        [enc setBuffer:V_h offset:0 atIndex:2];
        [enc setBuffer:O_h offset:0 atIndex:3];
        [enc setBuffer:dO_h offset:0 atIndex:4];
        [enc setBuffer:L_buf offset:0 atIndex:5];
        [enc setBuffer:dQ offset:0 atIndex:6];
        [enc setBuffer:dK offset:0 atIndex:7];
        [enc setBuffer:dV offset:0 atIndex:8];
        [enc setBytes:&curr_n length:sizeof(int) atIndex:9];
        [enc setBytes:&D length:sizeof(int) atIndex:10];
        [enc setBytes:&SCALE length:sizeof(float) atIndex:11];

        int b_stride = H * curr_n * D;
        int h_stride = curr_n * D;

        [enc setBytes:&b_stride length:sizeof(int) atIndex:12];
        [enc setBytes:&h_stride length:sizeof(int) atIndex:13];

        bool is_causal = false;
        [enc setBytes:&is_causal length:sizeof(bool) atIndex:14];

        int Br = 16;
        int num_blocks = (curr_n + Br - 1) / Br;
        MTLSize groupsGrid = MTLSizeMake(num_blocks, H, B);
        MTLSize threadsGroup = MTLSizeMake(32, 1, 1);

        [enc dispatchThreadgroups:groupsGrid
            threadsPerThreadgroup:threadsGroup];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        backwardTime =
            std::chrono::duration<double, std::milli>(end - start).count();
      }

      // verify backward
      float *dq_ptr = (float *)dQ.contents;
      float sum_dq = 0.0f;
      for (int k = 0; k < 100; ++k)
        sum_dq += std::abs(dq_ptr[k]);

      // Debug prints
      float *l_ptr = (float *)L_buf.contents;
      uint16_t *do_ptr = (uint16_t *)dO_h.contents;
      uint16_t *q_ptr_h = (uint16_t *)Q_h.contents;

      std::cout << "Debug Info (N=" << curr_n << "):" << std::endl;
      std::cout << "  Q[0] (Half): " << (float)q_ptr_h[0]
                << std::endl; // Garbage cast but visible non-zero
      std::cout << "  dO[0] (Half): " << (float)do_ptr[0] << std::endl;
      std::cout << "  L[0] (Float): " << l_ptr[0] << std::endl;
      std::cout << "  dQ[0] (Float): " << dq_ptr[0] << std::endl;
      std::cout << "  Sum(dQ): " << sum_dq << std::endl;

      std::string status = (sum_dq > 1e-9) ? "OK" : "ZERO/FAIL";
      std::cout << curr_n << ",N/A," << flashV4Time << "," << backwardTime
                << "," << status << std::endl;
    }
    if (csvFile.is_open()) {
      csvFile.close();
    }
  }
  return 0;
}
