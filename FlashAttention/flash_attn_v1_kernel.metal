#include <metal_stdlib>
using namespace metal;

// flash attention kernel V1
// grid: (n / br) threadgroups
// br = 32, bc = 32, d = 64
constant int Br = 32;
constant int Bc = 32;

kernel void flash_attention_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]])
{
    // shared mem
    threadgroup float Q_tile[32 * 64];
    threadgroup float K_tile[32 * 64];
    threadgroup float V_tile[32 * 64];
    
    // accumulator
    float o_acc[64];
    for(int i=0; i<64; ++i) o_acc[i] = 0.0f;
    
    float l = 0.0f; // sum of exp
    float m = -INFINITY; // max score
    
    uint tx = tid.x; // local thread id
    uint bx = bid.x; // block index for q
    
    uint row_q = bx * Br + tx;
    
    // loading Q tile into shared memory
    if (row_q < (uint)N) {
        for (int d = 0; d < D; ++d) {
            Q_tile[tx * D + d] = Q[row_q * D + d];
        }
    } else {
        // padding for non-multiples
        for (int d = 0; d < D; ++d) Q_tile[tx * D + d] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // looping over K and V blocks
    int num_blocks_k = (N + Bc - 1) / Bc;
    
    for (int j = 0; j < num_blocks_k; ++j) {
        // Load K/V tiles
        
        if (tx < (uint)Bc) {
            uint row_k = j * Bc + tx;
            if (row_k < (uint)N) {
                for (int d = 0; d < D; ++d) {
                    K_tile[tx * D + d] = K[row_k * D + d];
                    V_tile[tx * D + d] = V[row_k * D + d];
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    K_tile[tx * D + d] = 0.0f;
                    V_tile[tx * D + d] = 0.0f;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // computing attention scores for the current block
        
        for (int k = 0; k < Bc; ++k) {
            // Dot product Q[tx] . K[k]
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += Q_tile[tx * D + d] * K_tile[k * D + d];
            }
            score *= scale;
            
            // Online softmax update
            float m_prev = m;
            m = max(m_prev, score);
            float p = exp(score - m);
            float correction = exp(m_prev - m);
            
            l = l * correction + p;
            
            for (int d = 0; d < D; ++d) {
                o_acc[d] = o_acc[d] * correction + p * V_tile[k * D + d];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // writing output to global memory
    if (row_q < (uint)N) {
        for (int d = 0; d < D; ++d) {
            O[row_q * D + d] = o_acc[d] / l;
        }
    }
}
