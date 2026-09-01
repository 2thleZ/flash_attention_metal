#include <metal_stdlib>
using namespace metal;

// FlashAttention Kernel V2 (Vectorized float4)
// D = 64 floats = 16 float4s
// Br = 32, Bc = 32
// Threadgroup memory uses float4 arrays

kernel void flash_attention_v2_kernel(
    device const float4* Q [[buffer(0)]],
    device const float4* K [[buffer(1)]],
    device const float4* V [[buffer(2)]],
    device float4* O [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& D [[buffer(5)]], // D is still 64 (floats)
    constant float& scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]])
{
    // D_vec = D / 4 = 16
    const int D_vec = 16;
    const int Bc_local = 16; // Reduced from 32 to fit double buffers
    
    // Double Buffering: Two sets of K/V tiles (A and B)
    // Q: 32*16=512 float4 = 8KB
    // K_A + K_B + V_A + V_B: 4 * 16*16 = 4*256 = 1024 float4 = 16KB
    // Total: 24KB < 32KB limit
    threadgroup float4 Q_tile[32 * 16];
    threadgroup float4 K_tile_A[16 * 16];
    threadgroup float4 V_tile_A[16 * 16];
    threadgroup float4 K_tile_B[16 * 16];
    threadgroup float4 V_tile_B[16 * 16];
    
    float4 o_acc[16];
    for(int i=0; i<16; ++i) o_acc[i] = float4(0.0f);
    
    float l = 0.0f;
    float m = -INFINITY;
    
    uint tx = tid.x; // 0..31
    uint bx = bid.x;
    
    uint row_q = bx * Br + tx;
    
    // loads q_tile
    if (row_q < (uint)N) {
        for (int d = 0; d < D_vec; ++d) {
            Q_tile[tx * D_vec + d] = Q[row_q * D_vec + d];
        }
    } else {
        for (int d = 0; d < D_vec; ++d) Q_tile[tx * D_vec + d] = float4(0.0f);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int num_blocks_k = (N + Bc_local - 1) / Bc_local;
    
    // preloads first block
    if (num_blocks_k > 0) {
        if (tx < (uint)Bc_local) { // Bound check
            uint row_k = tx;
            if (row_k < (uint)N) {
                for (int d = 0; d < D_vec; ++d) {
                    K_tile_A[tx * D_vec + d] = K[row_k * D_vec + d];
                    V_tile_A[tx * D_vec + d] = V[row_k * D_vec + d];
                }
            } else {
                for (int d = 0; d < D_vec; ++d) {
                    K_tile_A[tx * D_vec + d] = float4(0.0f);
                    V_tile_A[tx * D_vec + d] = float4(0.0f);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ping-pong pointers
    threadgroup float4* K_curr = K_tile_A;
    threadgroup float4* V_curr = V_tile_A;
    threadgroup float4* K_next = K_tile_B;
    threadgroup float4* V_next = V_tile_B;
    
    for (int j = 0; j < num_blocks_k; ++j) {
        // prefetch next block
        if (j + 1 < num_blocks_k) {
            if (tx < (uint)Bc_local) { // Bound check
                uint row_k_next = (j + 1) * Bc_local + tx;
                if (row_k_next < (uint)N) {
                    for (int d = 0; d < D_vec; ++d) {
                        K_next[tx * D_vec + d] = K[row_k_next * D_vec + d];
                        V_next[tx * D_vec + d] = V[row_k_next * D_vec + d];
                    }
                } else {
                    for (int d = 0; d < D_vec; ++d) {
                        K_next[tx * D_vec + d] = float4(0.0f);
                        V_next[tx * D_vec + d] = float4(0.0f);
                    }
                }
            }
        }
        
        // compute on current block
        for (int k = 0; k < Bc_local; ++k) {
            float score = 0.0f;
            #pragma clang loop unroll(full)
            for (int d = 0; d < D_vec; ++d) {
                score += dot(Q_tile[tx * D_vec + d], K_curr[k * D_vec + d]);
            }
            score *= scale;
            
            float m_prev = m;
            m = max(m_prev, score);
            float p = exp(score - m);
            float correction = exp(m_prev - m);
            
            l = l * correction + p;
            
            #pragma clang loop unroll(full)
            for (int d = 0; d < D_vec; ++d) {
                o_acc[d] = o_acc[d] * correction + p * V_curr[k * D_vec + d];
            }
        }
        
        // barrier for prefetch
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // swap
        threadgroup float4* tmp_k = K_curr;
        threadgroup float4* tmp_v = V_curr;
        K_curr = K_next;
        V_curr = V_next;
        K_next = tmp_k;
        V_next = tmp_v;
    }
    
    // Write Output
    if (row_q < (uint)N) {
        for (int d = 0; d < D_vec; ++d) {
            O[row_q * D_vec + d] = o_acc[d] / l;
        }
    }
}
