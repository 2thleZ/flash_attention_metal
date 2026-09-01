#include <metal_stdlib>
using namespace metal;

// flash attention v4 (half precision)
// uses simdgroup_half8x8 (tensor cores)

kernel void flash_attention_v4_half_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& batch_stride [[buffer(7)]],
    constant int& head_stride [[buffer(8)]],

    device float* L_out [[buffer(9)]], // [Batch, Heads, N]
    constant bool& is_causal [[buffer(10)]], // causal flag
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    // FA config
    const int Br = 16;
    const int Bc = 16;
    // D = 64
    
    // calc base indices for current batch and head
    uint batch_offset = bid.z * batch_stride + bid.y * head_stride;
    uint l_base_idx = batch_offset / D;

    // Pointers
    device const half* Q_curr = Q + batch_offset;
    device const half* K_curr = K + batch_offset;
    device const half* V_curr = V + batch_offset;
    device half* O_curr = O + batch_offset;
    device float* L_curr = L_out + l_base_idx;
    
    // shared memory
    threadgroup half Q_shared[16 * 64];
    threadgroup half K_trans_shared[64 * 16]; // Transposed K
    threadgroup half V_shared[16 * 64];
    
    // accumulators (2x8 tiles)
    simdgroup_half8x8 acc[2][8];
    for(int r=0; r<2; ++r)
        for(int c=0; c<8; ++c)
            acc[r][c] = simdgroup_half8x8((half)0.0h);
            
    // softmax stats
    float l = 0.0f; // sum of exp
    float m = -INFINITY; // max
    
    uint g_row = bid.x * Br;
    uint lane = simd_lane_id;

    // loading Q tile into shared memory (vectorized)
    device const uint4* Q_curr_vec = (device const uint4*)Q_curr;
    threadgroup uint4* Q_shared_vec = (threadgroup uint4*)Q_shared;
    
    for (int k = 0; k < 4; ++k) {
        uint vec_idx = lane + k * 32;
        uint r = vec_idx / 8;
        
        if (g_row + r < (uint)N) {
             Q_shared_vec[vec_idx] = Q_curr_vec[ (g_row * 8) + vec_idx ];
        } else {
             Q_shared_vec[vec_idx] = uint4(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // load q to regs
    simdgroup_half8x8 q_regs[2][8];
    for(int r=0; r<2; ++r) {
        for(int c=0; c<8; ++c) {
            simdgroup_load(q_regs[r][c], Q_shared, 64, ulong2(c*8, r*8));
        }
    }
    
    int num_blocks = (N + Bc - 1) / Bc;
    

    
    for (int j = 0; j < num_blocks; ++j) {
        uint g_col = j * Bc;

        // causal skip
        if (is_causal && g_col > g_row + Br - 1) continue;
        
        // loading k (transposed) and v
        for (int i = 0; i < 32; ++i) {
            int idx = lane + i * 32;
            if (idx < 16 * 64) {
                int r = idx / 64;
                int c = idx % 64;
                
                // v
                if (g_col + r < (uint)N) {
                    V_shared[idx] = V_curr[(g_col + r) * 64 + c];
                } else {
                    V_shared[idx] = 0.0h;
                }
                
                // k (trans)
                if (g_col + r < (uint)N) {
                   K_trans_shared[c * 16 + r] = K_curr[(g_col + r) * 64 + c];
                } else {
                   K_trans_shared[c * 16 + r] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // compute scores s = q * k^t
        simdgroup_half8x8 s_tiles[2][2];
        for(int r=0; r<2; ++r)
            for(int c=0; c<2; ++c)
                s_tiles[r][c] = simdgroup_half8x8((half)0.0h);
        
        // loop over d=64
        for(int k=0; k<8; ++k) {
            simdgroup_half8x8 k_tile;
            for(int c=0; c<2; ++c) {
                simdgroup_load(k_tile, K_trans_shared, 16, ulong2(c*8, k*8));
                for(int r=0; r<2; ++r) {
                    // Accumulate in standard order
                    // s = q * k + s
                    simdgroup_multiply_accumulate(s_tiles[r][c], q_regs[r][k], k_tile, s_tiles[r][c]);
                }
            }
        }
        
        // softmax update
        // checkpoint s_tiles to q_shared
        for(int r=0; r<2; ++r) {
            for(int c=0; c<2; ++c) {
                simdgroup_store(s_tiles[r][c], Q_shared, 16, ulong2(c*8, r*8));
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // causal masking
        if (is_causal) {
            // Parallelize masking with 32 threads.
            for(int i=0; i<32; ++i) {
                int idx = lane + i*32;
                if (idx < 16*16) {
                    int r = idx / 16;
                    int c = idx % 16;
                    
                    uint global_r = g_row + r;
                    uint global_c = g_col + c;
                    
                    if (global_c > global_r) {
                        Q_shared[idx] = (half)-INFINITY;
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float m_block = -INFINITY;
        float l_block = 0.0f;
        
        if (lane < 16) {
            int row = lane;
            float row_max = -INFINITY;
            for(int c=0; c<16; ++c) {
                 float val = (float)Q_shared[row * 16 + c] * scale;
                 if (val > row_max) row_max = val;
                 Q_shared[row * 16 + c] = (half)val;
            }
            
            float row_sum = 0.0f;
            for(int c=0; c<16; ++c) {
                // Compute Exp in Float
                float val = exp((float)Q_shared[row * 16 + c] - row_max);
                Q_shared[row * 16 + c] = (half)val;
                row_sum += val;
            }
            
            m_block = row_max;
            l_block = row_sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // correction logic
        half my_correction = (half)1.0h;
        if (lane < 16) {
             float m_prev = m;
             float m_new = max(m_prev, m_block);
             
             float corr_acc = exp(m_prev - m_new);
             float corr_p = exp(m_block - m_new);
             
             m = m_new;
             l = l * corr_acc + l_block * corr_p;
             
             my_correction = (half)corr_acc;
             
             // scale p in shared
             for(int c=0; c<16; ++c) {
                float v = (float)Q_shared[lane*16 + c];
                Q_shared[lane*16 + c] = (half)(v * corr_p);
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // diagonal correction matrix
        for(int i=0; i<32; ++i) {
            int idx = lane + i*32;
            if (idx < 16*16) K_trans_shared[idx] = 0.0h;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16) {
           K_trans_shared[lane * 16 + lane] = my_correction;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // apply correction to acc
        simdgroup_half8x8 acc_temp[2][8];
        for(int r=0; r<2; ++r)
            for(int c=0; c<8; ++c)
                acc_temp[r][c] = simdgroup_half8x8((half)0.0h);
        
        simdgroup_half8x8 corr_tiles[2][2];
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
            simdgroup_load(corr_tiles[r][c], K_trans_shared, 16, ulong2(c*8, r*8));
            
        for(int k=0; k<2; ++k) {
             for(int c=0; c<8; ++c) {
                  for(int r=0; r<2; ++r) {
                      simdgroup_multiply_accumulate(acc_temp[r][c], corr_tiles[r][k], acc[k][c], acc_temp[r][c]);
                  }
             }
        }
        for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) acc[r][c] = acc_temp[r][c];
        
        // accumulate p * v
        simdgroup_half8x8 p_tiles[2][2];
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
            simdgroup_load(p_tiles[r][c], Q_shared, 16, ulong2(c*8, r*8));
            
        for(int k=0; k<2; ++k) {
             simdgroup_half8x8 v_slices[8];
             for(int c=0; c<8; ++c) {
                 simdgroup_load(v_slices[c], V_shared, 64, ulong2(c*8, k*8));
             }
             for(int r=0; r<2; ++r) {
                 for(int c=0; c<8; ++c) {
                      simdgroup_multiply_accumulate(acc[r][c], p_tiles[r][k], v_slices[c], acc[r][c]);
                 }
             }
        }
    }
    
    // Final Division by l
    for(int i=0; i<32; ++i) {
        int idx = lane + i*32;
        if (idx < 16*16) K_trans_shared[idx] = 0.0h;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lane < 16) {
        K_trans_shared[lane * 16 + lane] = 1.0f / l;
        
        // store l = m + log(l)
        if (g_row + lane < (uint)N) {
             L_curr[g_row + lane] = m + log(l);
        }
    }
    // inv_l * acc -> output
    simdgroup_half8x8 l_tiles[2][2];
    for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
       simdgroup_load(l_tiles[r][c], K_trans_shared, 16, ulong2(c*8, r*8));
       
    for(int c=0; c<8; ++c) {
        for(int r=0; r<2; ++r) {
             simdgroup_half8x8 result((half)0.0h);
             for(int k=0; k<2; ++k) {
                 simdgroup_multiply_accumulate(result, l_tiles[r][k], acc[k][c], result);
             }
             
             if (g_row + r*8 < (uint)N) {
                 simdgroup_store(result, O_curr, D, ulong2(c*8, g_row + r*8));
             }
        }
    }
}
