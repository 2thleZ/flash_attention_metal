#include <metal_stdlib>
using namespace metal;

// FlashAttention Backward Kernel
// Inputs: Q, K, V, O, dO, L
// Outputs: dQ, dK, dV (accumulated via atomics)
// Precision: half computation, float accumulation for gradients

// Helper to atomic add float
inline void atomic_add_float(device atomic_float* addr, float val) {
    // Try built-in fetch_add if available in MSL 3.0+
    atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
    
    /* 
    // Fallback CAS loop
    float expected = atomic_load_explicit(addr, memory_order_relaxed);
    float desired;
    do {
        desired = expected + val;
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, desired, memory_order_relaxed, memory_order_relaxed));
    */
}

kernel void flash_attention_backward_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const half* O [[buffer(3)]],
    device const half* dO [[buffer(4)]],
    device const float* L [[buffer(5)]],
    device atomic_float* dQ [[buffer(6)]], // Output gradients (float for atomics)
    device atomic_float* dK [[buffer(7)]],
    device atomic_float* dV [[buffer(8)]],
    constant int& N [[buffer(9)]],
    constant int& D [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant int& batch_stride [[buffer(12)]],
    constant int& head_stride [[buffer(13)]],
    constant bool& is_causal [[buffer(14)]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    // simplified backward
    // recompute attention matrix p_ij and backprop
    // atomics for dk, dv
    
    const int Br = 16; // Block Q
    const int Bc = 16; // Block K
    // D = 64
    
    uint batch_offset = bid.z * batch_stride + bid.y * head_stride;
    uint l_base_idx = batch_offset / D; // L is [B, H, N]
    
    // Offsets
    device const half* Q_curr = Q + batch_offset;
    device const half* K_curr = K + batch_offset;
    device const half* V_curr = V + batch_offset;
    device const half* dO_curr = dO + batch_offset;
    device const float* L_curr = L + l_base_idx;
                                
    device atomic_float* dQ_curr = dQ + batch_offset;
    device atomic_float* dK_curr = dK + batch_offset;
    device atomic_float* dV_curr = dV + batch_offset;
    
    uint g_row = bid.x * Br;
    uint lane = simd_lane_id;
    
    // Shared Memory
    threadgroup half Q_shared[16*64];
    threadgroup half K_trans_shared[64*16];
    threadgroup half V_trans_shared[64*16]; 
    
    threadgroup half dO_shared[16*64];
    threadgroup float DS_shared[16*16]; 
    threadgroup half p_store[16*16]; // Dedicated buffer for S store
    
    // Gradient Accumulation Buffer (Float)
    // 16x64 floats = 4KB.
    threadgroup float grad_shared[16*64];
    
    // load q and do
    for (int i = 0; i < 32; ++i) {
        int idx = lane + i * 32;
        if (idx < 16 * 64) {
            int r = idx / 64; int c = idx % 64;
            if (g_row + r < (uint)N) Q_shared[idx] = Q_curr[(g_row + r) * 64 + c];
            else Q_shared[idx] = 0.0h;
        }
    }
    
    for (int i = 0; i < 32; ++i) {
        int idx = lane + i * 32;
        if (idx < 16 * 64) {
             int r = idx / 64; int c = idx % 64;
             if (g_row + r < (uint)N) dO_shared[idx] = dO_curr[(g_row + r) * 64 + c];
             else dO_shared[idx] = 0.0h;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // compute di = sum(do_row * o_row)
    float Di = 0.0f;
    if (lane < 16 && g_row + lane < (uint)N) {
         for(int d=0; d<64; ++d) {
             half val_o = O[(batch_offset + (g_row + lane)*64 + d)]; 
             half val_do = dO_shared[lane*64 + d];
             Di += (float)(val_o * val_do);
         }
    }
    
    // load l_i
    float Li = 0.0f;
    if (lane < 16 && g_row + lane < (uint)N) {
        Li = L_curr[g_row + lane];
    }
    
    // regs
    simdgroup_half8x8 q_regs[2][8];
    for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) 
        simdgroup_load(q_regs[r][c], Q_shared, 64, ulong2(c*8, r*8));
        
    simdgroup_half8x8 do_regs[2][8];
    for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) 
        simdgroup_load(do_regs[r][c], dO_shared, 64, ulong2(c*8, r*8));

    // accumulator for dq
    simdgroup_float8x8 dq_acc[2][8];
    for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) dq_acc[r][c] = simdgroup_float8x8(0.0f);

    // loop over k/v blocks
    int num_blocks = (N + Bc - 1) / Bc;
    int start_block = bid.x % num_blocks; // stagger start to avoid contention
    
    for (int jj = 0; jj < num_blocks; ++jj) {
        int j = (start_block + jj) % num_blocks;
        uint g_col = j * Bc;
        
        if (is_causal && g_col > g_row + Br - 1) continue;
        
        // load k (trans) and v (trans)
        for (int i = 0; i < 32; ++i) {
            int idx = lane + i * 32;
            if (idx < 16 * 64) {
                 int r = idx / 64; int c = idx % 64; 
                 // K Load (Transposed)
                 if (g_col + r < (uint)N) {
                     K_trans_shared[c * 16 + r] = K_curr[(g_col + r)*64 + c];
                 } else {
                     K_trans_shared[c * 16 + r] = 0.0h;
                 }
                 
                 // V load (Transposed)
                 if (g_col + r < (uint)N) {
                     V_trans_shared[c * 16 + r] = V_curr[(g_col + r)*64 + c];
                 } else {
                     V_trans_shared[c * 16 + r] = 0.0h;
                 }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // recompute s = q * k^t
        simdgroup_half8x8 s_avg[2][2]; 
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c) s_avg[r][c] = simdgroup_half8x8((half)0.0h);
        
        for(int k=0; k<8; ++k) {
            simdgroup_half8x8 k_tile;
            for(int c=0; c<2; ++c) {
                simdgroup_load(k_tile, K_trans_shared, 16, ulong2(c*8, k*8));
                for(int r=0; r<2; ++r) {
                    simdgroup_multiply_accumulate(s_avg[r][c], q_regs[r][k], k_tile, s_avg[r][c]);
                }
            }
        }
        
        // compute p = exp(s - li)
        // Store s_avg (half) to p_store (half buffer). Safe.
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
            simdgroup_store(s_avg[r][c], p_store, 16, ulong2(c*8, r*8));
            
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // causal masking
        if (is_causal) {
            for (int i=0; i<32; ++i) {
                int idx = lane + i*32; 
                if (idx < 16*16) {
                    int r = idx / 16; int c = idx % 16; 
                    uint global_r = g_row + r; 
                    uint global_c = g_col + c; 
                    
                    if (global_c > global_r) {
                        p_store[idx] = (half)-INFINITY;
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // applying exponential scale to scores
        if (lane < 16) {
             for(int c=0; c<16; ++c) {
                 half s_val_h = p_store[lane*16 + c];
                 float val = (float)s_val_h * scale;
                 float p_val = exp(val - Li); 
                 DS_shared[lane*16 + c] = p_val; // Write float
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // converting back to half precision in-place
        if (lane < 16) {
             for(int c=0; c<16; ++c) {
                  float f_val = DS_shared[lane*16 + c];
                  ((threadgroup half*)DS_shared)[lane*16 + c] = (half)f_val;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // computing dv += p^t * do
        simdgroup_float8x8 dv_accum[2][8]; 
        for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) dv_accum[r][c] = simdgroup_float8x8(0.0f);
        
        // transposing p matrix
        if (lane < 16) {
             threadgroup half* p_ptr = (threadgroup half*)DS_shared;
             for (int c=lane+1; c<16; ++c) {
                 half temp = p_ptr[lane*16 + c];
                 p_ptr[lane*16 + c] = p_ptr[c*16 + lane];
                 p_ptr[c*16 + lane] = temp;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        simdgroup_half8x8 pt_regs[2][2];
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
             simdgroup_load(pt_regs[r][c], (threadgroup half*)DS_shared, 16, ulong2(c*8, r*8));
             
        for(int k=0; k<2; ++k) { 
             for(int c=0; c<8; ++c) {
                 for(int r=0; r<2; ++r) {
                      simdgroup_multiply_accumulate(dv_accum[r][c], pt_regs[r][k], do_regs[k][c], dv_accum[r][c]);
                 }
             }
        }
        
        // computing dp = do * v^t
        simdgroup_half8x8 dp_regs[2][2];
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c) dp_regs[r][c] = simdgroup_half8x8((half)0.0h);
        
        for(int k=0; k<8; ++k) { 
             simdgroup_half8x8 vt_tile;
             for(int c=0; c<2; ++c) {
                 simdgroup_load(vt_tile, V_trans_shared, 16, ulong2(c*8, k*8)); // V^T
                 for(int r=0; r<2; ++r) {
                      simdgroup_multiply_accumulate(dp_regs[r][c], do_regs[r][k], vt_tile, dp_regs[r][c]);
                 }
             }
        }
        
        // computing ds = p * (dp - di)
        // transposing p back to original orientation
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16) {
             threadgroup half* p_ptr = (threadgroup half*)DS_shared;
             for (int c=lane+1; c<16; ++c) {
                 half temp = p_ptr[lane*16 + c];
                 p_ptr[lane*16 + c] = p_ptr[c*16 + lane];
                 p_ptr[c*16 + lane] = temp;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
             simdgroup_store(dp_regs[r][c], V_trans_shared, 16, ulong2(c*8, r*8)); // dP stored
             
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lane < 16) {
             threadgroup half* p_ptr = (threadgroup half*)DS_shared;
             for(int c=0; c<16; ++c) {
                 float p_val = (float)p_ptr[lane*16 + c]; 
                 float dp_val = (float)V_trans_shared[lane*16 + c]; 
                 float ds_val = p_val * (dp_val - Di) * scale;
                 
                 V_trans_shared[lane*16 + c] = (half)ds_val; 
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        simdgroup_half8x8 ds_regs[2][2];
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
             simdgroup_load(ds_regs[r][c], V_trans_shared, 16, ulong2(c*8, r*8));
             
        // computing dq += ds * k
        for(int k=0; k<2; ++k) { 
             simdgroup_half8x8 k_reg_tiles[8];
             for(int c=0; c<8; ++c) {
                  simdgroup_load(k_reg_tiles[c], K_curr + g_col*64, 64, ulong2(c*8, k*8));
             }
             for(int r=0; r<2; ++r) {
                  for(int c=0; c<8; ++c) {
                      simdgroup_multiply_accumulate(dq_acc[r][c], ds_regs[r][k], k_reg_tiles[c], dq_acc[r][c]);
                  }
             }
        }
        
        // computing dk += ds^t * q
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16) {
             for (int c=lane+1; c<16; ++c) {
                 half temp = V_trans_shared[lane*16 + c];
                 V_trans_shared[lane*16 + c] = V_trans_shared[c*16 + lane];
                 V_trans_shared[c*16 + lane] = temp;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // dS^T is in V_trans_shared
        
        simdgroup_half8x8 dst_regs[2][2];
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
             simdgroup_load(dst_regs[r][c], V_trans_shared, 16, ulong2(c*8, r*8));
             
        simdgroup_float8x8 dk_acc[2][8];
        for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) dk_acc[r][c] = simdgroup_float8x8(0.0f);
        
        for(int k=0; k<2; ++k) {
             for(int c=0; c<8; ++c) {
                  for(int r=0; r<2; ++r) {
                      simdgroup_multiply_accumulate(dk_acc[r][c], dst_regs[r][k], q_regs[k][c], dk_acc[r][c]);
                  }
             }
        }
        
        // accumulation of gradients via atomics
        for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) 
            simdgroup_store(dk_acc[r][c], grad_shared, 64, ulong2(c*8, r*8));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        for(int i=0; i<32; ++i) { 
             int idx = lane + i*32;
             if(idx < 16*64) {
                 int r = idx / 64; int c = idx % 64;
                 if(g_col + r < (uint)N) {
                      float val = grad_shared[idx]; 
                      atomic_add_float(&dK_curr[(g_col + r)*64 + c], val);
                 }
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) 
            simdgroup_store(dv_accum[r][c], grad_shared, 64, ulong2(c*8, r*8));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        for(int i=0; i<32; ++i) { 
             int idx = lane + i*32;
             if(idx < 16*64) {
                 int r = idx / 64; int c = idx % 64;
                 if(g_col + r < (uint)N) {
                      float val = grad_shared[idx]; 
                      atomic_add_float(&dV_curr[(g_col + r)*64 + c], val);
                 }
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    } 
    
    // storing dq via atomic store
    for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) 
        simdgroup_store(dq_acc[r][c], grad_shared, 64, ulong2(c*8, r*8));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    for(int i=0; i<32; ++i) {
         int idx = lane + i*32;
         if(idx < 16*64) {
             int r = idx / 64; int c = idx % 64;
             if(g_row + r < (uint)N) {
                  float val = grad_shared[idx];
                  atomic_store_explicit(&dQ_curr[(g_row + r)*64 + c], val, memory_order_relaxed);
             }
         }
    }
}
