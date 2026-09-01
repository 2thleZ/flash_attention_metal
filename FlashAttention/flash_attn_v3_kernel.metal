#include <metal_stdlib>
using namespace metal;

// flash attention v3 (matrix intrinsics)
// br=16, bc=16, d=64
// 32 threads/group (simdgroup)

kernel void flash_attention_simd_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const int Br = 16;
    const int Bc = 16;
    // D = 64; (implicit)
    
    // shared mem
    threadgroup half Q_shared[16 * 64];
    threadgroup half K_trans_shared[64 * 16];
    threadgroup half V_shared[16 * 64];
    
    // Output accumulators: 16x64 result -> 2x8 tiles of 8x8
    simdgroup_half8x8 acc[2][8];
    for(int r=0; r<2; ++r)
        for(int c=0; c<8; ++c)
            acc[r][c] = simdgroup_half8x8(0.0h);
            
    float l = 0.0f;
    float m = -INFINITY;
    
    uint global_row = bid.x * Br; // global row offset
    uint lane = simd_lane_id; // thread_index_in_simdgroup [0, 31]

    // 1. loading Q into sharedmem
    for (int i = 0; i < 32; ++i) {
        int idx = lane + i * 32; // strided cooperative loading - each thread loads one of the 16*64 elements 32 times
        if (idx < 16 * 64) {
            int r = idx / 64;
            int c = idx % 64;
            if (global_row + r < (uint)N) {
                Q_shared[idx] = Q[(global_row + r) * 64 + c]; // Buffer #0 is half
            } else {
                Q_shared[idx] = 0.0h;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // load q into registers
    // 2x8 tiles
    simdgroup_half8x8 q_regs[2][8];
    for(int r=0; r<2; ++r) {
        for(int c=0; c<8; ++c) {
            // simdgroup_load(destination, source (shared or device mem), width_of_matrix, top_left_corner)
            // width_of_matrix: to read the next row of values from the source for the 8*8 tile
            // ulong(x, y): location of top-left corner of 8*8 tile
            simdgroup_load(q_regs[r][c], Q_shared, 64, ulong2(c*8, r*8));
        }
    }
    
    int num_blocks = (N + Bc - 1) / Bc;
    
    for (int j = 0; j < num_blocks; ++j) {
        uint g_col = j * Bc; // global starting index for the j-th block of K and V
        
        // loading K (transposed) and V into shared mem
        // 32 threads, 1024 elements, each thread handles 32 elements
        for (int i = 0; i < 32; ++i) {
            int idx = lane + i*32;
         
            int r = idx / 64;
            int c = idx % 64;

            // loads V
            if (g_col + r < (uint)N) {
                V_shared[idx] = V[(g_col + r) * 64 + c];
            } else {
                V_shared[idx] = 0.0h;
            }

            // loads K (transposed)
            if (g_col + r < (uint)N) {
                K_trans_shared[c * 16 + r] = K[(g_col + r) * 64 + c];
            } else {
                K_trans_shared[c * 16 + r] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // computing scores s = q * k^t
        // 16x16 result, 2x2 tiles
        
        simdgroup_half8x8 s_tiles[2][2];
        for(int r=0; r<2; ++r)
            for(int c=0; c<2; ++c)
                s_tiles[r][c] = simdgroup_half8x8(0.0h);
        
        // loop 'k' over dimension d=64
        for(int k=0; k<8; ++k) {
            simdgroup_half8x8 k_tile; // 8x8 slice of k^t
            
            // loop over output columns of S (2 tiles)
            for(int c=0; c<2; ++c) {
                // loads K^T tile, row start: k*8, col start: c*8.
                // stride is 16 (width of K_trans_shared)
                simdgroup_load(k_tile, K_trans_shared, 16, ulong2(c*8, k*8));
                
                // multiply with q tiles
                for(int r=0; r<2; ++r) {
                    simdgroup_multiply_accumulate(s_tiles[r][c], q_regs[r][k], k_tile, s_tiles[r][c]);
                }
            }
        }
        
        // storing S to shared for scalar reduction
        for(int r=0; r<2; ++r) {
            for(int c=0; c<2; ++c) {
                simdgroup_store(s_tiles[r][c], Q_shared, 16, ulong2(c*8, r*8));
            }
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // scalar softmax on q_shared (16x16)
        // first 16 threads handle one row each
        
        float m_block = -INFINITY;
        float l_block = 0.0f;
        
        if (lane < 16) {
            int row = lane;
            // find row max
            float row_max = -INFINITY;
            for(int c=0; c<16; ++c) {
                 // Q_shared is half. row_max is float.
                 float val = (float)Q_shared[row * 16 + c] * scale;
                 if (val > row_max) row_max = val;
                 Q_shared[row * 16 + c] = (half)val;
            }
            
            // exponentials & their sum over the row
            float row_sum = 0.0f;
            for(int c=0; c<16; ++c) {
                float val = exp((float)Q_shared[row * 16 + c] - row_max);
                Q_shared[row * 16 + c] = (half)val;
                row_sum += val;
            }
            
            m_block = row_max;
            l_block = row_sum;
        }
        
        // sync p
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // apply correction to acc
        
        // Spill-Scale-Reload - avoids 16 expensive MatMuls for scalar scaling
        
        // store Correction Factors
        // re-use available space in Q_shared [256..]
        
        float my_correction = 1.0f;
        if (lane < 16) {
             float m_prev = m;
             float m_new = max(m_prev, m_block);
             
             float corr_acc = exp(m_prev - m_new); // correction for global
             float corr_p = exp(m_block - m_new); // correction for current block
             
             m = m_new;
             l = l * corr_acc + l_block * corr_p;
             
             my_correction = corr_acc;
             
             // scaling P in shared memory
             for(int c=0; c<16; ++c) {
                Q_shared[lane*16 + c] *= (half)corr_p;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Store correction factors
        if (lane < 16) {
             Q_shared[256 + lane] = (half)my_correction;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // 2. Spill acc to K_trans_shared (16x64)
        for(int r=0; r<2; ++r) {
            for(int c=0; c<8; ++c) {
                 // storing to K_trans_shared (treated as 16x64 with stride 64)
                 simdgroup_store(acc[r][c], K_trans_shared, 64, ulong2(c*8, r*8));
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. scalar scale in place in shared_mem (threadgroup_mem)
        // 32 threads. 1024 elements. Each thread handles 32 elements.
        for(int i=0; i<32; ++i) {
             int idx = lane + i*32; // 0..1023
             int r = idx / 64; // Row index 0..15
             half scale = Q_shared[256 + r];
             K_trans_shared[idx] *= scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // 4. Reload Acc
        for(int r=0; r<2; ++r) {
            for(int c=0; c<8; ++c) {
                 simdgroup_load(acc[r][c], K_trans_shared, 64, ulong2(c*8, r*8));
            }
        }
        
        // 5. Accumulate P * V (Standard)
        // P is in Q_shared (16x16)
        // V is in V_shared (16x64).
        // Load P tiles
        simdgroup_half8x8 p_tiles[2][2];
        for(int r=0; r<2; ++r) {
            for(int c=0; c<2; ++c) {
                simdgroup_load(p_tiles[r][c], Q_shared, 16, ulong2(c*8, r*8));
            }
        }
            
        // Loop over inner dim D_p=16 (2 tiles)
        for(int k=0; k<2; ++k) {
             // Load V tiles slice k
             simdgroup_half8x8 v_slices[8]; // row k, cols 0..7
             for(int c=0; c<8; ++c) {
                 simdgroup_load(v_slices[c], V_shared, 64, ulong2(c*8, k*8));
             }
             
             // Multiply
             for(int r=0; r<2; ++r) {
                 for(int c=0; c<8; ++c) {
                      simdgroup_multiply_accumulate(acc[r][c], p_tiles[r][k], v_slices[c], acc[r][c]);
                 }
             }
        }
    }
    
    // final division by l
    // constructing diagonal inverse L matrix - diag(l)^(-1)
    for(int i=0; i<32; ++i) {
        int idx = lane + i*32;
        if (idx < 16*16) K_trans_shared[idx] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lane < 16) {
        K_trans_shared[lane * 16 + lane] = 1.0f / l; // diagonal indices for the 16*16 matrix
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // applying InvL * Acc -> Output
    
    simdgroup_half8x8 l_tiles[2][2];
    for(int r=0; r<2; ++r)
        for(int c=0; c<2; ++c)
            simdgroup_load(l_tiles[r][c], K_trans_shared, 16, ulong2(c*8, r*8));
       
    for(int c=0; c<8; ++c) { // for each column block
        for(int r=0; r<2; ++r) { // for each row block
             simdgroup_half8x8 result(0.0h);
             // Inv diag * acc_tile
             for(int k=0; k<2; ++k) {
                 simdgroup_multiply_accumulate(result, l_tiles[r][k], acc[k][c], result);
             }
             
             // result store
             // global row: global_row + r*8, global col: c*8
             if (global_row + r*8 < (uint)N) {
                 simdgroup_store(result, O, D, ulong2(c*8, global_row + r*8));
             }
        }
    }
}
