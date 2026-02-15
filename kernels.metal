#include <metal_stdlib>
using namespace metal;

// naive attention kernel
// o = softmax(q * k^t) * v
// q: [n, d]
// k: [n, d]
// v: [n, d]
// o: [n, d]
//
// grid: n threads (one per query row)
kernel void naive_attention_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= (uint)N) return;

    // Compute scores: s[id, :] = q[id, :] * k^t
    // Online softmax handles O(N) storage avoidance
    
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Weighted sum of V
    float acc[64];
    for (int d=0; d<64; ++d) acc[d] = 0.0f;

    // pass 1: find max for stability
    for (int j = 0; j < N; ++j) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[id * D + d] * K[j * D + d];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // pass 2: compute exp and weighted sum
    for (int j = 0; j < N; ++j) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[id * D + d] * K[j * D + d];
        }
        score *= scale;
        
        float p = exp(score - max_score);
        sum_exp += p;
        
        for (int d = 0; d < D; ++d) {
            acc[d] += p * V[j * D + d];
        }
    }

    // Write Output
    for (int d = 0; d < D; ++d) {
        O[id * D + d] = acc[d] / sum_exp;
    }
}

// flash attention kernel
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
    // Shared Memory
    threadgroup float Q_tile[32 * 64]; 
    threadgroup float K_tile[32 * 64]; 
    threadgroup float V_tile[32 * 64]; 
    
    // Accumulators
    float o_acc[64];
    for(int i=0; i<64; ++i) o_acc[i] = 0.0f;
    
    float l = 0.0f; // sum of exp
    float m = -INFINITY; // max score
    
    uint tx = tid.x; // local thread id
    uint bx = bid.x; // block index for q
    
    uint row_q = bx * Br + tx;
    
    // load q_tile
    // each thread loads one row of q
    if (row_q < (uint)N) {
        for (int d = 0; d < D; ++d) {
            Q_tile[tx * D + d] = Q[row_q * D + d];
        }
    } else {
        // padding for non-multiples
        for (int d = 0; d < D; ++d) Q_tile[tx * D + d] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // loop over k, v blocks
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
        
        // Compute Attention for this block
        // Each thread (query i) computes scores against all Bc keys
        
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
    
    // Write Output
    if (row_q < (uint)N) {
        for (int d = 0; d < D; ++d) {
            O[row_q * D + d] = o_acc[d] / l;
        }
    }
}

// flash attention v3 (matrix intrinsics)
// br=16, bc=16, d=64
// 32 threads/group (simdgroup)

kernel void flash_attention_simd_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const int Br = 16;
    const int Bc = 16;
    // const int D = 64; // Implicit 64
    
    // Shared Memory
    threadgroup float Q_shared[16 * 64];
    threadgroup float K_trans_shared[64 * 16]; // Transposed K: [D, Bc]
    threadgroup float V_shared[16 * 64];
    
    // Output Accumulators: 16x64 result -> 2x8 tiles of 8x8
    simdgroup_float8x8 acc[2][8]; 
    for(int r=0; r<2; ++r) 
        for(int c=0; c<8; ++c) 
            acc[r][c] = simdgroup_float8x8(0.0f);
            
    float l = 0.0f;
    float m = -INFINITY;
    
    uint g_row = bid.x * Br;
    uint lane = simd_lane_id;

    // 1. Load Q into Shared
    for (int i = 0; i < 32; ++i) {
        int idx = lane + i * 32; // 0..1023
        if (idx < 16 * 64) {
            int r = idx / 64;
            int c = idx % 64;
            if (g_row + r < (uint)N) {
                Q_shared[idx] = Q[(g_row + r) * 64 + c];
            } else {
                Q_shared[idx] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // load q into registers
    // 2x8 tiles
    simdgroup_float8x8 q_regs[2][8];
    for(int r=0; r<2; ++r) {
        for(int c=0; c<8; ++c) {
            simdgroup_load(q_regs[r][c], Q_shared, 64, ulong2(c*8, r*8));
        }
    }
    
    int num_blocks = (N + Bc - 1) / Bc;
    
    for (int j = 0; j < num_blocks; ++j) {
        uint g_col = j * Bc;
        
        // Load K (transposed) and V into Shared Memory
        // Vectorized load (128-bit)
        
        device const uint4* K_curr_vec = (device const uint4*)K; 
        device const uint4* V_curr_vec = (device const uint4*)V; 
        threadgroup uint4* V_shared_vec = (threadgroup uint4*)V_shared;
        
        // 4 vectors per thread
        for (int k = 0; k < 4; ++k) {
            uint vec_idx = lane + k * 32;
            
            uint r = vec_idx / 8; // 0..15
            uint c8 = vec_idx % 8; // 0..7
            
            uint4 val_k = uint4(0);
            uint4 val_v = uint4(0);
            
            if (g_col + r < (uint)N) {
                 uint global_row_idx = g_col + r;
                 uint global_vec_offset_k_v = global_row_idx * (D/8) + c8;

                 val_k = K_curr_vec[global_vec_offset_k_v];
                 val_v = V_curr_vec[global_vec_offset_k_v];
            }
            
            // store v
            V_shared_vec[vec_idx] = val_v;
            
            // store k (unpack)
            thread uint4* kp = &val_k;
            thread half* val_k_ptr = (thread half*)kp;
            
            for(int x=0; x<8; ++x) {
                int col = c8 * 8 + x;
                K_trans_shared[col * 16 + r] = val_k_ptr[x];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // compute scores s = q * k^t
        // 16x16 result, 2x2 tiles
        
        simdgroup_float8x8 s_tiles[2][2];
        for(int r=0; r<2; ++r) 
            for(int c=0; c<2; ++c) 
                s_tiles[r][c] = simdgroup_float8x8(0.0f);
        
        // loop 'k' over dimension d=64
        for(int k=0; k<8; ++k) {
            simdgroup_float8x8 k_tile; // 8x8 slice of k^t
            
            // Loop over output columns of S (2 tiles)
            for(int c=0; c<2; ++c) {
                // Load K^T tile. Row start: k*8. Col start: c*8.
                // Stride is 16 (width of K_trans_shared)
                simdgroup_load(k_tile, K_trans_shared, 16, ulong2(c*8, k*8));
                
                // multiply with q tiles
                for(int r=0; r<2; ++r) {
                    simdgroup_multiply_accumulate(s_tiles[r][c], q_regs[r][k], k_tile, s_tiles[r][c]);
                }
            }
        }
        
        // Online Softmax: Store S to shared for scalar reduction
        
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
        
        // only run on first 16 lanes
        if (lane < 16) {
            int row = lane;
            // find row max
            float row_max = -INFINITY;
            for(int c=0; c<16; ++c) {
                 float val = Q_shared[row * 16 + c] * scale;
                 if (val > row_max) row_max = val;
                 Q_shared[row * 16 + c] = val; // Store scaled
            }
            
            // Exponentials
            float row_sum = 0.0f;
            for(int c=0; c<16; ++c) {
                float val = exp(Q_shared[row * 16 + c] - row_max);
                Q_shared[row * 16 + c] = val; // Store P
                row_sum += val;
            }
            
            m_block = row_max;
            l_block = row_sum;
        }
        
        // sync p
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // apply correction to acc
        
        // IMPLEMENTATION: Spill-Scale-Reload Strategy
        // This avoids 16 expensive MatMuls for scalar scaling.
        
        // 1. Store Correction Factors
        // Use available space in Q_shared [256..]
        
        float my_correction = 1.0f;
        if (lane < 16) {
             float m_prev = m;
             float m_new = max(m_prev, m_block);
             
             float corr_acc = exp(m_prev - m_new);
             float corr_p = exp(m_block - m_new); 
             
             m = m_new;
             l = l * corr_acc + l_block * corr_p;
             
             my_correction = corr_acc;
             
             // Scale P in shared memory
             for(int c=0; c<16; ++c) {
                Q_shared[lane*16 + c] *= corr_p;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Store correction factors
        if (lane < 16) {
             Q_shared[256 + lane] = (half)my_correction;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // 2. Spill Acc to K_trans_shared (16x64)
        for(int r=0; r<2; ++r) {
            for(int c=0; c<8; ++c) {
                 // Store to K_trans_shared (treated as 16x64 with stride 64)
                 simdgroup_store(acc[r][c], K_trans_shared, 64, ulong2(c*8, r*8));
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. Scalar Scale in Place in Shared Memory
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
        // P is in Q_shared (16x16). (Correctly scaled).
        // V is in V_shared (16x64).
        // Load P tiles
        simdgroup_float8x8 p_tiles[2][2];
        for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
            simdgroup_load(p_tiles[r][c], Q_shared, 16, ulong2(c*8, r*8));
            
        // Loop over inner dim D_p=16 (2 tiles)
        for(int k=0; k<2; ++k) {
             // Load V tiles slice k
             simdgroup_float8x8 v_slices[8]; // row k, cols 0..7
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
    
    // Final Division by l
    // Construct Diagonal Inverse L matrix.
    for(int i=0; i<32; ++i) { 
        int idx = lane + i*32; 
        if (idx < 16*16) K_trans_shared[idx] = 0.0f; 
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lane < 16) {
        K_trans_shared[lane * 16 + lane] = 1.0f / l;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply InvL * Acc -> Output Store
    
    simdgroup_float8x8 l_tiles[2][2];
    for(int r=0; r<2; ++r) for(int c=0; c<2; ++c)
       simdgroup_load(l_tiles[r][c], K_trans_shared, 16, ulong2(c*8, r*8));
       
    for(int c=0; c<8; ++c) { // For each column block
        for(int r=0; r<2; ++r) { // For each row block
             simdgroup_float8x8 result(0.0f);
             // Multiply Diag * Tile
             for(int k=0; k<2; ++k) {
                 simdgroup_multiply_accumulate(result, l_tiles[r][k], acc[k][c], result);
             }
             
             // Store Result
             // Global Row: g_row + r*8. Global Col: c*8.
             // We need to handle bounds?
             // Assuming padded N/D for simplicity or unsafe store? 
             // Let's use robust store.
             if (g_row + r*8 < (uint)N) { 
                 simdgroup_store(result, O, D, ulong2(c*8, g_row + r*8));
             }
        }
    }
}

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
    
    // load q_tile
    if (row_q < (uint)N) {
        for (int d = 0; d < D_vec; ++d) {
            Q_tile[tx * D_vec + d] = Q[row_q * D_vec + d];
        }
    } else {
        for (int d = 0; d < D_vec; ++d) Q_tile[tx * D_vec + d] = float4(0.0f);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int num_blocks_k = (N + Bc_local - 1) / Bc_local;
    
    // preload first block
    if (num_blocks_k > 0) {
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ping-pong pointers
    threadgroup float4* K_curr = K_tile_A;
    threadgroup float4* V_curr = V_tile_A;
    threadgroup float4* K_next = K_tile_B;
    threadgroup float4* V_next = V_tile_B;
    
    for (int j = 0; j < num_blocks_k; ++j) {
        // prefetch next block
        if (j + 1 < num_blocks_k) {
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
        
        // compute on current
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
    constant bool& is_causal [[buffer(10)]], // [NEW] Causal flag
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    // Configuration
    const int Br = 16;
    const int Bc = 16;
    // D = 64.
    
    // Grid Mapping
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

    // Load Q (Vectorized)
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
        
        // load k (transposed) and v
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
        K_trans_shared[lane * 16 + lane] = (half)(1.0f / l);
        
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
        
        // Apply exp scale (Load half from p_store, calc float, store float to DS_shared)
        if (lane < 16) {
             for(int c=0; c<16; ++c) {
                 half s_val_h = p_store[lane*16 + c];
                 float val = (float)s_val_h * scale;
                 float p_val = exp(val - Li); 
                 DS_shared[lane*16 + c] = p_val; // Write float
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // Convert Float P -> Half P in place in DS_shared
        // We pack 16x16 floats into 16x16 halfs (first half of buffer).
        if (lane < 16) {
             for(int c=0; c<16; ++c) {
                  float f_val = DS_shared[lane*16 + c];
                  ((threadgroup half*)DS_shared)[lane*16 + c] = (half)f_val;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // compute dv += p^t * do
        simdgroup_float8x8 dv_accum[2][8]; 
        for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) dv_accum[r][c] = simdgroup_float8x8(0.0f);
        
        // transpose p
        // DS_shared is now Half matrix at the start.
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
        
        // compute dp = do * v^t
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
        
        // compute ds = p * (dp - di)
        // P needs to be Transposed Back.
        // P is in DS_shared (Half).
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
             
        // compute dq += ds * k
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
        
        // compute dk += ds^t * q
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16) {
             for (int c=lane+1; c<16; ++c) {
                 half temp = V_trans_shared[lane*16 + c];
                 V_trans_shared[lane*16 + c] = V_trans_shared[c*16 + lane];
                 V_trans_shared[c*16 + lane] = temp;
             }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // dS^T in V_trans_shared
        
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
        
        // atomic update dk, dv
        // Store FLOAT accumulators to grad_shared (FLOAT)
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
    
    // store dq
    for(int r=0; r<2; ++r) for(int c=0; c<8; ++c) 
        simdgroup_store(dq_acc[r][c], grad_shared, 64, ulong2(c*8, r*8)); // grad_shared (FLOAT)
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
