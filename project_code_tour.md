# FlashAttention Project: The Code Tour

This guide walks you through your codebase (`kernels.metal` and `main.mm`) to help you explain *exactly* what you wrote. Open `kernels.metal` in a split pane while reading this.

---

## Part 1: The Baseline (Naive Attention)
**File:** `kernels.metal` (Lines 12 - 68)
**Function:** `naive_attention_kernel`

This is your control group. It represents the "standard" way a researcher might write attention, which is slow.

*   **The Grid:** One thread per query row (`grid: n threads`).
*   **The Loop (Line 39):** `for (int j = 0; j < N; ++j)`
    *   It loops over *every* key in the sequence.
*   **The Bottleneck (Line 42):** `Q[id * D + d] * K[j * D + d]`
    *   Every time this line runs, it fetches data from **Global Memory** (slow DRAM).
    *   For N=4096, it reads the entire K and V matrix 4096 times.
*   **Online Softmax (Line 29-57):**
    *   Even here, you used a smart trick. Instead of writing the $N \times N$ score matrix to memory, you compute `max_score` and `sum_exp` on the fly. This saved significant memory, even in the naive version.

---

## Part 2: The Speed Demon (V2 - Float4 Vectorization)
**File:** `kernels.metal` (Lines 572 - 702)
**Function:** `flash_attention_v2_kernel`

This is the code that got you the **8.5x speedup**. It relies on pure engineering efficiency, not fancy tensor cores.

### 1. Vector Types (Line 576)
```cpp
device float4* O [[buffer(3)]]
```
*   **Explanation:** You cast the pointers to `float4*`. The GPU now sees memory as chunks of 4 floats (128 bits). Every load/store instruction moves 4x the data.

### 2. Double Buffering (Lines 592-597)
```cpp
threadgroup float4 K_tile_A[16 * 16];
threadgroup float4 K_tile_B[16 * 16];
```
*   **Explanation:** You allocate **two** tiles in shared memory (Threadgroup Memory) for K and V.
*   **The "Ping-Pong" (Lines 640-694):**
    *   You use pointers `K_curr` and `K_next`.
    *   **In the loop:**
        *   Lines 647-660: You issue loads for the *next* block (`row_k_next`) into `K_next`.
        *   Lines 663-682: You compute math on the *current* block (`K_curr`).
    *   **Line 685 (`threadgroup_barrier`):** Critical. Ensures loading is done before we swap pointers.
    *   **Line 690:** You swap the pointers (`K_curr = K_next`).

### 3. Tiling (The "Flash" Mechanics)
*   **Line 609:** `Q_tile` is loaded only **once** per threadgroup.
*   **Line 645:** `for (int j = 0; j < num_blocks_k; ++j)`
    *   We stream K and V blocks from global memory into on-chip SRAM (`K_tile`).
    *   We reused the cached `Q_tile` against all these K blocks.
    *   **Result:** Q is read from global memory only once. K/V are read N/Bc times. Massive bandwidth saving.

---

## Part 3: The Pre-Silicon Expert Mode (V4 - Matrix Intrinsics)
**File:** `kernels.metal` (Lines 706 - 980)
**Function:** `flash_attention_v4_half_kernel`

This proves you can program proprietary hardware features (Apple AMX).

### 1. Matrix Intrinsics (Line 747)
```cpp
simdgroup_half8x8 acc[2][8];
```
*   **Explanation:** You aren't using `float`. You are using an opaque Metal object `simdgroup_half8x8`.
*   These live in special registers. You cannot access `acc[row][col]` directly!
*   **Line 939:** `simdgroup_multiply_accumulate(...)` -> This maps to a hardware instruction (like `FMA` on CPU or Tensor Core op on NVIDIA) that does an $8 \times 8$ matrix multiply in one go.

### 2. Spill-Scale-Reload (Lines 893-943)
*   User: "How did you handle softmax rescaling with opaque matrices?"
*   Code:
    *   **Line 843:** `simdgroup_store(...)`. You force the opaque tile out of registers into Shared Memory (`Q_shared`).
    *   **Line 910:** `Q_shared[...] *= corr_p`. Now it's just a normal array. You use scalar math to multiply every element by the correction factor.
    *   **Line 948:** `simdgroup_load(...)`. You load the fixed values back into the matrix registers to continue specifically the P*V step.

### 3. Causal Masking (Lines 794 & 849)
*   **Block Skip (Line 794):** `if (is_causal && g_col > g_row + Br - 1) continue;`
    *   If a whole block is in the "future", don't even load it.
*   **Fine-Grained Mask (Lines 849-866):**
    *   For the diagonal block, we have to mask specific elements.
    *   You calculated `global_c > global_r` and set the score to `-INFINITY` (Line 861).

---

## Part 4: The Backward Pass (Atomics)
**File:** `kernels.metal` (Lines 1002 - 1367)
**Function:** `flash_attention_backward_kernel`

### 1. The Strategy
*   Forward pass is easy: 1 Query Row -> 1 Output Row.
*   Backward pass is hard: 1 Query Row -> Impacts *many* K and V rows. Access pattern is transposed.

### 2. Atomic Accumulation (Line 1328)
```cpp
atomic_add_float(&dK_curr[(g_col + r)*64 + c], val);
```
*   **Why:** Multiple threadgroups might try to update the gradient for the same Key `dK[5]` at the same time.
*   **How:** You used `atomic_fetch_add_explicit`. This locks the memory address for a nanosecond to add the value safely.
*   **Smart Move:** You accumulate locally in `grad_shared` (Line 1319) first, then only do atomics to global memory at the very end of the block. This reduces memory contention by 1000x.
