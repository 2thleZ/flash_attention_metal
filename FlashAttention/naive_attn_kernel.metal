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

    // Compute scores: s = q * k^t
    // Online softmax handles O(N) storage avoidance
    
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // accumulating V weighted by attention scores
    float acc[64];
    for (int d=0; d<64; ++d) acc[d] = 0.0f;

    // pass 1: finding max score for numerical stability
    for (int j = 0; j < N; ++j) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[id * D + d] * K[j * D + d];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // pass 2: computing exponentials and accumulating weighted sum
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
