# TurboQuant KV Cache Quantization — Implementation Plan for llama.cpp

## Background & Goal

This plan implements TurboQuant KV cache quantization in a fork of llama.cpp. TurboQuant is a
two-stage algorithm from Google Research (ICLR 2026) that compresses the KV cache to 3 bits with
no accuracy loss and no training required.

**Papers:**

- TurboQuant: https://arxiv.org/abs/2504.19874
- PolarQuant: https://arxiv.org/abs/2502.02617
- QJL: https://arxiv.org/abs/2406.03482

**Reference implementation (QJL, Python/CUDA):** https://github.com/amirzandieh/QJL

**Existing community branch (builds/quantizes but incomplete):** https://github.com/mudler/llama.cpp/tree/feat/turbo-quant

The two algorithms are:

1. **PolarQuant** — the main compression stage. Takes K/V vectors, applies a random rotation
   (preconditioning), converts pairs of coordinates into polar form (radius + angle) recursively,
   then quantizes only the angles to N bits. The single surviving radius from each recursive step
   is stored separately. Because the post-rotation angle distribution is analytically predictable
   (tightly concentrated), no per-block normalization constants (zero-point, scale) need to be
   stored — eliminating the 1–2 bit overhead of traditional methods.
1. **QJL** — a 1-bit residual error correction stage. Applies a Johnson-Lindenstrauss random
   rotation to the quantization error left over from PolarQuant, then stores just the sign bit of
   each result. At attention time, an asymmetric estimator uses the full-precision query against
   these sign bits to produce an unbiased correction to the attention score.

Combined as TurboQuant: most bits go to PolarQuant angles; 1 residual bit per dimension goes to
QJL. Net result: ~3-bit KV cache with lossless attention quality.

-----

## Scope of This Implementation

Implement in three phases, each independently testable:

- **Phase 1:** PolarQuant-only (CPU, ~4-bit mode) — the correctness foundation
- **Phase 2:** QJL residual stage (CPU) — full TurboQuant at ~3 bits
- **Phase 3:** CUDA kernels for fused quantized attention — the performance win

A CPU-only Phase 1+2 still delivers the memory savings (the main benefit for consumer hardware).
Phase 3 is where the speedup benchmarked in the paper comes from.

-----

## Repository Setup

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-turboquant
cd llama.cpp-turboquant
git checkout -b feat/turboquant
```

Also clone the QJL reference repo for algorithm reference:

```bash
git clone https://github.com/amirzandieh/QJL.git ../QJL-reference
```

-----

## Phase 1: PolarQuant (CPU, correctness)

### 1.1 — Add new GGML types

**File: `ggml/include/ggml.h`**

Find the `ggml_type` enum. After the existing `GGML_TYPE_IQ4_XS` or similar entries, add:

```c
GGML_TYPE_POLARQUANT_4 = <next_available_id>,  // 4-bit PolarQuant angles, KV cache use only
GGML_TYPE_TURBOQUANT_3 = <next_available_id>,  // 3-bit TurboQuant (PolarQuant + QJL residual)
GGML_TYPE_COUNT,
```

**File: `ggml/src/ggml.c`**

In the `type_traits` array, add entries for the new types. Follow the pattern of `GGML_TYPE_Q4_0`.
Set `blck_size` to 64 (process 64 dimensions per block — matches QJL reference impl), and compute
`type_size` based on the storage layout described in 1.2.

### 1.2 — Define the quantized block layout

**File: `ggml/src/ggml-quants.h`**

Add the block structs. The PolarQuant block stores:

```c
// PolarQuant 4-bit block: 64 dimensions -> 32 angle pairs, each 4 bits
// Plus the surviving radius from the recursive polar transform
#define POLAR_BLOCK_SIZE 64

typedef struct {
    ggml_half radius;          // Final radius after recursive polar decomposition (fp16)
    uint8_t   angles[POLAR_BLOCK_SIZE / 2]; // Packed 4-bit quantized angles (32 bytes)
} block_polarquant4;
static_assert(sizeof(block_polarquant4) == 2 + 32, "wrong block_polarquant4 size");

// TurboQuant 3-bit block: PolarQuant angles at ~2.5 bits + QJL 1-bit residual
typedef struct {
    ggml_half  radius;                      // Final radius (fp16)
    uint8_t    angles[POLAR_BLOCK_SIZE / 4 * 3 / 2]; // ~2.5-bit packed angles (adjust to fit)
    uint8_t    qjl_signs[POLAR_BLOCK_SIZE / 8]; // 1 sign bit per dimension = 8 bytes
} block_turboquant3;
```

NOTE: The exact bit-packing arithmetic should match the target bit budget. For a clean 3-bit total:
use 2-bit angles (24 bytes for 64 dims) + 1-bit QJL signs (8 bytes) + 2 bytes radius = 34 bytes
for 64 values = ~4.25 bits/value. Adjust the angle bit-width vs QJL tradeoff to hit 3.0 bits/value
exactly, following Table 1 in the PolarQuant paper for the optimal split.

### 1.3 — Implement the PolarQuant encode/decode functions

**File: `ggml/src/ggml-quants.c`**

#### Random rotation matrix

PolarQuant requires a fixed random orthogonal rotation matrix applied before quantization. Use a
structured Hadamard-based random rotation (same as the QJL reference impl uses) so it’s fast and
requires no storage:

```c
// Apply a seeded random Hadamard rotation to vector x of length d (must be power of 2)
// Uses the Fast Walsh-Hadamard Transform (FWHT) + random sign flips
static void polar_random_rotation(float * x, int d, uint64_t seed) {
    // 1. Apply random diagonal sign flips (D matrix)
    // Use a simple LCG or xorshift seeded with 'seed' to generate signs
    uint64_t rng = seed;
    for (int i = 0; i < d; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; // xorshift64
        x[i] *= (rng & 1) ? 1.0f : -1.0f;
    }
    // 2. Apply FWHT (in-place, unnormalized)
    for (int len = 1; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i+j], v = x[i+j+len];
                x[i+j] = u + v;
                x[i+j+len] = u - v;
            }
        }
    }
    // 3. Normalize
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= scale;
}
```

Use a fixed global seed (e.g., `0xDEADBEEF1234ULL`) so the same rotation is applied consistently
at encode and decode time with no per-vector storage.

#### Recursive polar transform (encode)

```c
// Recursively convert a Cartesian vector to polar representation.
// Output: angles[] has (d-1) angles; final_radius is the single surviving magnitude.
// Based on Algorithm 1 in the PolarQuant paper (arxiv:2502.02617).
static float polar_encode_recursive(const float * x, float * angles, int d) {
    if (d == 1) return fabsf(x[0]);
    if (d == 2) {
        float r = sqrtf(x[0]*x[0] + x[1]*x[1]);
        angles[0] = atan2f(x[1], x[0]);  // angle in [-pi, pi]
        return r;
    }
    // Split: process even-indexed pairs first, collect radii
    float radii[d/2];
    float sub_angles[d/2];
    for (int i = 0; i < d/2; i++) {
        float r = sqrtf(x[2*i]*x[2*i] + x[2*i+1]*x[2*i+1]);
        sub_angles[i] = atan2f(x[2*i+1], x[2*i]);
        angles[i] = sub_angles[i];   // store this level's angles
        radii[i] = r;
    }
    // Recurse on the radii vector to get the next level
    return polar_encode_recursive(radii, angles + d/2, d/2);
}
```

#### Angle quantization

After rotation, the PolarQuant insight is that the angle distribution post-rotation is
approximately uniform on [-π, π] and concentrated near 0 for the inner angles. Quantize uniformly:

```c
static uint8_t quantize_angle_4bit(float angle) {
    // Map [-pi, pi] -> [0, 15]
    float normalized = (angle + M_PI) / (2.0f * M_PI);  // [0, 1]
    int q = (int)(normalized * 16.0f);
    return (uint8_t)GGML_CLAMP(q, 0, 15);
}

static float dequantize_angle_4bit(uint8_t q) {
    return ((float)q / 16.0f) * 2.0f * M_PI - M_PI;
}
```

#### Full quantize function

```c
void quantize_row_polarquant4(const float * GGML_RESTRICT x,
                               void * GGML_RESTRICT vy, int64_t k) {
    assert(k % POLAR_BLOCK_SIZE == 0);
    block_polarquant4 * y = (block_polarquant4 *)vy;
    float rotated[POLAR_BLOCK_SIZE];
    float angles[POLAR_BLOCK_SIZE - 1];

    for (int i = 0; i < k / POLAR_BLOCK_SIZE; i++) {
        const float * src = x + i * POLAR_BLOCK_SIZE;
        memcpy(rotated, src, POLAR_BLOCK_SIZE * sizeof(float));

        // 1. Apply random rotation
        polar_random_rotation(rotated, POLAR_BLOCK_SIZE, 0xDEADBEEF1234ULL);

        // 2. Recursive polar transform
        float radius = polar_encode_recursive(rotated, angles, POLAR_BLOCK_SIZE);

        // 3. Store radius as fp16
        y[i].radius = GGML_FP32_TO_FP16(radius);

        // 4. Pack 4-bit quantized angles
        for (int j = 0; j < POLAR_BLOCK_SIZE / 2; j++) {
            uint8_t a0 = quantize_angle_4bit(angles[2*j]);
            uint8_t a1 = (j*2+1 < POLAR_BLOCK_SIZE - 1)
                         ? quantize_angle_4bit(angles[2*j+1]) : 0;
            y[i].angles[j] = (a1 << 4) | a0;
        }
    }
}
```

#### Dequantize function

Implement `dequantize_row_polarquant4` as the inverse: unpack angles, reconstruct Cartesian vector
from polar coordinates working recursively in reverse, apply the *inverse* random rotation
(transpose of the Hadamard rotation — for Hadamard this is itself, just re-apply), and scale by
the stored radius.

### 1.4 — Register the types with the GGML dispatch table

**File: `ggml/src/ggml.c`** (in `ggml_type_traits` initializer):

```c
[GGML_TYPE_POLARQUANT_4] = {
    .type_name         = "pq4",
    .blck_size         = POLAR_BLOCK_SIZE,
    .type_size         = sizeof(block_polarquant4),
    .is_quantized      = true,
    .to_float          = (ggml_to_float_t)dequantize_row_polarquant4,
    .from_float        = quantize_row_polarquant4,
    .from_float_ref    = quantize_row_polarquant4,
    .vec_dot           = ggml_vec_dot_polarquant4_q8_0,  // see 1.5
    .vec_dot_type      = GGML_TYPE_Q8_0,
},
```

### 1.5 — Implement a scalar vec_dot for correctness testing

For Phase 1, implement a simple dequantize-then-dot product. This is not fast but will be
correct. Real performance comes in Phase 3.

```c
void ggml_vec_dot_polarquant4_q8_0(int n, float * GGML_RESTRICT s,
                                    size_t bs, const void * GGML_RESTRICT vx,
                                    size_t bx, const void * GGML_RESTRICT vy,
                                    size_t by, int nrc) {
    // Dequantize both sides, do scalar dot product
    float tmp_x[n], tmp_y[n];
    dequantize_row_polarquant4(vx, tmp_x, n);
    dequantize_row_q8_0(vy, tmp_y, n);
    float dot = 0.0f;
    for (int i = 0; i < n; i++) dot += tmp_x[i] * tmp_y[i];
    *s = dot;
}
```

### 1.6 — Wire into llama.cpp KV cache

**File: `src/llama.cpp`** — function `llama_kv_cache_init`

Find where KV cache tensors are allocated. Add `pq4` and `tq3` as valid `cache_type_k` /
`cache_type_v` options alongside existing `q8_0`, `q4_0` etc:

```cpp
// In the cache_type validation / selection block:
if (cparams.cache_type_k == GGML_TYPE_POLARQUANT_4 ||
    cparams.cache_type_k == GGML_TYPE_TURBOQUANT_3) {
    // PolarQuant and TurboQuant are KV-cache-only types.
    // They do not support weight quantization — validate this is only
    // applied to the KV cache tensors.
}
```

**File: `src/arg.cpp`** or wherever `--cache-type-k` is parsed:

Add `pq4` and `tq3` as valid string values mapping to the new enum entries.

### 1.7 — Phase 1 test

Build and run a basic correctness test before proceeding:

```bash
cmake -B build -DGGML_CUDA=OFF
cmake --build build --config Release -j$(nproc)

# Run perplexity on a small model with PolarQuant KV cache
./build/bin/llama-perplexity \
    -m models/llama-3.2-1b-instruct.gguf \
    --cache-type-k pq4 \
    --cache-type-v pq4 \
    -f datasets/wikitext-2.txt \
    --chunks 10

# Compare against baseline (no KV quant)
./build/bin/llama-perplexity \
    -m models/llama-3.2-1b-instruct.gguf \
    -f datasets/wikitext-2.txt \
    --chunks 10
```

Expected: pq4 perplexity should be within ~0.5 of baseline. If it’s much worse, the angle
quantization range or the rotation implementation has a bug.

-----

## Phase 2: QJL Residual Stage (Full TurboQuant, CPU)

### 2.1 — QJL algorithm overview

QJL stores a 1-bit sign vector that captures the residual error from PolarQuant. At decode time,
this sign vector provides a bias-correction term added to the attention score.

The key is the **asymmetric estimator**: at attention time, the query Q is kept in full precision,
and the inner product `Q · K_quantized` is estimated as:

```
estimate = PolarQuant_reconstruct(K_q) · Q  +  QJL_correction(K_signs, Q)
```

where the QJL correction is:

```
correction = (pi/2) * (1/sqrt(m)) * sum_i( sign_i * (R_jl @ Q)_i )
```

Here `R_jl` is the same fixed random rotation, `m` is the dimension, and `sign_i` is the stored
sign bit. See Section 3 of the QJL paper for the full derivation.

### 2.2 — Extend the block struct

Update `block_turboquant3` (defined in 1.2) to actually include the QJL sign bits. The full layout:

```c
typedef struct {
    ggml_half  radius;                       // fp16, surviving radius from PolarQuant
    uint8_t    angles[POLAR_BLOCK_SIZE / 4]; // 2-bit angles: 64 dims * 2 bits / 8 = 16 bytes
    uint8_t    qjl_signs[POLAR_BLOCK_SIZE / 8]; // 1 bit per dim: 64 / 8 = 8 bytes
} block_turboquant3;
// Total: 2 + 16 + 8 = 26 bytes for 64 values = 3.25 bits/value
```

Adjust `angles` bit-width and/or block size to hit exactly 3 bits/value if needed.

### 2.3 — Implement TurboQuant encode

```c
void quantize_row_turboquant3(const float * GGML_RESTRICT x,
                               void * GGML_RESTRICT vy, int64_t k) {
    block_turboquant3 * y = (block_turboquant3 *)vy;
    float rotated[POLAR_BLOCK_SIZE];
    float angles[POLAR_BLOCK_SIZE - 1];
    float reconstructed[POLAR_BLOCK_SIZE];
    float residual[POLAR_BLOCK_SIZE];

    for (int i = 0; i < k / POLAR_BLOCK_SIZE; i++) {
        const float * src = x + i * POLAR_BLOCK_SIZE;
        memcpy(rotated, src, POLAR_BLOCK_SIZE * sizeof(float));

        // Stage 1: PolarQuant (same as before but 2-bit angles)
        polar_random_rotation(rotated, POLAR_BLOCK_SIZE, 0xDEADBEEF1234ULL);
        float radius = polar_encode_recursive(rotated, angles, POLAR_BLOCK_SIZE);
        y[i].radius = GGML_FP32_TO_FP16(radius);
        // Pack 2-bit angles (quantize to [0,3] range)
        for (int j = 0; j < POLAR_BLOCK_SIZE / 4; j++) {
            uint8_t packed = 0;
            for (int b = 0; b < 4; b++) {
                int idx = j*4 + b;
                float a = (idx < POLAR_BLOCK_SIZE - 1) ? angles[idx] : 0.0f;
                uint8_t qa = (uint8_t)(((a + M_PI) / (2.0f * M_PI)) * 4.0f);
                qa = GGML_CLAMP(qa, 0, 3);
                packed |= (qa << (b * 2));
            }
            y[i].angles[j] = packed;
        }

        // Stage 2: Compute PolarQuant residual
        // Reconstruct what PolarQuant would decode to
        dequantize_block_turboquant3_polar_only(&y[i], reconstructed);
        for (int j = 0; j < POLAR_BLOCK_SIZE; j++) {
            residual[j] = rotated[j] - reconstructed[j];
        }

        // Stage 3: Apply QJL to residual — store sign bits
        // Apply same random rotation to residual (reuse polar_random_rotation)
        polar_random_rotation(residual, POLAR_BLOCK_SIZE, 0xDEADBEEF1234ULL);
        for (int j = 0; j < POLAR_BLOCK_SIZE / 8; j++) {
            uint8_t signs = 0;
            for (int b = 0; b < 8; b++) {
                if (residual[j*8 + b] >= 0.0f) signs |= (1 << b);
            }
            y[i].qjl_signs[j] = signs;
        }
    }
}
```

### 2.4 — Implement the asymmetric attention estimator

For attention computation with TurboQuant keys, the decoder cannot simply dequantize K and
dot with Q. It must use the asymmetric estimator. This requires a custom attention path.

**File: `ggml/src/ggml-quants.c`**

```c
// Asymmetric TurboQuant dot product: full-precision Q dotted with TurboQuant K
// q: full precision query vector [d]
// k_block: TurboQuant encoded key block
// returns: estimated dot product
float turboquant3_asymmetric_dot(const float * q, const block_turboquant3 * k, int d) {
    float rotated_q[d];
    memcpy(rotated_q, q, d * sizeof(float));

    // 1. Rotate Q with the same fixed rotation used during K encoding
    polar_random_rotation(rotated_q, d, 0xDEADBEEF1234ULL);

    // 2. PolarQuant component: reconstruct K from PolarQuant angles, dot with rotated Q
    float k_reconstructed[d];
    dequantize_block_turboquant3_polar_only(k, k_reconstructed);
    float polar_dot = 0.0f;
    for (int i = 0; i < d; i++) polar_dot += rotated_q[i] * k_reconstructed[i];

    // 3. QJL correction: (pi/2) * (1/sqrt(d)) * sum_i(sign_i * rotated_q_i)
    float qjl_sum = 0.0f;
    for (int j = 0; j < d / 8; j++) {
        uint8_t signs = k->qjl_signs[j];
        for (int b = 0; b < 8; b++) {
            float sign = (signs & (1 << b)) ? 1.0f : -1.0f;
            qjl_sum += sign * rotated_q[j*8 + b];
        }
    }
    float qjl_correction = (float)(M_PI / 2.0) * (1.0f / sqrtf((float)d)) * qjl_sum;

    return polar_dot + qjl_correction;
}
```

### 2.5 — Hook asymmetric estimator into ggml_flash_attn_ext

This is the most invasive change. The flash attention operator in `ggml/src/ggml-cpu/ggml-cpu.c`
(and potentially `ggml/src/ggml-cuda/fattn.cu`) needs a code path that dispatches to
`turboquant3_asymmetric_dot` when the K cache tensor type is `GGML_TYPE_TURBOQUANT_3`.

Find the inner loop of `ggml_flash_attn_ext_f32` and add:

```c
if (k->type == GGML_TYPE_TURBOQUANT_3) {
    // Use asymmetric estimator instead of standard dot product
    const block_turboquant3 * k_block = (const block_turboquant3 *)
        ((const char *)k->data + ik * nb1);
    kq = turboquant3_asymmetric_dot(q_row, k_block, head_dim);
} else {
    // existing path
}
```

### 2.6 — Phase 2 test

```bash
cmake --build build --config Release -j$(nproc)

./build/bin/llama-perplexity \
    -m models/llama-3.2-1b-instruct.gguf \
    --cache-type-k tq3 \
    --cache-type-v pq4 \  # V cache uses PolarQuant (TurboQuant is for K only per paper)
    -f datasets/wikitext-2.txt \
    --chunks 10
```

Also run the needle-in-a-haystack test to validate long-context behaviour doesn’t degrade:

```bash
./build/bin/llama-cli \
    -m models/llama-3.2-1b-instruct.gguf \
    --cache-type-k tq3 \
    -c 8192 \
    -p "$(cat tests/needle_haystack_prompt.txt)"
```

-----

## Phase 3: CUDA Fused Attention Kernels (Performance)

This phase is optional but delivers the 8x attention speedup benchmarked in the paper.

### 3.1 — New CUDA kernel file

**File: `ggml/src/ggml-cuda/fattn-turboquant.cuh`**

Create a CUDA kernel that performs the full TurboQuant attention computation without dequantizing
to fp32. The key operations:

1. Load Q row into shared memory (fp16 or bf16)
1. Apply Hadamard transform to Q in shared memory (fast, in-place)
1. For each K block:
   a. Compute polar dot product using 2-bit angles and stored radius
   b. Compute QJL correction using XOR of sign bits with sign(rotated_Q)
   c. Sum to get attention logit
1. Softmax + weighted sum of V

The QJL step (3b) is particularly GPU-friendly: it reduces to a popcount over XOR of two bit
vectors, which maps directly to `__popc` in CUDA.

```cuda
__device__ float turboquant_kq_dot(
    const half * __restrict__ q_rot,   // pre-rotated Q in shared mem
    const uint8_t * angles,            // 2-bit packed angles
    const uint8_t * qjl_signs,        // 1-bit sign vector
    half radius,                       // stored radius
    int d)
{
    // 1. Reconstruct K from polar angles (dot product only, not full reconstruction)
    float polar_dot = polar_dot_from_angles(q_rot, angles, __half2float(radius), d);

    // 2. QJL correction via popcount
    // Load sign bits of rotated Q
    uint8_t q_signs[d/8];
    for (int i = 0; i < d/8; i++) {
        uint8_t qs = 0;
        for (int b = 0; b < 8; b++) {
            if (__half2float(q_rot[i*8+b]) >= 0.0f) qs |= (1 << b);
        }
        q_signs[i] = qs;
    }
    int popcount_sum = 0;
    for (int i = 0; i < d/8; i++) {
        popcount_sum += __popc((uint32_t)(q_signs[i] ^ qjl_signs[i]));
    }
    // Convert popcount to correction (negatives = d/2 - popcount)
    float agree = (float)(d - popcount_sum);
    float disagree = (float)popcount_sum;
    float qjl_correction = (float)(M_PI / 2.0) / sqrtf((float)d) * (agree - disagree);

    return polar_dot + qjl_correction;
}
```

### 3.2 — Dispatch in fattn.cu

**File: `ggml/src/ggml-cuda/fattn.cu`**

Add a new dispatch case in `ggml_cuda_flash_attn_ext`:

```cpp
if (k->type == GGML_TYPE_TURBOQUANT_3) {
    ggml_cuda_flash_attn_ext_turboquant(ctx, dst);
    return;
}
```

### 3.3 — CMakeLists update

**File: `ggml/src/ggml-cuda/CMakeLists.txt`**

Add:

```cmake
target_sources(ggml-cuda PRIVATE fattn-turboquant.cu)
```

### 3.4 — Phase 3 benchmark

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Benchmark tokens/sec: TurboQuant vs fp16 KV cache
./build/bin/llama-bench \
    -m models/llama-3.2-8b-instruct.gguf \
    --cache-type-k tq3 -pg 1,512 -pg 1,2048

./build/bin/llama-bench \
    -m models/llama-3.2-8b-instruct.gguf \
    -pg 1,512 -pg 1,2048
```

Expected: generation speed similar or faster than fp16 at long contexts. Memory usage should
be ~5x lower for the KV cache.

-----

## Files Modified / Created (Summary)

|File                                     |Change                                                        |
|-----------------------------------------|--------------------------------------------------------------|
|`ggml/include/ggml.h`                    |Add `GGML_TYPE_POLARQUANT_4`, `GGML_TYPE_TURBOQUANT_3` to enum|
|`ggml/src/ggml.c`                        |Register type traits for both new types                       |
|`ggml/src/ggml-quants.h`                 |Add `block_polarquant4`, `block_turboquant3` structs          |
|`ggml/src/ggml-quants.c`                 |Implement all encode/decode/dot functions                     |
|`ggml/src/ggml-cpu/ggml-cpu.c`           |Hook asymmetric estimator into `ggml_flash_attn_ext`          |
|`ggml/src/ggml-cuda/fattn.cu`            |Add TurboQuant dispatch                                       |
|`ggml/src/ggml-cuda/fattn-turboquant.cuh`|New CUDA kernel (Phase 3)                                     |
|`ggml/src/ggml-cuda/CMakeLists.txt`      |Add new CUDA source                                           |
|`src/llama.cpp`                          |Validate new cache types in `llama_kv_cache_init`             |
|`src/arg.cpp`                            |Add `pq4` and `tq3` as valid `--cache-type-k/v` values        |

-----

## Key Constants & Hyperparameters

|Parameter                |Value                                  |Source                                         |
|-------------------------|---------------------------------------|-----------------------------------------------|
|Block size               |64                                     |Matches QJL reference impl, power of 2 for FWHT|
|Random rotation seed     |Fixed (e.g., `0xDEADBEEF1234ULL`)      |Same seed used at encode + decode              |
|Angle bits (PolarQuant-4)|4                                      |Phase 1                                        |
|Angle bits (TurboQuant-3)|2                                      |Phase 2 — tune vs QJL budget                   |
|QJL bits                 |1 (sign only)                          |Per QJL paper                                  |
|Rotation type            |Randomized Hadamard (FWHT + sign flips)|Fast, no storage needed                        |
|QJL correction constant  |π/2 · 1/√d                             |Theorem 1 in QJL paper                         |

-----

## Recommended Build Order

1. Implement struct layouts and type registration (no logic yet) → build to check compilation
1. Implement `polar_random_rotation` → unit test in isolation with Python reference
1. Implement `polar_encode_recursive` + `quantize_row_polarquant4` + matching decode
1. Wire into KV cache, run perplexity test (Phase 1 complete)
1. Add QJL signs to block struct, implement `quantize_row_turboquant3`
1. Implement `turboquant3_asymmetric_dot`, hook into flash attention (Phase 2 complete)
1. CUDA kernel (Phase 3, optional)

-----

## Reference Implementations to Consult

The QJL reference repo at https://github.com/amirzandieh/QJL contains:

- `qjl_kernel/` — CUDA kernel implementing the JL transform and inner product estimator
  in C++/CUDA. Study `qjl_kernel.cu` for the GPU implementation of the sign-bit dot product.
- `models/` — modified LLaMA attention modules in Python showing the asymmetric estimator
  integration into transformer attention. This is the clearest reference for Phase 2 step 2.5.

The PolarQuant paper appendix (arxiv:2502.02617) contains pseudocode for Algorithm 1 (the
recursive polar transform) and the quantization grid for angles.

-----

## Notes for Claude Code

- Do not modify any existing quantization types. Only add new ones.
- The random rotation seed must be identical at quantize and dequantize time. Do not derive
  it from the tensor data — use a compile-time constant.
- The FWHT requires input length to be a power of 2. POLAR_BLOCK_SIZE=64 satisfies this.
- The V cache can use PolarQuant (not TurboQuant). TurboQuant’s asymmetric estimator is
  specific to the Q·K dot product. V vectors are accessed differently (weighted sum, not dot).
- Test with small models (1B–3B) first. Perplexity delta should be < 0.5 at 3-bit total.
- If perplexity is bad, the most likely causes are: wrong rotation inverse, angle range
  mismatch (angles[d-1] never set), or missing normalization in the recursive step.