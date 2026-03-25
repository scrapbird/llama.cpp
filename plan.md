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

**File: `ggml/src/ggml-common.h`** (where all block structs like `block_q4_0` are defined)

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

Use fixed global seeds so the same rotation is applied consistently at encode and decode time
with no per-vector storage. PolarQuant and QJL must use DIFFERENT seeds:

```c
#define POLARQUANT_ROTATION_SEED 0xDEADBEEF1234ULL
#define QJL_ROTATION_SEED        0xCAFEBABE5678ULL  // Must differ from PolarQuant seed
```

#### Recursive polar transform (encode)

```c
// Recursively convert a Cartesian vector to polar representation.
// Output: angles[] has (d-1) angles; final_radius is the single surviving magnitude.
// depths[] tracks recursion depth of each angle (needed for quantization range).
// Based on Algorithm 1 in the PolarQuant paper (arxiv:2502.02617).
//
// IMPORTANT: At depth 0, inputs are rotated Cartesian coords (can be negative),
// so atan2 gives angles in [-π, π]. At depth > 0, inputs are radii (always ≥ 0),
// so atan2 gives angles only in [0, π/2]. The quantizer must use the correct range.
static float polar_encode_recursive(const float * x, float * angles,
                                     int * depths, int d, int depth) {
    if (d == 1) return fabsf(x[0]);
    if (d == 2) {
        float r = sqrtf(x[0]*x[0] + x[1]*x[1]);
        angles[0] = atan2f(x[1], x[0]);
        depths[0] = depth;
        return r;
    }
    // Split: process even-indexed pairs first, collect radii
    float radii[d/2];
    for (int i = 0; i < d/2; i++) {
        float r = sqrtf(x[2*i]*x[2*i] + x[2*i+1]*x[2*i+1]);
        angles[i] = atan2f(x[2*i+1], x[2*i]);
        depths[i] = depth;
        radii[i] = r;
    }
    // Recurse on the radii vector (depth+1 since radii are always ≥ 0)
    return polar_encode_recursive(radii, angles + d/2, depths + d/2, d/2, depth + 1);
}
```

#### Angle quantization

After rotation, the first-level angles (from Cartesian coordinate pairs) are approximately
uniform on [-π, π]. However, **inner-level angles** (from recursion on radii, which are always
≥ 0) are restricted to [0, π/2]. Using [-π, π] for inner angles wastes 3/4 of the quantization
range. The quantizer must track recursion depth and use the correct range:

```c
// depth=0: first-level angles from Cartesian pairs → range [-π, π]
// depth>0: inner angles from radii pairs (always ≥ 0) → range [0, π/2]
static uint8_t quantize_angle_4bit(float angle, int depth) {
    float lo, hi;
    if (depth == 0) { lo = -M_PI; hi = M_PI; }
    else            { lo = 0.0f;  hi = M_PI / 2.0f; }
    float normalized = (angle - lo) / (hi - lo);  // [0, 1]
    int q = (int)(normalized * 16.0f);
    return (uint8_t)GGML_CLAMP(q, 0, 15);
}

static float dequantize_angle_4bit(uint8_t q, int depth) {
    float lo, hi;
    if (depth == 0) { lo = -M_PI; hi = M_PI; }
    else            { lo = 0.0f;  hi = M_PI / 2.0f; }
    return ((float)q / 16.0f) * (hi - lo) + lo;
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
    int   depths[POLAR_BLOCK_SIZE - 1];  // recursion depth per angle

    for (int i = 0; i < k / POLAR_BLOCK_SIZE; i++) {
        const float * src = x + i * POLAR_BLOCK_SIZE;
        memcpy(rotated, src, POLAR_BLOCK_SIZE * sizeof(float));

        // 1. Apply random rotation
        polar_random_rotation(rotated, POLAR_BLOCK_SIZE, 0xDEADBEEF1234ULL);

        // 2. Recursive polar transform (depth=0 for first level)
        float radius = polar_encode_recursive(rotated, angles, depths,
                                              POLAR_BLOCK_SIZE, /*depth=*/0);

        // 3. Store radius as fp16
        y[i].radius = GGML_FP32_TO_FP16(radius);

        // 4. Pack 4-bit quantized angles (using depth-aware range)
        for (int j = 0; j < POLAR_BLOCK_SIZE / 2; j++) {
            uint8_t a0 = quantize_angle_4bit(angles[2*j], depths[2*j]);
            uint8_t a1 = (j*2+1 < POLAR_BLOCK_SIZE - 1)
                         ? quantize_angle_4bit(angles[2*j+1], depths[2*j+1]) : 0;
            y[i].angles[j] = (a1 << 4) | a0;
        }
    }
}
```

#### Dequantize function

Implement `dequantize_row_polarquant4` as the inverse: unpack angles using depth-aware
dequantization (first d/2 angles use [-π,π] range, remaining inner angles use [0,π/2] range),
reconstruct Cartesian vector from polar coordinates working recursively in reverse, apply the
*inverse* random rotation (transpose of the Hadamard rotation — for Hadamard this is itself,
just re-apply), and scale by the stored radius. The depth layout is deterministic from the block
size so no per-block depth storage is needed — derive it at decode time.

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
    .vec_dot_type      = GGML_TYPE_Q8_0,  // NOTE: TurboQuant bypasses vec_dot in flash attn
                                           // (uses custom asymmetric path). This is set for
                                           // compatibility with code that inspects the field.
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

**File: `src/llama-kv-cache.h`** (and associated `.cpp` files) — KV cache class

Find where KV cache tensors are allocated. Add `pq4` and `tq3` as valid `cache_type_k` /
`cache_type_v` options alongside existing `q8_0`, `q4_0` etc:

```cpp
// In the cache_type validation / selection block:
if (cparams.cache_type_k == GGML_TYPE_POLARQUANT_4 ||
    cparams.cache_type_k == GGML_TYPE_TURBOQUANT_3) {
    // PolarQuant and TurboQuant are KV-cache-only types.
    // They do not support weight quantization — validate this is only
    // applied to the KV cache tensors.

    // Reject non-power-of-2 head dimensions (required by recursive polar transform + FWHT)
    GGML_ASSERT(head_dim > 0 && (head_dim & (head_dim - 1)) == 0 &&
                "PolarQuant/TurboQuant requires power-of-2 head_dim");
}
```

**File: `common/arg.cpp`** (where `--cache-type-k` is parsed):

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
        // IMPORTANT: QJL MUST use a DIFFERENT random rotation than PolarQuant.
        // Using the same rotation violates the JL independence guarantee.
        polar_random_rotation(residual, POLAR_BLOCK_SIZE, QJL_ROTATION_SEED);
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
    // 1. PolarQuant component: rotate Q with PolarQuant's rotation, dot with reconstructed K
    float pq_rotated_q[d];
    memcpy(pq_rotated_q, q, d * sizeof(float));
    polar_random_rotation(pq_rotated_q, d, POLARQUANT_ROTATION_SEED);

    float k_reconstructed[d];
    dequantize_block_turboquant3_polar_only(k, k_reconstructed);
    float polar_dot = 0.0f;
    for (int i = 0; i < d; i++) polar_dot += pq_rotated_q[i] * k_reconstructed[i];

    // 2. QJL correction: rotate Q with QJL's SEPARATE rotation, dot with sign bits
    //    IMPORTANT: QJL uses a different random projection than PolarQuant.
    //    The JL guarantee requires an independent random matrix.
    float qjl_rotated_q[d];
    memcpy(qjl_rotated_q, q, d * sizeof(float));
    polar_random_rotation(qjl_rotated_q, d, QJL_ROTATION_SEED);

    float qjl_sum = 0.0f;
    for (int j = 0; j < d / 8; j++) {
        uint8_t signs = k->qjl_signs[j];
        for (int b = 0; b < 8; b++) {
            float sign = (signs & (1 << b)) ? 1.0f : -1.0f;
            qjl_sum += sign * qjl_rotated_q[j*8 + b];
        }
    }
    float qjl_correction = (float)(M_PI / 2.0) * (1.0f / sqrtf((float)d)) * qjl_sum;

    return polar_dot + qjl_correction;
}
```

### 2.5 — Hook asymmetric estimator into ggml_flash_attn_ext

This is the most invasive change. The flash attention operator in `ggml/src/ggml-cpu/ops.cpp`
(and potentially `ggml/src/ggml-cuda/fattn.cu`) needs a code path that dispatches to
`turboquant3_asymmetric_dot` when the K cache tensor type is `GGML_TYPE_TURBOQUANT_3`.

Find the inner loop of `ggml_compute_forward_flash_attn_ext` in `ops.cpp` and add:

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
1. Apply TWO Hadamard transforms to Q in shared memory: one with PolarQuant seed, one with QJL seed
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
    // NOTE: q_rot here must be rotated with the QJL seed (separate from PolarQuant seed).
    // In practice, pre-rotate Q with QJL's rotation in shared memory alongside the
    // PolarQuant rotation.
    // Load sign bits of QJL-rotated Q
    uint8_t q_signs[d/8];
    for (int i = 0; i < d/8; i++) {
        uint8_t qs = 0;
        for (int b = 0; b < 8; b++) {
            if (__half2float(q_rot_qjl[i*8+b]) >= 0.0f) qs |= (1 << b);
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
|`ggml/src/ggml-common.h`                 |Add `block_polarquant4`, `block_turboquant3` structs          |
|`ggml/src/ggml-quants.h`                 |Add function declarations for encode/decode/dot               |
|`ggml/src/ggml-polarquant.c` (NEW)       |All encode/decode/dot/rotation functions (separate file, following community branch pattern)|
|`ggml/src/ggml-polarquant.h` (NEW)       |Header for polarquant functions                               |
|`ggml/src/ggml-cpu/ops.cpp`              |Hook asymmetric estimator into flash attention                |
|`ggml/src/ggml-cuda/fattn.cu`            |Add TurboQuant dispatch                                       |
|`ggml/src/ggml-cuda/fattn-turboquant.cuh`|New CUDA kernel (Phase 3)                                     |
|`ggml/src/ggml-cuda/CMakeLists.txt`      |Add new CUDA source                                           |
|`src/llama-kv-cache.h` (+ `.cpp`)        |Validate new cache types in KV cache class                    |
|`common/arg.cpp`                         |Add `pq4` and `tq3` as valid `--cache-type-k/v` values        |

-----

## Key Constants & Hyperparameters

|Parameter                |Value                                        |Source                                         |
|-------------------------|----------------------------------------------|-----------------------------------------------|
|Block size               |64                                            |Matches QJL reference impl, power of 2 for FWHT|
|PolarQuant rotation seed |Fixed (e.g., `0xDEADBEEF1234ULL`)             |Same seed at encode + decode                   |
|QJL rotation seed        |Fixed, DIFFERENT from PolarQuant (e.g., `0xCAFEBABE5678ULL`)|JL guarantee requires independent projection|
|Angle bits (PolarQuant-4)|4                                             |Phase 1                                        |
|Angle bits (TurboQuant-3)|2                                             |Phase 2 — tune vs QJL budget                   |
|QJL bits                 |1 (sign only)                                 |Per QJL paper                                  |
|Rotation type            |Randomized Hadamard (FWHT + sign flips)       |Fast, no storage needed                        |
|QJL correction constant  |π/2 · 1/√d                                    |Theorem 1 in QJL paper                         |
|Angle range (depth 0)    |[-π, π]                                       |First-level: rotated Cartesian coords          |
|Angle range (depth > 0)  |[0, π/2]                                      |Inner levels: radii are always ≥ 0             |

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
- PolarQuant and QJL each need their own fixed rotation seed. Do not reuse the same seed for
  both — the JL guarantee requires an independent random projection for the residual stage.
- Each seed must be identical at quantize and dequantize time. Do not derive seeds from tensor
  data — use compile-time constants.
- The FWHT requires input length to be a power of 2. POLAR_BLOCK_SIZE=64 satisfies this.
- Angle quantization ranges differ by recursion depth: first-level angles (from Cartesian
  pairs) span [-π, π]; inner-level angles (from radii pairs) span [0, π/2]. The depth layout
  is deterministic from the block size, so no per-block storage is needed.
- The V cache can use PolarQuant (not TurboQuant). TurboQuant’s asymmetric estimator is
  specific to the Q·K dot product. V vectors are accessed differently (weighted sum, not dot).
- Test with small models (1B–3B) first. Perplexity delta should be < 0.5 at 3-bit total.
- If perplexity is bad, the most likely causes are: wrong rotation inverse, angle range
  mismatch (using [-π,π] for inner angles instead of [0,π/2]), angles[d-1] never set,
  using the same rotation seed for both PolarQuant and QJL, or missing normalization in
  the recursive step.

-----

## Resolved Questions

### Q1: Radii sign handling in recursive polar transform
**Answer:** No sign correction needed. Radii are always ≥ 0 by definition. Inner-level angles
naturally fall in [0, π/2]. The depth-aware quantizer (first level [-π, π], inner levels
[0, π/2]) is the correct approach. Confirmed by PolarQuant paper and community implementations.

### Q2: Bit budget split for 3.0 bits/value
**Answer:** TurboQuant_prod uses **2-bit Lloyd-Max + 1-bit QJL = exactly 3.0 bits/value**,
uniform across all coordinates. The split is fixed, not configurable per model.

**IMPORTANT DESIGN NOTE:** The TurboQuant paper's recommended "prod" variant uses Lloyd-Max
codebook quantization (optimal scalar quantizer for Gaussian distributions) as Stage 1, NOT
PolarQuant's recursive polar decomposition. After random rotation, coordinates are approximately
i.i.d. Gaussian, making Lloyd-Max optimal. The community branch (mudler) also uses this simpler
approach. The recursive polar transform described earlier in this plan is an alternative Stage 1
that works but is more complex than necessary.

**Decision:** Keep the recursive polar transform as specified (it is mathematically valid and
matches the PolarQuant paper). If implementation proves too complex or slow, fall back to
Lloyd-Max codebook quantization (simpler, community-validated). See "Community Branch Analysis"
section below for Lloyd-Max implementation details.

### Q3: vec_dot_type and flash attention
**Answer:** Flash attention's non-tiled path (used for quantized KV types) DOES use `vec_dot`.
Q is converted to `vec_dot_type` (e.g., Q8_0) before calling `vec_dot` with quantized K. The
tiled path (F16/F32 only) bypasses vec_dot entirely but doesn't support quantized KV.

**Impact on TurboQuant:** The standard vec_dot path would quantize Q to Q8_0, defeating the
asymmetric estimator. However, TurboQuant REQUIRES a custom attention path anyway (for the QJL
correction term), so we bypass the standard vec_dot dispatch entirely. The custom path in
`ggml_compute_forward_flash_attn_ext` (ops.cpp) keeps Q in full precision.

**Decision:** Register `vec_dot_type = GGML_TYPE_Q8_0` for compatibility with code that checks
it, but the actual TurboQuant attention path never calls it. Add a comment noting this.

### Q4: Block size vs head_dim alignment
**Answer:** PolarQuant requires power-of-2 dimensions (recursive halving). The papers do not
discuss padding for non-power-of-2 dims. All major LLMs use head_dim=128 (power of 2):
LLaMA 2/3, Mistral, Qwen2 all use 128.

**Decision:** Reject non-power-of-2 head_dim with a clear error message at KV cache init time.
Add validation: `GGML_ASSERT(head_dim > 0 && (head_dim & (head_dim - 1)) == 0)`. This is
conservative — zero-padding to next power of 2 is a future option if needed.

⚠ **REVISIT LATER:** Some models may use head_dim=80 or 96. If this becomes a blocker, add
zero-padding support. The Hadamard rotation and polar transform would operate on the padded
dimension; unpad after dequantization.

### Q5: Community branch analysis
**Answer:** See dedicated section below.

### Q6: CUDA shared memory for two Q rotations
**Answer:** Storing two rotated Q copies (PolarQuant + QJL) adds **0.55–4.43% overhead** for
head_dim=128 across ncols=1–8. This is negligible relative to the 48KB shared memory budget.

For tight configs (e.g., Deepseek with head_dim=576, already at 47.4KB/48KB), a sequential
strategy works: rotate Q with PolarQuant seed → compute all polar dots → rotate Q with QJL
seed → compute all QJL corrections. This avoids storing both copies simultaneously.

**Decision:** Use simultaneous storage (Strategy B) by default. Add a sequential fallback for
large head_dim configs where shared memory is tight. Document in CUDA kernel comments.

| head_dim | One Q copy (bytes) | Two copies overhead | % of 48KB |
|----------|--------------------|--------------------|-----------|
| 64       | 144                | +144               | 0.29%     |
| 128      | 272                | +272               | 0.55%     |
| 256      | 528                | +528               | 1.07%     |

-----

## Community Branch Analysis (mudler/llama.cpp feat/turbo-quant)

**Not rebasing — using as reference only.** Key implementation details for our reference:

### Architecture
- Random orthogonal rotation via QR decomposition (Haar-distributed)
- Lloyd-Max codebook quantization (NOT recursive polar transform)
- Dense bit packing into 3-bit and 4-bit formats
- **No QJL residual stage** — this is Stage 1 only

### Types Added
- `GGML_TYPE_TBQ3_0` (3-bit, ~3.06 bits/value with radius overhead)
- `GGML_TYPE_TBQ4_0` (4-bit, ~4.06 bits/value with radius overhead)

### Block Structs
```c
// 3-bit: 96 bytes packed indices + 2 bytes L2 norm (block 0 only)
block_tbq3_0 { uint8_t qs[QK_K * 3 / 8]; ggml_half d; }

// 4-bit: 128 bytes packed nibbles + 2 bytes L2 norm
block_tbq4_0 { uint8_t qs[QK_K / 2]; ggml_half d; }
```

### Quantization Pipeline
1. Compute L2 norm, normalize to unit vector
2. Apply fixed random orthogonal matrix (deterministic seed per row)
3. Scale by √k, quantize to nearest Lloyd-Max codebook entry via binary search
4. 3-bit: 8 levels, codebook range [-2.15, 2.15]
5. 4-bit: 16 levels, codebook range [-2.73, 2.73]

### Files Added/Modified
| File | Change |
|------|--------|
| `ggml/src/ggml-turboq.c` (NEW) | Main quantize/dequantize logic |
| `ggml/src/ggml-turboq.h` (NEW) | Rotation + seed functions |
| `ggml/src/ggml-turboq-tables.h` (NEW) | Lloyd-Max codebooks + decision boundaries |
| `ggml/include/ggml.h` | Added TBQ3_0, TBQ4_0 enum entries |
| `ggml/src/ggml-common.h` | Block struct definitions |
| `ggml/src/ggml-quants.h` | Function declarations |
| `ggml/src/ggml-quants.c` | Validation macros |
| `ggml/src/CMakeLists.txt` | Build registration |
| `common/arg.cpp` | CLI cache type options |

### Key Differences From Our Plan
| Aspect | Our Plan (PolarQuant) | Community Branch |
|--------|----------------------|------------------|
| Stage 1 quantizer | Recursive polar decomposition | Lloyd-Max codebook |
| QJL residual (Stage 2) | Yes (1-bit signs) | No |
| Rotation | Hadamard (FWHT, no storage) | Dense orthogonal (QR, needs matrix storage or recomputation) |
| Angle handling | Depth-aware ranges | N/A (scalar codebook) |
| Complexity | Higher | Lower |

### Lessons From Community Branch
- **Separate files are cleaner** — `ggml-turboq.c/h` pattern avoids polluting ggml-quants.c.
  Consider creating `ggml-polarquant.c/h` for our implementation.
- **Per-row deterministic seed** (`turboq_seed_from_row`) — their seed varies per row. Our plan
  uses a global constant seed. Per-row seeds may give better statistical properties.
- **Lloyd-Max codebooks are precomputed** — stored in a tables header file. Simple and fast.
- **Thread-local rotation caching** — good optimization for CPU path.