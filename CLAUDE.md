# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

FlashAttention is a high-performance implementation of the attention mechanism for deep learning, optimized for NVIDIA and AMD GPUs. This repository contains FlashAttention, FlashAttention-2, and FlashAttention-3 implementations with custom CUDA/HIP kernels and Python interfaces.

## Build and Installation

### Main Package Installation

**From PyPI (recommended):**
```bash
pip install flash-attn --no-build-isolation
```

**From source:**
```bash
python setup.py install
```

**Control parallel compilation jobs** (important for systems with limited RAM):
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**Build environment variables:**
- `BUILD_TARGET`: Set to `cuda` or `rocm` (auto-detected by default)
- `FLASH_ATTENTION_FORCE_BUILD`: Set to `TRUE` to force local build instead of using prebuilt wheels
- `FLASH_ATTENTION_SKIP_CUDA_BUILD`: Set to `TRUE` to skip CUDA compilation (for sdist creation)
- `FLASH_ATTENTION_FORCE_CXX11_ABI`: Set to `TRUE` to force C++11 ABI
- `FLASH_ATTN_CUDA_ARCHS`: Semicolon-separated list of CUDA architectures (default: "80;90;100;110;120")
- `NVCC_THREADS`: Number of nvcc threads (default: 4)
- `MAX_JOBS`: Maximum parallel compilation jobs

### FlashAttention-3 (Hopper GPUs - H100/H800)

FlashAttention-3 is optimized for Hopper GPUs and requires CUDA >= 12.3 (CUDA 12.8 recommended).

```bash
cd hopper
python setup.py install
```

**Run tests:**
```bash
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```

### AMD ROCm Support

**Composable Kernel Backend (default):**
```bash
python setup.py install
```

**Triton Backend:**
```bash
pip install triton==3.2.0
cd flash-attention
git checkout main_perf
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
```

**Enable autotune:**
```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE" python $PATH_TO_CODE
```

## Testing

**Main tests:**
```bash
pytest -q -s tests/test_flash_attn.py
```

**ROCm Composable Kernel tests:**
```bash
pytest tests/test_flash_attn_ck.py
```

**AMD Triton tests:**
```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pytest tests/test_flash_attn_triton_amd.py
```

**Run specific test:**
```bash
pytest tests/test_flash_attn.py::test_name -v
```

## Architecture

### Directory Structure

- **`flash_attn/`** - Main Python package
  - `flash_attn_interface.py` - Primary Python API for FlashAttention
  - `flash_attn_triton.py` - Triton implementation
  - `flash_attn_triton_amd/` - AMD Triton backend
  - `cute/` - CuteDSL implementation (Python DSL for CUDA kernels)
  - `models/` - Reference model implementations (GPT, BERT, LLaMA, etc.)
  - `modules/` - PyTorch modules (MHA, MLP, embeddings)
  - `layers/` - Layer implementations (rotary embeddings, patch embeddings)
  - `ops/` - Fused operations (activations, layer norm, fused dense)
  - `losses/` - Loss functions (cross-entropy)

- **`csrc/`** - C++/CUDA source code
  - `flash_attn/` - CUDA kernels for NVIDIA GPUs (sm_80, sm_90)
  - `flash_attn_ck/` - Composable Kernel backend for AMD GPUs
  - `cutlass/` - CUTLASS library (submodule)
  - `composable_kernel/` - Composable Kernel library for AMD (submodule)
  - `fused_dense_lib/` - Fused dense layer kernels
  - `layer_norm/` - Layer normalization kernels

- **`hopper/`** - FlashAttention-3 for Hopper GPUs (sm_90, sm_100)

- **`tests/`** - Test suite
- **`benchmarks/`** - Performance benchmarks
- **`training/`** - Training scripts for GPT models

### Key Implementation Details

**GPU Architecture Support:**
- **NVIDIA CUDA:**
  - Ampere (sm_80): A100, RTX 3090
  - Ada (sm_89): RTX 4090
  - Hopper (sm_90): H100, H800
  - Blackwell (sm_100, sm_120): GB100, GB200
  - Thor (sm_110/sm_101): Upcoming architecture

- **AMD ROCm:**
  - MI200 (gfx90a)
  - MI300 (gfx942, gfx950)
  - RDNA GPUs (via Triton backend)

**Head Dimension Support:**
- Standard: 32, 64, 96, 128, 192, 256
- Head dim > 192 backward pass requires Ampere/Ada/Hopper GPUs

**Data Type Support:**
- FP16, BF16 (BF16 requires Ampere or newer)
- FP8 (forward only, Hopper GPUs)

**Feature Support:**
- Causal masking
- Sliding window (local) attention
- ALiBi (Attention with Linear Biases)
- Paged KV cache (PagedAttention)
- Variable sequence lengths
- Multi-query attention (MQA) and grouped-query attention (GQA)
- Softcapping
- Dropout
- Rotary embeddings (RoPE)

### Kernel Generation

CUDA kernels are generated for different head dimensions and configurations:
- Forward/backward kernels are instantiated separately for each head dimension (32, 64, 96, 128, 192, 256)
- Separate kernels for causal and non-causal attention
- Separate kernels for FP16 and BF16 data types
- Split-KV kernels for long context scenarios

For AMD ROCm, kernels are code-generated at build time:
```bash
python csrc/composable_kernel/example/ck_tile/01_fmha/generate.py -d fwd --output_dir build --receipt 2 --optdim 32,64,128,256
```

### CuteDSL Implementation

The `flash_attn/cute/` directory contains a Python DSL (Domain-Specific Language) implementation:
- Python-based kernel definitions that compile to CUDA
- Uses `cute.jit` decorator for just-in-time compilation
- Supports Ampere (sm_80), Hopper (sm_90), and Blackwell (sm_100) architectures
- Main files:
  - `flash_fwd.py` - Forward pass implementation
  - `flash_bwd.py` - Backward pass implementation
  - `flash_fwd_sm100.py` - Blackwell-specific forward pass
  - `flash_bwd_sm90.py` - Hopper-specific backward pass
  - `flash_bwd_sm100.py` - Blackwell-specific backward pass
  - `interface.py` - High-level Python interface
  - `mask.py`, `mask_definitions.py` - Attention mask support
  - `block_sparsity.py` - Block-sparse attention support

## Development Workflow

### Code Style

The repository uses Ruff for linting and formatting (configured in `.pre-commit-config.yaml`).

**Run linting:**
```bash
ruff check --fix flash_attn/cute/
```

**Run formatting:**
```bash
ruff format flash_attn/cute/
```

### Making Changes to CUDA Kernels

1. For NVIDIA CUDA: Modify files in `csrc/flash_attn/src/` or use the kernel generation script
2. For AMD ROCm CK: Modify files in `csrc/flash_attn_ck/` and regenerate kernels
3. For CuteDSL: Modify Python files in `flash_attn/cute/`
4. Rebuild the package: `python setup.py install`

### Requirements

**Build dependencies:**
- PyTorch >= 2.2
- CUDA >= 12.0 (NVIDIA) or ROCm >= 6.0 (AMD)
- `packaging` Python package
- `ninja` Python package (verify with `ninja --version && echo $?`)
- `psutil` (for build system)
- `einops` (for Python API)

**For Hopper (FlashAttention-3):**
- CUDA >= 12.3 (12.8+ recommended)

**For AMD Triton backend:**
- Triton 3.2.0

### Submodule Management

The repository uses git submodules for CUTLASS and Composable Kernel:
```bash
git submodule update --init csrc/cutlass
git submodule update --init csrc/composable_kernel
```

## Common Pitfalls

1. **Compilation memory usage:** Compiling FlashAttention requires significant RAM (~9GB per job). Use `MAX_JOBS` to limit parallel jobs on memory-constrained systems.

2. **Ninja not working:** If `ninja --version && echo $?` returns non-zero, reinstall: `pip uninstall -y ninja && pip install ninja`

3. **CUDA version mismatch:** Ensure your CUDA toolkit version matches PyTorch's CUDA version. Check with:
   ```bash
   nvcc --version  # CUDA toolkit version
   python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version
   ```

4. **Architecture targeting:** By default, kernels are compiled for sm_80, sm_90, sm_100, sm_110, sm_120. Set `FLASH_ATTN_CUDA_ARCHS` to compile for specific architectures only.

5. **Head dimension constraints:** Head dimensions must be <= 256 and are optimized for powers of 2 or multiples of 32.

6. **Causal mask alignment (v2.1+):** Causal masks are aligned to the bottom-right corner of the attention matrix, not top-left.

## API Reference

**Main functions:**
- `flash_attn_func(q, k, v, ...)` - Separate Q, K, V tensors
- `flash_attn_qkvpacked_func(qkv, ...)` - Packed QKV tensor
- `flash_attn_kvpacked_func(q, kv, ...)` - Packed KV tensor
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` - Variable-length sequences
- `flash_attn_with_kvcache(q, k_cache, v_cache, ...)` - KV cache for inference

All functions are defined in `flash_attn/flash_attn_interface.py`.

## Performance Considerations

- FlashAttention achieves 2-4x speedup over PyTorch's standard attention on A100 GPUs
- Memory usage is linear in sequence length (vs. quadratic for standard attention)
- For inference with KV caching, use `flash_attn_with_kvcache` for optimal performance
- For long sequences, split-KV kernels may be automatically selected
- Block sizes are automatically selected based on head dimension and GPU architecture
