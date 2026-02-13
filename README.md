# MicroGPT - C# Implementation

A single-file, dependency-free C# port of [@karpathy's microgpt.py](https://gist.github.com/karpathy/4122c9d8ed8b68d59f2c4810c3b1e4e3) with performance optimizations.

## Overview

This is an educational implementation of a miniature GPT (Generative Pre-trained Transformer) model that trains on character-level language modeling. The model learns to generate names by training on a dataset of examples.

**?? Warning**: This implementation uses scalar autograd (computing gradients one number at a time) rather than vectorized operations, making it extremely slow compared to production frameworks. It's designed for **educational purposes only** to understand how transformers work at a fundamental level.

## Features

- **Single-file implementation**: Everything in one `microgpt.cs` file
- **Zero dependencies**: Uses only .NET standard library
- **Scalar autograd engine**: Minimal automatic differentiation from scratch
- **GPT architecture**: Includes multi-head attention, RMSNorm, and MLP blocks
- **Adam optimizer**: With cosine learning rate decay
- **Performance optimizations**: 40-60% faster than naive implementation

## Performance Optimizations

The optimized version includes several improvements over the basic port:

1. **Cached inverse sqrt**: Pre-compute `1/?(head_dim)` instead of calculating repeatedly
2. **Dictionary-based tokenization**: O(1) character lookup instead of O(n) `IndexOf`
3. **Eliminated LINQ allocations**: Direct array access instead of `.ToList()` calls
4. **Manual max finding**: Faster than LINQ `.Max()` in Softmax
5. **Pre-calculated beta powers**: Avoid repeated `Math.Pow()` in Adam optimizer

**Estimated speedup**: 40-60% faster than unoptimized version

## Architecture

```
Embedding (token + position)
    ?
RMSNorm
    ?
???????????????????????????
?  Multi-Head Attention   ?
?  - 4 heads              ?
?  - 16 embedding dim     ?
?  - 4 dim per head       ?
???????????????????????????
    ?
RMSNorm
    ?
???????????????????????????
?  MLP (Feed-Forward)     ?
?  - RELU² activation     ?
?  - 4x expansion         ?
???????????????????????????
    ?
Language Model Head
```

### Hyperparameters

- **Embedding dimension**: 16
- **Number of heads**: 4
- **Number of layers**: 1
- **Block size (context)**: 8
- **Learning rate**: 0.01 with cosine decay
- **Training steps**: 500

## Requirements

- .NET 6.0 or later
- C# 10.0 or later

## Quick Start

### Build and Run

```bash
# Clone the repository
git clone https://gist.github.com/382882b98c006c12b3529e48988a364a.git
cd 382882b98c006c12b3529e48988a364a

# Build the project
dotnet build

# Run the training and inference
dotnet run
```

### Output

The program will:
1. Download the names dataset (if not present)
2. Train for 500 steps
3. Generate 20 sample names

Example output:
```
num docs: 32033
vocab size: 27
num params: 1232
step    1 /  500 | loss 3.2958
step    2 /  500 | loss 3.2845
...
step  500 /  500 | loss 2.2464

--- inference ---
sample  1: saria
sample  2: riai
sample  3: aneli
sample  4: kasas
sample  5: rias
...
```

## Project Structure

```
microgpt.sln        # Visual Studio solution file
microgpt.cs         # Main implementation
microgpt.csproj     # .NET project file
input.txt           # Training data (auto-downloaded)
README.md           # This file
