---
title: Port the LLaMa Model on iOS
seo_title: iOS LLaMa
summary: How to Port the LLaMa Model to Run on iOS Devices
description: How to Port the LLaMa Model to Run on iOS Devices
slug: llama-ios
author: Ai Doge

draft: false
date: 2023-09-01T21:21:46-05:00
lastmod: 2023-09-01T21:21:46-05:00
expiryDate: 
publishDate: 

feature_image: 
feature_image_alt: 

categories:
  - LLM
  - LLaMa
tags:
  - LLaMa
  - MPSX
  - ONNX

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

# Running Baby llama2 Model on iOS with Illustrate Llama App

## Introduction

In this blog post, we'll walk through the technical steps involved in running the Baby llama2 model from the [llama2.c GitHub repository](https://github.com/karpathy/llama2.c) on an iOS app called [Illustrate Llama](https://apps.apple.com/us/app/illustrate-llama/id6452017369). We'll cover the process of exporting the model to ONNX format, integrating it into the iOS app, and the challenges we faced along the way.

## Part 1: Exporting the Model to ONNX Format

### The Challenge

The first step in our journey was to export the pre-trained Baby llama2 model to ONNX format. However, we encountered a roadblock: the llama2.c project did not initially support ONNX export. The primary reason was that ONNX did not support the `Complex64` data type, which was used in the codebase.

For more context, you can refer to the GitHub issue [here](https://github.com/karpathy/llama2.c/issues/142).

### Our Solution

To overcome this challenge, we submitted a Pull Request ([PR #103](https://github.com/karpathy/llama2.c/pull/103)) to the llama2.c repository. The core idea behind the PR was to decompose the operations involving complex numbers into separate operations for the real and imaginary parts.

### Code Changes for ONNX Export

To resolve the issue with the `Complex64` data type, we made several changes to the codebase. Below are some of the key modifications:

#### Replacing Complex Numbers with Real and Imaginary Parts

Originally, the code used complex numbers for certain calculations. We replaced these with separate real and imaginary parts.

```python
# Original code
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

# Modified code
freqs_cos = torch.cos(freqs)  # real part
freqs_sin = torch.sin(freqs)  # imaginary part
```

#### Modifying the Rotary Embedding Function

The `apply_rotary_emb` function was also modified to accommodate the changes.

```python
# Original code
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

# Modified code
xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
```

#### Updating the Forward Method

The forward method in various classes was updated to use the new real and imaginary parts.

```python
# Original code
h = layer(h, freqs_cis)

# Modified code
h = layer(h, freqs_cos, freqs_sin)
```

By making these changes, we were able to successfully export the Baby llama2 model to ONNX format without any issues related to the `Complex64` data type.

## Part 2: Exporting to ONNX Format Post-PR

### The ONNX Export Code

After successfully merging our Pull Request, we were able to export the Baby llama2 model to ONNX format. Below is the Python code snippet that demonstrates how to perform the export:

```python
torch.onnx.export(model,
                  torch.from_numpy(input),
                  "./model_128.onnx",
                  verbose=True,
                  input_names=["input"],
                  output_names=["output"])
```

In this code snippet:

- `model`: The pre-trained Baby llama2 model.
- `torch.from_numpy(input)`: The input tensor converted from a NumPy array.
- `./model_128.onnx`: The path where the exported ONNX model will be saved.
- `verbose=True`: Enables verbose output to understand the export process.
- `input_names` and `output_names`: Specifies the names for the input and output nodes in the ONNX graph.

By running this code, the model is exported to an ONNX file named `model_128.onnx`, which can then be integrated into our iOS application.

## Part 3: Running the ONNX Model on iOS Devices

### Utilizing MPSX Library

To run the ONNX model on iOS, we leveraged a closed-source modification of the [MPSX library](https://github.com/prisma-ai/MPSX). MPSX is an excellent open-source project that allows you to load ONNX models on iOS using Swift and perform inference in a straightforward manner.

### Enhancements to MPSX

We made extensive enhancements to the MPSX library to support a more comprehensive set of ONNX operators and offer a more flexible way to invoke the model. These modifications enabled us to integrate the ONNX model seamlessly into our iOS application.

### Code Snippet: Loading and Running the ONNX Model

Below is a Swift code snippet that demonstrates how to load the ONNX model and perform inference:

```swift
func testLlama2_fp32() {
    let path = "model_path"
    let onnxModel = try! OnnxModel(path: path + "/llama2.sim.fp16.onnx")

    let inputConfigs = ["input": OnnxGraphInputConfig(shape: [1, 103], type: .int64)]
    let outputConfigs = ["output": OnnxGraphOutputConfig()]
    let globalConfig = OnnxGraphGlobalConfig(floatPrecision: .float32)
    let graphConfig = OnnxGraphConfig(inputConfigs: inputConfigs,
                                      outputConfigs: outputConfigs,
                                      globalConfig: globalConfig,
                                      gradConfig: nil)

    let graph = try! OnnxGraphBuilder().build(onnxModel: onnxModel, config: graphConfig)
    let input = Tensor.loadFromNpy(path: path + "/input.npy")!
  
    let output = graph.forward(inputs: ["input": input], outputs: ["output"])["output"]!
}
```

In this code snippet:

- `OnnxModel`: Class for loading the ONNX model.
- `OnnxGraphInputConfig` and `OnnxGraphOutputConfig`: Classes for configuring the input and output shapes and types.
- `OnnxGraphGlobalConfig`: Class for setting global configurations like float precision.
- `OnnxGraphBuilder`: Class for building the graph for inference.
- `Tensor.loadFromNpy`: Method for loading input data from an NPY file.

By running this code, you can perform inference using the ONNX model on your iOS device.
