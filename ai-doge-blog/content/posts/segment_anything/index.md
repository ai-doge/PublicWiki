---
title: Porting the Segment Anything Model to iOS
seo_title: Segment Anything
summary: Porting the Segment Anything Model to an iOS App
description: Porting the Segment Anything Open-Source Image Segmentation Algorithm to iOS
slug: segment_anything
author: ai-doge

draft: false
date: 2023-09-17T10:57:29+08:00
lastmod: 
expiryDate: 
publishDate: 

feature_image: 
feature_image_alt: 

categories:
  - Segmentation
tags:
  - SAM
  - Vision
series:

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

## Introduction to the Segment Anything iOS App

[Segment Anything](https://apps.apple.com/us/app/segment-anything/id6447527235) is an image segmentation app available for iPhone or iPad. The app is based on the open-source SAM (Segment Anything Model). All processing is done locally on your iPhone or iPad, requiring no network connection. The app has been optimized for smooth, reliable performance on your device.

## Exporting to Onnx Format

SAM relies on two models: the image encoder (vit) for extracting image features and the mask_decoder for obtaining the final segmentation mask.

### Export Image Encoder to Onnx

```python
sam_checkpoint = "./model/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Load input data
input_data = np.load("./model/image_encoder/input.npy")
input_tensor = torch.from_numpy(input_data).float()

# Export the image encoder model using torch.onnx
torch.onnx.export(
    sam.image_encoder,                      # model instance
    input_tensor,                           # input tensor
    "./model/image_encoder/image_encoder_vit.onnx",  # output ONNX file name
    verbose=True,
    export_params=True,                     # whether to export model parameters
    # opset_version=11,                      # ONNX opset version
    do_constant_folding=True,               # whether to perform constant folding optimization
    input_names=["input"],                  # input node names
    output_names=["output"],                # output node names
)
```

### 1.2 Export Mask Decoder to Onnx

To simplify the complexity of exporting the ONNX model, we have removed a few inputs: `mask_input`, `has_mask_input`, and `orig_im_size`. The `orig_im_size` will be defaulted to 1024x1024, which will greatly simplify the handling of variable-length dimensions in ONNX, making it easier for us to port it to iOS later. We temporarily do not need `mask_input`, so it is directly removed.

```python
# Export mask generator
import warnings
onnx_model = SamOnnxModel(sam, return_single_mask=False, return_extra_metrics=True)

dynamic_axes = {
    # "point_coords": {1: "num_points"},
    # "point_labels": {1: "num_points"},
}

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
batch_size = 1
num_points = 4
dummy_inputs = {
    "image_embeddings": torch.randn(batch_size, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(batch_size, num_points, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(batch_size, num_points), dtype=torch.float),
    # "mask_input": torch.randn(batch_size, 1, *mask_input_size, dtype=torch.float),
    # "has_mask_input": torch.tensor([batch_size], dtype=torch.float),
    # "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float),
}
output_names = [
    "masks",
    "scores",
    "stability_scores",
    "areas",
    "low_res_masks"]

onnx_model_path = "./model/mask_decoder/mask_decoder." + str(num_points) + ".onnx"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )    

print("model exported")
```

## Running the ONNX Model on iOS Devices

### Utilizing MPSX Library

To run the ONNX model on iOS, we leveraged a closed-source modification of the [MPSX library](https://github.com/prisma-ai/MPSX). MPSX is an excellent open-source project that allows you to load ONNX models on iOS using Swift and perform inference in a straightforward manner.

### Enhancements to MPSX

We made extensive enhancements to the MPSX library to support a more comprehensive set of ONNX operators and offer a more flexible way to invoke the model. These modifications enabled us to integrate the ONNX model seamlessly into our iOS application.

### Code Snippet: Loading and Running the ONNX Model
Below is a Swift code snippet that demonstrates how to load the ONNX model and perform inference:

```swift
let graph = buildGraphVit(path: folder + "image_encoder_vit.sim.f16.onnx",
                          floatPrecision: .float16, 
                          input: "input", 
                          output: "output", 
                          inputShape: Shape([1, 3, 1024, 1024]))
        
let input = Tensor.loadFromNpy(path: folder + "input.npy")!
let output = graph.forward(inputs: ["input": input], outputs: ["output"])["output"]!
```

## Interesting Tidbits

It's worth noting that we encountered the following error when running inference on the ViT model on iOS 17:

```bash
Input N1D1C133H133W128 and output N1D19C19H7W128 tensors must have the same number of elements
```

The root cause of this issue lies in the `/layers.1/blocks.0/reshape` layer. The input tensor shape is `[1, 133, 133, 128]`, and the output tensor shape is `[1, 19, 7, 19, 7, 128]`. This Reshape operation was not a problem on iOS 16 and earlier versions but throws an error on iOS 17.

After much deliberation, we found a workaround. We forced the output tensor shape from a 6-dimensional tensor to a 5-dimensional tensor, `[19, 7, 19, 7, 128]`. This does not change the semantics (we assume that the batch size is always 1 in our application). This method successfully bypasses the error.
