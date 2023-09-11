---
title: Implementing ImageGAN From Data Collection to Model Compression
seo_title: Implementing ImageGAN
summary: 
description: How to train a projected gan on landscapes/ clouds dataset, then port it on iPhone.
slug: port_image_gan
author: ai-doge

draft: false
date: 2023-08-10T11:43:24+08:00
lastmod: 
expiryDate: 
publishDate: 

feature_image: 
feature_image_alt: 

categories:
  - Image GAN
  - Colab
tags:
  - Image GAN
  - Colab Training
  - KMeans Weight Quant
series:

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

## Introduction

[Image GAN](https://apps.apple.com/us/app/image-gan/id1637359196) is an innovative application that merges art and technology to create unique visuals. Utilizing advanced GAN (Generative Adversarial Network) algorithms, the app offers a range of features:

- **AI Landscapes**: Explore AI-generated landscapes that capture the beauty of nature in a novel way.
- **Dynamic Cloudscapes**: Experience ever-changing cloud patterns, curated solely by artificial intelligence.
- **AI Masterpieces**: Witness artistry as AI channels the essence of legendary painters to create new masterworks.

For those interested in the technical aspects, this blog post will walk you through the process of implementing ImageGAN, focusing on data preparation and model training.

## Preparing Training Data

### Data Collection: Landscapes

To train our GAN model, we needed a dataset that was both high-quality and relevant to our application's focus on landscapes. We used web scraping techniques to collect free and commercially usable natural landscape photos. If you're interested, you can download the dataset from [this Google Drive link](https://drive.google.com/file/d/1zmHJS8DIqQ7vJmIFH4xoBYL7zHGi2-XO/view).

### Data Preprocessing

The collected images were preprocessed to fit the requirements of GAN training. Specifically, we resized the images to a uniform 512x512 pixel resolution. This step is crucial for the stability and effectiveness of the GAN model.

## Training the Projected GAN Model

### Choosing an Open-Source Project

For the model training, we chose the open-source project [Projected GAN](https://github.com/autonomousvision/projected-gan) as our base code. Projected GAN is an exceptional work that allows for rapid training convergence. It was presented in a NeurIPS 2021 paper titled "Projected GANs Converge Faster" by Axel Sauer, Kashyap Chitta, Jens Müller, and Andreas Geiger. The repository also provides a quick start Colab notebook for those interested in trying it out.

### Model Customization for Mobile Deployment

To ensure that the trained model could be efficiently deployed on mobile devices, we made some modifications to the model architecture. Specifically, we opted for the `fastgan_lite` model, which is a relatively lightweight version of the original model.

#### Code Modifications

We modified the `FastganSynthesis` class in the generator to adjust the `ngf` parameter from 128 to 64. This change effectively halved the size of the trained model without compromising the quality of generated images in our application.

Here's a snippet of the modified code:

```python
class FastganSynthesis(nn.Module):
    def __init__(self, ngf=64, z_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        # ... (rest of the code remains the same)
```

By making these adjustments, we were able to train a model that not only converges quickly but is also optimized for mobile deployment.

## Exporting the Trained Model to ONNX and Optimization

After the model training is complete, the next step is to export the model to ONNX (Open Neural Network Exchange) format so that it can run on different platforms and environments. This section will detail how to go about this process, with special emphasis on two key steps: model simplification and model compression.

### Exporting to ONNX Format

First, we use PyTorch's `torch.onnx.export` function to export the trained GAN model to ONNX format. Let's assume the exported model file is named `gan.onnx`.

```python
import torch.onnx
import torchvision.models as models

### Initialize the model and input
model = YourTrainedGANModel()
x = torch.randn(batch_size, 3, 512, 512, requires_grad=True)
torch_out = model(x)

### Export the model
torch.onnx.export(model,                     # model being run
                  x,                         # model input
                  "gan.onnx",                # where to save the model
                  export_params=True)        # store the trained parameter weights inside the model file
```

### Model Simplification: Using onnxsim

Once the model is exported, we use the `onnxsim` tool to simplify the model. This step usually removes many redundant layers, thereby accelerating model inference and reducing the layers that need to be supported on the device.

```bash
pip install onnxsim
onnxsim gan.onnx gan.sim.onnx
```

### Model Compression: Quantization

#### Using INT8 Precision

The original FP32 model size is close to 100MB, which is too large for a device-side application. Therefore, we plan to use INT8 precision to compress the model, which usually reduces the model size to a quarter of the original.

#### K-means Quantization

In GAN models, traditional quantization methods like scale symmetric quantization or scale zeroPoint asymmetric quantization may lead to significant model accuracy loss. To address this, we use the K-means algorithm for quantization.

Specifically, for each well-trained weight matrix in the network, we use the K-means algorithm to cluster its values into 256 classes. Then, we record the class centers and the class IDs corresponding to each weight.

This K-means quantization method has been tested to almost completely retain the original network's generated image quality. The Cosine Similarity with the original FP32 model's output results is above 0.999.

```python
# K-means Quantization Code Example (Pseudocode)
from sklearn.cluster import KMeans

def kmeans_quantize(weight_matrix):
    original_shape = weight_matrix.shape
    weight_matrix = weight_matrix.reshape(-1, 1)
    kmeans = KMeans(n_clusters=256, random_state=0).fit(weight_matrix)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    # Replace the original weights with the class centers
    quantized_matrix = centers[labels].reshape(original_shape)
    return quantized_matrix, centers, labels
```

## Conclusion

Implementing ImageGAN involved a series of carefully planned steps, from data collection to model training and optimization. The end result is a mobile application that leverages advanced GAN algorithms to create unique and captivating visuals. Stay tuned for future posts where we will discuss the deployment and performance optimization of ImageGAN on mobile devices.
