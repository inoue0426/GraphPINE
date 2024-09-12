# GIPNet: Graph-based Importance Propagation Network

![Screenshot 2024-09-03 at 23 03 03](https://github.com/user-attachments/assets/33c55657-b3fc-4f56-ac43-b89dbe605605)


This is the official implementation for **GIPNet: A novel Graph Neural Network for interpretable drug response prediction through importance propagation**.

GIPNet is a novel Graph Neural Network (GNN) designed to provide interpretable predictions of drug response through the propagation of gene importance scores. This project integrates multi-omics data with graph structural information to enhance the prediction accuracy and interpretability of drug-gene interactions, offering significant advancements in computational biology, personalized medicine, and drug discovery.

## Features

- **Interpretable Predictions**: GIPNet incorporates an innovative ImportancePropagationLayer, which updates and propagates gene importance scores across the network during training, allowing for detailed insights into drug mechanisms.
- **Multi-omics Integration**: The model uses gene expression, copy number variation, methylation, and mutation data to build a comprehensive gene-gene interaction network.
- **Advanced GNN Architecture**: GIPNet leverages state-of-the-art GNN layers such as GAT, GINE, and Graph Transformer to process graph data with edge attributes.
- **Scalable and Flexible**: The model is designed to handle large-scale biological datasets and can be adapted for various drug response prediction tasks.

## Quick Start Guide

This quick start guide provides instructions on how to run GIPNet predictions using both CPU and GPU, with the process completing in just a few seconds.

### Steps:
1. **Download the Repository:**  
   Begin by downloading the repository to your local machine.

2. **Unzip the Repository:**  
   Use the command `unzip [REPOSITORY_DIRECTORY].zip` to extract the files.

3. **Change Directory:**  
   Navigate into the repository directory with `cd [REPOSITORY_DIRECTORY]`.

4. **Build the Docker Image:**  
   Build your Docker image using `docker build -t [YOUR_DOCKER_USERNAME]/[YOUR_IMAGE_NAME] .`.

5. **Run the Docker Container:**  
   Start the Docker container using `docker run -it -p 9999:9999 [YOUR_DOCKER_USERNAME]/[YOUR_IMAGE_NAME]`.

   After starting the Docker container, access the Jupyter Notebook by navigating to `http://localhost:9999/notebooks/Tutorial.ipynb` in your web browser and run all cells. This notebook will guide you through the basic usage of GIPNet and demonstrate example predictions.

## Full-Size Prediction

Follow the same procedure as above and execute the command `python predict.py` to perform full-size predictions with a pre-trained model.

## Full-Size Training

For training, follow the initial steps as described above. When ready, run the training script with `python train.py --config [YOUR_CONFIG]`.


## Requirements

```bash
accelerate==0.33.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
scikit-learn==1.2.2
torch==2.4.0
torch_geometric==2.5.3
torch_scatter==2.1.2
torchaudio==2.4.0
torchvision==0.19.0
tqdm==4.66.5
requests==2.32.3
plotly==5.23.0
networkx==3.3
seaborn==0.13.2
```
