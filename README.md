
## SG-Fusion: A Swin-Transformer and Graph Convolution-Based Multi-Modal Deep Neural Network for Glioma Prognosis

**Summary:** We have developed a novel dual-modal integration approach that combines morphological attributes from histopathological images with genomic data, offering a comprehensive view of tumor characteristics and addressing the limitations inherent in single-modal analyses. In the histopathological modality, we utilize the Swin-Transformer structure to capture both local and global image features, enhanced by contrastive learning to refine the model's ability to identify similarities and differences among tumor presentations. For the genomic modality, we employ a graph convolutional network that capitalizes on the similarities in gene function and expression levels, ensuring a robust genomic analysis. Additionally, we integrate a cross-attention module that facilitates superior interaction and information integration between the two modalities, significantly boosting diagnostic performance. To further enhance the model's adaptability and reliability across various datasets and scenarios, we incorporate divergence-based regularization, thus improving its generalization capabilities.

![Overall pipline](/imgs/til_pipline.png "SG-Fusion pipline")

### Pre-requisites:

- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 ) with CUDA 11.8
- Python (3.9.13), torch_geometric(2.1*) matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), openslide-python (1.1.1), openslide (3.4.1), pandas (1.1.3), pillow (7.0.0), torch (2.1.0), torchvision (0.16.0), scikit-learn (0.22.1), tensorboardx (1.9)
- [Pretrained Swin Transformer v2 weights](https://drive.google.com/file/d/1thbfaUGJoDaNkaek6QgTYs5BlmTnZ958/view?usp=drive_link)


### Downloading TCGA Data

To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) and the [cBioPortal](https://www.cbioportal.org/).

### Selected Gene names

The selected key gene expression data can be downloaded via:

[Selected Key Genes](https://drive.google.com/file/d/1U42o9FJFjp3jLPKN3Js_0eoRlXm0RMfM/view?usp=sharing)

### Graph adjacy matrix

The constitution matrix based on gene-gene functional interaction and value similarity can be downloaded via:
[Adjacency Matrix](https://drive.google.com/file/d/1JRbn3A9awxrGtMHLmnh1NOj9OWPeQFbn/view?usp=sharing)


### Interpretability Analysis

For the pathology image analysis, we utilized GradCAM to visualize the feature maps extracted by the Swin-Transformer, highlighting critical areas in a Stage 4 glioma sample. For the genomic analysis, Integrated Gradients helped visualize the most influential genes, with HEY1 emerging as particularly significant due to its role in enhancing glioma cell survival through the Notch signaling pathway, emphasizing its potential as a therapeutic target.
![Model interpolation](/imgs/interpolation.png "interpolation")


