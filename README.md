# Amumo
### Understanding and Comparing Multi-Modal Models

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/45741696/255124758-3a7b6c57-9c1d-4044-b54d-684711622de8.jpg" width="500"/>

python framework for exploring CLIP models

set up conda environment:

```bash
conda create -n myenv python=3.9
activate myenv
pip install ipykernel
pip install ipywidgets
```


for local installation:
```bash
pip install -e .
```


for package installation:
```bash
pip install git+https://github.com/ginihumer/Amumo.git
```

# Understanding and Comparing Multi-Modal Models
### Exploring the Latent Space of CLIP-like Models (CLIP, CyCLIP, CLOOB) Using Inter-Modal Pairs (Featuring Amumo, Your Friendly Neighborhood Mummy)

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/45741696/255124758-3a7b6c57-9c1d-4044-b54d-684711622de8.jpg" width="500"/>

Check out the [Interactive article](https://jku-vds-lab.at/amumo) submitted to the [6th Workshop on Visualization for AI Explainability (VISxAI 2023)](https://visxai.io/).

Check out the [computational notebook to reproduce the results](https://github.com/ginihumer/Amumo/blob/main/notebooks/clip_article.ipynb) shown in the article or for using as a starting point for future investigations.

Check out the [computational notebook for exporting the data](https://github.com/ginihumer/Amumo/blob/main/notebooks/export_data.ipynb) used in the interactive article.


### Abstract
[Contrastive Language Image Pre-training (CLIP)](https://proceedings.mlr.press/v139/radford21a.html) and variations of this approach like [CyCLIP](https://proceedings.neurips.cc/paper_files/paper/2022/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf), or [CLOOB](https://proceedings.neurips.cc/paper_files/paper/2022/file/8078e76f913e31b8467e85b4c0f0d22b-Paper-Conference.pdf) are trained on image-text pairs with a contrastive objective. The goal of contrastive loss objectives is to minimize latent-space distances of data points that have the same underlying meaning. We refer to the particular cases of contrastive learning that CLIP-like models perform as multi-modal contrastive learning because they use two (or [more](https://arxiv.org/pdf/2305.05665.pdf)) modes of data (e.g., images and texts) where each mode uses their own encoder to generate a latent embedding space. More specifically, the objective that CLIP is optimized for minimizes the distances between image-text embeddings of pairs that have the same semantic meaning while maximizing the distances to all other combinations of text and image embeddings.
We would expect that such a shared latent space places similar concepts of images and texts close to each other. However, the reality is a bit more complicated...

