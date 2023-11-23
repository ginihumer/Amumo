# Amumo (Analyze Multi-Modal Models)
## Understanding and Comparing Multi-Modal Models
Amumo is a visual analytics prototype that allows users to explore latent spaces of inter-modal data pairs (in particular pairs of image and text embeddings as they can be retrieved by CLIP-like models).
It is implemented as a collection of ipywidgets and comes with a set of pre-defined datasets and bi-modal contrastive models like [CLIP](https://proceedings.mlr.press/v139/radford21a.html), [CyCLIP](https://proceedings.neurips.cc/paper_files/paper/2022/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf), and [CLOOB](https://proceedings.neurips.cc/paper_files/paper/2022/file/8078e76f913e31b8467e85b4c0f0d22b-Paper-Conference.pdf) to keep the overhead for users as low as possible. Additionally, users can define their own datasets and models to explore. Check out the ["getting started" notebook](https://github.com/ginihumer/Amumo/blob/main/notebooks/getting_started.ipynb) for first steps.

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/45741696/255124758-3a7b6c57-9c1d-4044-b54d-684711622de8.jpg" height="400"/>

### Examples
You can use Amumo to interactively explore bi-modal datasets...

<img src="https://github.com/ginihumer/Amumo/assets/45741696/d5bcf266-7e1a-4fd7-af09-2bbc0a5ab2ed"/>

... or compare various bi-modal models.

<img src="https://github.com/ginihumer/Amumo/assets/45741696/55681796-5124-4af9-b317-353f40b49605" height="400"/>


### Installation
Set up conda environment with ipywidgets:

```bash
conda create -n myenv python=3.9
activate myenv
pip install ipykernel
pip install ipywidgets
```

Local installation:
```bash
pip install -e .
```

Package installation:
```bash
pip install git+https://github.com/ginihumer/Amumo.git
```

Or you can create the conda environment from the .yml file:
```bash
conda env create -f environment.yml
```

## Troubleshooting

### SSL Certificate Verification Error
If you encounter an SSL certificate verification error, such as 'CERTIFICATE_VERIFY_FAILED,' while installing or running Amumo, it might be due to your system's SSL configuration. Here are a few steps you can take to address this issue: 
- Try updating your SSL certificate bundle
  ```bash
  pip install --upgrade certifi
  ```
- Try disabling SSL verification temporarily for your Python script.
  ```python
  import ssl
  ssl._create_default_https_context = ssl._create_unverified_context
  ```
# Understanding and Comparing Multi-Modal Models
## Exploring the Latent Space of CLIP-like Models (CLIP, CyCLIP, CLOOB) Using Inter-Modal Pairs (Featuring Amumo, Your Friendly Neighborhood Mummy)

### Abstract
[Contrastive Language Image Pre-training (CLIP)](https://proceedings.mlr.press/v139/radford21a.html) and variations of this approach like [CyCLIP](https://proceedings.neurips.cc/paper_files/paper/2022/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf), or [CLOOB](https://proceedings.neurips.cc/paper_files/paper/2022/file/8078e76f913e31b8467e85b4c0f0d22b-Paper-Conference.pdf) are trained on image-text pairs with a contrastive objective. The goal of contrastive loss objectives is to minimize latent-space distances of data points that have the same underlying meaning. We refer to the particular cases of contrastive learning that CLIP-like models perform as multi-modal contrastive learning because they use two (or [more](https://arxiv.org/pdf/2305.05665.pdf)) modes of data (e.g., images and texts) where each mode uses their own encoder to generate a latent embedding space. More specifically, the objective that CLIP is optimized for minimizes the distances between image-text embeddings of pairs that have the same semantic meaning while maximizing the distances to all other combinations of text and image embeddings.
We would expect that such a shared latent space places similar concepts of images and texts close to each other. However, the reality is a bit more complicated...

### Resources
Check out the [Interactive article](https://jku-vds-lab.at/amumo) submitted to the [6th Workshop on Visualization for AI Explainability (VISxAI 2023)](https://visxai.io/).

Check out the [computational notebook to reproduce the results](https://github.com/ginihumer/Amumo/blob/main/notebooks/clip_article.ipynb) shown in the article or for using as a starting point for future investigations.

Check out the [computational notebook for exporting the data](https://github.com/ginihumer/Amumo/blob/main/notebooks/export_data.ipynb) used in the interactive article.


## How to cite?

You may cite Amumo using the following bibtex:

```bibtex
@software{humer2023amumo,
  author = {Humer, Christina and Streit, Marc and Strobelt, Hendrik},
  title = {{Amumo (Analyze Multi-Modal Models)}},
  url = {https://github.com/ginihumer/Amumo},
  year = {2023}
}
```