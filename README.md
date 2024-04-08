# Understanding and Comparing Latent Space Characteristics of Multi-Modal Models
## Exploring the Latent Space of CLIP-like Models Using Inter-Modal Pairs

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/45741696/255124758-3a7b6c57-9c1d-4044-b54d-684711622de8.jpg" width="500"/>

Interactive article submitted to the [Journal of Visualization and Interaction](https://www.journalovi.org/).

### Abstract
**Introduction**
Multi-modal contrastive learning models are trained to map data from two or more modalities to a shared embedding space. This latent data representation can then be used for zero- or few-shot classification, cross-modal data retrieval, or generation tasks. Although remarkable results have been reported when testing multi-modal models on these tasks, understanding the latent representations remains challenging. In particular, many multi-modal models exhibit a phenomenon called the “modality gap”, leading to a latent space that cleanly separates the modalities.

**Conclusion**
This article introduces and compares three models trained on image-text pairs. We use these models and interactive visualizations to explain where the modality gap arises from, how it can be closed, and why closing it is important. In the second part, we introduce “Amumo”, a framework we implemented for analyzing multi-modal models. We describe various analysis tasks that can be performed with Amumo. In particular, Amumo can be used for (i) analyzing models, (ii) comparing models with each other, and (iii) analyzing multi-modal datasets. We demonstrate Amumo’s capabilities and generalizability using image, text, audio, and molecule data in combination with several different models.

**Implementation**
For smooth integration into research workflows, we implemented Amumo as a Python package with Jupyter widgets. We implemented the interactive visualizations in this article with JavaScript and plotly.js.

**Demonstration & Materials**
A minimal usage demonstration of Amumo is [deployed with MyBinder](https://mybinder.org/v2/gh/ginihumer/binder-repo/amumo?labpath=getting_started.ipynb). We also provide a demonstration of [analyzing CLOOME with Amumo](https://mybinder.org/v2/gh/ginihumer/binder-repo/cloome?labpath=modality_gap_cloome.ipynb). The code for the Amumo python package and guidelines on how to use it can be found in the [github repository](https://github.com/ginihumer/Amumo/).
