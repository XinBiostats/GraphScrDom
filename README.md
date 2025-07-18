
# GraphScrDom

**GraphScrDom** is a **semi-supervised** spatial domain detection tool designed for spatial transcriptomics data. It integrates **spatial gene expression** and **cell type composition** through a graph-based deep learning framework. Users can interactively annotate partial labels via histology-guided scribbles. GraphScrDom achieves accurate and biologically meaningful spatial domain identification, even with minimal supervision.

---

## 📦 Pipeline Overview

The pipeline consists of two main modules:
<p align="center">
  <img src="./assets/GraphScrDom.png" alt="GraphScrDom Overview" width="600"/>
</p>

---

### 📌 Module 1: Scribble Annotation

Use the **GraphScrDom** tool to manually annotate scribbles on the tissue image via free-form lasso selections, providing **partial supervision** for downstream training.  

We provide both **Windows** and **macOS** compatible versions of the Scribble Tool.
- [Download for Windows](https://drive.google.com/drive/folders/1nlIg2MkPhiym2701isVreEg7-iFmTdIR?usp=drive_link)
- [Download for macOS](https://drive.google.com/drive/folders/1LRcAEVUAGBs35fwEYALvZ1b6iZLBYAqN?usp=drive_link)

#### GUI Example:
<p align="center">
  <img src="./assets/GUI.png" alt="Scribble Tool Demo" width="600"/>
</p>

---

### 📌 Module 2: Domain Detection and Evaluation (Google Colab)

Use the generated annotations from **Module 1**, along with cell type compositions and optional ground-truth labels, to train and evaluate the **GraphScrDom** domain detection model in **Google Colab**.

#### 📥 Required Inputs:
- `anndata`: spatial transcriptomics data in .h5/.h5ad
- `scribble_output.csv`: partial labels from Scribble Tool  
- `deconvolution.csv`: cell type composition matrix (e.g., from **Cell2location**)  
- `annotation.csv` *(optional)*: ground-truth domain labels for evaluation only  

#### 💻 Colab Example:

**[Run this notebook in Google Colab](https://colab.research.google.com/drive/1HDQB4R5XMFmdpT4wFR9GZPdSnKuFRYc5?usp=drive_link)**

<p align="center">
  <img src="./assets/output.png" alt="GraphScrDom Output Example" width="600"/>
</p>
