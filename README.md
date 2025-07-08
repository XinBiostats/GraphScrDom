<p align="left">
  <img src="./assets/logo-removebg-preview.png" alt="GraphScrDom Logo" width="100"/>
</p>

# GraphScrDom

'add description here'

---

## 📦 Pipeline Overview

The pipeline consists of two main modules:

---

### 📌 Module 1: Scribble Annotation

Use the **GraphScrDom** tool to manually annotate scribbles on the tissue image via free-form lasso selections, providing **partial supervision** for downstream training.  
We provide both **Windows** and **macOS** compatible versions of the Scribble Tool.

#### GUI Example:
<p align="center">
  <img src="./assets/GUI.png" alt="Scribble Tool Demo" width="600"/>
</p>
_(Example: Scribble Tool with color-coded region selection)_

---

### 📌 Module 2: Domain Detection and Evaluation (Google Colab)

Use the generated annotations from **Module 1**, along with cell type compositions and optional ground-truth labels, to train and evaluate the **GraphScrDom** domain detection model in **Google Colab**.

#### 📥 Required Inputs:
- `scribble.csv`: partial labels from Scribble Tool  
- `deconvolution.csv`: cell type composition matrix (e.g., from **Cell2location**)  
- `annotation.csv` *(optional)*: ground-truth domain labels for evaluation only  

#### 💻 Colab Example:
*Add Google Colab screenshot here*  
_(Example: model training, clustering results, evaluation metrics)_
