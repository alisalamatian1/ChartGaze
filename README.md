# 📊👀 ChartGaze: Eye-Tracking Guided Attention for Chart Understanding

<p align="center">
  <br>
  <a href="https://arxiv.org/pdf/2509.13282"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href="https://huggingface.co/datasets/alisalam/ChartGaze"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-yellow"></a>
</p>

This repository contains the official code for the paper: **"ChartGaze: Enhancing Chart Understanding in LVLMs with Eye-Tracking Guided Attention Refinement"**.

We introduce a novel attention refinement method and a new eye-tracking dataset to improve the performance and interpretability of Large Vision-Language Models (LVLMs) on chart question answering (CQA). Our approach aligns model attention with human gaze patterns, leading to performance gains on non-instruction-tuned models.

### Dataset Link 📊
[https://huggingface.co/datasets/alisalam/ChartGaze](https://huggingface.co/datasets/alisalam/ChartGaze)

### Paper Link
[https://arxiv.org/pdf/2509.13282](https://arxiv.org/pdf/2509.13282)

---
## Abstract
Charts are a crucial visual medium for communicating and representing information. While Large Vision-Language Models (LVLMs) have made progress on chart question answering (CQA), the task remains challenging, particularly when models attend to irrelevant regions of the chart. In this work, we present ChartGaze, a new eye-tracking dataset that captures human gaze patterns during chart reasoning tasks. Through a systematic comparison of human and model attention, we find that LVLMs often diverge from human gaze, leading to reduced interpretability and accuracy. To address this, we propose a gaze-guided attention refinement that aligns image-text attention with human fixations. Our approach improves both answer accuracy and attention alignment, yielding gains of up to 2.56 percentage points across multiple models. These results demonstrate the promise of incorporating human gaze to enhance both the reasoning quality and interpretability of chart-focused LVLMs.

---

### 📂 Repository Structure
After downloading the dataset and models, please ensure your directory is structured as follows:

- `ChartGaze/`
  - `models/`
    - `ChartGemma/`
      - `model/`
      - `finetune.py`
    - `TinyLLaVA/`
      - `my_modelling/`
        - `TinyLLaVA-OpenELM-450M-SigLIP-0.89B_wAttn/`
          - `model.safetensors`  
      - `my_scripts/`
        - `lora_finetune_human_wAttn.sh`
      - `tinyllava/`
    - `InternVL/`

  - `data/`
    - `train/`
      - `attn_maps/`
      - `images/`
      - `data.json`
    - `val/`
      - `attn_maps/`
      - `images/`
      - `data.json`

---

### 🚀 Setup and Finetuning
Follow these steps to set up the project and finetune the models:

1.  **Download Models**:
    * **TinyLLaVA**: Download the **450M model** from the official repository: [https://github.com/TinyLLaVA/TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory?tab=readme-ov-file)
    * **ChartGemma**: Download the model from its official repository: [https://github.com/vis-nlp/ChartGemma](https://github.com/vis-nlp/ChartGemma)

2.  **Download Dataset**:
    * Our dataset is available on Hugging Face: [https://huggingface.co/datasets/alisalam/ChartGaze](https://huggingface.co/datasets/alisalam/ChartGaze)
    
      Note that to make things easier, you can just download the train and val folders here: https://huggingface.co/datasets/alisalam/ChartGaze/tree/main/data
    
3.  **Organize Files**:
    * Place the downloaded models and dataset into the directory structure shown above.

4.  **Run Finetuning Scripts**:
    * To finetune **TinyLLaVA**, execute:
        ```bash
        ChartGaze/models/TinyLLaVA/my_scripts/lora_finetune_human_wAttn.sh
        ```
    * To finetune **ChartGemma**, run:
        ```bash
        ChartGaze/models/ChartGemma/finetune.py
        ```
    * To finetune **InternVL**, please take a look at the README.md file under the models/InternVL folder


---

## Citation
```
@inproceedings{
  salamatian2025chartgaze,
  title={ChartGaze: Enhancing Chart Understanding in {LVLM}s with Eye-Tracking Guided Attention Refinement},
  author={Ali Salamatian and Amirhossein Abaskohi and Wan-Cyuan Fan and Mir Rayat Imtiaz Hossain and Leonid Sigal and Giuseppe Carenini},
  booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  url={https://openreview.net/forum?id=W1fNDoL7sv}
}
```
