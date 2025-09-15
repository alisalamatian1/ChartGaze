# 📊👀 ChartGaze: Eye-Tracking Guided Attention for Chart Understanding

This repository contains the official code for the paper: **"ChartGaze: Enhancing Chart Understanding in LVLMs with Eye-Tracking Guided Attention Refinement"**.

We introduce a novel attention refinement method and a new eye-tracking dataset to improve the performance and interpretability of Large Vision-Language Models (LVLMs) on chart question answering (CQA). Our approach aligns model attention with human gaze patterns, leading to performance gains on non-instruction-tuned models.

| Paper Link 📄 | Dataset Link 📊 |
| :--- | :--- |
| [TODO]| [https://huggingface.co/datasets/alisalam/ChartGaze](https://huggingface.co/datasets/alisalam/ChartGaze) |

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