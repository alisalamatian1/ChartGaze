---
license: apache-2.0
pipeline_tag: image-text-to-text
---
**<center><span style="font-size:2em;">TinyLLaVA</span></center>**

[![arXiv](https://img.shields.io/badge/Arxiv-2402.14289-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.14289)[![Github](https://img.shields.io/badge/Github-Github-blue.svg)](https://github.com/TinyLLaVA/TinyLLaVA_Factory)[![Demo](https://img.shields.io/badge/Demo-Demo-red.svg)](http://8843843nmph5.vicp.fun/#/)
TinyLLaVA has released a family of small-scale Large Multimodel Models(LMMs), ranging from 0.55B to 3.1B. Our best model, TinyLLaVA-Phi-2-SigLIP-3.1B, achieves better overall performance against existing 7B models such as LLaVA-1.5 and Qwen-VL.
### TinyLLaVA
Here, we introduce TinyLLaVA-OpenELM-450M-SigLIP-0.89B, which is trained by the [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) codebase. For LLM and vision tower, we choose [OpenELM-450M-Instruct](apple/OpenELM-450M-Instruct) and [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384), respectively. The dataset used for training this model is the The dataset used for training this model is the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) dataset.

### Usage
Execute the following test code:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
hf_path = 'jiajunlong/TinyLLaVA-OpenELM-450M-SigLIP-0.89B'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
model.cuda()
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
prompt="What are these?"
image_url="http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
output_text, genertaion_time = model.chat(prompt=prompt, image=image_url, tokenizer=tokenizer)
print('model output:', output_text)
print('runing time:', genertaion_time)
```
### Result

|                          model_name                          | gqa   | textvqa | sqa   | vqav2 | MME     | MMB   | MM-VET |
| :----------------------------------------------------------: | ----- | ------- | ----- | ----- | ------- | ----- | ------ |
| [TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B) | 60.3  | 51.7    | 60.3  | 76.9  | 1276.5  | 55.2  | 25.8   |
| [TinyLLaVA-0.89B](https://huggingface.co/jiajunlong/TinyLLaVA-OpenELM-450M-SigLIP-0.89B) | 53.87 | 44.02   | 54.09 | 71.74 | 1118.75 | 37.8  | 20     |

P.S. [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) is an open-source modular codebase for small-scale LMMs with a focus on simplicity of code implementations, extensibility of new features, and reproducibility of training results. This code repository provides standard training&evaluating pipelines, flexible data preprocessing&model configurations, and easily extensible architectures. Users can customize their own LMMs with minimal coding effort and less coding mistake.
TinyLLaVA Factory integrates a suite of cutting-edge models and methods. 
  - LLM currently supports OpenELM, TinyLlama, StableLM, Qwen, Gemma, and Phi. 
  - Vision tower currently supports CLIP, SigLIP, Dino, and combination of CLIP and Dino.
  - Connector currently supports MLP, Qformer, and Resampler.

