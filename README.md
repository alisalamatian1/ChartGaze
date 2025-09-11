# ChartGaze

- Paper link:

- Dataset link:

## Current Structure
- models folder:
    - ChartGemma
    - TinyLLaVA
        - my_modelling
        - my_scripts
        - tinyllava

After you download our dataset and the models, the structure should look like:
- models:
    - ChartGemma
        - model
    - TinyLLaVA
        - my_modelling: put the downloaded model.safetensors under TinyLLaVA-OpenELM-450M-SigLIP-0.89B_wAttn
        - my scripts
        - tinyllava
- data
    - train
        - attn_maps
        - images
        - data.json
    - val 
        - attn_maps
        - images
        - data.json

## How to set up
You need to first download each of the models. For Tinyllava refer to: https://github.com/TinyLLaVA/TinyLLaVA_Factory?tab=readme-ov-file (we used the 450M model).
For ChartGemma, refer to: https://github.com/vis-nlp/ChartGemma
Once you download the model and dataset and put them in the right location, you can use the ChartGaze/models/TinyLLaVA/my_scripts/lora_finetune_human_wAttn.sh script and ChartGaze/models/ChartGemma/finetune.py to finetune TinyLLaVA and ChartGemma on our data.