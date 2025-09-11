import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
import os, random, numpy as np, torch

from tinyllava.train.tinyllava_trainer import LLaVATrainer, CustomCallback
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # force deterministic CuDNN; may slow you down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # newer PyTorch: catch nondet ops
    torch.use_deterministic_algorithms(True, warn_only=True)

def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    set_all_seeds(training_arguments.seed)
    
    logger_setting(getattr(training_arguments, 'output_dir', None))
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    load_settings(model_arguments, data_arguments, training_arguments)
    # load pretrained checkpoint
    model = AutoModelForCausalLM.from_pretrained(training_arguments.pretrained_model_path, trust_remote_code=True)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(training_arguments.pretrained_model_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
    model.tokenizer = tokenizer
    model = training_recipe(model)
    print("Custom Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path)
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    # trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
    #                        tokenizer=tokenizer,
    #                        args=training_arguments,
    #                        callbacks=[CustomCallback(model, tokenizer, data_arguments.image_processor)],
    #                        **data_module)
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
