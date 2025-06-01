from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model_name = "microsoft/Phi-3-mini-4k-instruct"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load LoRA adapter
adapter_model = PeftModel.from_pretrained(base_model, "/kaggle/input/fine-tuned-model")

# Merge LoRA into base model
merged_model = adapter_model.merge_and_unload()

# Save merged model and tokenizer
output_dir = "/kaggle/working/merged-phi3-model"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)  # Ensure tokenizer files are saved

print(f"Merged model saved at {output_dir}")
