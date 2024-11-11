from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chatbot:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="./model_cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir="./model_cache",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    def generate_response(self, message: str) -> str:
        try:
            input_ids = self.tokenizer.encode(
                message + self.tokenizer.eos_token,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=100
            )

            response_ids = self.model.generate(
                input_ids,
                max_length=50,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=30,
                top_p=0.85,
                temperature=0.7,
                num_return_sequences=1
            )

            response = self.tokenizer.decode(
                response_ids[:, input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            return response or "I'm not sure how to respond to that."
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I encountered an error. Please try again."

