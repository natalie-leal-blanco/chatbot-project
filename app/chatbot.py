from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chatbot:
    def __init__(self):
        print("Loading model...")
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        )
        print("Model loaded!")

    def generate_response(self, message: str) -> str:
        try:
            inputs = self.tokenizer.encode(
                message + self.tokenizer.eos_token,
                return_tensors='pt'
            )
            
            outputs = self.model.generate(
                inputs,
                max_length=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(
                outputs[:, inputs.shape[-1]:][0],
                skip_special_tokens=True
            )
--simplify            
            return response or "I'm not sure how to respond to that."
        except Exception as e:
            print(f"Error: {str(e)}")
            return "I encountered an error. Please try again."
