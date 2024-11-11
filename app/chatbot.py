from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

class Chatbot:
    def __init__(self):
        print("Initializing chatbot...")
        try:
            self.model_name = "microsoft/DialoGPT-small"
            print(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}", file=sys.stderr)
            raise

    def generate_response(self, message: str) -> str:
        try:
            print(f"Generating response for: {message}")
            
            # Encode the input
            input_ids = self.tokenizer.encode(
                message + self.tokenizer.eos_token,
                return_tensors='pt'
            )

            # Generate response
            output_ids = self.model.generate(
                input_ids,
                max_length=100,  # Reduced for faster responses
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.6,  # Reduced temperature for more focused responses
                num_return_sequences=1,
                clean_up_tokenization_spaces=True
            )

            # Decode the response
            response = self.tokenizer.decode(
                output_ids[:, input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            
            print(f"Generated response: {response}")
            
            # Check if response is empty or too short
            if not response or len(response.strip()) < 2:
                return "Let me think about that for a moment..."
                
            return response
            
        except Exception as e:
            print(f"Error in response generation: {str(e)}", file=sys.stderr)
            return "I'm having trouble right now. Could you try again?"
