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
                self.model_name,
                padding_side='left'
            )
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )
            print("Model loaded successfully!")
            
            # Initialize conversation history
            self.conversation_history = []
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}", file=sys.stderr)
            raise

    def generate_response(self, message: str) -> str:
        try:
            print(f"Generating response for: {message}")
            
            # Encode the new user input
            new_input_ids = self.tokenizer.encode(
                message + self.tokenizer.eos_token,
                return_tensors='pt'
            )

            # Append to conversation history if it exists
            if self.conversation_history:
                bot_input_ids = torch.cat([self.conversation_history, new_input_ids], dim=-1)
            else:
                bot_input_ids = new_input_ids

            # Generate response
            response_ids = self.model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8,
                num_return_sequences=1
            )

            # Update conversation history
            self.conversation_history = response_ids

            # Decode and return the response
            response = self.tokenizer.decode(
                response_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            
            print(f"Generated response: {response}")
            return response or "I'm not sure how to respond to that."
            
        except Exception as e:
            print(f"Error generating response: {str(e)}", file=sys.stderr)
            return "I'm having trouble understanding. Could you rephrase that?"
