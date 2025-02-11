import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_argparse():
    parser = argparse.ArgumentParser(description='Language Model Inference Script')
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--num_generations', type=int, default=3, help='Number of generations per prompt'
    )
    parser.add_argument(
        '--max_length', type=int, default=100, help='Maximum length of generated text'
    )
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    return parser


def load_model(model_path):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto"
    )
    return tokenizer, model


def generate_text(model, tokenizer, prompt, num_generations, max_length, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    results = []
    for i in range(num_generations):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            results.append(generated_text)

    return results


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)

    test_prompts = [
        "Hello",
        "Tokyo is",
        "Artificial intelligence will",
        "In the future",
        "Japanese culture",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)

        generations = generate_text(
            model,
            tokenizer,
            prompt,
            args.num_generations,
            args.max_length,
            args.temperature,
            args.top_p,
        )

        for i, text in enumerate(generations, 1):
            print(f"\nGeneration {i}:")
            print(text)
            print("-" * 30)


if __name__ == "__main__":
    main()
