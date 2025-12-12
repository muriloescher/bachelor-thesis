import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct"
QWEN_MODEL = "qwen/qwen3-8b"

EXAMPLE_INPUT = "gosta\tAquele cliente gosta apenas de vinho branco ."
EXAMPLE_PREDICTION = "gostar\tV;IND;SG;3;PRS"
PROMPT = "Generate the lemma and morphological tags for the following inflected verb and context. IMPORTANT: Provide ONLY the answer in the exact format 'lemma<tab>tags' with NO explanations, NO commentary, NO additional text: "

print(f"API Key loaded: {os.environ.get('OPENROUTER_API_KEY')[:20]}..." if os.environ.get('OPENROUTER_API_KEY') else "API Key NOT loaded")

# Test with actual dataset format
ex_form, ex_context = EXAMPLE_INPUT.split('\t')
test_form = "tem"
test_context = "Ele tem um carro novo ."

prompt_content = f"Based on this example:\nInput: {ex_form}\nContext: {ex_context}\nPrediction: {EXAMPLE_PREDICTION}\n\n{PROMPT}{test_form}\t{test_context}\n\nAnswer (lemma and tags only, no explanation):"

print("\n" + "="*60)
print("TESTING WITH DATASET PROMPT FORMAT:")
print("="*60)
print(prompt_content)
print("="*60 + "\n")

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
  },
  data=json.dumps({
    "model": QWEN_MODEL,
    "messages": [
      {
        "role": "user",
        "content": prompt_content
      }
    ]
  })
)

print("Response:")
result = response.json()
print(json.dumps(result, indent=2))

if 'choices' in result and len(result['choices']) > 0:
    prediction = result['choices'][0]['message']['content'].strip()
    print(f"\nExtracted prediction: {prediction}")