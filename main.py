import os
import json
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
app.secret_key = 'your_secret_key'

openai_client = OpenAI(api_key=os.environ.get("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"))

mistral_client = MistralClient(api_key="4ifHJVbIpIoMXBaGQOMH2SWwlaQBIjPP")

tokenizer = BertTokenizer.from_pretrained('Fine-tuned BERT')
model = BertForSequenceClassification.from_pretrained('Fine-tuned BERT')

device = torch.device("cpu")
model.to(device)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json

    if 'user_input' not in data or 'backend' not in data:
        return jsonify({"error": "Missing 'user_input' or 'backend' key in JSON payload"}), 400

    statement = data['user_input']
    backend = data['backend']

    format = {"probabilities": {"credible": int, "misleading": int}, "commentary": str}
    message_content = f"""Classify the following health-related text into one of two categories: 'Credible', 'Misleading'. 
    Always provide the answer formatted as {format}. 
    Don't quote the statement in the commentary. 
    You must enclose the commentary string in single quotation marks, but never use any quotation marks inside the commentary string.
    Probabilities must be given in range 0-100.
    If the statement is not health-related, return 0 for both classes and the following commentary:
    "The provided text is not health-related."
    If the statement contains personal data (name, surname, birthdate, residence, id), extend the commentary by the following remark:
    "\n(your personal information has been anonymised on the server)"
    Statement = {statement}"""

    if backend == 'gpt4':
        chat_completion = openai_client.chat.completions.create(messages=[{"role": "user", "content": message_content}], model="gpt-4o-2024-05-13")
        generated_text = chat_completion.choices[0].message.content
    elif backend == 'mistral':
        messages = [ChatMessage(role="user", content=message_content)]
        chat_response = mistral_client.chat(model="mistral-large-latest", messages=messages)
        generated_text = chat_response.choices[0].message.content
    elif backend == 'BERT fine-tuned':

        encoded_dict = tokenizer.encode_plus(
            statement,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_label = torch.argmax(logits, dim=1).cpu().numpy()[0]

        if predicted_label == 0:
            label = 'credible'
        else:
            label = 'misleading'

        result_dict = {
            "probabilities": {
                "credible": int(probabilities[0] * 100),
                "misleading": int((probabilities[1] + probabilities[2] + probabilities[3]) * 100)
            },
            "commentary": "No commentary is available for the chosen backend."
        }
        generated_text = result_dict
    else:
        return jsonify({"error": "Invalid backend selection"}), 400

    if backend == 'gpt4' or backend == 'mistral':
        result_dict = json.loads(generated_text.replace("'", '"'))
        print(f"Result from {backend}: {result_dict}")
        return jsonify(result_dict)

    elif backend == 'BERT fine-tuned':
        result_dict = generated_text
        print(f"Result from {backend}: {result_dict}")
        return jsonify(result_dict)


if __name__ == '__main__':
    app.run(debug=True)
