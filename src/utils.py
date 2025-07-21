import openai 
import os 
import json

class OpenaiLLM:
    def __init__(self, gpt_model: str):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = gpt_model

    def generate_response(self, messages, new_token_limit, temperature):
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
			temperature=temperature,
			max_tokens=new_token_limit,
			top_p=0.95,
			frequency_penalty=0,
			presence_penalty=0,
			stop=None,
        )
		
        return chat_completion.choices[0].message.content
    

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def write_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_jsonlines(path):
    with open(path) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines if line.strip() != '']

def write_jsonlines(path, data):
    with open(path, "w") as f:
        for i, item in enumerate(data):
            item_str = json.dumps(item, ensure_ascii=False)
            f.write("\n" * (i > 0) + item_str)

