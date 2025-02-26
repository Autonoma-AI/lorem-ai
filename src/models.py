import ollama

def call_ollama(system: str, user: str) -> str:
    response = ollama.chat(
        # model='llama3.2',
        model='llama3.2-vision',
        # model='deepseek-r1:8b',
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        format='json',
        options={"temperature": 0}
    )

    return response['message']['content']

