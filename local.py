from openai import OpenAI

client = OpenAI(
  base_url='http://localhost:11434/v1',
  api_key='ollama'
)

dialog_history = []

while True:
    user_input = input("Введите ваше сообщение ('stop' для завершения): ")

    if user_input == 'stop':
        break
  
    dialog_history.append({
        'role': 'user',
        'content': user_input
    })

    response = client.chat.completions.create(
        model='gemma2:latest',
        messages=dialog_history
    )


    response_content = response_content = response.choices[0].message.content
    print("Ответ модели:", response_content)

    dialog_history.append({
        'role': 'assistant',
        'content': response_content
    })

    print("\n\n")

