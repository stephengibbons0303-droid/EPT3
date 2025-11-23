from openai import OpenAI

def call_llm(messages, api_key, model="gpt-4-turbo-preview", max_tokens=4096):
    """
    Sends a message history to the OpenAI API using the provided API key.
    Increased max_tokens to 4096 to support batch generation of multiple questions.
    Temperature increased to 0.9 for better diversity across questions.
    """
    if not api_key:
        return "Error: API Key is missing. Please enter it in the sidebar."

    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": messages[0]},
                {"role": "user", "content": messages[1]}
            ],
            temperature=0.9,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"
