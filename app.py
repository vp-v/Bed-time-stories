import gradio as gr
import os
from huggingface_hub import InferenceClient

# Recieving HF_TOKEN from environment variable for security
client = InferenceClient(token=os.getenv("HF_TOKEN")) 

def generate_story(prompt, age_group, length):
    system_message = (
        f"You are a creative storyteller writing safe, complete bedtime stories for children aged {age_group}. "
        "Write a story with a clear beginning, middle, and satisfying ending."
    )
    user_message = f"Write a detailed bedtime story with a beginning, middle, and end starting with: {prompt}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = client.chat_completion(
        model="google/gemma-2-2b-it",
        messages=messages,
        max_tokens= max(length, 750),
        temperature=0.8,
        top_p=0.95,
        stop=None
    ) 

    return response.choices[0].message.content.strip()

demo = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(label="Story Prompt", placeholder="Once upon a time..."),
        gr.Radio(["3-5 years", "6-8 years", "9-12 years"], label="Age", value="6-8 years"),
        gr.Radio([600, 800], label="Story Length (max tokens)", value=800)
    ],
    outputs=gr.Textbox(label="Story", lines=20),
    title="Bedtime Story Generator"
)

demo.launch(share=True)
