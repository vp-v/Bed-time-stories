# app.py
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
        max_tokens=length if length >= 600 else 600,
        temperature=0.9,
        top_p=0.95,
        stop=None
    ) 

    return response.choices[0].message.content.strip()

demo = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(label="Story Prompt", placeholder="Once upon a time..."),
        gr.Radio(["3-5 years", "6-8 years", "9-12 years"], label="Age", value="6-8 years"),
        gr.Slider(minimum=400, maximum=800, step=50, value=600, label="Length (max tokens)")
    ],
    outputs=gr.Textbox(label="Story", lines=15),
    title="Bedtime Story Generator"
)

demo.launch(share=True)
