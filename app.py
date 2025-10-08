# app.py
import gradio as gr
from huggingface_hub import InferenceClient

# Recieving HF_TOKEN from environment variable for security
client = InferenceClient(token=os.getenv("HF_TOKEN")) 

def generate_story(prompt, age_group, length):
    messages = [
        {
            "role": "system",
            "content": f"You are a children's story writer creating safe, educational content for ages {age_group}."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = client.chat_completion(
        messages=messages,
        model="google/gemma-2-2b-it",
        max_tokens=length,
        temperature=0.7
    )
    
    return response.choices[0].message.content

demo = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(label="Story Prompt", placeholder="Once upon a time..."),
        gr.Radio(["3-5 years", "6-8 years", "9-12 years"], label="Age", value="6-8 years"),
        gr.Slider(100, 400, 250, label="Length")
    ],
    outputs=gr.Textbox(label="Story", lines=15),
    title="🌙 Bedtime Story Generator"
)

demo.launch(share=True)


