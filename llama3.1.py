import transformers
import torch
import gradio as gr


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def chat(question, context):
  messages = [
    {"role": "system", "content": "You are a chatbot that answers questions based on the given context. If you can't find information in the context to give appropriate answer, let the user know that you there is no info in the context. Don't add any new info that is not in the context"},
    {"role": "user", "content": f"###question: {question} ###context: {context}"},
  ]
  outputs = pipeline(
      messages,
      max_new_tokens=512,
  )
  return outputs[0]["generated_text"][-1]['content']

chat_interface = gr.Interface(
    fn=chat,
    inputs=["text", "text"],
    outputs=["text"],
)

chat_interface.launch(share=True)
