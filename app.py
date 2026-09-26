from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Summariser", description="Text Summariser using T5", version=1.0)

#tokenisation & model
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

#device
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
model.to(device)

#Templating
templates = Jinja2Templates(directory=".")

#Input
class dialogueInput(BaseModel):
    dialogue:str

#clean data
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

#model
def summarize_dialogue(dialogue:str):
    dialogue = clean_data(dialogue) #clean

    #tokenise
    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    #generate the summary
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )

    #tokens--> text decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)

    return summary

@app.post("/summarize/")
async def summarize(dialogue_input: dialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary":summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

