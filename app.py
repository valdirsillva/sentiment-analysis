from model.predict import predict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict", response_class=JSONResponse)
async def predict_sentiment(body: TextInput):
    if not body.text.strip():
        return JSONResponse(content={"error": "O texto não pode ser vazio."}, status_code=400)
    return predict(body.text)