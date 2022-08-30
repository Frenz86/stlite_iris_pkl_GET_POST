from fastapi import FastAPI, Request,Depends,HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse
import pandas as pd
import uvicorn
from pydantic import BaseModel
import joblib

import warnings
warnings.filterwarnings("ignore")

classes = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}

class Feature_type(BaseModel):
    #description: Optional[str] = None questo è un campo opzionale
    feature1 : float = 3.0
    feature2 : float = 3.0
    feature3 : float = 3.0
    feature4 : float = 3.0

app = FastAPI(title="API1", description="with FastAPI by Daniele Grotti", version="1.0")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup") #define event handlers (functions) that need to be executed before the application starts up
def load_model():
    global model
    model = joblib.load("iris.pkl")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
async def predict_get(data: Feature_type= Depends()):              # depends() input nelle celle
    try:
        data = pd.DataFrame(data)
        data = data.T
        data.rename(columns=data.iloc[0], inplace = True)
        data= data.iloc[1:] #must have array
        
        y_pred = list(map(lambda x: classes[x], model.predict(data).tolist()))[0]
        return JSONResponse(y_pred)
    except:
        raise HTTPException(status_code=404, detail="error")

@app.post("/predict", response_class=HTMLResponse)
#async def predict_post(data: Feature_type= Depends()):
async def predict_post(data: Feature_type):
    try:
        data = pd.DataFrame(data).T
        data.rename(columns=data.iloc[0], inplace = True)
        data= data.iloc[1:] #must have array
        y_pred = list(map(lambda x: classes[x], model.predict(data).tolist()))[0]
        return JSONResponse(y_pred)
    except:
        raise HTTPException(status_code=404, detail="error") 

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)