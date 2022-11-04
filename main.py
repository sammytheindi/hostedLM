from fastapi import FastAPI,Request
# from pydantic import BaseModel
import uvicorn
import numpy as np
# from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch

app = FastAPI()

def get_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    return tokenizer,model

tokenizer,model = get_model()

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    user_input = data['userInput']
    current_sentence = data['currentSentence']
    
    inputs = tokenizer(current_sentence, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_token_id = torch.topk(logits[0, mask_token_index], 30000, dim=-1).indices
    predicted_list = list(map(lambda x: tokenizer.decode(x), predicted_token_id))[0].split(" ")
    
    
    predicted_list = map(lambda x: x.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("/", "").replace(":", "").replace(";", ""), predicted_list)
    predicted_list = list(filter(lambda x: (len(x) > 1) and (not x.isnumeric()) and ("##" not in x), predicted_list))

    # entered_char = 'of'
    return {"predictions": list(filter(lambda x: x[:len(user_input)] == user_input, predicted_list))[:10]}

if __name__ == "__main__":
    uvicorn.run("main:app",host='0.0.0.0', port=8080, reload=True)
