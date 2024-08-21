from transformers import BertTokenizer, BertModel
import torch
import random
model_path="model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
def get_vector(sentence):
    encoded_input = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt', max_length=512)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # print (model_output)
    # asd
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = model_output[1]
    print (sentence_embeddings)
    return sentence_embeddings

def normal(vector):
    vector=vector.tolist()[0]
    ss=sum([s**2 for s in vector])**0.5
    return [s/ss for s in vector]
with open("train_data",encoding="utf-8") as f:
    lines=[eval(s.strip())  for s in f.readlines()]
random.shuffle(lines)
for s1,s2,target in lines:

    v1=normal(get_vector(s1))
    v2=normal(get_vector(s2))
   
    #print (len(v1),len(v2))
    score=sum([a*b for a,b in zip(v1,v2)])
    print (score,target)
 