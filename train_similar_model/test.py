from transformers import BertTokenizer, BertModel
import torch

model_path="my_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
sentences = ['2009年11月发电机汽油中毒,当时没出事一直到2010年正月开始发现大小便失控.神智不清,记忆力下降,语言表达差,不认识人.当时住院开始吸高压氧,输液治疗.到现在已有20多天,不见疗效.想到你们医院治疗.但我们是第一次到你们医院,我想知道一下这个病会不会留下后遗症.我好怕']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling.
sentence_embeddings = model_output#[1]
print("Sentence embeddings:")
print(sentence_embeddings)
