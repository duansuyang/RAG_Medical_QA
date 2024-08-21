# 基于RAG的医疗大模型





下载chatglm2-6b

清华云盘下载器：https://github.com/chenyifanthu/THU-Cloud-Downloader
```
python thu_cloud_download.py \
    -l https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/  \
    -s "/xxx/chatglm2" \
    -f "chatglm2-6b/*"
```


my_model ： 训练好的bert模型

train_similar_model： 准备工作，组建向量数据库

    训练：train_recall_model.py
    
    此文件用于训练基于bert的dssm
    
    两个Bert模型先只放开最后一层，不够用再放开更多
    
    DSSM模型(双塔后计算余弦相似度过Sigmoid)
    
    保存时只存Bert模型，不保存dssm
    
    insert_knowledge.py
    
    应用bert进行tokenizer，将所有key生成的向量放入faiss索引库


对话：chat.py

    对于输入的问题，首先对query使用大模型进行多样化得到q1,q2,q3；然后分别提取向量后在向量数据库中进行召回得到更多相似的句子（加入关键词贡献进行过滤）。
    
    对召回后的句子使用二分类bert做相似度计算(精排)，过滤得分低的句子，再对相同句子进行累加去重。