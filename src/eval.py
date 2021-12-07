#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

from tqdm import tqdm


# In[2]:


reference_data = pd.read_csv('./VQA-Med-2021/Task1-VQA-2021-TestSet-w-GroundTruth/Task1-VQA-2021-TestSet-ReferenceAnswers.txt',sep='|', names=['imageid', 'ans1', 'ans2', 'ans3'])


# In[3]:


predictions = pd.read_csv('./vgg16_fusion-MLB.txt', sep='|', names=['imageids', 'answers'])
predictions.head()


# In[4]:


correct = 0
gt_answers = []
pred_answers = []

for en, idx in enumerate(predictions.imageids.tolist()):
    tmp_ = reference_data[reference_data.imageid==idx].reset_index()
    assert len(tmp_)==1
    
    pred_answers.append(predictions.answers[en].strip())
    if predictions.answers[en].strip()==tmp_.ans1[0]:
        gt_answers.append(tmp_.ans1[0])
    elif predictions.answers[en].strip()==tmp_.ans2[0]:
        gt_answers.append(tmp_.ans2[0])
    elif predictions.answers[en].strip()==tmp_.ans3[0]:
        gt_answers.append(tmp_.ans3[0])
    else:
        gt_answers.append(tmp_.ans1[0])
    if predictions.answers[en].strip()==tmp_.ans1[0] or predictions.answers[en]==tmp_.ans2[0] or predictions.answers[en]==tmp_.ans3[0]:
        correct+=1

print("Test accuracy:", correct*100/len(predictions.imageids))


# In[5]:


from sklearn.metrics import f1_score
f1_score(gt_answers, pred_answers, average='micro')


# In[6]:


bleu_score = []

smoothie = SmoothingFunction().method0
stops = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
    

    
for en, idx in tqdm(enumerate(predictions.imageids)):
    tmp_ = reference_data[reference_data.imageid==idx].reset_index()
    assert len(tmp_)==1
    
    
    
    candidate = [stemmer.stem(w.lower()) for w in word_tokenize(str(predictions.answers[en])) if w not in stops]
    w_ = [str(tmp_.ans1[0]), str(tmp_.ans2[0]), str(tmp_.ans3[0])]
    
    reference = []
    for tw in w_:
        reference.append([stemmer.stem(w.lower()) for w in word_tokenize(str(tw)) if w not in stops])
    
    if len(candidate)==0 and len(reference[0])==0:
        tmp_score = 1
    else:
        tmp_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    bleu_score.append(tmp_score)
        
print("Bleu score:", np.mean(bleu_score))


# In[ ]:





# In[ ]:




