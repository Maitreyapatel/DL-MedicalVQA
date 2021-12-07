#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[2]:


len(os.listdir('./dataset/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_images/'))


# In[3]:


len(os.listdir('./dataset/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_images/'))


# In[6]:


get_ipython().system('cp ./dataset/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_images/* ./dataset/TrainingDataset/images/')


# In[7]:


get_ipython().system('cp ./dataset/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_images/* ./dataset/TrainingDataset/images/')


# In[8]:


len(os.listdir('./dataset/TrainingDataset/images/'))


# In[2]:


data1 = pd.read_csv('./dataset/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_QA_pairs.txt', sep='|', names=['imageid', 'question', 'answer'])


# In[3]:


data2 = pd.read_csv('./dataset/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_QA_Pairs.txt', sep='|', names=['imageid', 'question', 'answer'])


# In[4]:


data=pd.concat([data1, data2])


# In[5]:


data.shape


# In[8]:


data.to_csv('./dataset/TrainingDataset/training_qa.txt', sep='|', index=False, header=False)


# In[7]:


data.head()


# In[ ]:




