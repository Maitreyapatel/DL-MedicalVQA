#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import numpy as np
import sys
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from scipy.io import loadmat
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


import pickle
from tqdm import tqdm
import json
import random
from collections import Counter

from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel


# In[4]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import itertools

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

from torch.utils.tensorboard import SummaryWriter


# In[5]:


import transformers
import tokenizers
from transformers import BertTokenizer, BertModel


# In[6]:


cfg = {
    'max_len': 128,
    'lr': 2e-5,
    'warmup_steps': 5,
    'epochs': 250
}


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


class medical_dataset(Dataset):
    def __init__(self, config=None, answer_map=None, image_path='./dataset/TrainingDataset/images/', qa_file="./dataset/TrainingDataset/training_qa.txt", train=True):
            
        assert answer_map!=None
        assert config!=None
        
        self.image_path = image_path
        self.qa_file = qa_file
        self.config = config
        self.train = train
        
        self.answer_map = answer_map
        self.data = pd.read_csv(qa_file, sep='|', names=['imageid', 'question', 'answer'])
           
        # print(Counter(self.data.answer.tolist()))
        
        self.transforms = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        question = self.data.question[index]
        answer = self.data.answer[index]
        image_idx = self.data.imageid[index]
        
        ## add visual feature related steps
        visuals = Image.open(join(self.image_path, f'{image_idx}.jpg'))
        visuals = self.transforms(visuals)
        
        target = torch.from_numpy(np.array([self.answer_map[answer]])).long()
        
        tmp = self.process_data(question, self.config['max_len'])
        question_tokens = {
            'ids': tmp['ids'],
            'mask': tmp['mask'],
        }
        
        return visuals, question_tokens, target

    def __len__(self):
        return len(self.data)


# In[8]:


# file_path = './dataset/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_QA_pairs.txt'
# data = pd.read_csv(file_path, sep='|', names=['imageid', 'question', 'answer'])
# data.head()


# In[9]:


# answer_map = {}
# ct = 0

# for i in data.answer.unique():
#     answer_map[i] = ct
#     ct+=1


# In[10]:


# with open('./answer_map.pickle', 'wb') as handle:
#     pickle.dump(answer_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


# In[11]:


with open('./answer_map.pickle', 'rb') as handle:
    answer_map = pickle.load(handle)
    
len(answer_map)


# In[12]:


TrainData = medical_dataset(config=cfg, answer_map=answer_map, image_path='./dataset/TrainingDataset/images/', qa_file='./dataset/TrainingDataset/training_qa.txt')
ValData = medical_dataset(config=cfg, answer_map=answer_map, image_path='./dataset/VQA-Med-2021-Tasks-1-2-NewValidationSets/ImageCLEF-2021-VQA-Med-New-Validation-Images/', qa_file='./dataset/VQA-Med-2021-Tasks-1-2-NewValidationSets/VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt', train=False)


# In[13]:


get_ipython().run_cell_magic('time', '', 'a,b,c = ValData.__getitem__(0)')


# In[14]:


TrainDataLoader = DataLoader(TrainData, batch_size=16, shuffle=True, num_workers=4)  # num_workers=0 for windows OS
ValDataLoader = DataLoader(ValData, batch_size=128, shuffle=False, num_workers=4)  # num_workers=0 for windows OS


# In[15]:


class MinhmulFusion(nn.Module):

    def __init__(self, dim_v=None, dim_q=None, dim_h=512):
        super(MinhmulFusion, self).__init__()
        # Modules
        self.dim_v = dim_v
        self.dim_q = dim_q
        self.dim_h = dim_h
        self.dropout = 0
        self.activation = 'relu'
        
        if dim_v:
            self.linear_v = nn.Linear(dim_v, dim_h)
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if dim_q:
            self.linear_q = nn.Linear(dim_q, dim_h)
        else:
            print('Warning fusion.py: no question embedding before fusion')
        
    def forward(self, input_v, input_q):
        # visual (cnn features)
        if self.dim_v:
            x_v = F.dropout(input_v, p=self.dropout, training=self.training)
            x_v = self.linear_v(x_v)
            x_v = getattr(F, self.activation)(x_v)
        else:
            x_v = input_v
        # question (rnn features)
        if self.dim_q:
            x_q = F.dropout(input_q, p=self.dropout, training=self.training)
            x_q = self.linear_q(x_q)
            x_q = getattr(F, self.activation)(x_q)
        else:
            x_q = input_q
        # hadamard product
        # x_mm = torch.mul(x_q, x_q)
        x_q = torch.pow(x_q, 2)
        x_v = torch.pow(x_v, 1)

        x_mm = torch.mul(x_q, x_v)
        # x_mm = torch.mul(x_q, x_q)
        # x_mm = torch.mul(x_mm, x_v)
        return x_mm


# In[16]:


class mednet(nn.Module):
    def __init__(self, config, max_labels):
        super(mednet, self).__init__()
        self.vision = models.vgg16(pretrained=True)
        num_ftrs = self.vision.classifier[-1].in_features
        
        self.vision.classifier = nn.Sequential(*list(self.vision.classifier.children())[:-1])
        self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        
        print(f"Number of vision filters: {num_ftrs}")
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, max_labels)
        
        self.fusion = MinhmulFusion(dim_v=num_ftrs, dim_q=768, dim_h=512)
        

    def forward(self, visual=None, ids=None, mask=None):
        vision = self.vision(visual).view((ids.shape[0], -1))
        bert_out = self.bert(ids, mask)
        
        h = self.fusion(vision, bert_out.last_hidden_state[:,0])
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h


# In[17]:


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,
                 patience=7,
                 verbose=False,
                 delta=0,
                 path='./models/vgg16_fusion/checkpoint.pt',
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# In[18]:


class Trainer:
    def __init__(self,
                 trainloader,
                 vallaoder,
                 model_ft,
                 writer=None,
                 testloader=None,
                 checkpoint_path=None,
                 patience=10,
                 feature_extract=True,
                 print_itr=50,
                 config=None):
        self.trainloader = trainloader
        self.valloader = vallaoder
        self.testloader = testloader
        
        self.config=config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        print("==" * 10)
        print("Training will be done on ", self.device)
        print("==" * 10)

        self.model = model_ft        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids=[0,1])
        
        self.model = self.model.to(self.device)
        
        
        # Observe that all parameters are being optimized
        self.optimizer = optim.RAdam(self.model.parameters(), lr=self.config['lr'])
        
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps = len(self.trainloader)*self.config['warmup_steps'], # Default value in run_glue.py
                                            num_training_steps = len(self.trainloader)*self.config['epochs'])

        
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.writer = writer
        self.print_itr = print_itr

    def train(self, ep):
        self.model.train()

        running_loss = 0.0

        for en, (visuals, question, target) in tqdm(enumerate(self.trainloader)):
            self.optimizer.zero_grad()
            
            visuals = visuals.to(self.device)
            y = target.squeeze().to(self.device)
            
            ids = question['ids'].to(self.device)
            mask = question['mask'].to(self.device)

            outputs = self.model(visuals, ids, mask)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            if self.writer:
                self.writer.add_scalar('Train Loss', running_loss, ep*len(self.trainloader) + en)
            running_loss = 0
            

    def validate(self, ep):
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for en, (visuals, question, target) in tqdm(enumerate(self.valloader)):
                visuals = visuals.to(self.device)
                y = target.squeeze().to(self.device)
                
                ids = question['ids'].to(self.device)
                mask = question['mask'].to(self.device)
                
                outputs = self.model(visuals, ids, mask)
                loss = self.criterion(outputs, y)

                y_pred_softmax = torch.log_softmax(outputs, dim = 1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                # self.tmp_sv_ = y_pred_tags
                
                correct += (y_pred_tags.detach().cpu().data.numpy() == y.detach().cpu().data.numpy()).sum()
                total += y_pred_tags.shape[0]
                
                # print statistics
                running_loss += loss.item()
        
        
        return running_loss / len(self.valloader), correct*100/total

    def perform_training(self, total_epoch):
        val_loss, acc = self.validate(0)

        print("[Initial Validation results] Loss: {} \t Acc: {}".format(
            val_loss, acc))

        for i in range(total_epoch):
            self.train(i + 1)
            val_loss, acc = self.validate(i + 1)
            print('[{}/{}] Loss: {} \t Acc: {}'.format(i+1, total_epoch, val_loss, acc))

            if self.writer:
                self.writer.add_scalar('Validation Loss', val_loss, (i + 1))
                self.writer.add_scalar('Validation Acc', acc, (i + 1))

            self.early_stopping(-acc, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        print("=" * 20)
        print("Training finished !!")
        print("=" * 20)


# In[19]:


model_ft = mednet(config=cfg, max_labels=len(answer_map))
writer = SummaryWriter('runs/vgg16_fusion_run_1')
trainer = Trainer(TrainDataLoader, ValDataLoader, model_ft, writer=writer, config=cfg)


# In[20]:


trainer.perform_training(cfg['epochs'])


# In[ ]:





# In[ ]:





# In[21]:


class medical_dataset_test(Dataset):
    def __init__(self, config=None, answer_map=None, image_path='./dataset/Task1-VQA-2021-TestSet-w-GroundTruth/VQA-500-Images/', qa_file="./dataset/Task1-VQA-2021-TestSet-w-GroundTruth/Task1-VQA-2021-TestSet-Questions.txt", train=True):
            
        assert answer_map!=None
        assert config!=None
        
        self.image_path = image_path
        self.qa_file = qa_file
        self.config = config
        self.train = train
        
        self.answer_map = answer_map
        self.data = pd.read_csv(qa_file, sep='|', names=['imageid', 'question'])
           
        # print(Counter(self.data.answer.tolist()))
        
        self.transforms = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        question = self.data.question[index]
        image_idx = self.data.imageid[index]
        
        ## add visual feature related steps
        visuals = Image.open(join(self.image_path, f'{image_idx}.jpg'))
        visuals = self.transforms(visuals)
        
        
        tmp = self.process_data(question, self.config['max_len'])
        question_tokens = {
            'ids': tmp['ids'],
            'mask': tmp['mask'],
        }
        
        return visuals, question_tokens, image_idx

    def __len__(self):
        return len(self.data)


# In[ ]:





# In[ ]:





# In[22]:


TestData = medical_dataset_test(config=cfg, answer_map=answer_map)
TestDataLoader = DataLoader(TestData, batch_size=128, shuffle=False, num_workers=4)  # num_workers=0 for windows OS


# In[23]:


inv_map = {v: k for k, v in answer_map.items()}


# In[24]:


trainer.model.load_state_dict(torch.load('./models/vgg16_fusion/checkpoint.pt'))
trainer.model.eval()
        
imageids = []
answers = []

running_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for en, (visuals, question, target) in tqdm(enumerate(TestDataLoader)):
        visuals = visuals.to(trainer.device)

        ids = question['ids'].to(trainer.device)
        mask = question['mask'].to(trainer.device)

        outputs = trainer.model(visuals, ids, mask)

        y_pred_softmax = torch.log_softmax(outputs, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        # self.tmp_sv_ = y_pred_tags
        
        for i in range(visuals.shape[0]):
            imageids.append(target[i])
            answers.append(inv_map[int(y_pred_tags[i])])


# In[25]:


answers[:5]


# In[26]:


pd.DataFrame({'imageids':imageids, 'answers':answers}).to_csv('vgg16_fusion.txt', sep='|', index=False, header=False)


# In[ ]:





# In[ ]:




