# pip install transformers
# pip install datasets
# pip install librosa
# pip install torch

"""파일 찾기 및 데이터프레임 정리"""
import os
import pandas as pd
import time
from tqdm import tqdm
import torch

# cuda 사용
cuda_id = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # colab에서는 쿠다사용이 안되나봄


# 폴더 경로 가져오기
PATH = os.path.dirname(os.path.abspath('ETRI 데이터셋'))
dir_name = 'ETRI 데이터셋'
dir_path = os.path.join(PATH, dir_name)

# 폴더 안에 wav파일들의 경로만 가져와서 저장하기
file_path_list = []
for path, dirs, files in os.walk(dir_path):
  #print(path)
  for file in files:
    file_path = os.path.join(path, file)
    #print(file_path)
    if file[-3:] != 'wav':
      continue
    #if file[:] not in file_name:
     # continue
    else:
      file_path_list.append(file_path)

# file name 추출
name = []
for path in file_path_list:
  name.append(path[-15:])

# emotion 추출
emotion = []
for path in file_path_list:
  if path[-8:-7] == 'n':
    emotion.append('neutral')
  elif path[-8:-7] == 'h':
    emotion.append('happy')
  elif path[-8:-7] == 's':
    emotion.append('sad')
  elif path[-8:-7] == 'a':
    emotion.append('angry')

dict = {'Name': name, 'Emotion': emotion, 'Path': file_path_list}  
df = pd.DataFrame(dict) 
df

# 감정 레이블 추가
df_emotion = df['Emotion'].replace({'neutral':0, 'happy':1, 'sad':2, 'angry':3})

# 데이터프로임으로 합치기
df = pd.concat([df, df_emotion],axis=1)
df.columns = ['file_name','emotion','path','labels']


# 감정별 몇개의 데이터 있는지 파악하기
print(df["emotion"].value_counts())

# 데이터 사용 비율(colab에서 전체 데이터 돌리면 런타임 에러 남)
percentage = 57
df = df.sample(frac=percentage/100)

# train/test 데이터 쪼개기(나중에 "from sklearn.model_selection import train_test_split"로 바꾸기)
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)

# 감정별 몇개의 데이터 있는지 파악하기
print(train_df["emotion"].value_counts())

"""모델 돌리기"""
from transformers import Wav2Vec2FeatureExtractor
from datasets import Dataset
import librosa

print(Dataset)

def map_to_array(example):
    speech, _ = librosa.load(example["path"], sr=16000, mono=True) # sr: sample rate(높을 수록 정교해짐), mono: 모노로 변경
    example["speech"] = speech
    return example

train_data = Dataset.from_pandas(train_df).map(map_to_array)
test_data = Dataset.from_pandas(test_df).map(map_to_array)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

train_encodings = feature_extractor(list(train_data["speech"]), sampling_rate=16000, padding=True, return_tensors="pt")
test_encodings = feature_extractor(list(test_data["speech"]), sampling_rate=16000, padding=True, return_tensors="pt")

# Turn data into a Dataset object
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

train_dataset = EmotionDataset(train_encodings, list(train_data["labels"]))
test_dataset = EmotionDataset(test_encodings, list(test_data["labels"]))

"""Loading the Model and Optimizer"""

from transformers import HubertForSequenceClassification
from torch.optim import AdamW

# Loading the model
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
model.to(device)

# Loading the optimizer
optim = AdamW(model.parameters(), lr=1e-5)

"""Training"""

# Prediction function
def predict(outputs):
    probabilities = torch.softmax(outputs["logits"], dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return predictions

# Training
from torch.utils.data import DataLoader

# Set the number of epoch
epoch = 9

# Start training
model.train()

train_loss = list()
train_accuracies = list()
for epoch_i in range(epoch):
    print('Epoch %s/%s' % (epoch_i + 1, epoch))
    time.sleep(0.3)

    # Get training data by DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    correct = 0
    count = 0
    epoch_loss = list()
    
    pbar = tqdm(train_loader)
    for batch in pbar:
        optim.zero_grad()
        input_ids = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optim.step()
        
        # make predictions
        predictions = predict(outputs)

        # count accuracy
        correct += predictions.eq(labels).sum().item()
        count += len(labels)
        accuracy = correct * 1.0 / count

        # show progress along with metrics
        pbar.set_postfix({
            'Loss': '{:.3f}'.format(loss.item()),
            'Accuracy': '{:.3f}'.format(accuracy)
        })
        
        # record the loss for each batch
        epoch_loss.append(loss.item())
        
    pbar.close()
    
    # record the loss and accuracy for each epoch
    train_loss += epoch_loss
    train_accuracies.append(accuracy)

# 시각화
import matplotlib.pyplot as plt
import numpy as np

# Plot Iteration vs Training Loss
plt.plot(train_loss, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iteration vs Training Loss")  
plt.legend()
plt.show()

# Plot Epoch vs Training Accuracy
acc_X = np.arange(len(train_accuracies))+1                          
plt.plot(acc_X, train_accuracies,"-", label="Training Accuracy")
plt.xticks(acc_X)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epoch vs Training Accuracy")  
plt.legend()
plt.show()

"""Testing"""

# Testing
from torch.utils.data import DataLoader

# Get test data by DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Start testing
model.eval()

with torch.no_grad():
    
    correct = 0
    count = 0
    record = {"labels":list(), "predictions":list()}
    
    pbar = tqdm(test_loader)
    for batch in pbar:
        input_ids = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        # make predictions
        predictions = predict(outputs)

        # count accuracy
        correct += predictions.eq(labels).sum().item()
        count += len(labels)
        accuracy = correct * 1.0 / count

        # show progress along with metrics
        pbar.set_postfix({
            'loss': '{:.3f}'.format(loss.item()),
            'accuracy': '{:.3f}'.format(accuracy)
        })
    
        # record the results
        record["labels"] += labels.cpu().numpy().tolist()
        record["predictions"] += predictions.cpu().numpy().tolist()
        
    pbar.close()
    
time.sleep(0.3)
print("The final accuracy on the test dataset: %s%%" % round(accuracy*100,4))

# Convert test record to a pandas DataFrame object
from pandas.core.frame import DataFrame
df_record = DataFrame(record)
df_record.columns = ["Ground Truth","Model Prediction"]

def get_emotion(label_id):
    return model.config.id2label[label_id]
    
df_record["Ground Truth"] = df_record.apply(lambda x: get_emotion(x["Ground Truth"]), axis=1)
df_record["Model Prediction"] = df_record.apply(lambda x: get_emotion(x["Model Prediction"]), axis=1)

# Concat test texts and test records
df = pd.concat([test_df.reset_index(), df_record["Model Prediction"]], axis=1)
df["emotion"] = df.apply(lambda x: x["emotion"][:3], axis=1)

# Show test result
# pd.set_option('display.max_rows', None)    # Display all rows
# df

# Show incorrect predictions 
df[df["emotion"]!=df["Model Prediction"]]

# Display the Confusion Matrix
import seaborn as sns
crosstab = pd.crosstab(df_record["Ground Truth"],df_record["Model Prediction"])
sns.heatmap(crosstab, cmap='Oranges', annot=True, fmt='g', linewidths=5)
accuracy = df_record["Ground Truth"].eq(df_record["Model Prediction"]).sum() / len(df_record["Ground Truth"])
plt.title("Confusion Matrix (Accuracy: %s%%)" % round(accuracy*100,2))
plt.show()

"""## **Save Model**"""

#학습시키고 저장할때 꼭 모델명 확인!! (덮어쓰기방지)
torch.save(model,f'ETRI_model_{percentage}_{epoch}.pt')