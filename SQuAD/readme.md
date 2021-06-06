# Information Retrieval Approach to Question Answering for Stanford's Question Answering Dataset



# Creating Sentence Embeddings for SQuAD


```python
import pandas as pd
```

## Converting Json to Pandas DataFrame for better Visualization


```python
X = pd.read_json("data/train-v1.1.json")
```


```python
X.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'title': 'University_of_Notre_Dame', 'paragra...</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'title': 'Beyoncé', 'paragraphs': [{'qas': [{...</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'title': 'Montana', 'paragraphs': [{'qas': [{...</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'title': 'Genocide', 'paragraphs': [{'qas': [...</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'title': 'Antibiotics', 'paragraphs': [{'qas'...</td>
      <td>1.1</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the SQuAD dataset is structured and close-domained i.e. it only contains selected question answer pairs.
Context is the paragraph from the article within which the answer is located. This text is under the key 'paragraphs'. Every Context contains a set of questions and their respective answers that are placed under the key 'answers'. The dataset also provides the index of the character where the answer to the question starts. Let's go ahead and convert this information into an excel sheet for better and easier data handling.


```python
X.iloc[6,0]['paragraphs'][56]
```




    {'context': 'P. Christiaan Klieger, an anthropologist and scholar of the California Academy of Sciences in San Francisco, writes that the vice royalty of the Sakya regime installed by the Mongols established a patron and priest relationship between Tibetans and Mongol converts to Tibetan Buddhism. According to him, the Tibetan lamas and Mongol khans upheld a "mutual role of religious prelate and secular patron," respectively. He adds that "Although agreements were made between Tibetan leaders and Mongol khans, Ming and Qing emperors, it was the Republic of China and its Communist successors that assumed the former imperial tributaries and subject states as integral parts of the Chinese nation-state."',
     'qas': [{'answers': [{'answer_start': 304,
         'text': 'the Tibetan lamas and Mongol khans'}],
       'id': '56ce2752aab44d1400b884d2',
       'question': 'Who does P. Christiaan Klieger claim to have had a mutual role of religious prelate?'},
      {'answers': [{'answer_start': 534,
         'text': 'the Republic of China and its Communist successors'}],
       'id': '56ce2752aab44d1400b884d3',
       'question': 'Who does P. Christiaan Klieger believe undertook the former imperial tributaries?'},
      {'answers': [{'answer_start': 56,
         'text': 'the California Academy of Sciences in San Francisco'}],
       'id': '56ce2752aab44d1400b884d4',
       'question': 'Where does P. Christiaan Klieger work?'},
      {'answers': [{'answer_start': 171, 'text': 'the Mongols'}],
       'id': '56ce2752aab44d1400b884d5',
       'question': 'Who was the  vice royalty of the Sakya regime established by?'},
      {'answers': [{'answer_start': 197,
         'text': 'patron and priest relationship'}],
       'id': '56ce2752aab44d1400b884d6',
       'question': 'The Sakya regime established what kind of relationship between the Tibetans and Mongol converts?'}]}




```python
contexts = []
questions = []
answer_texts = []
starts_at = []
```


```python
for data in range(X.shape[0]):
    title = X.iloc[data,0]['paragraphs']
    for subtitle in title:
        for qa in subtitle['qas']:
            questions.append(qa['question'])
            starts_at.append(qa['answers'][0]['answer_start'])
            answer_texts.append(qa['answers'][0]['text'])
            contexts.append(subtitle['context'])
```

Create a single Dataframe with all data


```python
df = pd.DataFrame({"context":contexts, "question": questions, "answer_start": starts_at, "text": answer_texts})
```


```python
df.to_csv("data/train.csv", index = None)
```


```python
df.shape
```




    (87599, 4)




## Using InferSent for creating embeddings and dumping data dictionary to pickle


```python
uniquecontext = list(df['context'].drop_duplicates().reset_index(drop= True))
```


```python
from textblob import TextBlob

tb = TextBlob(" ".join(uniquecontext))
sentences = [item.raw for item in tb.sentences]
```

#### Transfer Learning:  Import InferSent LSTM model with pretrained GloVe vectors for creating sentence embeddings


```python
import sys
sys.path.append('InferSent')
import torch

from models import InferSent
model_version = 1
MODEL_PATH = "InferSent/encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
use_cuda = True
model = model.cuda()
```


```python
model.set_w2v_path("InferSent/glove/glove.840B.300d.txt")
```


```python
model.build_vocab(sentences, tokenize=True)
```

    Found 88993(/109718) words with w2v vectors
    Vocab size : 88993
    

This Step is just for visualization purposes. If you don't have a GPU, this step will take a lot of time. Just load the pre created embeddings from embedding1.pkl and embedding2.pkl as specified below


```python
with open("data/embedding1.pickle", "rb") as f:
    d1 = pickle.load(f)

with open("data/embedding2.pickle", "rb") as f:
    d2 = pickle.load(f)
```


```python
dict_embeddings = {}
for i in range(len(sentences)):
    dict_embeddings[sentences[i]] = model.encode([sentences[i]], tokenize=True)
```


```python
questions = list(df["question"])
```


```python
len(questions)
```




    87599




```python
for i in range(len(questions)):
    dict_embeddings[questions[i]] = model.encode([questions[i]], tokenize=True)
```


```python
d1 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 0}
d2 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 1}
```


```python
import pickle

with open('data/embedding1.pickle', 'wb') as handle:
    pickle.dump(d1, handle)
with open('data/embedding2.pickle', 'wb') as handle:
    pickle.dump(d2, handle)
```


```python
X_train = pd.read_csv('data/train.csv')
```


```python
X_train
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>answer_start</th>
      <th>context</th>
      <th>question</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>515</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>To whom did the Virgin Mary allegedly appear i...</td>
      <td>Saint Bernadette Soubirous</td>
    </tr>
    <tr>
      <th>1</th>
      <td>188</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is in front of the Notre Dame Main Building?</td>
      <td>a copper statue of Christ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>279</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>The Basilica of the Sacred heart at Notre Dame...</td>
      <td>the Main Building</td>
    </tr>
    <tr>
      <th>3</th>
      <td>381</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is the Grotto at Notre Dame?</td>
      <td>a Marian place of prayer and reflection</td>
    </tr>
    <tr>
      <th>4</th>
      <td>92</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What sits on top of the Main Building at Notre...</td>
      <td>a golden statue of the Virgin Mary</td>
    </tr>
    <tr>
      <th>5</th>
      <td>248</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>When did the Scholastic Magazine of Notre dame...</td>
      <td>September 1876</td>
    </tr>
    <tr>
      <th>6</th>
      <td>441</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>How often is Notre Dame's the Juggler published?</td>
      <td>twice</td>
    </tr>
    <tr>
      <th>7</th>
      <td>598</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>What is the daily student paper at Notre Dame ...</td>
      <td>The Observer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>126</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>How many student news papers are found at Notr...</td>
      <td>three</td>
    </tr>
    <tr>
      <th>9</th>
      <td>908</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>In what year did the student paper Common Sens...</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>10</th>
      <td>119</td>
      <td>The university is the major seat of the Congre...</td>
      <td>Where is the headquarters of the Congregation ...</td>
      <td>Rome</td>
    </tr>
    <tr>
      <th>11</th>
      <td>145</td>
      <td>The university is the major seat of the Congre...</td>
      <td>What is the primary seminary of the Congregati...</td>
      <td>Moreau Seminary</td>
    </tr>
    <tr>
      <th>12</th>
      <td>234</td>
      <td>The university is the major seat of the Congre...</td>
      <td>What is the oldest structure at Notre Dame?</td>
      <td>Old College</td>
    </tr>
    <tr>
      <th>13</th>
      <td>356</td>
      <td>The university is the major seat of the Congre...</td>
      <td>What individuals live at Fatima House at Notre...</td>
      <td>Retired priests and brothers</td>
    </tr>
    <tr>
      <th>14</th>
      <td>675</td>
      <td>The university is the major seat of the Congre...</td>
      <td>Which prize did Frederick Buechner create?</td>
      <td>Buechner Prize for Preaching</td>
    </tr>
    <tr>
      <th>15</th>
      <td>487</td>
      <td>The College of Engineering was established in ...</td>
      <td>How many BS level degrees are offered in the C...</td>
      <td>eight</td>
    </tr>
    
  </tbody>
</table>
<p>87599 rows × 4 columns</p>
</div>




```python
dict_emb = dict(d1)
dict_emb.update(d2)
```


```python
len(dict_emb)
```




    179862




```python
X_train.dropna(inplace=True)
```


```python
X_train.shape
```




    (87598, 4)




```python
def get_target(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]: idx = i
    return idx

def process(train):
    
    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])
    train["target"] = train.apply(get_target, axis = 1)
    train['sent_emb'] = train['sentences'].apply(lambda x: [dict_emb[item][0] if item in\
                                                           dict_emb else np.zeros(4096) for item in x])
    train['quest_emb'] = train['question'].apply(lambda x: dict_emb[x] if x in dict_emb else np.zeros(4096) )
    
    return train
```


```python
import numpy as np

X_train = process(X_train)
```


```python
X_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>answer_start</th>
      <th>context</th>
      <th>question</th>
      <th>text</th>
      <th>sentences</th>
      <th>target</th>
      <th>sent_emb</th>
      <th>quest_emb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>515</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>To whom did the Virgin Mary allegedly appear i...</td>
      <td>Saint Bernadette Soubirous</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>5</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.11010079, 0.11422941, 0.115608975, 0.05489...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>188</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is in front of the Notre Dame Main Building?</td>
      <td>a copper statue of Christ</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>2</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.10951651, 0.11030627, 0.052100062, 0.03053...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>279</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>The Basilica of the Sacred heart at Notre Dame...</td>
      <td>the Main Building</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>3</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.011956477, 0.14930707, 0.026600495, 0.0527...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>381</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is the Grotto at Notre Dame?</td>
      <td>a Marian place of prayer and reflection</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>4</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.0711433, 0.05411832, -0.013959841, 0.05310...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>92</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What sits on top of the Main Building at Notre...</td>
      <td>a golden statue of the Virgin Mary</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>1</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.16133596, 0.1503958, 0.09225755, 0.0404580...</td>
    </tr>
  </tbody>
</table>
</div>




```python
def cosine_similarity(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item, x["quest_emb"][0]))
    return li

def pred_idx(distances):
    return np.argmin(distances)

def predictions(train):
    
    train["cosine_sim"] = train.apply(cosine_similarity, axis = 1)
    train["diff"] = (train["quest_emb"] - train["sent_emb"])**2
    train["euclidean_dis"] = train["diff"].apply(lambda x: list(np.sum(x, axis = 1)))
    del train["diff"]
    
    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: pred_idx(x))
    train["pred_idx_euc"] = train["euclidean_dis"].apply(lambda x: pred_idx(x))
    
    return train
```


```python
from scipy import spatial

predicted = predictions(X_train)
```

    e:\anaconda3\envs\squad\lib\site-packages\scipy\spatial\distance.py:698: RuntimeWarning: invalid value encountered in double_scalars
      dist = 1.0 - uv / np.sqrt(uu * vv)
    


```python
predicted.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>answer_start</th>
      <th>context</th>
      <th>question</th>
      <th>text</th>
      <th>sentences</th>
      <th>target</th>
      <th>sent_emb</th>
      <th>quest_emb</th>
      <th>cosine_sim</th>
      <th>euclidean_dis</th>
      <th>pred_idx_cos</th>
      <th>pred_idx_euc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>515</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>To whom did the Virgin Mary allegedly appear i...</td>
      <td>Saint Bernadette Soubirous</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>5</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.11010079, 0.11422941, 0.115608975, 0.05489...</td>
      <td>[0.42473626136779785, 0.3640499711036682, 0.34...</td>
      <td>[14.563859, 15.262213, 17.398178, 14.272491, 1...</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>188</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is in front of the Notre Dame Main Building?</td>
      <td>a copper statue of Christ</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>2</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.10951651, 0.11030627, 0.052100062, 0.03053...</td>
      <td>[0.45407456159591675, 0.32262009382247925, 0.3...</td>
      <td>[12.889506, 12.285219, 16.843704, 8.361172, 11...</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>279</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>The Basilica of the Sacred heart at Notre Dame...</td>
      <td>the Main Building</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>3</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.011956477, 0.14930707, 0.026600495, 0.0527...</td>
      <td>[0.3958578109741211, 0.2917083501815796, 0.309...</td>
      <td>[11.857297, 11.392319, 15.061656, 7.184714, 8....</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>381</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is the Grotto at Notre Dame?</td>
      <td>a Marian place of prayer and reflection</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>4</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.0711433, 0.05411832, -0.013959841, 0.05310...</td>
      <td>[0.49006974697113037, 0.4060605764389038, 0.45...</td>
      <td>[13.317537, 15.017247, 20.81268, 10.511387, 10...</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>92</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What sits on top of the Main Building at Notre...</td>
      <td>a golden statue of the Virgin Mary</td>
      <td>[Architecturally, the school has a Catholic ch...</td>
      <td>1</td>
      <td>[[0.05519996, 0.0501314, 0.047870375, 0.016248...</td>
      <td>[[0.16133596, 0.1503958, 0.09225755, 0.0404580...</td>
      <td>[0.4777514934539795, 0.2891119122505188, 0.341...</td>
      <td>[15.0888195, 11.612734, 16.684145, 9.71824, 12...</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>248</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>When did the Scholastic Magazine of Notre dame...</td>
      <td>September 1876</td>
      <td>[As at most other universities, Notre Dame's s...</td>
      <td>2</td>
      <td>[[0.09720327, 0.09345725, 0.054660242, 0.04843...</td>
      <td>[[0.016918724, 0.12084099, 0.013292058, 0.0587...</td>
      <td>[0.2747580409049988, 0.3731493353843689, 0.280...</td>
      <td>[11.473504, 16.305737, 14.419686, 11.785967, 1...</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>441</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>How often is Notre Dame's the Juggler published?</td>
      <td>twice</td>
      <td>[As at most other universities, Notre Dame's s...</td>
      <td>3</td>
      <td>[[0.09720327, 0.09345725, 0.054660242, 0.04843...</td>
      <td>[[0.07944553, 0.11071574, 0.11615732, 0.045065...</td>
      <td>[0.29136353731155396, 0.44691193103790283, 0.3...</td>
      <td>[12.094654, 19.268333, 17.051125, 12.115431, 1...</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>598</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>What is the daily student paper at Notre Dame ...</td>
      <td>The Observer</td>
      <td>[As at most other universities, Notre Dame's s...</td>
      <td>9</td>
      <td>[[0.09720327, 0.09345725, 0.054660242, 0.04843...</td>
      <td>[[0.0711433, 0.05411832, 0.02641398, 0.0866460...</td>
      <td>[0.24287956953048706, 0.38149863481521606, 0.3...</td>
      <td>[10.40575, 17.056553, 16.048374, 12.6742735, 1...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>126</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>How many student news papers are found at Notr...</td>
      <td>three</td>
      <td>[As at most other universities, Notre Dame's s...</td>
      <td>9</td>
      <td>[[0.09720327, 0.09345725, 0.054660242, 0.04843...</td>
      <td>[[0.06699271, 0.050647143, 0.118103534, 0.0667...</td>
      <td>[0.18055570125579834, 0.3603665828704834, 0.34...</td>
      <td>[7.8146877, 16.114155, 17.537537, 12.886263, 1...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>908</td>
      <td>As at most other universities, Notre Dame's st...</td>
      <td>In what year did the student paper Common Sens...</td>
      <td>1987</td>
      <td>[As at most other universities, Notre Dame's s...</td>
      <td>7</td>
      <td>[[0.09720327, 0.09345725, 0.054660242, 0.04843...</td>
      <td>[[0.042654388, 0.13311043, 0.112292886, 0.0977...</td>
      <td>[0.2252199649810791, 0.36460912227630615, 0.26...</td>
      <td>[9.867832, 16.703403, 13.726372, 11.037147, 12...</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
predicted.to_csv("data/newdata.csv", index=None)
```


```python
def accuracy(target, predicted):
    
    acc = (target==predicted).sum()/len(target)
    
    return acc
```


```python
print(accuracy(predicted["target"], predicted["pred_idx_euc"]))
```

    0.4471106646270463
    


```python
print(accuracy(predicted["target"], predicted["pred_idx_cos"]))
```

    0.6333477933286148
    


```python
predicted.iloc[65000,:]
```




    answer_start                                                   546
    context          Socioeconomic factors, in combination with ear...
    question         What has led to many tragic instances of event...
    text                                                        Racism
    sentences        [Socioeconomic factors, in combination with ea...
    target                                                           3
    sent_emb         [[0.098147884, 0.1247487, 0.14421512, 0.060331...
    quest_emb        [[0.079051174, 0.08900199, 0.13616362, 0.06939...
    cosine_sim       [0.2518765926361084, 0.24583172798156738, 0.24...
    euclidean_dis           [11.059026, 11.683056, 11.02791, 3.251983]
    pred_idx_cos                                                     3
    pred_idx_euc                                                     3
    Name: 65000, dtype: object




```python
ct,k = 0,0
for i in range(predicted.shape[0]):
    if predicted.iloc[i,10] != predicted.iloc[i,5]:
        k += 1
        if predicted.iloc[i,11] == predicted.iloc[i,5]:
            ct += 1

ct, k
```




    (5534, 32118)




```python
label = []
for i in range(predicted.shape[0]):
    if predicted.iloc[i,10] == predicted.iloc[i,11]:
        label.append(predicted.iloc[i,10])
    else:
        label.append((predicted.iloc[i,10],predicted.iloc[i,10]))
```


```python
ct = 0
for i in range(75206):
    item = predicted["target"][i]
    try:
        if label[i] == predicted["target"][i]: ct +=1
    except:
        if item in label[i]: ct +=1
```

# Combining Accuracy of features


```python
ct/75206
```




    0.6364385820280297




```python

```


# More techniques and Accuracy comparison on SQuAD


```python
import pandas as pd
data = pd.read_csv("data/newdata.csv").reset_index(drop=True)
```


```python
data.shape
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>answer_start</th>
      <th>context</th>
      <th>question</th>
      <th>text</th>
      <th>sentences</th>
      <th>target</th>
      <th>sent_emb</th>
      <th>quest_emb</th>
      <th>cosine_sim</th>
      <th>euclidean_dis</th>
      <th>pred_idx_cos</th>
      <th>pred_idx_euc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>515</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>To whom did the Virgin Mary allegedly appear i...</td>
      <td>Saint Bernadette Soubirous</td>
      <td>['Architecturally, the school has a Catholic c...</td>
      <td>5</td>
      <td>[array([ 0.05519996,  0.0501314 ,  0.04787038,...</td>
      <td>[[ 0.11010079  0.11422941  0.11560898 ...  0.0...</td>
      <td>[0.42473626136779785, 0.3640499711036682, 0.34...</td>
      <td>[14.563859, 15.262213, 17.398178, 14.272491, 1...</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>188</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is in front of the Notre Dame Main Building?</td>
      <td>a copper statue of Christ</td>
      <td>['Architecturally, the school has a Catholic c...</td>
      <td>2</td>
      <td>[array([ 0.05519996,  0.0501314 ,  0.04787038,...</td>
      <td>[[ 0.10951651  0.11030627  0.05210006 ... -0.0...</td>
      <td>[0.45407456159591675, 0.32262009382247925, 0.3...</td>
      <td>[12.889506, 12.285219, 16.843704, 8.361172, 11...</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>279</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>The Basilica of the Sacred heart at Notre Dame...</td>
      <td>the Main Building</td>
      <td>['Architecturally, the school has a Catholic c...</td>
      <td>3</td>
      <td>[array([ 0.05519996,  0.0501314 ,  0.04787038,...</td>
      <td>[[ 0.01195648  0.14930707  0.02660049 ...  0.0...</td>
      <td>[0.3958578109741211, 0.2917083501815796, 0.309...</td>
      <td>[11.857297, 11.392319, 15.061656, 7.184714, 8....</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>381</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What is the Grotto at Notre Dame?</td>
      <td>a Marian place of prayer and reflection</td>
      <td>['Architecturally, the school has a Catholic c...</td>
      <td>4</td>
      <td>[array([ 0.05519996,  0.0501314 ,  0.04787038,...</td>
      <td>[[ 0.0711433   0.05411832 -0.01395984 ... -0.0...</td>
      <td>[0.49006974697113037, 0.4060605764389038, 0.45...</td>
      <td>[13.317537, 15.017247, 20.81268, 10.511387, 10...</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>92</td>
      <td>Architecturally, the school has a Catholic cha...</td>
      <td>What sits on top of the Main Building at Notre...</td>
      <td>a golden statue of the Virgin Mary</td>
      <td>['Architecturally, the school has a Catholic c...</td>
      <td>1</td>
      <td>[array([ 0.05519996,  0.0501314 ,  0.04787038,...</td>
      <td>[[0.16133596 0.1503958  0.09225755 ... 0.06351...</td>
      <td>[0.4777514934539795, 0.2891119122505188, 0.341...</td>
      <td>[15.0888195, 11.612734, 16.684145, 9.71824, 12...</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
import ast
data = data[data["sentences"].apply(lambda x: len(ast.literal_eval(x)))<11].reset_index(drop=True)
```


```python
def create_features(data):
    train = pd.DataFrame()
     
    for k in range(len(data["euclidean_dis"])):
        dis = ast.literal_eval(data["euclidean_dis"][k])
        for i in range(len(dis)):
            train.loc[k, "column_euc_"+"%s"%i] = dis[i]
    
    print("Done")
    
    for k in range(len(data["cosine_sim"])):
        dis = ast.literal_eval(data["cosine_sim"][k].replace("nan","1"))
        for i in range(len(dis)):
            train.loc[k, "column_cos_"+"%s"%i] = dis[i]
            
    train["target"] = data["target"]
    return train
```


```python
train = create_features(data)
```

    Done
    


```python
train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column_euc_0</th>
      <th>column_euc_1</th>
      <th>column_euc_2</th>
      <th>column_euc_3</th>
      <th>column_euc_4</th>
      <th>column_euc_5</th>
      <th>column_euc_6</th>
      <th>column_euc_7</th>
      <th>column_euc_8</th>
      <th>column_euc_9</th>
      <th>...</th>
      <th>column_cos_1</th>
      <th>column_cos_2</th>
      <th>column_cos_3</th>
      <th>column_cos_4</th>
      <th>column_cos_5</th>
      <th>column_cos_6</th>
      <th>column_cos_7</th>
      <th>column_cos_8</th>
      <th>column_cos_9</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.563859</td>
      <td>15.262213</td>
      <td>17.398178</td>
      <td>14.272491</td>
      <td>13.339654</td>
      <td>9.336262</td>
      <td>15.720997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.364050</td>
      <td>0.347755</td>
      <td>0.394242</td>
      <td>0.371025</td>
      <td>0.185690</td>
      <td>0.351921</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.889506</td>
      <td>12.285219</td>
      <td>16.843704</td>
      <td>8.361172</td>
      <td>11.918098</td>
      <td>17.601221</td>
      <td>14.929260</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.322620</td>
      <td>0.355004</td>
      <td>0.271561</td>
      <td>0.392342</td>
      <td>0.384383</td>
      <td>0.362597</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.857297</td>
      <td>11.392319</td>
      <td>15.061656</td>
      <td>7.184714</td>
      <td>8.465475</td>
      <td>13.927310</td>
      <td>12.249870</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.291708</td>
      <td>0.309919</td>
      <td>0.223061</td>
      <td>0.265975</td>
      <td>0.293025</td>
      <td>0.288712</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.317537</td>
      <td>15.017247</td>
      <td>20.812680</td>
      <td>10.511387</td>
      <td>10.947038</td>
      <td>16.777027</td>
      <td>17.992474</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.406061</td>
      <td>0.456177</td>
      <td>0.353780</td>
      <td>0.373973</td>
      <td>0.368208</td>
      <td>0.450643</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.088819</td>
      <td>11.612734</td>
      <td>16.684145</td>
      <td>9.718240</td>
      <td>12.873976</td>
      <td>17.225935</td>
      <td>14.960226</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.289112</td>
      <td>0.341493</td>
      <td>0.286430</td>
      <td>0.384779</td>
      <td>0.362687</td>
      <td>0.347337</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
train.apply(max, axis = 0)
```




    column_euc_0    64.051070
    column_euc_1    70.236873
    column_euc_2    57.691376
    column_euc_3    56.178434
    column_euc_4    58.976979
    column_euc_5    50.716496
    column_euc_6    63.920295
    column_euc_7          NaN
    column_euc_8          NaN
    column_euc_9          NaN
    column_cos_0     1.466755
    column_cos_1     1.606138
    column_cos_2     1.552745
    column_cos_3     1.544334
    column_cos_4     1.542615
    column_cos_5     1.477041
    column_cos_6     1.553627
    column_cos_7          NaN
    column_cos_8          NaN
    column_cos_9          NaN
    target           9.000000
    dtype: float64




```python
subset1 = train.iloc[:,:10].fillna(60)
subset2 = train.iloc[:,10:].fillna(1)
```


```python
train2 = pd.concat([subset1, subset2],axis=1, join_axes=[subset1.index])
```


```python
train2.apply(max, axis = 0)
```




    column_euc_0    64.051070
    column_euc_1    70.236873
    column_euc_2    60.000000
    column_euc_3    60.000000
    column_euc_4    60.000000
    column_euc_5    60.000000
    column_euc_6    63.920295
    column_euc_7    60.000000
    column_euc_8    60.000000
    column_euc_9    60.000000
    column_cos_0     1.466755
    column_cos_1     1.606138
    column_cos_2     1.552745
    column_cos_3     1.544334
    column_cos_4     1.542615
    column_cos_5     1.477041
    column_cos_6     1.553627
    column_cos_7     1.450005
    column_cos_8     1.118746
    column_cos_9     1.023689
    target           9.000000
    dtype: float64




```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(train2.iloc[:,:-1])
```


```python
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X,
train.iloc[:,-1], train_size=0.8, random_state = 5)
```

    e:\anaconda3\envs\squad\lib\site-packages\sklearn\model_selection\_split.py:2179: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)
    


```python
from sklearn import linear_model, metrics

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
```

    Multinomial Logistic regression Train Accuracy :  0.6365078199574125
    Multinomial Logistic regression Test Accuracy :  0.6385690789473685
    


```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
rf.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, rf.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, rf.predict(test_x)))
```

    Multinomial Logistic regression Train Accuracy :  0.7692047874293266
    Multinomial Logistic regression Test Accuracy :  0.6732260338345865
    


```python

```

