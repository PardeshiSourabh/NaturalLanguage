
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
    <tr>
      <th>16</th>
      <td>46</td>
      <td>The College of Engineering was established in ...</td>
      <td>In what year was the College of Engineering at...</td>
      <td>1920</td>
    </tr>
    <tr>
      <th>17</th>
      <td>126</td>
      <td>The College of Engineering was established in ...</td>
      <td>Before the creation of the College of Engineer...</td>
      <td>the College of Science</td>
    </tr>
    <tr>
      <th>18</th>
      <td>271</td>
      <td>The College of Engineering was established in ...</td>
      <td>How many departments are within the Stinson-Re...</td>
      <td>five</td>
    </tr>
    <tr>
      <th>19</th>
      <td>155</td>
      <td>The College of Engineering was established in ...</td>
      <td>The College of Science began to offer civil en...</td>
      <td>the 1870s</td>
    </tr>
    <tr>
      <th>20</th>
      <td>496</td>
      <td>All of Notre Dame's undergraduate students are...</td>
      <td>What entity provides help with the management ...</td>
      <td>Learning Resource Center</td>
    </tr>
    <tr>
      <th>21</th>
      <td>68</td>
      <td>All of Notre Dame's undergraduate students are...</td>
      <td>How many colleges for undergraduates are at No...</td>
      <td>five</td>
    </tr>
    <tr>
      <th>22</th>
      <td>155</td>
      <td>All of Notre Dame's undergraduate students are...</td>
      <td>What was created at Notre Dame in 1962 to assi...</td>
      <td>The First Year of Studies program</td>
    </tr>
    <tr>
      <th>23</th>
      <td>647</td>
      <td>All of Notre Dame's undergraduate students are...</td>
      <td>Which organization declared the First Year of ...</td>
      <td>U.S. News &amp; World Report</td>
    </tr>
    <tr>
      <th>24</th>
      <td>358</td>
      <td>The university first offered graduate degrees,...</td>
      <td>The granting of Doctorate degrees first occurr...</td>
      <td>1924</td>
    </tr>
    <tr>
      <th>25</th>
      <td>624</td>
      <td>The university first offered graduate degrees,...</td>
      <td>What type of degree is an M.Div.?</td>
      <td>Master of Divinity</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1163</td>
      <td>The university first offered graduate degrees,...</td>
      <td>Which program at Notre Dame offers a Master of...</td>
      <td>Alliance for Catholic Education</td>
    </tr>
    <tr>
      <th>27</th>
      <td>92</td>
      <td>The university first offered graduate degrees,...</td>
      <td>In what year was a Master of Arts course first...</td>
      <td>1854</td>
    </tr>
    <tr>
      <th>28</th>
      <td>757</td>
      <td>The university first offered graduate degrees,...</td>
      <td>Which department at Notre Dame is the only one...</td>
      <td>Department of Pre-Professional Studies</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4</td>
      <td>The Joan B. Kroc Institute for International P...</td>
      <td>What institute at Notre Dame studies  the reas...</td>
      <td>Joan B. Kroc Institute for International Peace...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>87569</th>
      <td>209</td>
      <td>Sikhism is practiced primarily in Gurudwara at...</td>
      <td>Where can a temple of the Jain faith be found?</td>
      <td>Gyaneshwar</td>
    </tr>
    <tr>
      <th>87570</th>
      <td>355</td>
      <td>Sikhism is practiced primarily in Gurudwara at...</td>
      <td>Kathmandu valley is home to about how many Bah...</td>
      <td>300</td>
    </tr>
    <tr>
      <th>87571</th>
      <td>427</td>
      <td>Sikhism is practiced primarily in Gurudwara at...</td>
      <td>Where is the Baha'i national office located in...</td>
      <td>Shantinagar, Baneshwor</td>
    </tr>
    <tr>
      <th>87572</th>
      <td>633</td>
      <td>Sikhism is practiced primarily in Gurudwara at...</td>
      <td>About what percentage of the Nepali population...</td>
      <td>4.2</td>
    </tr>
    <tr>
      <th>87573</th>
      <td>728</td>
      <td>Sikhism is practiced primarily in Gurudwara at...</td>
      <td>About how many Christian houses of worship exi...</td>
      <td>170</td>
    </tr>
    <tr>
      <th>87574</th>
      <td>46</td>
      <td>Institute of Medicine, the central college of ...</td>
      <td>Of what university is the Institute of Medicin...</td>
      <td>Tribhuwan</td>
    </tr>
    <tr>
      <th>87575</th>
      <td>123</td>
      <td>Institute of Medicine, the central college of ...</td>
      <td>In what part of Kathmandu is the Institute of ...</td>
      <td>Maharajgunj</td>
    </tr>
    <tr>
      <th>87576</th>
      <td>219</td>
      <td>Institute of Medicine, the central college of ...</td>
      <td>When did the Institute of Medicine begin to of...</td>
      <td>1978</td>
    </tr>
    <tr>
      <th>87577</th>
      <td>425</td>
      <td>Institute of Medicine, the central college of ...</td>
      <td>What does KUSMS stand for?</td>
      <td>Kathmandu University School of Medical Sciences</td>
    </tr>
    <tr>
      <th>87578</th>
      <td>377</td>
      <td>Institute of Medicine, the central college of ...</td>
      <td>What institution of tertiary education is know...</td>
      <td>National Academy of Medical Sciences</td>
    </tr>
    <tr>
      <th>87579</th>
      <td>0</td>
      <td>Football and Cricket are the most popular spor...</td>
      <td>Along with cricket, what sport is highly popul...</td>
      <td>Football</td>
    </tr>
    <tr>
      <th>87580</th>
      <td>160</td>
      <td>Football and Cricket are the most popular spor...</td>
      <td>What body oversees soccer in Nepal?</td>
      <td>All Nepal Football Association</td>
    </tr>
    <tr>
      <th>87581</th>
      <td>498</td>
      <td>Football and Cricket are the most popular spor...</td>
      <td>How many people can fit in Dasarath Rangasala ...</td>
      <td>25,000</td>
    </tr>
    <tr>
      <th>87582</th>
      <td>430</td>
      <td>Football and Cricket are the most popular spor...</td>
      <td>In what part of Kathmandu is Dasarath Rangasal...</td>
      <td>Tripureshwor</td>
    </tr>
    <tr>
      <th>87583</th>
      <td>628</td>
      <td>Football and Cricket are the most popular spor...</td>
      <td>Who assisted Nepal in renovating Dasarath Rang...</td>
      <td>Chinese</td>
    </tr>
    <tr>
      <th>87584</th>
      <td>54</td>
      <td>The total length of roads in Nepal is recorded...</td>
      <td>As of 2004, how many kilometers of road existe...</td>
      <td>17,182</td>
    </tr>
    <tr>
      <th>87585</th>
      <td>289</td>
      <td>The total length of roads in Nepal is recorded...</td>
      <td>Why is travel in Kathmandu mainly via automobi...</td>
      <td>hilly terrain</td>
    </tr>
    <tr>
      <th>87586</th>
      <td>500</td>
      <td>The total length of roads in Nepal is recorded...</td>
      <td>What highway connecting Kathmandu to elsewhere...</td>
      <td>BP</td>
    </tr>
    <tr>
      <th>87587</th>
      <td>457</td>
      <td>The total length of roads in Nepal is recorded...</td>
      <td>In what direction out of Kathmandu does the Pr...</td>
      <td>west</td>
    </tr>
    <tr>
      <th>87588</th>
      <td>466</td>
      <td>The total length of roads in Nepal is recorded...</td>
      <td>If one wished to travel north out of Kathmandu...</td>
      <td>Araniko</td>
    </tr>
    <tr>
      <th>87589</th>
      <td>71</td>
      <td>The main international airport serving Kathman...</td>
      <td>What is Nepal's primary airport for internatio...</td>
      <td>Tribhuvan International Airport</td>
    </tr>
    <tr>
      <th>87590</th>
      <td>134</td>
      <td>The main international airport serving Kathman...</td>
      <td>Starting in the center of Kathmandu, how many ...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>87591</th>
      <td>297</td>
      <td>The main international airport serving Kathman...</td>
      <td>How many airlines use Tribhuvan International ...</td>
      <td>22</td>
    </tr>
    <tr>
      <th>87592</th>
      <td>698</td>
      <td>The main international airport serving Kathman...</td>
      <td>From what city does Arkefly offer nonstop flig...</td>
      <td>Amsterdam</td>
    </tr>
    <tr>
      <th>87593</th>
      <td>734</td>
      <td>The main international airport serving Kathman...</td>
      <td>Who operates flights between Kathmandu and Ist...</td>
      <td>Turkish Airlines</td>
    </tr>
    <tr>
      <th>87594</th>
      <td>229</td>
      <td>Kathmandu Metropolitan City (KMC), in order to...</td>
      <td>In what US state did Kathmandu first establish...</td>
      <td>Oregon</td>
    </tr>
    <tr>
      <th>87595</th>
      <td>414</td>
      <td>Kathmandu Metropolitan City (KMC), in order to...</td>
      <td>What was Yangon previously known as?</td>
      <td>Rangoon</td>
    </tr>
    <tr>
      <th>87596</th>
      <td>476</td>
      <td>Kathmandu Metropolitan City (KMC), in order to...</td>
      <td>With what Belorussian city does Kathmandu have...</td>
      <td>Minsk</td>
    </tr>
    <tr>
      <th>87597</th>
      <td>199</td>
      <td>Kathmandu Metropolitan City (KMC), in order to...</td>
      <td>In what year did Kathmandu create its initial ...</td>
      <td>1975</td>
    </tr>
    <tr>
      <th>87598</th>
      <td>0</td>
      <td>Kathmandu Metropolitan City (KMC), in order to...</td>
      <td>What is KMC an initialism of?</td>
      <td>Kathmandu Metropolitan City</td>
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


```python

```
