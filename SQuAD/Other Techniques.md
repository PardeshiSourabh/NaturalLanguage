
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
