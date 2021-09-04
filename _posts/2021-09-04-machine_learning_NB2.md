# [4.5] 나이브 베이즈 (Naïve Bayes) 알고리즘

- 지도학습 알고리즘 > '분류'에 이용; 확률 기반 알고리즘

## 예제2) 베르누이 나이브 베이즈를 활용한 스팸 메일 분류


```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
```

### Data 가져오기


```python
email_list = [
                {'email title': 'free game only today', 'spam': True},
                {'email title': 'cheapest flight deal', 'spam': True},
                {'email title': 'limited time offer only today only today', 'spam': True},
                {'email title': 'today meeting schedule', 'spam': False},
                {'email title': 'your flight schedule attached', 'spam': False},
                {'email title': 'your credit card statement', 'spam': False}
             ]

df = pd.DataFrame(email_list)
df
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
      <th>email title</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>free game only today</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cheapest flight deal</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>limited time offer only today only today</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>today meeting schedule</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>your flight schedule attached</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>your credit card statement</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Data 가공하기


베르누이 나이브 베이즈 알고리즘은 '0' 또는 '1'로 Data의 특징이 표현됐을 때 사용하는 모델이다.


현재 Data에는 스팸의 여부가 'True', 'False' 등의 bool 형식으로 되어있는데, 이를 'True' = 1로, 'False' = 0으로 매칭해서 바꿔준 후 'label'이라는 Column에 새로 추가하였다.


```python
df['label'] = df['spam'].map({True:1, False:0})
```


```python
df_x = df['email title']
df_y = df['label']
```


```python
cv = CountVectorizer(binary = True)
x_traincv = cv.fit_transform(df_x)
```


```python
encoded_input = x_traincv.toarray()
encoded_input
```




    array([[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
           [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
           [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]])




```python
cv.vocabulary_
```




    {'free': 6,
     'game': 7,
     'only': 11,
     'today': 15,
     'cheapest': 2,
     'flight': 5,
     'deal': 4,
     'limited': 8,
     'time': 14,
     'offer': 10,
     'meeting': 9,
     'schedule': 12,
     'your': 16,
     'attached': 0,
     'credit': 3,
     'card': 1,
     'statement': 13}




```python
cv.inverse_transform(encoded_input[0])
```




    [array(['free', 'game', 'only', 'today'], dtype='<U9')]




```python
cv.get_feature_names()
```




    ['attached',
     'card',
     'cheapest',
     'credit',
     'deal',
     'flight',
     'free',
     'game',
     'limited',
     'meeting',
     'offer',
     'only',
     'schedule',
     'statement',
     'time',
     'today',
     'your']



### 모델 학습하기


```python
bnb = BernoulliNB()
y_train = df_y.astype('int')  # 현재는 float type
bnb.fit(x_traincv, y_train)
```




    BernoulliNB()



### 테스트 Data 가공하기


```python
test_email_list = [
                {'email title': 'free flight offer', 'spam': True},
                {'email title': 'hey traveler free flight deal', 'spam': True},
                {'email title': 'limited free game offer', 'spam': True},
                {'email title': 'today flight schedule', 'spam': False},
                {'email title': 'your credit card attached', 'spam': False},
                {'email title': 'free credit card offer only today', 'spam': False}
             ]

test_df = pd.DataFrame(test_email_list)
test_df['label'] = test_df['spam'].map({True:1, False:0})

test_x = test_df['email title']
test_y = test_df['label']

x_testcv = cv.transform(test_x)
```

### 테스트


```python
predictions = bnb.predict(x_testcv)
```


```python
accuracy_score(test_y, predictions)
```




    0.8333333333333334



## 예제3) 다항분포 나이브 베이즈를 활용한 영화 리뷰 분류


```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```

### Data 가공하기


```python
review_list = [
                {'movie_review': 'this is great great movie. I will watch again', 'type': 'positive'},
                {'movie_review': 'I like this movie', 'type': 'positive'},
                {'movie_review': 'amazing movie in this year', 'type': 'positive'},
                {'movie_review': 'cool my boyfriend also said the movie is cool', 'type': 'positive'},
                {'movie_review': 'awesome of the awesome movie ever', 'type': 'positive'},
                {'movie_review': 'shame I wasted money and time', 'type': 'negative'},
                {'movie_review': 'regret on this move. I will never never what movie from this director', 'type': 'negative'},
                {'movie_review': 'I do not like this movie', 'type': 'negative'},
                {'movie_review': 'I do not like actors in this movie', 'type': 'negative'},
                {'movie_review': 'boring boring sleeping movie', 'type': 'negative'}
             ]

df = pd.DataFrame(review_list)
df
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
      <th>movie_review</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>this is great great movie. I will watch again</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I like this movie</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>amazing movie in this year</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cool my boyfriend also said the movie is cool</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>awesome of the awesome movie ever</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>5</th>
      <td>shame I wasted money and time</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>6</th>
      <td>regret on this move. I will never never what m...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I do not like this movie</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>8</th>
      <td>I do not like actors in this movie</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>9</th>
      <td>boring boring sleeping movie</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['label'] = df['type'].map({'positive':1, 'negative':0})
```


```python
df_x = df['movie_review']
df_y = df['label']
```


```python
cv = CountVectorizer()
x_traincv = cv.fit_transform(df_x)

encoded_input = x_traincv.toarray()
encoded_input
```




    array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
            0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2,
            0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])




```python
cv.vocabulary_
```




    {'this': 30,
     'is': 15,
     'great': 13,
     'movie': 19,
     'will': 35,
     'watch': 33,
     'again': 1,
     'like': 16,
     'amazing': 3,
     'in': 14,
     'year': 36,
     'cool': 8,
     'my': 20,
     'boyfriend': 7,
     'also': 2,
     'said': 26,
     'the': 29,
     'awesome': 5,
     'of': 23,
     'ever': 11,
     'shame': 27,
     'wasted': 32,
     'money': 17,
     'and': 4,
     'time': 31,
     'regret': 25,
     'on': 24,
     'move': 18,
     'never': 21,
     'what': 34,
     'from': 12,
     'director': 9,
     'do': 10,
     'not': 22,
     'actors': 0,
     'boring': 6,
     'sleeping': 28}




```python
cv.inverse_transform(encoded_input[0])
```




    [array(['again', 'great', 'is', 'movie', 'this', 'watch', 'will'],
           dtype='<U9')]




```python
cv.get_feature_names()
```




    ['actors',
     'again',
     'also',
     'amazing',
     'and',
     'awesome',
     'boring',
     'boyfriend',
     'cool',
     'director',
     'do',
     'ever',
     'from',
     'great',
     'in',
     'is',
     'like',
     'money',
     'move',
     'movie',
     'my',
     'never',
     'not',
     'of',
     'on',
     'regret',
     'said',
     'shame',
     'sleeping',
     'the',
     'this',
     'time',
     'wasted',
     'watch',
     'what',
     'will',
     'year']



### 모델 학습하기


```python
mnb = MultinomialNB()
y_train = df_y.astype('int')
mnb.fit(x_traincv, y_train)
```




    MultinomialNB()



### 테스트 Data 가공하기


```python
test_feedback_list = [
                {'movie_review': 'great great great movie ever', 'type': 'positive'},
                {'movie_review': 'I like this amazing movie', 'type': 'positive'},
                {'movie_review': 'my boyfriend said great movie ever', 'type': 'positive'},
                {'movie_review': 'cool cool cool', 'type': 'positive'},
                {'movie_review': 'awesome boyfriend said cool movie ever', 'type': 'positive'},
                {'movie_review': 'shame shame shame', 'type': 'negative'},
                {'movie_review': 'awesome director shame movie boring movie', 'type': 'negative'},
                {'movie_review': 'do not like this movie', 'type': 'negative'},
                {'movie_review': 'I do not like this boring movie', 'type': 'negative'},
                {'movie_review': 'aweful terrible boring movie', 'type': 'negative'}
             ]

test_df = pd.DataFrame(test_feedback_list)
test_df['label'] = test_df['type'].map({'positive':1, 'negative':0})

test_x = test_df['movie_review']
test_y = test_df['label']
```

### 테스트


```python
x_testcv = cv.transform(test_x)
predictions = mnb.predict(x_testcv)
```


```python
accuracy_score(test_y, predictions)
```




    1.0


