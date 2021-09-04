---
title: "[Machine Learning] 나이브 베이즈(Naive Bayes) 알고리즘을 활용해 스팸 메일/영화 리뷰 분류하기!"
excerpt: "나이브 베이즈(Naive Bayes) 알고리즘 중에서 베르누이 나이브 베이즈와 다항분포 나이브 베이즈를 이용하여 스팸 메일/영화 리뷰를 분류해보았다."

categories:
  - Machine Learning

tags:
  - Python
  - Machine Learning
  - Supervised Learning
  - Classification
  - Naive Bayes
  - scikit-learn

comments: true

mathjax: true

toc: true
toc_sticky: true

date: 2021-09-04
last_modified_at: 2021-09-04
---


오늘은 지난 포스트에 이어서 나이브 베이즈(Naive Bayes) 알고리즘에 대한 예제를 더 진행해보려고 한다.


이번 예제는 2가지이다.


첫 번째 예제는 '베르누이 나이브 베이즈(Bernoulli Naive Bayes) 알고리즘'을 이용하여 스팸 메일을 분류하는 예제이다.


두 번째 예제는 '다항분포 나이브 베이즈(Multinomial Naive Bayes) 알고리즘'을 이용하여 긍정적/부정적 영화 리뷰를 분류하는 예제이다.


전체적인 나이브 베이즈 알고리즘에 대한 설명을 보고 싶다면 아래의 이전 포스트를 보길 바란다!


[이전 포스트](https://jeongxihoon.github.io/machine%20learning/machine_learning_NB/)


　


　


## 예제2) 베르누이 나이브 베이즈를 활용한 스팸 메일 분류


이메일 제목을 보고 스팸인지 아닌지 분류하는 예제이다.


```python
# 필요한 모듈 Import하기

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


예제에서 사용할 Data를 살펴보면,


'email title'(이메일 제목)과 'spam'(스팸 메일인지 아닌지) 이렇게 2개의 column으로 이루어진 Dataframe이다.


　


　


### Data 가공하기


베르누이 나이브 베이즈 알고리즘은 '0' 또는 '1'로 Data의 특징이 표현됐을 때 사용하는 모델이다.


현재 Data에는 스팸의 여부가 'True', 'False' 등의 bool type으로 되어있는데, 이를 'True' = 1로, 'False' = 0으로 매칭해서 바꿔준 후 'label'이라는 Column에 새로 추가하였다.


```python
df['label'] = df['spam'].map({True:1, False:0})
```


```python
df_x = df['email title']
df_y = df['label']
```


또한, 베르누이 나이브 베이즈의 입력 Data는 **고정된 크기의 벡터**이어야 한다.


따라서 Scikitlearn의 `CountVectorizer()`함수를 이용하였다.


이 함수를 이용하면 Dataframe 안에 있는 모든 단어들을 포함하여 고정된 길이의 벡터를 만들어 해당 Data를 표현할 수 있다!


그리고 이전 포스트에서 설명했듯이, 베르누이 나이브 베이즈는 0 또는 1로 Data의 특징이 표현됐을 때 사용하는 알고리즘이다.


따라서 `CountVectorizer()`함수에 `binary = True`옵션을 추가해 특정 단어가 출현하면 1을, 아니면 0을 벡터 인자로 갖도록 설정하였다.


```python
cv = CountVectorizer(binary = True)
x_traincv = cv.fit_transform(df_x)
```


각 row의 'email title' Data를 벡터로 변환한 것을 `x_traincv`라는 변수에 저장하였다.


이를 array type으로 변환하여 `encoded_input`이라는 변수에 저장하여 출력해보았다.


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



이렇게 보니 뭘 의미하는지 직관적으로 알기 힘들다.


설명하자면, `df['email title']`에 존재하는 모든 단어들이 총 17개여서(중복되는 단어는 1개로 계산) 벡터의 크기가 17로 설정되었다.


총 6개의 row가 존재하기에 벡터의 크기가 17인 벡터가 6개 반환된 것이다.


각 벡터의 위치에 대응하는 단어를 보기 위해서 다음과 같은 코드를 실행하였다.



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



위에서부터 순서대로 0 ~ 16번 자리에 대응되는 단어이다.


추가적으로, 첫 번째 row에 해당하는 벡터를 역변환하면 다음과 같다.


```python
cv.inverse_transform(encoded_input[0])
```




    [array(['free', 'game', 'only', 'today'], dtype='<U9')]


　


　


### 모델 학습하기


가공한 Data를 바탕으로 모델을 학습하는 코드는 다음과 같다.


```python
bnb = BernoulliNB()
y_train = df_y.astype('int')  # 현재는 float type
bnb.fit(x_traincv, y_train)
```


　


　


### 테스트 Data 가공하기


다음은 학습된 모델을 테스트하기 위해 사용할 테스트 Data를 가공하는 코드이다.


테스트 Data 또한 위에서 보았던 학습 Data와 마찬가지로 동일한 형태를 띄며,


가공하는 방식도 역시 동일하다.


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


위에서 가공한 테스트 Data와 학습된 모델을 이용하여 모델을 테스트하였다.


```python
predictions = bnb.predict(x_testcv)
```


```python
accuracy_score(test_y, predictions)
```




    0.8333333333333334


모델을 테스트한 결과, 최종적으로 약 0.833(83.33%)의 정확도를 보였다.


　


　


## 예제3) 다항분포 나이브 베이즈를 활용한 영화 리뷰 분류


```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```

### Data 가져오기


영화 리뷰를 보고 긍정적(positive)인지, 부정적(negative)인지 분류하는 예제이다.


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


예제에서 사용할 Data를 살펴보면,


'movie_review'(영화 리뷰)와 'type'(긍정적/부정적 리뷰) 이렇게 2개의 column으로 이루어진 Dataframe이다.


　


　


### Data 가공하기


현재 Data에는 리뷰가 긍정적인지 부정적인지 'positive', 'negative' 등의 str type으로 되어있다. 이를 'positive' = 1로, 'negative' = 0으로 매칭해서 바꿔준 후 'label'이라는 Column에 새로 추가하였다.


```python
df['label'] = df['type'].map({'positive':1, 'negative':0})
```


```python
df_x = df['movie_review']
df_y = df['label']
```


다항분포 나이브 베이즈 알고리즘은 Data의 특징이 **출현 횟수**로 표현됐을 때 사용하는 모델이다.


따라서 `CountVectorizer()`함수를 통해 'movie_review'에 해당하는 Data들을 벡터로 변환하였다.


위의 '예제2'에서 사용했던 방식과 동일한데, 한 가지 차이가 있다.


바로 `binary = True` 옵션을 사용하지 않는다는 점이다.


위에서 언급했듯이 다항분포 나이브 베이즈 알고리즘은 **출현 횟수**로 표현된 Data의 특징을 사용하는 모델이다.


따라서 단어가 몇 번 언급되었는지를 벡터로 표현하기 위해 해당 옵션을 사용하지 않았다.


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



실행 결과, 벡터의 크기가 37인 벡터가 10개 생성된 것을 알 수 있다.


이는 해당 Dataframe에 총 37개의 단어가 등장했으며, 총 10개의 row로 구성되어있기 때문이다.


벡터의 인덱스와 단어가 어떻게 매칭되었는지 알아보기 위해서 다음과 같은 코드를 실행하였다.


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


위에서부터 순서대로 0 ~ 36번 벡터의 요소 인덱스에 해당하는 단어이다.


추가적으로, 첫 번째 row에 해당하는 벡터를 역변환하면 다음과 같다.


```python
cv.inverse_transform(encoded_input[0])
```




    [array(['again', 'great', 'is', 'movie', 'this', 'watch', 'will'],
           dtype='<U9')]


　


　


### 모델 학습하기


가공한 Data를 바탕으로 모델을 학습하는 코드는 다음과 같다.


```python
mnb = MultinomialNB()
y_train = df_y.astype('int')
mnb.fit(x_traincv, y_train)
```


　


　


### 테스트 Data 가공하기


다음은 학습된 모델을 테스트하기 위해 사용할 테스트 Data를 가공하는 코드이다.


테스트 Data 또한 위에서 보았던 학습 Data와 마찬가지로 동일한 형태를 띄며,


가공하는 방식도 역시 동일하다.


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


위에서 가공한 테스트 Data와 학습된 모델을 이용하여 모델을 테스트하였다.


```python
x_testcv = cv.transform(test_x)
predictions = mnb.predict(x_testcv)
```


```python
accuracy_score(test_y, predictions)
```




    1.0


모델을 테스트한 결과, 최종적으로 1.0(100%)의 정확도를 보였다.


　


　


## 마무리


이로써 총 3가지 유형의 나이브 베이즈 모델에 대한 예제 소개도 모두 끝났다.


다음 포스트에서는 '앙상블' 알고리즘에 대해서 포스트할 예정이다.


그럼 다음 포스트에서 보도록 하겠다!
