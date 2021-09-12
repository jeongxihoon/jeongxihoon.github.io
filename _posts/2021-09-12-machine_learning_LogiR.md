# [4.9] 로지스틱 회귀(Logistic Regression)

## 이론


> 선형회귀의 결과를 input으로 받아서 특정 레이블로 분류하는 알고리즘

## 실습1) 단일 입력 로지스틱 회귀


> 한 개의 input을 받아서 0 또는 1을 출력하는 로지스틱 회귀모델


```python
# 필요한 모듈 불러오기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
```

### 로지스틱 회귀모델 만들기


```python
model = Sequential()

# 선형 회귀 레이어 생성
model.add(Dense(input_dim = 1, units = 1))

# 출력값을 시그모이드 함수에 연결
model.add(Activation('sigmoid'))

# 비용함수로 '크로스 엔트로피' 설정
model.compile(loss = 'binary_crossentropy',
             optimizer = 'sgd', metrics = ['binary_accuracy'])
```

### Data 생성


```python
X = np.array([-2, -1.5, -1, 1.25, 1.62, 2])
Y = np.array([0, 0, 0, 1, 1, 1])
```

### 모델 학습


```python
model.fit(X, Y, epochs = 300, verbose = 0)
```




    <tensorflow.python.keras.callbacks.History at 0x7f602b5d2050>




```python
model.predict(X)
```




    array([[0.15690681],
           [0.22062483],
           [0.3009807 ],
           [0.7397466 ],
           [0.7949467 ],
           [0.84207463]], dtype=float32)




```python
model.predict([-1000, 1000])
```




    array([[0.],
           [1.]], dtype=float32)



### 모델 요약


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 1)                 2         
    _________________________________________________________________
    activation (Activation)      (None, 1)                 0         
    =================================================================
    Total params: 2
    Trainable params: 2
    Non-trainable params: 0
    _________________________________________________________________



```python
model.layers[0].weights
```




    [<tf.Variable 'dense/kernel:0' shape=(1, 1) dtype=float32, numpy=array([[0.8387929]], dtype=float32)>,
     <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32, numpy=array([-0.00383937], dtype=float32)>]




```python
model.layers[0].get_weights()
```




    [array([[0.8387929]], dtype=float32), array([-0.00383937], dtype=float32)]



## 실습2) 다중 입력 로지스틱 회귀


> 2개 이상의 input을 받아서 0 또는 1을 출력하는 로지스틱 회귀모델


실습에서는 2개의 input을 받는 다중 입력 로지스틱 회귀모델을 구현해보았다.


그 중에서도 대표적인 예인 'AND 연산'을 실습해보았다.


```python
# 필요한 모듈 불러오기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
```

### 로지스틱 회귀모델 만들기


```python
model = Sequential()

# 선형 회귀 레이어 생성
model.add(Dense(input_dim = 2, units = 1))

# 출력값을 시그모이드 함수에 연결
model.add(Activation('sigmoid'))

# 비용함수로 '크로스 엔트로피' 설정
model.compile(loss = 'binary_crossentropy',
             optimizer = 'sgd', metrics = ['binary_accuracy'])
```

### Data 생성


```python
X = np.array([(0,0), (0,1), (1,0), (1,1)])
Y = np.array([0, 0, 0, 1])
```

### 모델 학습


```python
model.fit(X, Y, epochs = 5000, verbose = 0)
```




    <tensorflow.python.keras.callbacks.History at 0x7f60702d2ad0>




```python
model.predict([(0,0), (0,1), (1,0), (1,1)])
```




    array([[0.0371345 ],
           [0.23177478],
           [0.20865399],
           [0.6734855 ]], dtype=float32)



### 모델 요약


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 1)                 3         
    _________________________________________________________________
    activation_1 (Activation)    (None, 1)                 0         
    =================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    _________________________________________________________________



```python
model.layers[0].weights
```




    [<tf.Variable 'dense_1/kernel:0' shape=(2, 1) dtype=float32, numpy=
     array([[1.9223089],
            [2.0570502]], dtype=float32)>,
     <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([-3.255367], dtype=float32)>]




```python
model.layers[0].get_weights()
```




    [array([[1.9223089],
            [2.0570502]], dtype=float32),
     array([-3.255367], dtype=float32)]



## 실습3) 다중 분류 로지스틱 회귀(소프트맥스, Softmax)


> M개의 input을 받아서 N개의 클래스로 분류하는 로지스틱 회귀모델


보통 다중 분류 로지스틱 회귀모델을 '소프트맥스(Softmax)'라고 부른다.


이번 실습 예제에서는 앙상블 알고리즘에서 사용했던 'MNIST 손글씨 숫자 데이터셋'을 이용하여 손글씨 숫자를 0 ~ 9까지 분류하였다.


```python
# 필요한 모듈 불러오기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
```

### Data 가져오기


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```


```python
print('train data (count, row, column) : {}'.format(X_train.shape))
print('test data (count, row, column) : {}'.format(X_test.shape))
```

    train data (count, row, column) : (60000, 28, 28)
    test data (count, row, column) : (10000, 28, 28)


위에서 볼 수 있듯이, 손글씨 Data는 가로 28px, 세로 28px로 구성되어있다.


학습 Data는 총 6 만개, 테스트 Data는 총 1만 개가 있다.


```python
print('Sample from X_train : {}'.format(y_train[0]))
print('Sample from X_test : {}'.format(y_test[0]))
```

    Sample from X_train : 5
    Sample from X_test : 7


위에서 볼 수 있듯이, 'y_train'과 'y_test' Data에는 해당하는 숫자 Data가 담겨있다.

### Data 정규화


원래 기존의 Data에는 1개의 픽셀에 0 ~ 255까지의 수치 정보가 저장되어있다.


하지만 경사하강법으로 모델을 학습할 때 더 쉽고 빠르게 parameter들을 구하기 위해서 각각의 픽셀 수치를 255로 나누어서 정규화하였다.


즉, 모든 픽셀 수치들을 0 ~ 1의 값으로 바꿔준 것이다.


```python
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
```

### Data 단순화


현재 Data들은 28 x 28의 Dimension을 가지고 있으며, 행/열 지역적 정보를 보여준다.


하지만, 실습에서는 이러한 지역적 정보를 사용하지 않고, 단순히 정규화시킨 숫자로만 분류할 것이기 때문에 784(28 x 28)의 길이를 가진 Data로 단순화 하였다.


```python
input_dim = 784  # 28 x 28

X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (60000, 784)
    (60000,)
    (10000, 784)
    (10000,)


### 소프트맥스(Softmax)


모델이 구별해야하는 숫자는 0 ~ 9, 총 10개이다.


'실습3'을 시작할 때 말했던 것처럼, 다중 분류 로지스틱 회귀모델인 '소프트맥스'는 M개의 입력을 받아 N개의 클래스로 분류하는 모델이라고 했다.


따라서 이 정의를 실습에 적용해보면, 이번 모델은 10개의 숫자로 분류하기 위해서 소프트맥스 모델에 10개의 로지스틱 회귀를 구현하였다.


또한, 출력값을 길이가 10인 배열(array)로 나타내도록 하였다.


즉 [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]로 출력값이 나오게 되는데, 이는 순서대로 0 ~ 9를 의미한다.


예를 들어 출력값이 '[0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0]'이면 이 모델은 가장 높은 확률을 가진 인덱스인 '2'를 결과로 출력하게 된다!


모델이 학습할 때 최적의 Parameter를 구하기 위해 실제값과의 크로스 엔트로피를 계산해야하기 때문에, 실제값들도 '원 핫 인코딩(One Hot Encoding)'을 통해서 출력값과 같은 형태로 바꾸어주었다.


'원 핫 인코딩'이란 한 개의 요소는 True(1), 나머지느 False(0)으로 만드는 기법이다. (위의 출력값과 비슷한 형태라고 생각하면 된다.)


```python
num_classes = 10

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
```


```python
print(y_train[0])
```

    [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]


위의 결과를 보면 원 핫 인코딩을 통해서 '5'의 수치 Data가 변환된 것을 확인할 수 있다.


```python
model = Sequential()
model.add(Dense(input_dim = input_dim, units = 10, activation = 'softmax'))
```

위는 최종적으로 모델을 구현하는 코드이다.


input은 총 길이가 784인 array이고, output은 10개의 시그모이드 값(0 ~ 1사이의 확률값)을 가지고 있으면서 길이가 10인 array이다.


활성화함수(`activation` 옵션)는 'softmax'로 지정함으로써 소프트맥스 모델을 구현하였다.

### 모델 학습


```python
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 2048, epochs = 100, verbose = 0)
```




    <tensorflow.python.keras.callbacks.History at 0x7f6008486610>



### 모델 테스트


```python
score = model.evaluate(X_test, y_test)
print('Test accuracy : {}'.format(score[1]))
```

    313/313 [==============================] - 2s 6ms/step - loss: 0.4227 - accuracy: 0.8903
    Test accuracy : 0.8902999758720398


테스트 결과, 약 89%의 정확도를 보였다.

### 모델 요약


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_2 (Dense)              (None, 10)                7850      
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________


총 Parameter의 개수는 7,850개이다. 이는 한 개의 로지스틱 회귀에 784(28 x 28)개의 회귀계수($\omega$)와 편향(bias, $b$)값을 더해 총 785개의 Parameter를 가지고 있기 때문이다.


모델에서는 총 10개의 로지스틱 회귀를 포함하고 있기 때문에 7850(785 x 10)개의 Parameter를 가지고 있다.


```python
model.layers[0].weights
```




    [<tf.Variable 'dense_2/kernel:0' shape=(784, 10) dtype=float32, numpy=
     array([[-0.02456667, -0.04687213,  0.00973451, ...,  0.00650328,
             -0.04734281,  0.02048253],
            [-0.08443252, -0.02385858,  0.05318164, ...,  0.00630102,
             -0.05599174, -0.05512647],
            [ 0.01461462,  0.08575616, -0.00956853, ..., -0.07055015,
              0.05416236,  0.05256601],
            ...,
            [-0.03499135,  0.02363621, -0.02695451, ..., -0.07299899,
              0.06626979,  0.00636236],
            [ 0.01819825,  0.05894279, -0.07297193, ...,  0.04544663,
             -0.03761313, -0.05542137],
            [-0.00860635, -0.06920259, -0.0515497 , ...,  0.02180584,
              0.03760223,  0.05680607]], dtype=float32)>,
     <tf.Variable 'dense_2/bias:0' shape=(10,) dtype=float32, numpy=
     array([-0.0817981 ,  0.16934235, -0.03331714, -0.05922424,  0.05676099,
             0.15661375, -0.02059836,  0.09341711, -0.24232659, -0.03886966],
           dtype=float32)>]



Parameter를 더 자세히 보면 위와 같다.


(784, 10)의 shape을 가진 Parameter는 로지스틱 회귀에 존재하는 회귀계수($\omega$)이고,


아래에 (10,)의 shape을 가진 Parameter는 편향(bias, $b$)값이다.
