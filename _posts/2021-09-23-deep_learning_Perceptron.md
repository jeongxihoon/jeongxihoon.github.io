---
title: "[Deep Learning] 퍼셉트론(Perceptron)으로 AND/OR 연산 구현하기!"
excerpt: "단일 퍼셉트론(Perceptron)으로 AND/OR 연산을 구현해보았다."

categories:
  - Deep Learning

tags:
  - Python
  - Deep Learning
  - Perceptron
  - Tensorflow

comments: true

mathjax: true

toc: true
toc_sticky: true

date: 2021-09-23
last_modified_at: 2021-09-23
---


딥러닝(Deep Learning)을 주제로 다시 돌아왔다!


오늘 소개할 것은 퍼셉트론(Perceptron)이다.


그 중에서도 단일 퍼셉트론으로 가장 기본 연산인 'AND'와 'OR' 연산을 구현해보았다.


　


　


## 퍼셉트론(Perceptron)


> 2개의 Input이 있을 때, 하나의 뉴런으로 2개의 Input을 계산하여 0 또는 1을 Output으로 출력하는 모델


퍼셉트론(Perceptron)은 2개의 Input을 받는다. 이를 $x_1$, $x_2$라고 하자.


퍼셉트론은 이 2개의 Input에 가중치를 곱하고, 편향값(Bias)를 더해서 값을 구한다.


수식으로 표현하면 다음과 같다.


$z = w_1x_1 + w_2x_2 + bias$


Input을 통해 구해진 $z$값은 활성화 함수($a$, Activation Function)의 Input으로 들어가 0 또는 1이라는 값을 출력하게 된다.


이때 퍼셉트론은 스텝 함수(Step Function)을 활성화 함수로 사용하는데, 스텝 함수는 다음과 같다.


- $z < 0$일 경우, $a(z) = 0$
- $z >= 0$일 경우, $a(z) = 1$


AND 연산의 경우 예를 들어보면 다음과 같다.


$w_1$, $w_2$ 모두 0.6의 값을 갖고, bias = -1이라고 하자.


만약 $x_1 = 0$, $x_2 = 0$인 경우,

- $z = 0 + 0 + -1 = -1
- $a(-1) = 0$


따라서 퍼셉트론은 2개의 Input을 받아서 0이라는 Output을 출력하게 되는 것이다.


아래의 실습을 통해서 더 자세하게 다뤄보겠다.


　


　


## 퍼셉트론으로 AND/OR 연산 구현하기


Python의 `tensorflow` 라이브러리를 통해 AND/OR 연산을 구현해보았다.


```python
# 필요한 모듈 불러오기


import tensorflow as tf
```


기본적인 AND/OR 연산의 Input과 Output을 정리해보면 다음의 표와 같다.


| $x_1$ | $x_2$ | $x_1$ AND $x_2$ | $x_1$ OR $x_2$ |
|-------|-------|-----------------|----------------|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 |


　


　


### Data를 함수를 통해 저장하기


코드를 실행하는 과정에서 Data를 간편하게 불러오기 위해서 AND/OR Data를 함수를 통해 불러오도록 하였다.


```python
T = 1.0
F = 0.0
bias = 1.0



def get_AND_data():
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    y = [
        [F],
        [F],
        [F],
        [T]
    ]
    
    return X, y

def get_OR_data():
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    y = [
        [F],
        [T],
        [T],
        [T]
    ]
    
    return X, y
```


　


　


### Perceptron Class 구현하기


다음은 퍼셉트론 클래스를 생성하는 코드이다.


```python
class Perceptron:
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([3, 1]))
        # X, y, bias 총 3개의 Input을 변수(Variable)로 설정
        # random.normal(); 정규분포에서 값 randomly 생성([3, 1] Array로)
        # Variable(); 텐서플로우에서 사용하는 변수 클래스 선언
        
    def train(self, X):
        err = 1
        epoch, max_epochs = 0, 20  # 최대 20번까지 학습하도록 설정
        while err > 0.0 and epoch < max_epochs:
            epoch += 1
            self.optimize(X)
            err = self.mse(y, self.pred(X)).numpy()
            # MSE(평균제곱오차) 계산 후, err 변수에 값 갱신
            print('epoch', epoch, 'mse', err)

    def pred(self, X):
        return self.step(tf.matmul(X, self.W))
        # matmul(); 행렬곱 계산
    
    def mse(self, y, y_hat):
        return tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
        # subtract(); 차(빼기) 계산
        # square(); 제곱 계산(x^2)
        # reduce_mean(); 전체의 값을 평균내서 차원 제거 후 1개의 스칼라 값으로 반환
    
    def step(self, x):  # 스텝 함수{ x > 0이면 1을 아니면 0을 반환하도록 }
        return tf.dtypes.cast(tf.math.greater(x, 0), tf.float32)
        # math.greater(a, b); a와 b 크기 비교 후, a가 크면 True, 아니면 False 반환
        # dtypes.cast(); 값을 새로운 type으로 재정의(위에서는 float32 type으로 값을 재정의)
    
    def optimize(self, X):
        delta = tf.matmul(X, tf.subtract(y, self.step(tf.matmul(X, self.W))), transpose_a = True)
        self.W.assign(self.W + delta)  # assign(); 새로운 값을 할당하는 것
        # self.W을 (self.W + delta)로 할당(재정의)하는 것
```


위의 코드 블럭 안에 주석을 통해 설명을 했으니 참고하길!!


~~나름 이해가 됐으면 좋겠다~~


　


　


### AND 연산 구현하기


AND 연산을 구현하기 위해 먼저 Data를 위에서 정의한 함수를 통해 불러온다.


```python
X, y = get_AND_data()
```


Data를 불러온 후, 다음과 같이 퍼셉트론 객체를 생성한다.


```python
perceptron = Perceptron()
```


불러온 Data를 통해 퍼셉트론 객체를 학습시킨다.


```python
perceptron.train(X)
```

    epoch 1 mse 0.75
    epoch 2 mse 0.25
    epoch 3 mse 0.25
    epoch 4 mse 0.5
    epoch 5 mse 0.25
    epoch 6 mse 0.25
    epoch 7 mse 0.25
    epoch 8 mse 0.5
    epoch 9 mse 0.25
    epoch 10 mse 0.0


　


　


### 테스트


다음 코드를 통해 테스트 결과를 확인하였다.


```python
print(perceptron.pred(X).numpy())
```

    [[0.]
     [0.]
     [0.]
     [1.]]


테스트 결과,


- (0,0) -> 0
- (0,1) -> 0
- (1,0) -> 0
- (1,1) -> 1


AND 연산 결과가 잘 출력된 것을 확인할 수 있다.


OR 연산도 동일한 퍼셉트론 클래스 객체에 Data만 `get_OR_data()`를 이용하여 설정해주면 된다.


이번 포스트에서 OR 연산은 생략하였다!


　


　


## 마무리


이번 포스트에서는 간단하게 단일 퍼셉트론을 이용하여 AND/OR 연산을 구현해보았다.


다음 포스트는 다층 퍼셉트론(Multi Layer Perceptron, MLP)을 이용하여 XOR 연산을 구현해보고,


MNIST Dataset을 이용하여 손글씨 숫자 분류하기 또한 다룰 예정이다.


Deep Learning 파트로 넘어오니까 확실히 이론 내용이 많아졌다.


이 내용들을 정리해서 포스트를 올리기도 상당히 어려울 것 같다.


복잡한 개념들을 이미지 없이, 그리고 한정적인 수식으로 설명하기에는 한계가 있을 것 같아서...!


그래서 아마 Deep Learning 파트에서는 이론에 대해서 정말 간략하게나 정리 없이 포스팅을 하게 될 것 같다.


아무튼 다음 포스트에서 보도록 하겠다!


　


　


P.S. 군학점 강의도 듣느랴, 머신러닝/딥러닝 독학도 하느랴 정신이 없다.


정해진 스케줄대로 하는게 아니라 '이거 하다가, 저거 하다가' 왔다갔다해서 그런 것 같다.


뭔가 스케줄을 짜서 해야할 것 같다...
