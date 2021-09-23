# Neuron(Node) / 뉴런(노드)

## AND, OR 연산 구현하기

### Node Class 생성


```python
import tensorflow as tf
```


```python
class Node:
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([2, 1]))
        self.b = tf.Variable(tf.random.normal([1, 1]))
        
    def __call__(self, x):
        return self.preds(x)
    
    def preds(self, x):
        out = tf.matmul(x, self.w)
        out = tf.add(out, self.b)
        out = tf.nn.sigmoid(out)
        return out
    
    def loss(self, y_pred, y):
        return tf.reduce_mean(tf.square(y_pred - y))
    
    def train(self, inputs, outputs, learning_rate):
        epochs = range(10000)
        for epoch in epochs:
            with tf.GradientTape() as t:
                current_loss = self.loss(self.preds(inputs), outputs)
                dw, db = t.gradient(current_loss, [self.w, self.b])
                self.w.assign_sub(learning_rate * dw)
                self.b.assign_sub(learning_rate * db)
```

### AND 연산


```python
inputs = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
outputs = tf.constant([[0.0], [0.0], [0.0], [1.0]])

node = Node()


# Train

node.train(inputs, outputs, 0.01)


# Test

assert node([[0.0, 0.0]]).numpy()[0][0] < 0.5
assert node([[0.0, 1.0]]).numpy()[0][0] < 0.5
assert node([[1.0, 0.0]]).numpy()[0][0] < 0.5
assert node([[1.0, 1.0]]).numpy()[0][0] >= 0.5
```

### OR 연산


```python
inputs = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
outputs = tf.constant([[0.0], [0.0], [0.0], [1.0]])

node = Node()


# Train

node.train(inputs, outputs, 0.01)


# Test

assert node([[0.0, 0.0]]).numpy()[0][0] < 0.5
assert node([[0.0, 1.0]]).numpy()[0][0] >= 0.5
assert node([[1.0, 0.0]]).numpy()[0][0] >= 0.5
assert node([[1.0, 1.0]]).numpy()[0][0] >= 0.5
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-4-1e8d02b22e88> in <module>
         13 
         14 assert node([[0.0, 0.0]]).numpy()[0][0] < 0.5
    ---> 15 assert node([[0.0, 1.0]]).numpy()[0][0] >= 0.5
         16 assert node([[1.0, 0.0]]).numpy()[0][0] >= 0.5
         17 assert node([[1.0, 1.0]]).numpy()[0][0] >= 0.5


    AssertionError: 


## 퍼셉트론


> 2개의 Input이 있을 때, 하나의 뉴런으로 2개의 Input을 계산하여 0 또는 1을 Output으로 출력하는 모델


```python
import tensorflow as tf
```

| $x_1$ | $x_2$ | $x_1$ AND $x_2$ | $x_1$ OR $x_2$ |
|-------|-------|-----------------|----------------|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 |

### Data를 함수를 통해 저장하기


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

def get_XOR_data():
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
        [F]
    ]
    
    return X, y
```

### Perceptron Class 구현하기


```python
class Perceptron:
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([3, 1]))
        
    def train(self, X):
        err = 1
        epoch, max_epochs = 0, 20
        while err > 0.0 and epoch < max_epochs:
            epoch += 1
            self.optimize(X)
            err = self.mse(y, self.pred(X)).numpy()
            print('epoch', epoch, 'mse', err)

    def pred(self, X):
        return self.step(tf.matmul(X, self.W))
    
    def mse(self, y, y_hat):
        return tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
    
    def step(self, x):
        return tf.dtypes.cast(tf.math.greater(x, 0), tf.float32)
    
    def optimize(self, X):
        delta = tf.matmul(X, tf.subtract(y, self.step(tf.matmul(X, self.W))), transpose_a = True)
        self.W.assign(self.W + delta)
```

### AND 연산 구현하기


```python
X, y = get_AND_data()
```


```python
perceptron = Perceptron()
```


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


```python
print(perceptron.pred(X).numpy())
```

    [[0.]
     [0.]
     [0.]
     [1.]]



```python
X, y = get_XOR_data()
```


```python
perceptron.train(X)
```

    epoch 1 mse 0.25
    epoch 2 mse 0.5
    epoch 3 mse 0.25
    epoch 4 mse 0.5
    epoch 5 mse 0.5
    epoch 6 mse 0.5
    epoch 7 mse 0.5
    epoch 8 mse 0.5
    epoch 9 mse 0.5
    epoch 10 mse 0.5
    epoch 11 mse 0.5
    epoch 12 mse 0.5
    epoch 13 mse 0.5
    epoch 14 mse 0.5
    epoch 15 mse 0.5
    epoch 16 mse 0.5
    epoch 17 mse 0.5
    epoch 18 mse 0.5
    epoch 19 mse 0.5
    epoch 20 mse 0.5

