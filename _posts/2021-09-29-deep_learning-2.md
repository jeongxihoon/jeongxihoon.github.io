---
title: "[Deep Learning] 난관에 봉착했다...!"
excerpt: "딥러닝 모델을 사지방에서 학습시키는 것은 어려운 것일까..."

categories:
  - Deep Learning

comments: true

mathjax: true

date: 2021-09-29
last_modified_at: 2021-09-29
---


지난 '퍼셉트론으로 AND/OR 연산 구현하기!' 포스트 이후로 거의 일주일 만에 쓰는 포스트이다.


일주일간 무엇을 하고 지냈는지 짧게 소개하자면,


일단 학교 수업 듣고 복습을 했고, 책으로 딥러닝 공부를 좀 했다.


CNN이랑 RNN에 대한 이론 공부를 한 상태이다!


그냥 책으로 보고 내 스스로 이해하는 것은 너무 얄팍하게 배우는 것 같아서 유튜브로 김성범 교수님 강의를 찾아 들었다.


세상에 이렇게 질좋은 강의를 유튜브에 그냥 올려주시다니ㅠㅠㅠ


너무나도 이해가 잘됐고, 상당히 흥미로웠다.


　


　


아무튼 그렇게 공부를 나름 하면서 일주일을 보내고,


오늘 책에 나와있는 CNN 알고리즘의 실습 예제를 하려고 코드를 다 써보고 실행을 시켰는데 **난관에 봉착했다.**


**모델 학습이 너무나도 오래 걸린다...**


사실 MLP(다층 퍼셉트론)로 XOR 연산 구현하는 코드도 학습량이 상당해서 모델 학습에만 20분을 쏟았다.


하지만 XOR 연산은 상당히 단순한 학습이였는데도 20분이 걸렸다.


근데 지금 CNN 알고리즘을 학습시켜놓고 포스트를 쓰고 있는데


10분정도 지났는데 아직 1 epoch의 1/3도 학습을 하지 못했다...


epoch도 5로 설정해서 그렇게 많은 주기로 학습을 지정하지도 않았는데 이렇게 오래걸리면 어떡해...


이번에도 MNIST 숫자 Dataset을 이용했다.


포스트의 많은 실습에서 게속해서 사용했던 그 Dataset이다.


뒤로 갈수록 복잡한 학습이 더 많아질텐데 약간 한계가 있을 것 같다.


마냥 학습이 다 될 때까지 기다리는 것도 무리고,


만약 중간에 오류라도 나서 처음부터 다시 학습을 해야된다면... 감당 불가능...


아무래도 사지방에서 계속적인 딥러닝 공부는 무리가 있을 듯 싶다.


지금 하고 있는 책도 딥러닝 알고리즘이 3~4개 정도밖에 안남아서 다음 책을 고민 중인데


아무래도 다시 데이터 분석과 가벼운 머신러닝 쪽으로 방향을 돌려야할 것 같다!


사실 나도 실제 Data를 전처리하고, 뭐하고 해서 유의미한 결과를 만드는 걸 해보고 싶긴 하다!


그래프도 막 그리고 어?


아무튼, 이 망할 놈의 사지방 컴퓨터는 무슨 CPU를 쓰길래 연산이 이렇게 느린거야


지금 구름IDE로 스윽 보니까 CPU 사용량이 92~101%를 왔다갔다 한다. ~~101%는 어째서 가능한거임?~~


일단 되는대로 학습해서 최대한 마무리하는 쪽으로 해보는 걸로!!!