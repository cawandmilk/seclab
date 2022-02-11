# seclab

### Hyper-parameters

| method | FGS | PGD | CW | BIM | MIM |
|---|:---:|:---:|:---:|:---:|:---:|
| eps |-|-|-|-|-|
| eps_iter | - | 0.01 | - | 0.01 | 0.01 |
| lr | - | - | 5e-3 | - | - |
| nb_iter | - | 40 | - | 40 | 40 |
| max_iterations | - | - | 100 | - | - |
| norm | L_inf | L_inf | L_2 | L_inf | L_inf |
| clip_min | 0 | 0 | 0 | 0 | 0 |
| clip_max | 1 | 1 | 1 | 1 | 1 |
| rand_init | - | True | - | - | - |
| rand_minmax | - | 0.3 | - | - | - |
| targeted | False | False | False | False | False |
| confidence | - | - | 40 | - | - |
| init_const | - | - | 1e-2 | - | - |
| binary_search_steps | - | - | 5 | - | - |
| decay_factor | - | - | - | - | 1 |

* eps (epsilon): 왜곡의 크기 (variable)
* eps_iter (a): 각 attack iteration에 대한 step size
* lr: 적대적 예제 생성을 위한 learning rate (CW 한정)
* nb_iter: Attack iteration 반복 횟수
* max_iterations: 최대로 반복할 iteration의 횟수 (CW 한정)
* norm: L_2 or L_inf
* clip_min: Box-constraints의 최솟값
* clip_max: Box-constraints의 최댓값
* rand_init: 왜곡을 rand_mimnax-boundary에서 수행할 것인지 여부
* rand_minmax: 왜곡을 uniform distribution에서 선택하기 위한 norm ball의 크기
* targeted: Targeted attack or Untargeted attack 결정
* confidence (kappa): 적대적 예제의 신뢰도 요구 사항
* init_const (c): 초기 c값
* binary_search_steps: 상수 c를 찾기 위한 binary search step
* decay_factor: momentum의 감소 비율

### Accuracies for Adversarial Examples (%): MNIST 

|epsilon|FGS|PGD|CW|BIM|MIM|
|:---:|:---:|:---:|:---:|:---:|:---:|
|-|-|-|87.60|-|-|
|0.1|75.30|31.70|-|31.60|35.20|
|0.2|45.55|0.55|-|0.50|1.25|
|0.3|28.00|0.10|-|0.15|0.25|
|0.4|17.15|0.10|-|0.15|0.25|
|0.5|12.30|0.00|-|0.15|0.25|
|0.6|11.15|0.10|-|0.15|0.25|

### Accuracies for Adversarial Examples (%): CIFAR-10 

|epsilon|FGS|PGD|CW|BIM|MIM|
|:---:|:---:|:---:|:---:|:---:|:---:|
|-|-|-|0.0|-|-|
|0.1|29.95|4.45|-|24.30|23.80|
|0.2|30.80|1.70|-|24.30|23.80|
|0.3|30.55|1.80|-|24.25|23.80|
|0.4|30.50|1.10|-|24.25|23.80|
|0.5|30.55|1.45|-|24.25|23.80|
|0.6|30.85|1.35|-|24.25|23.80|

See [result.csv](./result.csv)

### Adversarial Examples (Google Drive)

* For MNIST: [here](https://drive.google.com/file/d/1Go1IeKV80YSFuFO4I1JK9cOR06icLH18/view?usp=sharing)
* For CIFAR-10: [here](https://drive.google.com/file/d/1HDxzaiEUfcwup7qkq2AeenLHiwKTMLbJ/view?usp=sharing)