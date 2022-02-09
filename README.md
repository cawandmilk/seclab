# seclab

### Hyper-parameters

| method | FGS | PGD | CW | BIM | MIM |
|---|:---:|:---:|:---:|:---:|:---:|
| eps | 0.3 | 0.3 | - | 0.3 | 0.3 |
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

* eps (epsilon): 왜곡의 크기
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


### Results (MNIST)

|method|FGS|PGD|CW|BIM|MIN|
|:---:|:---:|:---:|:---:|:---:|:---:|
|exec_time (s)|0.37|1.47|6621.95|2.56|2.56|
|adv_acc (%)|28.60|<span style="color: skyblue">**0.05**</span>|<span style="color:firebrick">**87.60**</span>|0.15|0.25|

See [result.json](./result.json).


### Results (CIFAR-10)

|method|FGS|PGD|CW|BIM|MIN|
|:---:|:---:|:---:|:---:|:---:|:---:|
|exec_time (s)|1.28|15.60|4041.94|15.39|20.48|
|adv_acc (%)|<span style="color:firebrick">**28.95**</span>|1.35|<span style="color:skyblue">**0.0**</span>|22.65|21.80|

See [result.json](./result.json).
