# seclab

### Hyper-parameters

| method | FGS | PGD | CW | BIM | MIM |
|---|:---:|:---:|:---:|:---:|:---:|
| eps | 0.3 | 0.3 | - | 0.3 | 0.3 |
| step_size (eps_iter) | - | 0.01 | - | 0.01 | 0.01 |
| iter (nb_iter, max_iter) | - | 40 | <1000 | 40 | 40 |
| norm | L_inf | L_inf | L_2 | L_inf | L_inf |
| clip_min | 0 | 0 | 0 | 0 | 0 |
| clip_max | 1 | 1 | 1 | 1 | 1 |
| rand_minmax | - | 0.3 | - | - | False |
| binary_search_steps | - | - | 5 | - | - |
| targeted | False | False | False | False | False |
| confidence | - | - | 0 | - | - |
| init_const | - | - | 1e-2 | - | - |
| lr | - | - | 5e-3 | - | - |
| decay_factor | - | - | - | - | 1 |

* eps (\epsilon): 왜곡의 크기
* step_size: 
