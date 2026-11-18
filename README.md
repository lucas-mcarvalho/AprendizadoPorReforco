# Aprendizado por Reforço — Q-Learning no CliffWalking

Implementação do algoritmo **Q-Learning** (com uma matriz Q em NumPy, sem bibliotecas de RL)
para resolver um problema de **planejamento de trajetória**: o ambiente
**CliffWalking-v1** do Gymnasium.


## Vídeo da apresentação
https://github.com/user-attachments/assets/83d411a2-eaf0-456d-b9e2-fdc7b937c0f3

## O problema
<https://gymnasium.farama.org/environments/toy_text/cliff_walking/>

```
   .  .  .  .  .  .  .  .  .  .  .  .
   .  .  .  .  .  .  .  .  .  .  .  .
   .  .  .  .  .  .  .  .  .  .  .  .
   S  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  G
```

- Grade de 4 × 12 = **48 estados** discretos; **4 ações**: cima, direita, baixo, esquerda
- O agente sai de **S** e precisa chegar em **G** sem cair no penhasco (▓)
- Recompensa **−1** a cada passo e **−100** ao cair no penhasco (o agente volta para S)
- Então o melhor caminho é o mais curto que não cai: **13 passos**, recompensa **−13**

## O algoritmo
A matriz Q (48 × 4) começa zerada. A cada passo o agente escolhe uma ação com a política
**ε-greedy** (com probabilidade ε faz uma ação aleatória, senão faz a de maior Q), observa a
recompensa `r` e o novo estado `s'` e atualiza:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

| Parâmetro | Valor | Significado |
|---|---|---|
| α (alpha) | 0,1 | taxa de aprendizado |
| γ (gamma) | 0,99 | peso das recompensas futuras |
| ε (epsilon) | 0,1 | chance de explorar |
| Episódios | 500 | máximo de 200 passos cada |

## Como executar
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python q_learning.py              # treina (≈ 3 s) e gera os gráficos
.venv/bin/python q_learning.py --visualizar # também abre uma janela com o agente andando
```
O terminal mostra a matriz Q nos episódios 1, 10, 50 e 500, a política final em setas e o
caminho percorrido. Os gráficos e a matriz Q completa (CSV) ficam em `resultados/`.

## Resultados
- A política aprendida leva o agente de S a G em **13 passos (recompensa −13)**, que é o
  ótimo: sobe uma casa, anda pela beira do penhasco e desce no G.
- Nos 50 primeiros episódios a recompensa média foi −172,5; nos 50 últimos, −68,0. A média
  final não chega a −13 porque durante o treino o agente ainda explora 10% das vezes, e um
  passo aleatório na beira do penhasco custa −100.
- A matriz Q final mostra o que o agente aprendeu: na linha logo acima do penhasco,
  Q(s, baixo) fica perto de −100, enquanto as outras ações valem cerca de −1 a −12
  (≈ −número de passos que faltam até G).

| Gráfico | O que mostra |
|---|---|
| `resultados/evolucao_matriz_q.png` | max Q e melhor ação de cada estado após os episódios 1, 10, 50 e 500 |
| `resultados/matriz_q.png` | a matriz Q final completa, um mapa para cada ação |
| `resultados/politica_aprendida.png` | política final e o caminho do agente |
| `resultados/recompensa_por_episodio.png` | recompensa e número de passos por episódio |

## Arquivos
- `q_learning.py` — código-fonte
- `resultados/` — gráficos e `matriz_q.csv`
- `apresentacao/` — slides (`slides_qlearning.tex`/`.pdf`) e roteiro do vídeo
