# Roteiro do vídeo — leitura completa (≈ 4 min 30 s, limite de 5 min)

> O que está entre colchetes **[assim]** é só indicação do que fazer. Não é para ler em voz alta.

---

## [Slide 1 — Capa]

Olá, eu sou o Lucas. Neste vídeo vou mostrar a implementação do algoritmo Q-Learning, que é um
algoritmo de aprendizado por reforço, para resolver um problema de planejamento de trajetória.

## [Slide 2 — O problema]

O problema escolhido foi o CliffWalking, que é um ambiente da biblioteca Gymnasium.

É uma grade com 4 linhas e 12 colunas, ou seja, 48 estados. O agente começa no S, no canto
de baixo à esquerda, e tem que chegar no G, no canto de baixo à direita. Entre os dois fica um
penhasco. Em cada estado o agente pode fazer 4 ações: ir para cima, para a direita, para baixo
ou para a esquerda.

Cada passo dá uma recompensa de menos 1, então o agente quer chegar rápido. Se ele cair no
penhasco, ganha menos 100 e volta para o início. Então o melhor caminho é esse em vermelho:
subir uma casa, andar pela beira do penhasco e descer no G. São 13 passos.

O agente não conhece o mapa. Ele tem que descobrir isso sozinho, por tentativa e erro.

## [Slide 3 — Como funciona o Q-Learning]

Para isso o Q-Learning usa uma tabela chamada matriz Q. Ela tem uma linha para cada estado e
uma coluna para cada ação, então aqui ela é 48 por 4. Cada valor diz o quanto vale fazer
aquela ação naquele estado. No começo ela é toda zero.

A cada passo, o agente escolhe uma ação. Na maioria das vezes ele escolhe a ação com o maior
valor na tabela, mas em 10% das vezes ele escolhe uma ação aleatória, para explorar. Isso se
chama política epsilon-greedy.

Depois ele recebe a recompensa e atualiza a tabela com esta fórmula. Ela pega a recompensa
recebida mais o melhor valor do próximo estado, e puxa o valor antigo um pouco nessa direção.
O alfa controla o tamanho desse passo, e o gama dá o peso das recompensas futuras.

Isso se repete por 500 episódios. Um episódio vai do S até o G.

## [Slide 4 — Resultados]

Aqui está o resultado. A cor é o maior valor Q de cada estado e a seta é a melhor ação. A
linha vermelha é o caminho que o agente faz depois de treinado: exatamente os 13 passos do
caminho ótimo.

No começo do treino a recompensa média era de menos 172, porque o agente caía muito. No final
foi para menos 68. Não chega a menos 13 durante o treino porque ele continua explorando 10% das
vezes, e um passo aleatório na beira do penhasco custa menos 100.

E na linha logo acima do penhasco, o valor da ação "baixo" ficou perto de menos 100. Ou seja,
o agente aprendeu que descer ali é péssimo.

Agora vou mostrar o código.

---

## [Abrir o `q_learning.py`]

O código foi feito em Python. Usei o Gymnasium só para o ambiente. O Q-Learning foi
implementado na mão, com uma matriz do NumPy.

**[Mostrar as constantes no topo]**
Aqui ficam os hiperparâmetros: alfa 0,1, gama 0,99, epsilon 0,1 e 500 episódios.

**[Mostrar a função `escolhe_acao`]**
Esta é a política epsilon-greedy. Sorteia um número: se for menor que epsilon, faz uma ação
aleatória. Senão, pega a ação de maior valor na linha da matriz Q.

**[Mostrar a seção 2 — Treinamento]**
Aqui a matriz Q é criada com zeros, 48 por 4. Para cada episódio, o ambiente é reiniciado e o
agente vai dando passos. **[apontar as duas linhas da atualização]** Estas duas linhas são o
Q-Learning: calcula o alvo, que é a recompensa mais gama vezes o máximo do próximo estado, e
atualiza o valor Q com a taxa alfa.

## [Rodar no terminal: `.venv/bin/python q_learning.py --visualizar`]

Vou rodar. O treino leva só uns 3 segundos.

**[Rolar a saída até a matriz Q do episódio 1, depois 10, 50 e 500]**
O programa mostra a matriz Q em alguns episódios, para dar para ver ela sendo atualizada.
No episódio 1 quase tudo ainda é zero. Ao longo dos episódios os valores vão ficando mais
negativos e mais organizados. No 500, para o estado 36, que é o início, a ação "direita", que
cai no penhasco, vale menos 101, e as outras valem cerca de menos 12.

**[Mostrar a política em setas e o caminho]**
Aqui está a política final em setas e o caminho: 13 passos, recompensa menos 13.

**[A janela do agente abre — deixar ele andar até o G]**
E esta janela mostra o agente treinado andando pela beira do penhasco até o objetivo.

**[Abrir `resultados/evolucao_matriz_q.png`]**
Por fim, este gráfico mostra a evolução da matriz Q. Depois do episódio 1 só a parte da esquerda,
por onde o agente andou, foi atualizada. Com o tempo os valores se espalham pela grade, e no final todas
as setas da linha acima do penhasco apontam para a direita, em direção ao G.

Era isso. Obrigado!

---

**Lembretes**
- O vídeo tem limite de **5 minutos**. Treine a leitura uma vez cronometrando.
- Rode o código uma vez antes de gravar, para garantir que a `.venv` está funcionando.
- A janela do `--visualizar` abre só depois que os gráficos são salvos e anda a 3 passos por
  segundo (cerca de 5 s).
