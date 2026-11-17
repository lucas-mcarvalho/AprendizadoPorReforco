"""
Teste diagnóstico - Aprendizado por Reforço
Algoritmo Q-Learning para planejamento de trajetória

Ambiente: CliffWalking (Gymnasium) - "caminhada à beira do penhasco"
  https://gymnasium.farama.org/environments/toy_text/cliff_walking/
  - Grade de 4 linhas x 12 colunas = 48 estados discretos
  - 4 ações: 0 = cima, 1 = direita, 2 = baixo, 3 = esquerda
  - O agente começa no canto inferior esquerdo (S) e precisa chegar ao
    canto inferior direito (G). Entre os dois fica o penhasco.
  - Recompensa: -1 a cada passo e -100 ao cair no penhasco (volta para S)

Q-Learning (Watkins, 1989):
  Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Metodologia:
  1. Matriz Q 48 x 4 começa zerada
  2. Treino por N episódios com política epsilon-greedy (exploração)
  3. A matriz Q é exibida em alguns episódios para mostrar a atualização
  4. Avaliação da política gulosa final (sem exploração)

Uso:
  python q_learning.py              # treina e gera gráficos
  python q_learning.py --visualizar # também abre uma janela com o agente andando
"""

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
PASTA_SAIDA = "resultados"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Hiperparâmetros do Q-Learning
ALPHA = 0.1         # taxa de aprendizado
GAMMA = 0.99        # fator de desconto (peso das recompensas futuras)
EPSILON = 0.1       # probabilidade de explorar (ação aleatória)
EPISODIOS = 500
MAX_PASSOS = 200    # limite de passos por episódio

LINHAS, COLUNAS = 4, 12
ACOES = ["cima", "direita", "baixo", "esquerda"]
SETAS = ["↑", "→", "↓", "←"]
INICIO, OBJETIVO = 36, 47
PENHASCO = set(range(37, 47))

# Episódios em que a matriz Q é guardada para mostrar a evolução
EPISODIOS_SNAPSHOT = [1, 10, 50, EPISODIOS]


def linha(titulo=""):
    print("\n" + "=" * 70)
    if titulo:
        print(titulo)
        print("=" * 70)


def estado_para_posicao(s):
    return s // COLUNAS, s % COLUNAS


def escolhe_acao(Q, s, epsilon, rng):
    """Política epsilon-greedy: explora com prob. epsilon, senão usa a melhor ação."""
    if rng.random() < epsilon:
        return int(rng.integers(len(ACOES)))
    # desempate aleatório entre ações com o mesmo valor (ex.: Q ainda zerada)
    melhores = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(melhores))


def imprime_matriz_q(Q, estados):
    print(f"  {'estado':>6} {'(lin,col)':>9} " + " ".join(f"{a:>9}" for a in ACOES))
    for s in estados:
        lin, col = estado_para_posicao(s)
        valores = " ".join(f"{v:>9.2f}" for v in Q[s])
        print(f"  {s:>6} {f'({lin},{col})':>9} {valores}")


def imprime_politica(Q):
    for lin in range(LINHAS):
        celulas = []
        for col in range(COLUNAS):
            s = lin * COLUNAS + col
            if s == INICIO:
                celulas.append("S")
            elif s == OBJETIVO:
                celulas.append("G")
            elif s in PENHASCO:
                celulas.append("▓")
            else:
                celulas.append(SETAS[int(np.argmax(Q[s]))])
        print("   " + "  ".join(celulas))


def episodio_guloso(env, Q):
    """Executa a política aprendida sem exploração e devolve o caminho."""
    s, _ = env.reset(seed=SEED)
    caminho, total = [s], 0
    for _ in range(MAX_PASSOS):
        a = int(np.argmax(Q[s]))
        s, r, terminou, truncou, _ = env.step(a)
        caminho.append(s)
        total += r
        if terminou or truncou:
            break
    return caminho, total


parser = argparse.ArgumentParser()
parser.add_argument("--visualizar", action="store_true",
                    help="abre uma janela mostrando o agente treinado andando na grade")
args = parser.parse_args()

# ----------------------------------------------------------------------
# 1. Ambiente
# ----------------------------------------------------------------------
env = gym.make("CliffWalking-v1")
n_estados = env.observation_space.n
n_acoes = env.action_space.n
rng = np.random.default_rng(SEED)

linha("1. AMBIENTE")
print(f"Ambiente: {env.spec.id}")
print(f"Estados: {n_estados} (grade {LINHAS} x {COLUNAS}) | Ações: {n_acoes} {ACOES}")
print(f"Início S = estado {INICIO} | Objetivo G = estado {OBJETIVO}")
print("Recompensas: -1 por passo | -100 ao cair no penhasco (▓)")
print("\nMapa (S = início, G = objetivo, ▓ = penhasco):")
for lin in range(LINHAS):
    print("   " + "  ".join(
        "S" if s == INICIO else "G" if s == OBJETIVO else "▓" if s in PENHASCO else "."
        for s in range(lin * COLUNAS, (lin + 1) * COLUNAS)))

# ----------------------------------------------------------------------
# 2. Treinamento com Q-Learning
# ----------------------------------------------------------------------
Q = np.zeros((n_estados, n_acoes))
recompensas, passos, quedas = [], [], []
snapshots = {}

# estados mostrados no terminal: a linha de baixo da grade e a de cima do penhasco
ESTADOS_EXIBIDOS = [24, 25, 34, 35, 36]

linha("2. TREINAMENTO (Q-Learning)")
print(f"alpha = {ALPHA} | gamma = {GAMMA} | epsilon = {EPSILON} | episódios = {EPISODIOS}")

for ep in range(1, EPISODIOS + 1):
    s, _ = env.reset(seed=SEED if ep == 1 else None)
    total, n_quedas = 0, 0
    for t in range(MAX_PASSOS):
        a = escolhe_acao(Q, s, EPSILON, rng)
        s_novo, r, terminou, truncou, _ = env.step(a)

        # Atualização do Q-Learning (off-policy: usa o MAX do próximo estado)
        alvo = r + GAMMA * np.max(Q[s_novo]) * (not terminou)
        Q[s, a] += ALPHA * (alvo - Q[s, a])

        total += r
        n_quedas += r == -100
        s = s_novo
        if terminou or truncou:
            break

    recompensas.append(total)
    passos.append(t + 1)
    quedas.append(n_quedas)

    if ep in EPISODIOS_SNAPSHOT:
        snapshots[ep] = Q.copy()
        print(f"\n--- Matriz Q após o episódio {ep} "
              f"(recompensa = {total}, passos = {t + 1}, quedas = {n_quedas}) ---")
        imprime_matriz_q(Q, ESTADOS_EXIBIDOS)

media_ini = np.mean(recompensas[:50])
media_fim = np.mean(recompensas[-50:])
print(f"\nRecompensa média nos 50 primeiros episódios: {media_ini:.1f}")
print(f"Recompensa média nos 50 últimos episódios:   {media_fim:.1f}")

# ----------------------------------------------------------------------
# 3. Política aprendida e avaliação gulosa
# ----------------------------------------------------------------------
linha("3. POLÍTICA APRENDIDA (melhor ação de cada estado)")
imprime_politica(Q)

caminho, total_guloso = episodio_guloso(env, Q)
print(f"\nExecutando a política aprendida sem exploração (epsilon = 0):")
print(f"  Passos até o objetivo: {len(caminho) - 1} | Recompensa total: {total_guloso:.0f}")
print(f"  Caminho (estados): {' -> '.join(map(str, caminho))}")
print("  O caminho ótimo tem 13 passos (recompensa -13): sobe, anda pela beira do "
      "penhasco e desce no G.")

# Matriz Q completa em CSV
df_q = pd.DataFrame(Q.round(3), columns=ACOES)
df_q.insert(0, "coluna", [estado_para_posicao(s)[1] for s in range(n_estados)])
df_q.insert(0, "linha", [estado_para_posicao(s)[0] for s in range(n_estados)])
df_q.insert(0, "estado", range(n_estados))
df_q["melhor_acao"] = [ACOES[int(np.argmax(Q[s]))] for s in range(n_estados)]
df_q.to_csv(f"{PASTA_SAIDA}/matriz_q.csv", index=False)

# ----------------------------------------------------------------------
# 4. Gráficos
# ----------------------------------------------------------------------
def media_movel(x, janela=10):
    return np.convolve(x, np.ones(janela) / janela, mode="valid")


def desenha_grade(ax, Q, titulo, caminho=None, anotar=True):
    V = Q.max(axis=1).reshape(LINHAS, COLUNAS).copy()
    mascara = np.zeros_like(V, dtype=bool)
    for s in PENHASCO:
        mascara[estado_para_posicao(s)] = True
    V_plot = np.ma.masked_array(V, mascara)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#5a3a29")
    im = ax.imshow(V_plot, cmap=cmap)
    for s in range(n_estados):
        lin, col = estado_para_posicao(s)
        if s in PENHASCO:
            ax.text(col, lin, "▓", ha="center", va="center", color="white", fontsize=8)
        elif s == OBJETIVO:
            ax.text(col, lin, "G", ha="center", va="center", color="white",
                    fontsize=12, weight="bold")
        else:
            txt = SETAS[int(np.argmax(Q[s]))] if Q[s].any() else "·"
            if s == INICIO:
                txt = "S" + txt
            ax.text(col, lin - (0.15 if anotar else 0), txt, ha="center", va="center",
                    color="white", fontsize=11, weight="bold")
            if anotar:
                ax.text(col, lin + 0.28, f"{V[lin, col]:.1f}", ha="center",
                        va="center", color="white", fontsize=6)
    if caminho:
        lin_c = [estado_para_posicao(s)[0] for s in caminho]
        col_c = [estado_para_posicao(s)[1] for s in caminho]
        ax.plot(col_c, lin_c, color="tab:red", lw=2.5, alpha=0.8)
    ax.set_xticks(range(COLUNAS))
    ax.set_yticks(range(LINHAS))
    ax.set_title(titulo)
    return im


# 4.1 Recompensa e passos por episódio
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(recompensas, alpha=0.3, color="tab:blue", label="por episódio")
ax1.plot(range(9, EPISODIOS), media_movel(recompensas), color="tab:blue",
         label="média móvel (10)")
ax1.axhline(-13, color="k", ls="--", lw=1, label="ótimo (-13)")
ax1.set_ylim(max(min(recompensas), -500), 0)
ax1.set_ylabel("Recompensa total")
ax1.set_title("Aprendizado do agente ao longo dos episódios")
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)
ax2.plot(passos, alpha=0.3, color="tab:green")
ax2.plot(range(9, EPISODIOS), media_movel(passos), color="tab:green")
ax2.axhline(13, color="k", ls="--", lw=1)
ax2.set_xlabel("Episódio")
ax2.set_ylabel("Passos até o fim")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/recompensa_por_episodio.png", dpi=150)
plt.close(fig)

# 4.2 Política final + caminho guloso
fig, ax = plt.subplots(figsize=(10, 4))
im = desenha_grade(ax, Q, f"Política aprendida e caminho percorrido "
                          f"({len(caminho) - 1} passos, recompensa {total_guloso:.0f})",
                   caminho=caminho)
fig.colorbar(im, ax=ax, label="max Q(s, a)", shrink=0.8)
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/politica_aprendida.png", dpi=150)
plt.close(fig)

# 4.3 Matriz Q completa: um mapa por ação
#     (escala de cores limitada em -20 para não ser dominada pelo -100 do penhasco)
fig, eixos = plt.subplots(2, 2, figsize=(14, 6))
cmap = plt.get_cmap("viridis").copy()
cmap.set_under("#2b0a3d")
cmap.set_bad("#5a3a29")
for a, ax in enumerate(eixos.flat):
    Qa = Q[:, a].reshape(LINHAS, COLUNAS)
    mascara = np.zeros_like(Qa, dtype=bool)
    for s in PENHASCO | {OBJETIVO}:
        mascara[estado_para_posicao(s)] = True
    im = ax.imshow(np.ma.masked_array(Qa, mascara), cmap=cmap, vmin=-20, vmax=0)
    for s in range(n_estados):
        lin, col = estado_para_posicao(s)
        rotulo = "▓" if s in PENHASCO else "G" if s == OBJETIVO else f"{Qa[lin, col]:.1f}"
        cor = "black" if (Qa[lin, col] > -6 and s not in PENHASCO | {OBJETIVO}) else "white"
        ax.text(col, lin, rotulo, ha="center", va="center", color=cor, fontsize=7)
    ax.set_title(f"Q(s, {ACOES[a]} {SETAS[a]})")
    ax.set_xticks(range(COLUNAS))
    ax.set_yticks(range(LINHAS))
fig.colorbar(im, ax=eixos, label="Q(s, a)", shrink=0.8, extend="min")
fig.suptitle("Matriz Q final (48 estados x 4 ações)")
fig.savefig(f"{PASTA_SAIDA}/matriz_q.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4.4 Evolução da matriz Q
fig, eixos = plt.subplots(len(snapshots), 1, figsize=(9, 2.6 * len(snapshots)))
for ax, (ep, Qs) in zip(eixos, snapshots.items()):
    desenha_grade(ax, Qs, f"Após o episódio {ep}", anotar=True)
fig.suptitle("Evolução da matriz Q (cor = max Q, seta = melhor ação)")
fig.tight_layout()
fig.savefig(f"{PASTA_SAIDA}/evolucao_matriz_q.png", dpi=150)
plt.close(fig)

linha("4. ARQUIVOS GERADOS")
for arq in ["recompensa_por_episodio.png", "politica_aprendida.png", "matriz_q.png",
            "evolucao_matriz_q.png", "matriz_q.csv"]:
    print(f"  {PASTA_SAIDA}/{arq}")
env.close()

# ----------------------------------------------------------------------
# 5. Visualização opcional do agente treinado
# ----------------------------------------------------------------------
if args.visualizar:
    env_vis = gym.make("CliffWalking-v1", render_mode="human")
    env_vis.metadata["render_fps"] = 3
    episodio_guloso(env_vis, Q)
    env_vis.close()
