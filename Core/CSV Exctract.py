import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn

#CAMINHOS E DISPOSITIVO
Base = Path(__file__).resolve().parent.parent
ARQ = Base / 'dados_processados' / 'treino.csv'
ARQ_VAL = Base / 'dados_processados' / 'validacao.csv'
ARQ_TESTE = Base / 'dados_processados' / 'teste.csv'
ARQUIVO = Base / 'archive'            # CSVs originais (nomes de pilotos, equipes, GPs)

PASTA_GRAF = Base / 'graficos'
PASTA_GRAF.mkdir(exist_ok=True)

dtype = torch.float
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(device)


def coluna(i, arq=ARQ):
    c = pd.read_csv(arq, usecols=[i])
    c = torch.tensor(c.values, dtype=dtype, device=device)
    return c


# IMPORTAÇÃO DOS DADOS
# identificação de pilotos, equipes, ano e corridas (não entram na rede)
RaceID = pd.read_csv(ARQ, usecols=[0])
RaceID = np.array(RaceID)
RaceID = np.transpose(RaceID)

Ano = pd.read_csv(ARQ, usecols=[1])
Ano = np.array(Ano)
Ano = np.transpose(Ano)

PilotoID = pd.read_csv(ARQ, usecols=[3])
PilotoID = np.array(PilotoID)
PilotoID = np.transpose(PilotoID)

EquipeID = pd.read_csv(ARQ, usecols=[4])
EquipeID = np.array(EquipeID)
EquipeID = np.transpose(EquipeID)

# ---------- Atributos de entrada ----------
Grid            = coluna(5)
LargadaPitlane  = coluna(6)
GapPole         = coluna(7)
PilPontos       = coluna(8)
PilPos          = coluna(9)
PilVitorias     = coluna(10)
EqpPontos       = coluna(11)
EqpPos          = coluna(12)
PilMediaPos     = coluna(13)
PilTaxaPodio    = coluna(14)
EqpMediaPos     = coluna(15)
EqpTaxaAbandono = coluna(16)
PilExperiencia  = coluna(17)
PilIdade        = coluna(18)
RodadaFrac      = coluna(19)

# ---------- Target ----------
Podio = coluna(20)

Xt = torch.hstack((Grid, LargadaPitlane, GapPole, PilPontos, PilPos, PilVitorias,
                   EqpPontos, EqpPos, PilMediaPos, PilTaxaPodio, EqpMediaPos,
                   EqpTaxaAbandono, PilExperiencia, PilIdade, RodadaFrac))
Yt = Podio
print('Treino:', Xt.shape, Yt.shape)

# ---------- Validação ----------
Xval = torch.hstack([coluna(i, ARQ_VAL) for i in range(5, 20)])
Yval = coluna(20, ARQ_VAL)
print('Validação:', Xval.shape, Yval.shape)

# ---------- Teste ----------
Xte = torch.hstack([coluna(i, ARQ_TESTE) for i in range(5, 20)])
Yte = coluna(20, ARQ_TESTE)
ids_te = pd.read_csv(ARQ_TESTE, usecols=[0, 1, 3, 4])   # raceId, year, driverId, constructorId
print('Teste:', Xte.shape, Yte.shape)


# CONFIGURAÇÕES
m = Xt.shape[1]          # nº de atributos (15)
q = Xt.shape[0]          # nº de amostras de treino
N1, N2, N3 = 32, 16, 8   # neurônios das 3 camadas ocultas
eta = 0.001              # taxa de aprendizado padrão
numepoc = 100            # nº de épocas
batch = 64               # tamanho do lote padrão

# Peso extra para a classe pódio (compensa o desbalanceamento ~14% x 86%)
peso_pos = (Yt == 0).sum() / (Yt == 1).sum()
print('peso_pos =', peso_pos.item())


# MODELO
# cria o modelo (Sucumba-Gustavo Zampier)
def criar_modelo(ativacao):
    return nn.Sequential(
        nn.Linear(m, N1),  ativacao(),
        nn.Linear(N1, N2), ativacao(),
        nn.Linear(N2, N3), ativacao(),
        nn.Linear(N3, 1)                 # saída sem ativação: a loss aplica a sigmoide
    ).to(device)


# 4. TREINAMENTO
def treinar(ativacao, nome, eta=eta, numepoc=numepoc, batch=batch):
    torch.manual_seed(42)         
    model = criar_modelo(ativacao)
    criterion = nn.BCEWithLogitsLoss(pos_weight=peso_pos)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    hist = {'E_treino': [], 'E_val': [], 'acc_val': []}
    melhor_loss, melhor_ep, melhor_estado = float('inf'), 0, None

    for ep in range(numepoc):
        model.train()
        ordem = torch.randperm(q, device=device)
        soma = 0
        for k in range(0, q, batch):
            idx = ordem[k:k + batch]
            output = model(Xt[idx])
            loss = criterion(output, Yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soma += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            out_val = model(Xval)
            e_val = criterion(out_val, Yval).item()
            acc_val = ((torch.sigmoid(out_val) >= 0.5).float() == Yval).float().mean().item()

        # vetor de erros a cada iteração (época)
        hist['E_treino'].append(soma / q)
        hist['E_val'].append(e_val)
        hist['acc_val'].append(acc_val)
        print(f'[{nome}] Epoch [{ep+1}/{numepoc}], Loss treino: {soma/q:.4f}, '
              f'Loss val: {e_val:.4f}, Acc val: {acc_val:.3f}')

        # guarda os pesos da melhor época (menor erro de validação)
        if e_val < melhor_loss:
            melhor_loss, melhor_ep = e_val, ep + 1
            melhor_estado = {n: p.clone() for n, p in model.state_dict().items()}

    model.load_state_dict(melhor_estado)
    hist['melhor_ep'] = melhor_ep
    hist['melhor_loss'] = melhor_loss
    print(f'[{nome}] melhor época: {melhor_ep} (loss val {melhor_loss:.4f})\n')
    return model, hist


# MÉTRICAS E GRÁFICOS
def metricas(model, X, Y):
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(X))
    prev = (prob >= 0.5).float()
    vp = ((prev == 1) & (Y == 1)).sum().item()   # verdadeiro positivo
    fp = ((prev == 1) & (Y == 0)).sum().item()   # falso positivo
    fn = ((prev == 0) & (Y == 1)).sum().item()   # falso negativo
    vn = ((prev == 0) & (Y == 0)).sum().item()   # verdadeiro negativo
    precisao = vp / (vp + fp) if vp + fp else 0
    recall   = vp / (vp + fn) if vp + fn else 0
    f1       = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0
    return {'acuracia': (vp + vn) / len(Y), 'precisao': precisao, 'recall': recall,
            'f1': f1, 'VP': vp, 'FP': fp, 'FN': fn, 'VN': vn}


def plot_curvas(res, titulo, arquivo):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for nome, (model, hist) in res.items():
        ax[0].plot(hist['E_val'], label=nome)
        ax[1].plot(hist['acc_val'], label=nome)
    ax[0].set_title(f'{titulo} – loss de validação'); ax[0].set_xlabel('Época'); ax[0].set_ylabel('Loss')
    ax[1].set_title(f'{titulo} – acurácia de validação'); ax[1].set_xlabel('Época'); ax[1].set_ylabel('Acurácia')
    ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.savefig(PASTA_GRAF / arquivo, dpi=150); plt.show()


ativacoes = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid, 'LeakyReLU': nn.LeakyReLU}

#funções de ativação
res_ativ = {nome: treinar(ativ, nome) for nome, ativ in ativacoes.items()}

tab_ativ = pd.DataFrame({nome: metricas(model, Xval, Yval) for nome, (model, hist) in res_ativ.items()}).T
tab_ativ['melhor_ep'] = [hist['melhor_ep'] for model, hist in res_ativ.values()]
tab_ativ['melhor_loss_val'] = [hist['melhor_loss'] for model, hist in res_ativ.values()]
print('\n=== Comparação das ativações (validação) ===')
print(tab_ativ.round(3))
tab_ativ.round(4).to_csv(PASTA_GRAF / 'tabela_ativacoes.csv')

melhor_nome = tab_ativ['f1'].astype(float).idxmax()
melhor_ativ = ativacoes[melhor_nome]
print('Melhor ativação (maior F1 na validação):', melhor_nome)
plot_curvas(res_ativ, 'Funções de ativação', 'exp1_ativacoes.png')

#taxa de aprendizado
taxas = [0.01, 0.001, 0.0001]
res_lr = {f'lr={lr}': treinar(melhor_ativ, f'lr={lr}', eta=lr) for lr in taxas}
melhor_lr = min(taxas, key=lambda lr: res_lr[f'lr={lr}'][1]['melhor_loss'])
print('Melhor taxa de aprendizado:', melhor_lr)
plot_curvas(res_lr, f'Taxa de aprendizado ({melhor_nome})', 'exp2_taxa_aprendizado.png')

#tamanho do batch
batches = [16, 64, 256]
res_batch = {f'batch={b}': treinar(melhor_ativ, f'batch={b}', eta=melhor_lr, batch=b) for b in batches}
melhor_batch = min(batches, key=lambda b: res_batch[f'batch={b}'][1]['melhor_loss'])
print('Melhor batch:', melhor_batch)
plot_curvas(res_batch, f'Batch ({melhor_nome}, lr={melhor_lr})', 'exp3_batch.png')

# Tabela-resumo dos experimentos de hiperparâmetros
resumo = pd.DataFrame(
    [{'experimento': nome, 'melhor_ep': h['melhor_ep'], 'melhor_loss_val': h['melhor_loss'],
      **metricas(mdl, Xval, Yval)} for nome, (mdl, h) in {**res_lr, **res_batch}.items()])
print('\n=== Resumo taxa de aprendizado e batch (validação) ===')
print(resumo.round(3).to_string(index=False))
resumo.round(4).to_csv(PASTA_GRAF / 'tabela_hiperparametros.csv', index=False)

# ===== Modelo final: treino x validação (justifica as épocas)
model_final, hist_final = treinar(melhor_ativ, 'FINAL', eta=melhor_lr, batch=melhor_batch)

plt.figure(figsize=(8, 4))
plt.plot(hist_final['E_treino'], label='treino')
plt.plot(hist_final['E_val'], label='validação')
plt.axvline(hist_final['melhor_ep'] - 1, color='gray', ls='--',
            label=f'melhor época ({hist_final["melhor_ep"]})')
plt.title(f'Modelo final – {melhor_nome}, lr={melhor_lr}, batch={melhor_batch}')
plt.xlabel('Época'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout(); plt.savefig(PASTA_GRAF / 'exp4_treino_x_validacao.png', dpi=150); plt.show()

# Vetor de erros completo do modelo final
print('\nVetor de erros (loss de treino por época):')
print(np.round(hist_final['E_treino'], 4))
print('Vetor de erros (loss de validação por época):')
print(np.round(hist_final['E_val'], 4))


# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (uma única vez)
print('\n=== RESULTADO NO CONJUNTO DE TESTE ===')
m_teste = metricas(model_final, Xte, Yte)
for chave, valor in m_teste.items():
    print(f'{chave:>9}: {valor:.3f}' if isinstance(valor, float) else f'{chave:>9}: {valor}')
pd.DataFrame([m_teste]).round(4).to_csv(PASTA_GRAF / 'metricas_teste.csv', index=False)

# Matriz de confusão do teste
mc = np.array([[m_teste['VN'], m_teste['FP']], [m_teste['FN'], m_teste['VP']]])
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(mc, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, mc[i, j], ha='center', va='center', fontsize=14,
                color='white' if mc[i, j] > mc.max() / 2 else 'black')
ax.set_xticks([0, 1]); ax.set_xticklabels(['não pódio', 'pódio'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['não pódio', 'pódio'])
ax.set_xlabel('Previsto'); ax.set_ylabel('Real'); ax.set_title('Matriz de confusão – teste')
plt.tight_layout(); plt.savefig(PASTA_GRAF / 'matriz_confusao_teste.png', dpi=150); plt.show()

# --- Top-3 por corrida: os 3 mais prováveis de cada GP = pódio previsto ---
model_final.eval()
with torch.no_grad():
    prob_te = torch.sigmoid(model_final(Xte)).cpu().numpy().ravel()

prev = ids_te.copy()
prev['prob'] = prob_te
prev['podio_real'] = Yte.cpu().numpy().ravel().astype(int)
prev['rank_prev'] = prev.groupby('raceId')['prob'].rank(ascending=False, method='first').astype(int)
prev['podio_prev'] = (prev['rank_prev'] <= 3).astype(int)

acerto_top3 = prev.loc[prev['podio_prev'] == 1, 'podio_real'].mean()
print(f'\nTop-3 por corrida: {acerto_top3:.1%} dos pilotos previstos no pódio realmente subiram ao pódio')

# --- Cruzamento dos IDs com os nomes ---
drivers      = pd.read_csv(ARQUIVO / 'drivers.csv')
constructors = pd.read_csv(ARQUIVO / 'constructors.csv')
races        = pd.read_csv(ARQUIVO / 'races.csv')
results      = pd.read_csv(ARQUIVO / 'results.csv')
drivers['piloto'] = drivers['forename'] + ' ' + drivers['surname']

prev = (prev.merge(drivers[['driverId', 'piloto']], on='driverId')
            .merge(constructors[['constructorId', 'name']].rename(columns={'name': 'equipe'}), on='constructorId')
            .merge(races[['raceId', 'name']].rename(columns={'name': 'gp'}), on='raceId')
            .merge(results[['raceId', 'driverId', 'positionOrder']].rename(columns={'positionOrder': 'pos_real'}),
                   on=['raceId', 'driverId']))

podio_prev = prev[prev['podio_prev'] == 1].sort_values(['year', 'raceId', 'rank_prev'])
print(podio_prev[['year', 'gp', 'rank_prev', 'piloto', 'equipe', 'prob', 'pos_real']]
      .head(15).round(3).to_string(index=False))
podio_prev.to_csv(PASTA_GRAF / 'podios_previstos_teste.csv', index=False)

print('\nArquivos salvos em:', PASTA_GRAF)