import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn 

Base = Path(__file__).resolve().parent.parent
ARQ = Base / 'dados_processados' / 'treino.csv'
ARQ_VAL = Base / 'dados_processados' / 'validacao.csv'

dtype = torch.float
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(device)

def coluna(i, arq=ARQ):
    c = pd.read_csv(arq, usecols=[i])
    c = torch.tensor(c.values, dtype=dtype, device=device)
    return c

#identificaçao de pilotos, equipes, ano e corridas
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

#Validacao
Xval = torch.hstack([coluna(i, ARQ_VAL) for i in range(5, 20)])
Yval = coluna(20, ARQ_VAL)
print(Xval.shape, Yval.shape)

print(Xt.shape)
print(Yt.shape)

#Configs
m = Xt.shape[1]
q = Xt.shape[0]
N1,N2,N3 = 32,16,8
eta = 0.001
numepoc = 100
batch = 64

#Peso Extra
peso_pos = (Yt == 0).sum() / (Yt == 1).sum()
print('peso_pos =', peso_pos.item())

#cria o modelo (Sucumba-Gustavo Zampier)
def criar_modelo(ativacao):
    return nn.Sequential(
        nn.Linear(m, N1),  ativacao(),
        nn.Linear(N1, N2), ativacao(),
        nn.Linear(N2, N3), ativacao(),
        nn.Linear(N3, 1)
    ).to(device)

#treino do modelo
def treinar(ativacao, nome):
    torch.manual_seed(42)
    model = criar_modelo(ativacao)
    #loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=peso_pos)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    hist = {'E_treino': [], 'E_val': [], 'acc_val': []}

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

        #precisao do treino
        model.eval()
        with torch.no_grad():
            out_val = model(Xval)
            e_val = criterion(out_val, Yval).item()
            acc_val = ((torch.sigmoid(out_val) >= 0.5).float() == Yval).float().mean().item()

        #print da perda
            hist['E_treino'].append(soma / q)
            hist['E_val'].append(e_val)
            hist['acc_val'].append(acc_val)
            print(f'[{nome}] Epoch [{ep+1}/{numepoc}], Loss treino: {soma/q:.4f}, '
                f'Loss val: {e_val:.4f}, Acc val: {acc_val:.3f}')

    return model,hist

ativacoes = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid, 'LeakyReLU': nn.LeakyReLU}
resultados = {}

#grafico de erro
for nome, ativ in ativacoes.items():
    resultados[nome] = treinar(ativ, nome)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for nome, (model, hist) in resultados.items():
    ax[0].plot(hist['E_val'], label=nome)
    ax[1].plot(hist['acc_val'], label=nome)
ax[0].set_title('Loss'); ax[0].set_xlabel('Epoca'); ax[0].legend()
ax[1].set_title('Precisao de validacao por epoca'); ax[1].set_xlabel('epoca'); ax[1].legend()
plt.tight_layout()
plt.savefig(Base / 'comparacao_ativacoes.png', dpi=150)
plt.show()