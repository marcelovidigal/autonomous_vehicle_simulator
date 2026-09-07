# PRD — Autonomous Vehicle Simulator

> Documentos relacionados: [Plano de trabalho](plano-de-trabalho.md) · [Arquitetura](arquitetura.md)
> Status: rascunho · Data: 2026-09-06

## 1. Visão do produto

Uma aplicação **web didática** onde uma rede neural rasa aprende — diante do usuário — a
dirigir um carro num circuito sinuoso usando apenas sensores de distância. O usuário vê
**ao mesmo tempo** o carro na pista e os **neurônios acendendo**, e pode **mexer nos
hiperparâmetros e mandar re-treinar**, observando como cada escolha muda o aprendizado.

**Frase-guia:** "Ver a rede aprender a dirigir, e entender por quê."

O app abre por padrão na **Solução 1 — Algoritmo Genético**. Uma **barra de menu superior**
alterna entre **Solução 1 (RNA + AG)**, **Solução 2 (DQN)** e **Sobre**. Cada simulação tem
um botão **Tutorial** que abre o tutorial **exclusivo daquela técnica** (com **Voltar** e
download em PDF). As duas soluções resolvem o mesmo problema com o mesmo modelo de
sensores/carro, permitindo compará-las.

## 2. Público-alvo

| Persona | Contexto | O que espera |
|---|---|---|
| **Estudante** de IA/programação | Primeiro contato com redes neurais e otimização | Intuição visual: o que é um neurônio, uma ativação, um hiperparâmetro |
| **Professor / divulgador** | Aula, palestra, oficina | Rodar ao vivo, mexer em 1–2 parâmetros e mostrar o efeito; link para compartilhar |
| **Curioso técnico** | Viu um vídeo de "IA aprende a dirigir" | Brincar sem instalar nada, entender o mecanismo |

Não é para: pesquisa séria de RL, simulação veicular realista, competição de performance.

## 3. Objetivos e não objetivos

### Objetivos
- **O1** — Tornar tangível a relação *sensores → rede → direção* e *hiperparâmetro → aprendizado*.
- **O2** — Aprendizado visível em **< 1–2 min** de treino no navegador, numa máquina comum.
- **O3** — **Zero instalação**: abre por URL; roda 100% no cliente.
- **O4** — Também rodar no desktop a partir do mesmo código (aula sem internet).
- **O5** — Código pequeno e legível, aproveitável como material de estudo.

### Não objetivos
- Física veicular realista (pneus, transferência de carga, aerodinâmica).
- Multiplayer, contas, persistência em nuvem, *leaderboard*.
- 3D. Perseguição de estado-da-arte em RL.
- Editor visual de pistas dentro do app (pistas vêm de JSON; edição é ferramenta offline).

## 4. Histórias de usuário

1. Como estudante, **assisto** a população de carros tentando completar a volta e vejo a
   maioria melhorar a cada geração.
2. Como estudante, **passo o mouse/foco** no carro líder e vejo o **grafo de neurônios**
   reagindo às curvas (entradas dos sensores mudando, saída de direção mudando).
3. Como professor, **aumento a taxa de mutação** e clico **Aplicar**, e mostro que o
   aprendizado fica mais errático.
4. Como professor, **reduzo a camada oculta para 0** e mostro que ainda dá para dirigir
   uma pista simples (controle quase linear).
5. Como usuário, **troco de pista** e vejo a rede treinada antes falhar, motivando um novo
   treino.
6. Como usuário, **salvo** o melhor genoma (download JSON) e **carrego** depois.
7. Como usuário, aperto **Acelerar** e o treino corre sem animação até estabilizar, depois
   assisto só o campeão.
8. Como usuário, aperto **Reset** e tudo volta aos valores padrão.

## 5. Requisitos funcionais

(Referência cruzada com o plano de trabalho; IDs iguais.)

| ID | Requisito | Prioridade |
|---|---|---|
| RF1 | Carro controlado pela rede a partir de sensores normalizados | Must |
| RF2 | Detecção de colisão / saída de pista encerra a corrida do carro | Must |
| RF3 | Sensores configuráveis: nº (3–9), FOV, alcance | Must |
| RF4 | Treinar até completar 1 volta sem colidir em ≥1 pista | Must |
| RF5 | Visualização em tempo real de pista, carro e raios dos sensores | Must |
| RF6 | Grafo da rede sincronizado com o carro em foco: nós coloridos pela ativação **e arestas destacadas pelo sinal** (peso × ativação) | Must |
| RF7 | Painel para editar todos os hiperparâmetros listados no §7 | Must |
| RF8 | Botões: Iniciar/Pausar, Avançar 1 geração, **Aplicar** (aplica os sliders + reinicia o treino; realça quando há mudança pendente; tecla `a`), Reset, Acelerar | Must |
| RF9 | Salvar/Carregar genoma e config como JSON (download/upload) | Should |
| RF10 | ≥2 pistas embutidas, troca em runtime | Must |
| RF11 | HUD: geração, fitness (melhor/média), carros vivos, progresso/voltas | Must |
| RF12 | Curva fitness × geração ao vivo | Should |
| RF13 | *Presets* de hiperparâmetros ("estável", "agressivo", "mínimo") | Could |
| RF14 | *Tooltips* explicando cada hiperparâmetro | Should |
| RF16 | **Barra de menu superior** (abas RNA+AG / DQN / Sobre); app abre na Solução 1 | Must |
| RF17 | **Solução 2 — DQN**: rede Q por gradiente (backprop manual) + *replay* + rede-alvo + ε-greedy; treina até 1 volta; **carro-demonstração a ritmo fixo** + grafo da rede Q + curva de recompensa/episódio; sliders: ocultos, lr, γ, ε; botão **Aplicar** (mesma semântica do RF8) | Must |
| RF18 | **Tela Sobre** com a descrição do projeto | Must |
| RF19 | **Tutorial por técnica**, aberto de dentro da simulação, com **Voltar**; conteúdo exclusivo (raycast no início; prós/contras; código passo a passo); **Baixar PDF** | Must |
| RF20 | **Menu Configurações**: (a) idioma do app — **English (en-US)** (padrão) / **Português (pt-BR)**; (b) **velocidade da simulação** — 30 / 60 / 120 FPS / sem limite. Ambos aplicam na hora e persistem (localStorage no web / `~/.avs_settings.json` no desktop) | Must |
| RF21 | **Tela Sobre**: botão **Copiar link** com a URL pública (aparece quando `src/app/links.py:PUBLISHED_URL` estiver preenchido) | Should |

## 6. UX e layout

Tela única, três regiões (responsivo; em telas estreitas vira abas):

```
┌───────────────────────────────┬───────────────────────┐
│                               │  REDE NEURAL          │
│      PISTA + CARROS           │  (grafo de neurônios, │
│      + raios de sensor        │   cor = ativação)     │
│      + HUD (canto)            │                       │
│                               ├───────────────────────┤
│                               │  FITNESS × GERAÇÃO    │
├───────────────────────────────┴───────────────────────┤
│  PAINEL: sliders de hiperparâmetros · pista ·          │
│  [Iniciar/Pausar] [+1 ger] [Aplicar] [Reset] [Pista]   │
│  [Acelerar] [Salvar] [Carregar]                        │
└───────────────────────────────────────────────────────┘
```

- **Carro em foco**: por padrão o líder da geração; realçado; é dele o grafo de neurônios.
- **Cores das ativações**: divergente azul(−)/vermelho(+); legenda pequena fixa.
- **Feedback de treino**: barra/ível "geração X, melhor fitness Y"; curva atualizando.
- **Estados**: ocioso · treinando · pausado · "acelerando (sem render)".
- **Primeiro carregamento** (web): tela de *loading* explicando que baixa o runtime uma vez.
- **Acessibilidade**: navegação por teclado nos controles; não depender só de cor (usar
  também tamanho/rótulo nos nós).

## 7. Hiperparâmetros (contrato da UI)

Padrões conforme implementados em `core/config.py`. **P** = exposto como slider no painel do MVP (8 no total); os demais ficam no padrão e só mudam via API/JSON.

| Grupo | Parâmetro | Faixa / opções | Padrão | P |
|---|---|---|---|:-:|
| Rede | camada oculta (neurônios) | 0–16 | 6 | ✓ |
| Rede | ativação | tanh · relu | tanh | |
| Sensores | quantidade | 3–9 | 5 | ✓ |
| Sensores | FOV total | 60°–200° | 160° | |
| Sensores | alcance | 50–300 (u) | 150 | |
| Física | velocidade máx. | 40–200 | 140 | |
| Física | aceleração | 20–200 | 120 | |
| Física | taxa de esterço | 60–300 °/s | 170 | |
| Física | atrito (por passo) | 0–1 | 0.02 | |
| Física | dt | fixo | 1/60 | |
| GA | população | 10–80 | 30 | ✓ |
| GA | gerações (limite) | 5–200 | 45 | |
| GA | taxa de mutação | 0–1 | 0.12 | ✓ |
| GA | força de mutação (σ) | 0.01–1 | 0.18 | ✓ |
| GA | elitismo | 0–8 | 2 | ✓ |
| GA | torneio (k) | 2–8 | 3 | |
| GA | crossover | on/off | on | |
| GA | seed | inteiro | 0 | |
| Fitness | peso progresso | 0–2 | 1.0 | |
| Fitness | peso velocidade | 0–1 | 0.4 | ✓ |
| Fitness | peso suavidade | 0–1 | 0.05 | |
| Fitness | penalidade colisão | 0–5 | 1.0 | ✓ |
| Fitness | bônus de volta | 0–2 | 0.5 | |
| Simulação | passos máx./genoma | 200–5000 | 1300 | |
| Simulação | pista | circuito_1..3 | "circuito_1" | (botão) |

Alterar um slider **não** aplica sozinho: o botão **Aplicar** realça e, ao ser clicado
(ou tecla `a`), reconstrói a config e reinicia o treino. *Tooltips* explicam o efeito.

## 8. Métricas de sucesso

| Métrica | Alvo | Situação |
|---|---|---|
| Tempo até "primeira volta completa" (config padrão, seed 0, desktop) | < 30 s | ~12 s (≈ geração 5) ✅ |
| Treino completo de 45 gerações (desktop) | < 90 s | ~55 s ✅ |
| FPS de visualização (web / desktop) | ≥ 20 / ≥ 30 | a medir no navegador |
| *Site* estático próprio | ≤ 200 KB | ~76 KB ✅ (runtime Python/NumPy vem do CDN pygbag, ~10 MB, cacheado) |
| LOC do núcleo (`core/`) | < 1500 | 831 ✅ |
| Testes | núcleo coberto + 1 teste de aprendizado | 46 testes, verdes ✅ |
| Determinismo: mesma seed ⇒ mesma curva de fitness | 100% | ✅ (teste `test_trainer`) |

## 9. Restrições

- **Núcleo só com `numpy` + stdlib** (compatível com Pyodide/WASM).
- **Sem threads** no caminho web: treino incremental cooperativo.
- **Sem backend** na entrega principal: site estático (GitHub Pages / HF Spaces *Static*).
- Multiplataforma desktop (Windows/macOS/Linux) pelo mesmo código.
- Licença: código aberto (MIT) por ser material didático.

## 10. Fora de escopo (v1)

Editor de pista no app · NEAT · 3D · som · gravação de vídeo embutida ·
i18n além de en-US/pt-BR · mobile-first (funciona, mas não é o alvo).

## 11. Marcos de release

- **M1 — Núcleo prova o conceito** (headless): fases 0–3 do plano. Um carro aprende a volta.
- **M2 — App desktop jogável**: fases 4–6. Visual completo + painel + re-treino.
- **M3 — Web público**: fase 7. URL no GitHub Pages + espelho HF Spaces.
- **M4 — Didático polido**: fase 8. Pistas, presets, tooltips, README, roteiro de aula.
- **M5 (opcional) — Trilha B**: fase 9. FastAPI + PixiJS/D3.

## 12. Riscos de produto

| Risco | Impacto | Mitigação |
|---|---|---|
| Aprendizado lento demais na config padrão | Usuário desiste antes de ver resultado | Ajustar defaults com base em medições; preset "rápido"; botão Acelerar |
| Grafo de neurônios "bonito mas confuso" | Não cumpre o objetivo didático | Rótulos claros, poucos neurônios, legenda, tooltip "o que estou vendo" |
| Download inicial assusta no 1º acesso | Abandono | Tela de loading explicando; peso ≤ 15 MB; cache do navegador |
| Diferença de comportamento web × desktop | Confunde em aula | Mesmo núcleo determinístico; documentar só as diferenças de FPS/carregamento |
