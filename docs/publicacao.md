# Plano de publicação — GitHub Pages + Hugging Face Spaces

> **Nada aqui foi executado.** É a proposta para você aprovar. A publicação usa as suas
> contas (GitHub / Hugging Face) e credenciais, então cada passo é feito por você (ou
> autorizado explicitamente).

## 0. Pré-requisitos (uma vez)

- [ ] `git` instalado e conta no GitHub.
- [ ] Conta no Hugging Face + `huggingface_hub` (`pip install huggingface_hub`) ou `git` com token.
- [ ] Verificação local do build web (bloqueia o resto):
  ```bash
  uv pip install -e ".[web,docs]"
  bash scripts/serve_web.sh          # estatico em 127.0.0.1 -> abra http://127.0.0.1:8080
  ```
  Abra por **`127.0.0.1`**, NAO `localhost`, e NAO use `python -m pygbag` para servir
  (força COEP quebrado). Em `127.0.0.1` + estático o preview e identico ao Pages/Spaces:
  o pygbag baixa runtime + NumPy do CDN publico. Publicar = `bash scripts/build_web.sh` ->
  `build/web/`.
  Conferir no navegador: a pista aparece, o carro anda, o grafo de neurônios pulsa,
  os sliders + "Re-treinar" funcionam, FPS ≥ ~20. (Não consigo validar isto por você —
  é o único ponto ainda não verificado.)

## 1. Repositório Git

- [x] `git init -b main` na raiz + commits iniciais já feitos.
- [x] `.gitignore` cobre `.venv/`, `build/`, `dist/`, `*.whl`, `best_genome.json`,
      `.avs_settings.json` e os logs locais (`prompts/prompts_gerais_v1.md`,
      `docs/traceback_v1.md`).
- [x] Pasta física renomeada para `autonomous_vehicle_simulator`; diretório de estado do
      Claude Code copiado para o slug novo.
- [ ] Criar o repo no GitHub (`gh repo create autonomous_vehicle_simulator --public --source . --push`
      ou pela interface) e `git push -u origin main`.

## 2. GitHub Pages (automático via Actions)

Já existe o workflow [`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml):
faz `pip install pygbag`, roda `scripts/build_web.sh`, adiciona `.nojekyll` e publica `build/web/`.

- [ ] No GitHub: **Settings → Pages → Build and deployment → Source = GitHub Actions**.
- [ ] `git push` na `main` (ou **Actions → Deploy to GitHub Pages → Run workflow**).
- [ ] Aguardar o job (~2–4 min). URL final:
      `https://SEU_USUARIO.github.io/autonomous_vehicle_simulator/`.
- [ ] Abrir a URL, repetir o checklist visual do passo 0.

Notas:
- Funciona em *single-thread* (sem `SharedArrayBuffer`); o Pages não permite headers
  COOP/COEP e o app não precisa deles.
- 1º carregamento baixa o runtime do CDN do pygbag (~10 MB, cacheado depois).

## 3. Hugging Face Spaces (SDK: Static)

- [ ] `huggingface.co/new-space` → **SDK = Static**, nome `autonomous-vehicle-simulator`, visibilidade à sua escolha.
- [ ] Publicar o conteúdo do build na raiz do Space. Script pronto:
  ```bash
  SPACE_REPO=https://huggingface.co/spaces/SEU_USUARIO/autonomous-vehicle-simulator \
    bash scripts/deploy_hf.sh
  ```
  Ele roda `scripts/build_web.sh`, clona o Space, copia `build/web/*` + o `README.md` com o
  cabeçalho YAML ([`scripts/assets/hf_space_readme.md`](../scripts/assets/hf_space_readme.md)) e dá `git push`.
- [ ] Aguardar o Space "Running". URL:
      `https://huggingface.co/spaces/SEU_USUARIO/autonomous-vehicle-simulator`.
- [ ] Checklist visual do passo 0 na URL do Space.

## 4. Depois de publicado

- [ ] Colocar as duas URLs no `README.md` e no [PRD](prd.md).
- [ ] (Opcional) "pinar" o Space e adicionar screenshot/GIF no README.
- [ ] Registrar este trabalho como um prompt no [log](../prompts/prompts_gerais_v1.md) se você quiser.

## Riscos / pontos de atenção

| Item | Observação |
|---|---|
| Verificação no navegador | Ainda **não feita**. É o passo 0 e trava a publicação. |
| `scripts/build_web.sh` usa bash | No Windows: Git Bash ou WSL. O Actions roda em Ubuntu (ok). |
| Nome do repo ≠ `autonomous_vehicle_simulator` | Ajustar a URL do Pages e o caminho no workflow se mudar. |
| Tamanho do 1º download (~10 MB CDN) | Aceitável; considerar uma tela de "carregando" custom via `--template` depois. |
| Caminho relativo dos assets | pygbag gera tudo relativo; funciona em subpasta (`/autonomous_vehicle_simulator/`) e na raiz (Spaces). |
