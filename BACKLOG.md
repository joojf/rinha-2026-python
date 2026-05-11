# Backlog

## Pronto
- [x] Estrutura de pastas e arquivos (`api/`, `searcher/`, `nginx/`, `data/`)
- [x] Função `normalize()` com as 14 dimensões conforme `REGRAS_DE_DETECCAO.md`
- [x] 9 testes unitários para `normalize()` (incluindo os exemplos exatos da doc)
- [x] Searcher com FAISS `IndexIVFScalarQuantizer` (QT_8bit), cache de índice em disco
- [x] Carregamento streaming do dataset gzip (duas passagens, sem estourar memória)
- [x] `docker-compose.yml` com limites de CPU/RAM dentro do budget (1 CPU, 350 MB)
- [x] Submodule `rinha-ref/` apontando para o repo oficial

---

## A fazer

### 1. Dataset
- [ ] Baixar os arquivos de referência com `bash scripts/download-data.sh`
- [ ] Confirmar que o formato de `references.json.gz` é um objeto por linha (ajustar `_stream_records` se necessário)

### 2. Build e smoke test local
- [ ] `docker compose build` — garantir que as imagens compilam sem erro em `linux/amd64`
- [ ] `docker compose up` — verificar nos logs que o searcher termina de construir/carregar o índice
- [ ] `curl -s localhost:9999/ready` deve retornar `ok` (200)
- [ ] Testar o endpoint com um payload de exemplo:
  ```bash
  curl -s -X POST localhost:9999/fraud-score \
    -H 'Content-Type: application/json' \
    -d @rinha-ref/resources/example-payloads.json | head -c 200
  ```

### 3. Teste de carga com k6
- [ ] Instalar k6 (`brew install k6`)
- [ ] Rodar o script oficial: `k6 run rinha-ref/test/test.js`
- [ ] Analisar `results.json`: checar `failure_rate`, `p99`, `final_score`

### 4. Tuning do índice FAISS
- [ ] Se `failure_rate > 5%`: aumentar `NPROBE` (32 → 64) em `searcher/main.py`
- [ ] Se `p99 > 100ms`: reduzir `NPROBE` (32 → 16) ou investigar gargalo de rede
- [ ] Deletar cache (`data/index.faiss`, `data/labels.npy`) e reconstruir após qualquer mudança de parâmetro
- [ ] Registrar resultados de cada configuração testada

### 5. Preparar submissão
- [ ] Criar branch `submission` com apenas `docker-compose.yml`, `nginx.conf` e `info.json`
- [ ] Preencher `info.json` com dados do participante
- [ ] Verificar que o `docker-compose.yml` da branch `submission` usa imagens publicadas (não `build:`)
- [ ] Abrir PR no repo oficial adicionando `participants/<github-user>.json`

### 6. Testes de prévia
- [ ] Abrir issue no repo oficial com `rinha/test` na descrição para rodar prévia
- [ ] Analisar o comentário com o resultado e iterar conforme necessário
