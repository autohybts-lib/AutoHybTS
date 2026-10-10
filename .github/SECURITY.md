# Política de segurança

## Versões suportadas

A AutoHybTS está em pré-alfa. Correções de segurança saem apenas sobre a versão mais
recente da branch `main`.

## Como relatar uma vulnerabilidade

**Não abra uma issue pública.** Use o relato privado do GitHub:
aba **Security → Report a vulnerability** deste repositório.

Inclua, se possível:

- a versão (`autohybts.__version__`) ou o commit;
- os passos para reproduzir;
- o impacto esperado.

O relato é respondido assim que possível. A correção sai numa versão nova, com crédito a
quem relatou, se a pessoa quiser.

## Fora do escopo

- Carregar com `load_model(path, trusted=True)` um modelo de origem desconhecida: o
  pickle executa código por definição. Use `signing_key=` ou `sha256=` para conferir o
  artefato antes de carregá-lo.

---

*English:* please report vulnerabilities privately via **Security → Report a
vulnerability**, not through public issues.
