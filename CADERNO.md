# Caderno do Nous — Notas de Arquitetura e Visão

## Data: 2026-02-15

---

## 1. O Reframe: De Protocolo Inter-Agente pra Sistema Nervoso do Agente

O Nous não é um protocolo de comunicação. É a **infraestrutura de integridade cognitiva** de um agente.

A mesma infraestrutura (confidence, logical atoms, contracts, concept graphs) deve operar não na fronteira entre dois agentes, mas **nas fronteiras internas de um agente só**: perceber → lembrar → consolidar → crer → agir.

O nome já dizia isso. Nous é νοῦς — intelecto, a faculdade de entender. Faz mais sentido como infraestrutura interna do entendimento do que como protocolo de troca de mensagens.

---

## 2. Conexão com o Memoria-Lab (Avenue B)

O memoria-lab testa a tese "Everything is Memory" — memória é o primitivo que transforma um LLM stateless em agente de verdade. As reviews da Amazon são o dataset de agora, a pergunta é maior.

O Nous encaixa como a **segunda metade** desse problema:
- **Memoria-lab** responde: "como o agente forma entendimento do mundo"
- **Nous** responde: "como o agente age sobre esse entendimento sem corromper o que sabe"

O ciclo completo: o agente percebe eventos (Avenue B), forma memória (consolidação), desenvolve crenças (Eixo 4), e eventualmente age com base nessas crenças. O Nous entra no último passo — a **ponte entre memória interna e ação externa**.

### Quando o Nous vira necessidade?
Quando o Eixo 4 (belief formation) estiver rodando. Aí o agente tem crenças sobre o mundo, e o passo natural é agir sobre elas. A ponte entre "eu acredito que todos os roteadores Linksys falham" e a ação que isso gera precisa de fidelidade semântica.

---

## 3. As Quatro Mudanças Necessárias

Cada uma mapeia diretamente pros eixos do caderno do memoria-lab:

### 3.1 Confidence como Propriedade da Memória (Eixo 1: Profundidade)

Hoje `ConfidenceMap` vive dentro de `NousProtocolMessage`. Precisa viver independente — como trait que qualquer struct de memória implementa.

Dimensões cognitivas (diferentes das comunicativas atuais):
- **factual** — o agente viu o evento diretamente?
- **recency** — quanto tempo faz? (degrada com o tempo)
- **granularity** — quanto detalhe sobreviveu à compressão?
- **corroboration** — quantas outras memórias confirmam isso?
- **source_diversity** — veio de uma review ou de 50?

Uma memória não é só "o que aconteceu" — é "o que aconteceu, com que certeza, vista de quantos ângulos, há quanto tempo".

O confidence propagation rastrearia como uma memória consolidada há 50 turnos, com confiança degradada, deveria gerar uma ação menos assertiva do que uma memória fresca.

### 3.2 Logical Layer na Consolidação (Problema Mais Urgente)

O momento mais perigoso do memoria-lab é a consolidação. O agente comprime 150 memórias live em insights consolidados. É exatamente aí que:
- "todos os roteadores Linksys falharam" → "roteadores Linksys tinham problemas" (perda de quantificador)
- "nenhum iPod decepcionou em 2004" → "iPods eram populares em 2004" (perda de negação)

Isso é **corrupção cognitiva invisível** — o agente mentindo pra si mesmo durante o sono. E nem sabe.

A Logical Layer deveria:
1. Extrair atoms das memórias live antes da consolidação
2. Extrair atoms do resultado consolidado
3. Validar que quantificadores, negações e constraints foram preservados
4. Se não foram: flag, reconsolidar, ou anotar a perda

### 3.3 Contracts entre Fases Cognitivas (O Mais Original)

Ninguém está fazendo design-by-contract **dentro** de um agente. A ideia de que percepção→memória→consolidação→crença→ação são handoffs com precondições e fallbacks trata o agente como **um sistema distribuído consigo mesmo**. E é exatamente isso que ele é.

```
Percepção → Live Memory
  pre: "input tem conteúdo processável"
  post: "memória extraída preserva entidades e sentimento"

Live Memory → Consolidação
  pre: "pelo menos N memórias no buffer"
  post: "insights preservam quantificadores e negações das memórias originais"
  fallback: reconsolidar com granularidade maior

Consolidação → Crença
  pre: "insight tem corroboration >= threshold"
  post: "crença é consistente com crenças existentes"
  fallback: marcar como tentativa, não como estabelecida

Crença → Ação
  pre: "confidence propagated >= action_threshold"
  post: "ação é proporcional à confiança"
  fallback: escalate to human
```

### 3.4 Concept Graph como Mapa de Memória (Eixo 3: Meta-conhecimento)

O `ConceptGraphBuilder` hoje agrupa mensagens por similaridade. No memoria-lab, agruparia memórias consolidadas — dando ao agente um mapa de "o que eu sei sobre roteadores", "o que eu sei sobre iPods", "onde meu conhecimento tem vazios".

Isso é meta-cognição: o agente sabendo o que sabe. "Eu tenho 200 memórias sobre roteadores e 3 sobre impressoras" é informação que deveria mudar como ele age nos dois domínios.

---

## 4. Valor Imediato: Duas Ferramentas Independentes

Antes do reframe cognitivo, o Nous tem valor de mercado imediato como duas ferramentas isoladas:

### 4.1 Confidence Tracking pra Pipelines LLM

Qualquer sistema que usa LLM em cadeia (RAG → summarization → decisão) sofre de **degradação de confiança invisível**. O usuário recebe uma resposta com ar de certeza que na verdade passou por três transformações lossy.

**Exemplo concreto (Alfredinho):** Review do cliente passa por transcrição (confidence drop), extração estruturada (confidence drop), match com ofertas (confidence drop). O vendedor deveria saber que a confiança factual é 0.95 mas a completeness é 0.40 porque só metade da conversa foi captada.

### 4.2 Logical Validator pra Vector Search

O embedding blindness é um problema real e pouco endereçado. "Delete all" vs "delete some" com 0.998 de similaridade é um **bug silencioso que existe em todo sistema RAG em produção hoje**.

Se o Nous oferecesse só isso — um validador que senta em cima de qualquer busca por similaridade e diz "os embeddings concordam mas a lógica conflita" — já seria útil pra qualquer empresa que usa vector search.

---

## 5. Cenários de Alto Impacto

### Orquestração Financeira Autônoma
"Transferir para a conta X" vs "transferir da conta X" — similaridade semântica altíssima, consequência oposta. O contract system do Nous é exatamente o que esse cenário precisa.

### Infraestrutura como Código por Agentes
"Scale down to 2 instances" vs "scale down to 0 instances" — diferença entre economizar e derrubar produção. Embeddings não pegam isso.

### Cadeias Longas de Agentes (Telephone Game)
Research → analyst → decision → execution. Cada handoff perde nuance. O confidence propagation torna essa degradação visível e acionável.

---

## 6. O Que Falta (Roadmap)

### Curto prazo
1. **Middleware, não protocolo completo** — ninguém vai reescrever seu sistema pra NousProtocolMessage. Todo mundo colocaria um validador entre duas chamadas existentes.
2. **Python bindings (PyO3)** — ecossistema de agentes é esmagadoramente Python (LangChain, CrewAI, AutoGen). `pip install nous-confidence`, `pip install nous-logical`.
3. **Benchmarks públicos** — dataset real (HotpotQA, Natural Questions), similarity search puro vs similarity + logical validation. Mostrar os falsos positivos que a logical layer pega.

### Médio prazo (pós Avenue B)
4. **Extrair `nous-confidence` e `nous-logical` como crates independentes**
5. **Adaptar confidence pra dimensões cognitivas** baseado nos resultados do Avenue B
6. **ConsolidationValidator** usando Logical Layer
7. **Contracts intra-agente** operando entre fases cognitivas

### Princípio orientador
> "Os dados dizem a arquitetura, não o contrário."

O Avenue B está rodando. Quando terminar, os resultados informam exatamente quais dimensões de confidence e quais contratos cognitivos implementar primeiro. Adaptar o Nous agora seria especular. Adaptar depois dos resultados seria engenharia informada por evidência.

---

## 7. Estado Atual do nous-rs

- **Path**: `~/nous-rs` | **GitHub**: https://github.com/Ebaoj/nous-rs
- **66 arquivos Rust** | **7,420 LOC** | **92 testes** | Zero clippy warnings
- **3 crates**: nous-core (24 tests), nous-protocol (38 tests), nous-runtime (30 tests)
- **Facade**: `nous/` com prelude
- **Exemplo**: `cargo run --example basic_flow -p nous`
- **Rewrite do TS original** (`~/nous`, ~14K LOC) aproveitando o type system do Rust
