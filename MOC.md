---
tipo: moc
titulo: "Mapa del Libro — Finetunning Library"
tags:
  - moc
  - indice
---

# Mapa del Libro

Hub principal del vault. Desde aquí llegas a todos los capítulos y conceptos.

## Los 7 capítulos

1. [[01-fundamentos-transformers-y-pretraining]] — Arquitectura Transformer, atención, pretraining, leyes de escala
2. [[02-supervised-finetuning]] — SFT, chat templates, loss masking, chain of thought
3. [[03-lora-adaptacion-de-bajo-rango]] — LoRA desde primeros principios, hiperparámetros, Multi-LoRA
4. [[04-qlora-cuantizacion-4bit]] — Cuantización NF4, Double Quantization, Paged Optimizers
5. [[05-rlhf-alineacion-llms]] — RLHF, PPO, DPO, KL divergence
6. [[06-grpo-y-variantes]] — GRPO, DAPO, GSPO, Dr. GRPO
7. [[07-finetuning-multimodal-vision-tts]] — Vision-language models, TTS, codec neuronal

---

## Vista tabular (requiere plugin Dataview)

Si tienes el plugin **Dataview** instalado, las siguientes queries se renderizan automáticamente:

### Capítulos en orden con sus metadatos

```dataview
TABLE tema, dificultad, length(conceptos_centrales) as "# conceptos centrales"
FROM "capitulos"
SORT file.name ASC
```

### Conceptos por tipo

```dataview
TABLE tipo, capitulo_central, length(aparece_en) as "#menciones"
FROM "conceptos"
SORT tipo ASC, file.name ASC
```

### Solo técnicas de fine-tuning

```dataview
LIST
FROM "conceptos"
WHERE tipo = "técnica"
SORT file.name ASC
```

### Conceptos huérfanos (sin capítulo central)

```dataview
LIST
FROM "conceptos"
WHERE capitulo_central = null
SORT file.name ASC
```

---

## Navegar por tags

- `#técnica/lora`, `#técnica/qlora`, `#técnica/rlhf`, `#técnica/sft`, `#técnica/grpo`, `#técnica/dpo`
- `#concepto/función-pérdida`, `#concepto/kl-divergence`, `#concepto/atención`
- `#modelo/transformer`, `#modelo/vision-language-model`, `#modelo/neural-audio-codec`
- `#nivel/introducción`, `#nivel/intermedio`, `#nivel/avanzado`

---

## Flujo de construcción (resumen de CLAUDE.md)

1. **redactor** — artículo en `articulos/` → capítulo en `capitulos/NN-slug.md`
2. **visualizador** — añade diagramas Mermaid
3. **scanner** (paralelo, secuencial al índice) — pobla `assets/indice-conceptos.yml`
4. **enlazador** (paralelo) — añade frontmatter, wikilinks, tags
5. **generador-conceptos** — genera `conceptos/*.md` como nodos-hub del grafo
6. **unificador** — compila todo a `output/libro-completo.docx`
