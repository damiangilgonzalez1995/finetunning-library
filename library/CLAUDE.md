# Finetunning Library

Biblioteca/libro sobre fine-tuning de LLMs basada en artículos técnicos de The Neural Maze.

## Flujo de trabajo

Cuando el usuario pega un artículo técnico, sigue estos pasos EN ORDEN.

⚠️ REGLA CRÍTICA: Usa SIEMPRE `subagent_type` con el nombre exacto del agente registrado.
NUNCA uses `subagent_type: "general-purpose"`. NUNCA inyectes el prompt del agente en el campo `prompt` del Agent tool.
Los agentes ya tienen su propio system prompt en `.claude/agents/`. Solo pásales la tarea concreta.

### Paso 1 — Redactar el capítulo
```
Agent tool → subagent_type: "redactor"
prompt: "Redacta un capítulo basado en este artículo: [contenido o referencia al artículo]"
```

### Paso 2 — Añadir diagramas
```
Agent tool → subagent_type: "visualizador"
prompt: "Añade diagramas Mermaid al capítulo capitulos/NN-nombre.md"
```

### Paso 3 — Escanear conceptos (alimentar el índice de Obsidian)

Para cada capítulo afectado, lanza un `scanner`. Son ligeros y **se pueden lanzar en paralelo** (una sola respuesta con N Agent tool calls):

```
Agent tool → subagent_type: "scanner"
prompt: "Escanea el capítulo 03-lora-desde-primeros-principios.md"
```

El scanner actualiza `assets/indice-conceptos.yml` con los conceptos detectados. Nunca lee otros capítulos.

### Paso 4 — Enlazar el capítulo (wikilinks + tags + frontmatter)

Una vez el índice está poblado, lanza un `enlazador` por capítulo (secuencial, un capítulo a la vez):

```
Agent tool → subagent_type: "enlazador"
prompt: "Enlaza el capítulo 03-lora-desde-primeros-principios.md"
```

El enlazador añade frontmatter YAML, `[[wikilinks]]` internos y `#tags` consultando solo el índice y el capítulo objetivo. Contexto mínimo garantizado.

### Paso 5 — Regenerar notas de conceptos (nodos-hub del grafo de Obsidian)

Una sola invocación. El agente ejecuta `python scripts/generate-concept-notes.py` que lee el índice y regenera `conceptos/*.md`:

```
Agent tool → subagent_type: "generador-conceptos"
prompt: "Regenera las notas de conceptos a partir del índice actualizado."
```

Cada concepto del índice se materializa como una nota `.md` con `[[wikilink]]` al capítulo central y a los capítulos donde aparece. Esto hace que los conceptos se vean como nodos propios en el graph view de Obsidian.

**Idempotente**: reescribe las notas cada vez. Si se edita una nota a mano, se pierde.

### Paso 6 — Generar Word
```
Agent tool → subagent_type: "unificador"
prompt: "Genera el documento Word final"
```

## Estructura del proyecto

```
capitulos/                  → Capítulos individuales en Markdown
conceptos/                  → Notas por concepto (generadas por generador-conceptos)
.claude/agents/             → Subagentes (redactor, visualizador, scanner, enlazador, generador-conceptos, unificador)
.obsidian/                  → Config del vault (graph.json filtra articulos/ y colorea por tag)
assets/imagenes/            → Diagramas .mmd y .png renderizados
assets/presentaciones/      → PDFs e imágenes de referencia
assets/indice-conceptos.yml → Índice global de conceptos (lo mantiene `scanner`)
MOC.md                      → Mapa del libro con queries Dataview
output/                     → Word final unificado
scripts/                    → Scripts de conversión (build-word.py, generate-concept-notes.py)
```

## Arquitectura scanner + enlazador + generador-conceptos (¿por qué?)

Evita saturar ventana de contexto. Ningún agente lee los 7 capítulos a la vez.

- **`scanner`** (modelo Haiku, paralelizable): lee **1 capítulo** y actualiza un índice YAML pequeño.
- **`enlazador`** (modelo Sonnet, secuencial por capítulo): lee **1 capítulo + el índice** y añade frontmatter, `[[wikilinks]]`, `#tags`.
- **`generador-conceptos`** (modelo Haiku, 1 invocación): ejecuta script Python que lee el índice y regenera `conceptos/*.md`.

El índice (`assets/indice-conceptos.yml`) actúa como memoria compartida: pequeño, persistente entre sesiones, consultable.

## Regla clave del redactor

NO traducir ni reformatear el artículo. ENSEÑAR con ejemplos numéricos concretos, analogías memorables y desglose paso a paso de fórmulas.
