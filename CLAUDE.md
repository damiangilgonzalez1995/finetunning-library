# Contexto del repo (temporal)

> ⚠️ **CLAUDE.md temporal a nivel de raíz**. Pronto habrá una parte de código y este archivo se reescribirá entonces. Por ahora sirve para dar contexto a sesiones lanzadas desde la raíz.

## Qué es esto

Monorepo personal en transición. Hoy contiene una sola pieza:

- **`library/`** — biblioteca/libro sobre fine-tuning de LLMs. Vault de Obsidian con su propia cadena de agentes Claude Code (`redactor`, `visualizador`, `scanner`, `enlazador`, `generador-conceptos`, `unificador`).

Pronto se añadirá una parte de **código** como hermana de `library/`.

## Reglas para sesiones lanzadas desde la raíz

1. **No trabajes con los capítulos del libro desde aquí.** Si el usuario pide redactar/enlazar/escanear capítulos, indícale que abra una sesión dentro de `library/` con:
   ```bash
   cd library && claude
   ```
   Los agentes del libro (`redactor`, `visualizador`, etc.) solo se cargan desde ahí. Desde la raíz no están disponibles.

2. **El `library/CLAUDE.md`** contiene el flujo de trabajo completo del libro (6 pasos: redactar → diagramas → escanear → enlazar → regenerar conceptos → Word). Si el usuario hace una pregunta sobre el flujo, léelo desde ahí.

3. **`library/` es un vault de Obsidian.** Los `.md` dentro pueden tener frontmatter YAML, `[[wikilinks]]` y `#tags`. No los tomes como Markdown plano si vas a editarlos.

## Estructura

```
finetunning-library/
├── .git/
├── README.md           ← también temporal
├── CLAUDE.md           ← este archivo
└── library/            ← libro de fine-tuning (vault Obsidian)
    ├── .claude/        ← agentes específicos del libro
    ├── .obsidian/
    ├── capitulos/  conceptos/  assets/
    ├── scripts/    output/     docs/
    ├── CLAUDE.md   MOC.md      README.md
```
