# Finetunning Library

Biblioteca/libro sobre fine-tuning de LLMs. Transforma artículos técnicos en documentación accesible con tres niveles de profundidad y diagramas visuales.

## Estructura

```
capitulos/             → Capítulos individuales en Markdown
assets/imagenes/       → Diagramas Mermaid (.mmd) y sus PNGs renderizados
assets/presentaciones/ → PDFs e imágenes de referencia
output/                → Documento Word final unificado
scripts/               → Scripts de conversión (Mermaid→PNG, MD→Word)
```

## Comandos de Claude Code

| Comando | Descripción |
|---------|-------------|
| `/redactor` | Pega un artículo técnico → genera capítulo con estructura de 3 niveles |
| `/visualizador` | Toma un capítulo → añade diagramas Mermaid |
| `/unificador` | Combina todos los capítulos → genera Word final |

## Flujo de trabajo

1. Pega un artículo técnico y ejecuta `/redactor`
2. Ejecuta `/visualizador` sobre el capítulo generado
3. Repite para cada artículo
4. Ejecuta `/unificador` para generar el libro completo en Word

## Requisitos

- [Pandoc](https://pandoc.org/installing.html) — Conversión MD → Word
- [@mermaid-js/mermaid-cli](https://github.com/mermaid-js/mermaid-cli) — Renderizado Mermaid → PNG (`npm install -g @mermaid-js/mermaid-cli`)