# Finetunning Library — Design Spec

## Objetivo

Crear una biblioteca/libro sobre fine-tuning de LLMs que transforme artículos técnicos en documentación accesible con tres niveles de profundidad, diagramas visuales y exportación a Word.

## Fuente de datos

El usuario pega artículos técnicos directamente en la conversación. No hay scraping ni carga automática de archivos.

## Estructura de carpetas

```
finetunning-library/
├── .claude/
│   └── commands/
│       ├── redactor.md
│       ├── visualizador.md
│       └── unificador.md
├── capitulos/
│   ├── 01-finetuning-landscape.md
│   └── ...
├── assets/
│   ├── imagenes/          # PNGs generados desde Mermaid
│   └── presentaciones/    # PDFs e imágenes de referencia del usuario
├── output/
│   └── libro-completo.docx
├── scripts/
│   ├── mermaid-to-png.sh
│   └── build-word.sh
└── README.md
```

## Agentes (Slash Commands)

### 1. `/redactor`

**Input:** Artículo técnico pegado por el usuario.
**Output:** Capítulo en Markdown guardado en `capitulos/`.

Estructura obligatoria por cada concepto/sección:

1. **Palabras clave** — Tags relevantes en formato `code`.
2. **Nivel alto** — Qué es y por qué importa. Analogías, contexto general.
3. **Nivel medio** — Cómo funciona. Mecanismo, comparación con alternativas.
4. **Nivel bajo** — Detalles técnicos: hiperparámetros, consideraciones de implementación, gotchas.
5. **Resumen** — Bloque resumido al final de cada sección.

El capítulo debe incluir:
- Título con número de capítulo y nombre descriptivo.
- Referencia a la fuente original.
- Separadores `---` entre secciones principales.

### 2. `/visualizador`

**Input:** Un capítulo ya redactado de `capitulos/`.
**Output:** Diagramas Mermaid insertados en el .md + archivos `.mmd` en `assets/imagenes/`.

Tipos de diagramas según el contenido:
- **Flujo:** Pipelines de entrenamiento, secuencias de pasos.
- **Arquitectura:** Componentes del modelo, capas, bloques.
- **Comparativos:** Diferencias entre técnicas (ej: CPT vs SFT).
- **Secuencia:** Interacciones entre componentes.
- **Cualquier otro** que mejore la comprensión visual.

Cada diagrama debe:
- Tener un título descriptivo.
- Estar referenciado en el texto como imagen (`![titulo](assets/imagenes/nombre.png)`).
- Tener su archivo `.mmd` correspondiente para regeneración.

### 3. `/unificador`

**Input:** Todos los capítulos en `capitulos/` (ordenados por prefijo numérico).
**Output:** `output/libro-completo.docx`.

Pasos:
1. Lee todos los `.md` de `capitulos/` en orden.
2. Ejecuta `scripts/mermaid-to-png.sh` para renderizar todos los `.mmd` a PNG.
3. Ejecuta `scripts/build-word.sh` para compilar el Word final con Pandoc.

## Scripts

### `scripts/mermaid-to-png.sh`

- Busca todos los archivos `.mmd` en `assets/imagenes/`.
- Los renderiza a PNG usando `mmdc` (mermaid-cli).
- Sobreescribe PNGs existentes si el `.mmd` es más reciente.

### `scripts/build-word.sh`

- Concatena todos los archivos `.md` de `capitulos/` en orden numérico.
- Usa Pandoc para convertir a Word (`output/libro-completo.docx`).
- Incluye las imágenes referenciadas desde `assets/`.

## Herramientas requeridas

- **Node.js + @mermaid-js/mermaid-cli** (`mmdc`) — Renderizado de Mermaid a PNG.
- **Pandoc** — Conversión de Markdown a Word (.docx).

## Flujo de trabajo típico

1. El usuario pega un artículo técnico.
2. Ejecuta `/redactor` → se genera el capítulo en `capitulos/`.
3. Ejecuta `/visualizador` sobre ese capítulo → se añaden diagramas Mermaid.
4. Repite 1-3 para cada artículo.
5. Ejecuta `/unificador` → se genera `output/libro-completo.docx`.
