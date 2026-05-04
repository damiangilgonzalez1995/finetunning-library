# Guía de Supervised Fine-Tuning: epochs, hiperparámetros y cómo saber si va bien

Referencia práctica para entrenar modelos con SFT (Supervised Fine-Tuning). Conceptos básicos, configuraciones típicas por tamaño de dataset, y cómo interpretar lo que ves en el log.

---

## Vocabulario fundamental

### Sample

Una muestra del dataset. Para SFT de un LLM, cada sample es un par (prompt, respuesta esperada). Para un VLM, también incluye una o varias imágenes.

### Batch

Un grupo de N samples procesados juntos en un mismo forward pass del modelo. El batch size influye directamente en la VRAM consumida y la calidad del gradiente.

### Step

Una **actualización de los pesos del modelo**. La unidad fundamental del entrenamiento.

```
1 step = forward pass + backward pass + optimizer.step()
```

Tras un step, los pesos del modelo son distintos a antes.

### Gradient accumulation

Truco de memoria: en lugar de procesar un batch enorme de golpe (puede no caber en VRAM), procesas N micro-batches pequeños y **acumulas los gradientes** sin actualizar pesos. Tras los N micro-batches, el optimizer aplica todos los gradientes acumulados → 1 step.

```
batch_efectivo = per_device_train_batch_size × gradient_accumulation_steps
```

Ejemplo: `batch_size=4`, `accumulation=2` → batch efectivo 8. Procesas 8 muestras antes de actualizar pesos. Equivalente en aprendizaje a `batch_size=8` directo, pero con la VRAM de `batch_size=4`.

### Epoch

Una pasada completa por TODO el dataset. Si tienes 1000 samples y batch efectivo 8, una epoch son `1000 / 8 = 125 steps`.

```
steps_por_epoch = len(train_dataset) / batch_efectivo
```

### Learning rate (LR)

Tamaño del paso del optimizer. Cuánto se "mueve" cada peso en cada update.

```
W_nuevo = W_viejo - learning_rate × gradiente
```

LR muy alto → entrenamiento inestable, loss explota.
LR muy bajo → entrenamiento lentísimo, no converge en tiempo razonable.

---

## Cuántas epochs entrenar (la regla práctica)

| Tamaño del dataset (samples) | Epochs típicos |
|---|---|
| < 500 | **3-5** |
| 500-5.000 | **2-3** |
| 5.000-50.000 | **1-2** |
| > 50.000 | **1** (a veces incluso menos: max_steps explícito) |

**Regla mental**: cuanto más pequeño el dataset, más epochs necesitas para que el modelo "vea suficiente" cada muestra. Pero también más riesgo de overfitting.

### Señales de subentrenamiento (poco)

- La loss sigue cayendo claramente cuando paras.
- La eval_loss también sigue bajando.
- El modelo en inferencia da respuestas cercanas pero no exactas.

**Solución**: más epochs.

### Señales de sobreentrenamiento (demasiado)

- La loss de TRAIN sigue bajando pero la EVAL sube o se estanca.
- El modelo en inferencia "memoriza" muestras de train casi literalmente.
- En muestras de test da peor calidad que con menos epochs.

**Solución**: menos epochs, o usar `early stopping`. Quédate con el checkpoint del epoch donde eval_loss fue mínima.

---

## Hiperparámetros típicos para SFT con LoRA

Punto de partida razonable que rara vez falla:

```python
# LoRA config
r            = 16
lora_alpha   = 16
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # atención
                  "gate_proj", "up_proj", "down_proj"]      # MLP

# Training config
per_device_train_batch_size = 4       # ajustar a VRAM disponible
gradient_accumulation_steps = 2       # → batch efectivo 8
num_train_epochs            = 3       # 2-4 según tamaño dataset
learning_rate               = 2e-4    # típico LoRA (más alto que full FT)
lr_scheduler_type           = "cosine" # baja suave al final
warmup_ratio                = 0.03    # primer 3% calentamiento
optim                       = "adamw_8bit"  # ahorra VRAM
weight_decay                = 0.01
seed                        = 42
```

### Rank LoRA según tamaño del modelo

**Heurística clave**: cuanto más grande el modelo base, **menos rank LoRA necesitas**. La razón: modelos grandes ya vienen "casi entrenados" para tu tarea (mejor zero-shot), por tanto el delta que LoRA debe aprender es pequeño y cabe en un subespacio low-rank chico.

| Tamaño modelo | Rank `r` recomendado | `lora_alpha` recomendado |
|---|---|---|
| < 1B (ej. LFM-450M) | **64** | 64 (factor 1.0) |
| 1-3B | 32-64 | igual a r |
| 3-7B | **16-32** | 2× r (compensa rank bajo) |
| 7-13B | 8-16 | 2× r |
| 13B+ | 8 | 2× r |

**Excepción QLoRA (`load_in_4bit=True`)**: sube r en un nivel respecto a la tabla. La cuantización 4bit pierde precisión, y LoRA tiene que aprender DOS cosas a la vez: el delta de tarea + compensar la pérdida de la cuantización. Por eso QLoRA suele requerir r=64 incluso en modelos 7B+.

### Cuándo `alpha = r` vs `alpha = 2*r`

LoRA se aplica como `W' = W + (alpha/r) * B*A`. El factor `alpha/r` controla **cuánto pesa la contribución LoRA** sobre los pesos base:

| Configuración | Factor `alpha/r` | Cuándo usar |
|---|---|---|
| `alpha = r` | 1.0 | rank alto (r=64+); LoRA tiene capacidad de sobra, no necesitas amplificar |
| `alpha = 2*r` | 2.0 | rank bajo (r=8-32); compensas la menor capacidad amplificando la señal |

**Intuición**: si bajas r (menos capacidad bruta de los adapters), súbele alpha proporcionalmente para que la señal LoRA "pegue más fuerte" en cada paso de entrenamiento. Es la heurística más común en la literatura.

**Casos donde quieres `alpha < r`** (factor < 1.0): cuando estás haciendo multi-task fine-tuning y NO quieres que el modelo olvide su capacidad general. Atenúa LoRA → preserva el modelo base.

### Learning rate por estrategia

| Técnica | LR típico | Por qué |
|---|---|---|
| **LoRA** | 1e-4 a 2e-4 | Adaptadores tienen pocos parámetros, toleran LR alto |
| **QLoRA** | 1e-4 a 2e-4 | Igual que LoRA |
| **Full fine-tuning** | 5e-6 a 5e-5 | Modificas pesos preentrenados, hay que ir suave |
| **Pretraining desde cero** | 1e-4 a 5e-4 | Pesos random, gran rango |

### Batch efectivo

| Batch efectivo | Cuándo usar |
|---|---|
| 1-4 | datasets MUY pequeños (<200), gradiente ruidoso pero exploras más |
| 8-16 | **rango óptimo para SFT** con datasets de cientos a miles |
| 32-128 | datasets grandes, hardware potente, training estable |
| 256+ | pretraining a escala (LLM grandes, requiere multi-GPU) |

---

## Cómo interpretar la loss en el log

### Loss típica al inicio del entrenamiento

| Modelo | Loss inicial típica |
|---|---|
| LLM base sin afinar | 3.0 - 8.0 |
| LLM ya instruct (Llama-3-Instruct, Qwen-Chat) | 1.5 - 3.5 |
| VLM compacto (LFM2.5-VL, Qwen2.5-VL) | 2.0 - 4.0 |

### Loss objetivo al final

| Estado | Train loss típica | Eval loss |
|---|---|---|
| Subentrenado | bajando rápido aún | bajando rápido aún |
| Bien entrenado | 0.3 - 0.8 | similar a train, ~10-20% más alta |
| Overfitting | 0.05 - 0.2 | sube respecto al mínimo |

**Si train_loss y eval_loss bajan en paralelo** → el modelo está aprendiendo bien.
**Si train_loss baja pero eval_loss sube** → overfitting, para.
**Si train_loss no baja** → LR demasiado bajo, dataset mal formado, o problema de datos.

---

## Estrategias de evaluación y checkpoints

```python
eval_strategy    = "epoch"   # cuándo evaluar: "no", "steps", "epoch"
save_strategy    = "epoch"   # cuándo guardar checkpoint
save_total_limit = 2         # máx 2 checkpoints en disco (los demás se borran)
load_best_model_at_end = True  # al terminar, carga el de menor eval_loss
metric_for_best_model = "eval_loss"
```

### Por qué siempre `eval_strategy="epoch"` o `"steps"`

Sin eval durante training, no detectas overfitting hasta tener el modelo final (y entonces es tarde). Con eval por epoch, ves la curva train_loss vs eval_loss y sabes cuándo parar.

### Por qué `save_strategy = "epoch"` + `save_total_limit = 2`

Te permite quedarte con el MEJOR checkpoint, no el último. Si epoch 2 tuvo `eval_loss = 0.5` y epoch 3 subió a `eval_loss = 0.7` (overfitting), el `load_best_model_at_end = True` recupera automáticamente el de epoch 2.

`save_total_limit = 2` evita que te llene el disco si haces muchas epochs (cada checkpoint puede pesar GBs).

---

## Loss vs accuracy: dos métricas distintas

Importante distinguir:

| | `loss` (cross-entropy) | accuracy / f1 / etc |
|---|---|---|
| Qué mide | qué probabilidad asigna el modelo al token correcto | si el output del modelo es correcto |
| Cómo se calcula | en cada step automáticamente, sin necesidad de generar | requiere `model.generate(...)` y comparar texto |
| Coste | gratis (ya se calcula en el forward) | caro (generación token a token) |
| Cuándo se calcula | cada N steps de training | normalmente al final, o cada N epochs |

**Loss baja ≠ accuracy alta automáticamente.** Puede haber loss baja pero outputs malformateados (en JSON, por ejemplo). Por eso siempre debes hacer eval con `model.generate(...)` al final.

---

## Cuánta VRAM necesitas (estimación rápida)

Para entrenar con LoRA un modelo de N parámetros en `bfloat16`:

| Componente | VRAM aprox |
|---|---|
| Pesos del modelo (frozen) | `2 × N` bytes |
| Adapters LoRA (trainable) | ~0.5-2% del modelo |
| Gradientes (solo de los adapters) | mismo tamaño que adapters |
| Estados del optimizer (Adam) | `2 × adapters` (dos momentos) |
| Activaciones | depende del batch_size y secuencia |

Estimación práctica para un modelo de 7B en LoRA + batch 4 + secuencia 2048: **~10-14 GB VRAM**.

Para QLoRA (pesos en 4-bit): **~5-8 GB VRAM**.

Para full fine-tuning: **~15-20× los parámetros en GB** (ej: 7B → 100-150 GB → necesitas H100 o varias A100).

---

## Checklist antes de lanzar un run de SFT

- [ ] Dataset bien formateado (validado leyendo 2-3 muestras a mano)
- [ ] Train y eval splits separados (NO contaminados — si Sentinel-2, separar por fecha, no aleatorio)
- [ ] Batch efectivo 8-16
- [ ] LR adecuado a la técnica (2e-4 LoRA, 5e-5 full FT)
- [ ] `eval_strategy = "epoch"` activado
- [ ] `save_strategy = "epoch"` + `save_total_limit = 2`
- [ ] `load_best_model_at_end = True` para recuperar el mejor checkpoint
- [ ] `report_to` configurado (`comet_ml`, `wandb`, `tensorboard` o `none`)
- [ ] `seed` fijado para reproducibilidad
- [ ] Inferencia ANTES del entrenamiento ejecutada (baseline)
- [ ] VRAM verificada con un primer run corto (`max_steps=10`) antes de la run completa

---

## Errores comunes y cómo evitarlos

### "max_steps y num_train_epochs definidos a la vez"

`max_steps > 0` GANA siempre y `num_train_epochs` se ignora silenciosamente. Si quieres usar epochs, **elimina max_steps** o pon `max_steps = -1`.

### "El modelo no aprende nada (loss plana)"

- Comprueba que el dataset esté bien (lee muestras a mano).
- Sube el LR (de 2e-5 a 2e-4 si estás con LoRA).
- Verifica que el optimizer no esté congelando los pesos por algún bug del setup.

### "La loss explota (NaN, inf)"

- LR demasiado alto (bájalo a la mitad).
- Sin warmup → añade `warmup_ratio = 0.03`.
- Dataset con ejemplos muy raros (outliers numéricos, secuencias gigantescas).

### "Out of Memory (OOM)"

- Reduce `per_device_train_batch_size` (1 si hace falta).
- Sube `gradient_accumulation_steps` para mantener batch efectivo.
- Activa `gradient_checkpointing = True`.
- Si es VLM con imágenes grandes, reduce resolución de las imágenes.

### "Loss baja en train pero eval_loss sube"

Overfitting. Reduce epochs, sube `lora_dropout` (0.05-0.1), añade `weight_decay`.

---

## Resumen mental

> **Para SFT con LoRA en datasets pequeños** (cientos a miles de samples), el setup que rara vez falla es:
>
> 3 epochs, batch efectivo 8, LR 2e-4, cosine scheduler con warmup_ratio 0.03, eval_strategy="epoch", save al final del best checkpoint.
>
> A partir de ahí, ajustas según lo que veas en la curva de loss.
