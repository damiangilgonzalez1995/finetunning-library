
---
capitulo: 01
titulo: "El mapa del territorio: Transformers, pretraining y la pipeline de entrenamiento de LLMs"
aliases:
  - "Capítulo 1"
  - "Cap 1"
  - "Fundamentos Transformers"
  - "Pretraining"
tema: "arquitectura-y-pretraining"
subtemas: [transformer, pretraining, modelo-base]
dificultad: "introducción"
tipo: "lección"
estado: "completo"
conceptos_centrales:
  - transformer
  - atención
  - self-attention
  - pretraining
  - causal-language-modeling
  - modelo-base
prerequisitos: []
relacionados:
  - "[[02-supervised-finetuning]]"
  - "[[03-lora-adaptacion-de-bajo-rango]]"
  - "[[05-rlhf-alineacion-llms]]"
tags:
  - arquitectura/transformer
  - concepto/atención
  - concepto/self-attention
  - técnica/pretraining
  - concepto/causal-language-modeling
  - concepto/modelo-base
  - técnica/knowledge-distillation
  - nivel/introducción
  - tipo/lección
  - estado/completo
---

# Capítulo 1 — El mapa del territorio: Transformers, pretraining y la pipeline de entrenamiento de LLMs

> Basado en "The Finetuning Landscape — A Map of Modern LLM Training" y "Modern Pretraining Strategies: A Hands-On Guide" (The Neural Maze, Finetuning Sessions · Lesson 1 / Lab 1).

Antes de hablar de fine-tuning, hay que entender de dónde viene el modelo que vamos a afinar. Es como intentar aprender a tunear un motor sin saber cómo funciona la combustión interna: puedes seguir una receta, pero no entenderás por qué algo falla ni cómo arreglarlo. Este capítulo construye esa base. No es un desvío — es el mapa sin el que el resto del libro no tiene sentido.

---

## De las RNNs al Transformer: por qué cambió todo

Antes de que los Transformers se convirtieran en el estándar universal de la inteligencia artificial, el campo de procesamiento de lenguaje natural estaba dominado por las redes neuronales recurrentes — en inglés Recurrent Neural Networks, o RNNs. Y dentro de ellas, la arquitectura más capaz era la LSTM (Long Short-Term Memory, memoria a largo-corto plazo), diseñada para recordar patrones a lo largo de secuencias.

La idea detrás de las RNNs es intuitiva: para entender una frase, vas leyendo palabra por palabra, y en cada paso actualizas un "estado oculto" que se supone que resume todo lo que has leído hasta ese momento. Es como leer un libro y tomar notas en un post-it que sobrescribes continuamente: al llegar al capítulo 20, el post-it ya no recuerda bien lo del capítulo 1.

Eso es exactamente el problema. Las RNNs y LSTMs sufrían de lo que se llama el problema de las dependencias de largo alcance: cuando una frase es larga o la información relevante está lejos, el estado oculto "olvida" los detalles iniciales antes de poder usarlos. Tradúcelo a la práctica: si una oración tiene 200 palabras y el sujeto está en la primera, la red tenía dificultades para conectar el sujeto con el verbo 190 palabras más tarde.

Además, existía un problema estructural de rendimiento: las RNNs procesan los tokens en secuencia, uno después del otro. Eso las hace inherentemente lentas de entrenar, porque no puedes paralelizar el procesamiento sobre los tokens de una misma frase. En la era del hardware masivamente paralelo (GPUs, TPUs), esa limitación era especialmente dolorosa.

La solución vino de una dirección inesperada — no reemplazando las RNNs con algo completamente nuevo, sino añadiéndoles un componente que resultó ser tan poderoso que terminó haciendo las RNNs irrelevantes: el mecanismo de atención.

---

## El mecanismo de atención: un buscador diferencial

La atención fue introducida originalmente como una mejora a los modelos encoder–decoder basados en RNNs para tareas de traducción automática. Para entender por qué fue revolucionaria, hay que entender el problema que resolvía.

Los modelos de traducción de la época funcionaban así: un encoder (codificador) leía toda la frase de entrada y la comprimía en un único vector de tamaño fijo — una especie de "resumen" de la frase. Luego un decoder (decodificador) intentaba generar la traducción a partir de ese vector. El problema: comprimir "The cat that the dog chased sat on the mat near the window" en un vector de 512 números y esperar que el decodificador pueda generar una traducción fiel es pedir demasiado. Información se perdía.

La atención propuso algo elegante: en lugar de comprimir todo en un vector, ¿por qué no dejar que el decodificador consulte directamente los estados del encoder en cada paso de la decodificación? La metáfora útil aquí es un sistema de recuperación de información. Imagina que tienes una base de datos de fichas:

- Cada ficha tiene una **clave** (key, o $k$) que describe su contenido.
- Cada ficha tiene un **valor** (value, o $v$) que es el contenido real.
- Cuando buscas algo, formulas una **consulta** (query, o $q$) que expresa qué estás buscando.

El mecanismo compara tu consulta contra todas las claves, calcula una puntuación de relevancia para cada una, normaliza esas puntuaciones (con una función softmax, que las convierte en probabilidades que suman 1), y luego produce una combinación ponderada de todos los valores. Las fichas más relevantes para tu consulta contribuyen más; las menos relevantes contribuyen poco o nada.

Pongamos un ejemplo con números pequeños. Supón que estás decodificando la palabra "chased" en una traducción y tienes tres fichas del encoder: "dog" ($k_1$), "cat" ($k_2$), "mat" ($k_3$). Tu consulta $q$ representa "quiero saber quién realiza la acción de perseguir". El mecanismo calcula:

- Puntuación("dog") = 4.2 → muy relevante
- Puntuación("cat") = 1.1 → algo relevante  
- Puntuación("mat") = 0.1 → casi irrelevante

Después de la softmax, estas puntuaciones se convierten en pesos que suman 1 (por ejemplo, 0.93, 0.06, 0.01). El output de la atención es $0.93 \cdot v_{dog} + 0.06 \cdot v_{cat} + 0.01 \cdot v_{mat}$ — básicamente, el valor de "dog" con pequeñas contribuciones de los demás. El modelo puede "mirar" exactamente donde necesita.

> **Descripción visual:** Diagrama de flujo horizontal. A la izquierda, tres óvalos morados representan las entradas: "Consulta Query q", "Claves Keys k" y "Valores Values v". Query y Claves convergen en un rectángulo amarillo "Puntuación q·k", que fluye a otro rectángulo amarillo "Softmax pesos α". Desde Softmax y Valores, ambas flechas confluyen en un rectángulo amarillo "Suma ponderada", que desemboca en un óvalo verde "Output contextual". Fondo blanco, tipografía sans-serif, estilo limpio.

La clave de este mecanismo es que es diferencial (differentiable): las puntuaciones se aprenden durante el entrenamiento vía gradiente descendente. El modelo aprende, por sí solo, qué vale la pena atender en cada contexto.

---

## Self-attention: cuando la secuencia se lee a sí misma

La atención original conectaba dos secuencias distintas: la de entrada (encoder) con la de salida (decoder). Pero alguien hizo la pregunta obvia: ¿qué pasa si aplicamos el mismo mecanismo dentro de una sola secuencia?

Eso es la self-attention (auto-atención): cada token de la secuencia produce su propia consulta, su propia clave, y su propio valor, y luego "asiste" a todos los demás tokens — incluyéndose a sí mismo. El resultado es que la representación de cada token se actualiza con información de todos los demás tokens de la misma frase.

Considera la frase: "El banco aprobó el préstamo porque tenía buenas reservas." ¿A qué se refiere "tenía"? ¿Al banco o al solicitante? Para un humano es obvio — "el banco tenía buenas reservas". Con self-attention, el token "tenía" puede computar alta atención hacia "banco" y baja atención hacia otros candidatos, resolviendo la ambigüedad en la representación vectorial misma.

El proceso técnico es el siguiente. Dado un token en posición $i$ con representación $x_i$:

1. Se proyecta $x_i$ con tres matrices de pesos distintas para obtener $q_i$, $k_i$, $v_i$.
2. La puntuación del token $i$ respecto al token $j$ es $s_{ij} = q_i \cdot k_j$ (producto punto).
3. Para estabilizar la magnitud de las puntuaciones (que crecen con la dimensión del vector), se dividen entre $\sqrt{d_k}$, donde $d_k$ es la dimensión de las claves.
4. Se aplica softmax sobre todas las puntuaciones $s_{ij}$ para obtener pesos de atención $\alpha_{ij}$.
5. La nueva representación del token $i$ es $\sum_j \alpha_{ij} \cdot v_j$.

La fórmula compacta que resume todo esto es:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

donde $Q$, $K$, $V$ son matrices que apilan las consultas, claves y valores de todos los tokens de la secuencia. Esta operación se puede hacer sobre toda la secuencia de golpe, en paralelo — algo imposible con las RNNs. Es exactamente esto lo que hace al Transformer tan eficiente de entrenar.

Hay una consecuencia crucial que vale la pena subrayar: la self-attention no tiene noción inherente de posición. Para el mecanismo, los tokens son como un conjunto desordenado — no importa si "perro" está al principio o al final de la frase. Esa información hay que inyectarla por separado.

---

## Codificación posicional: enseñarle al modelo que el orden importa

Dado que la self-attention trata los tokens como un conjunto sin orden, el Transformer original añade vectores de codificación posicional (positional encodings) directamente a las representaciones de los tokens antes de procesarlos. Estos vectores codifican la posición de cada token en la secuencia de manera que el modelo puede aprender a usarlos.

La elección del paper original "Attention Is All You Need" fue usar funciones sinusoidales a distintas frecuencias. Cada dimensión del vector de posición usa una sinusoide diferente:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

donde $pos$ es la posición del token (0, 1, 2...), $i$ es el índice de la dimensión, y $d_{model}$ es la dimensión del modelo. Las dimensiones de baja frecuencia capturan información gruesa sobre posición (¿estoy al principio, al medio, al final?), mientras que las dimensiones de alta frecuencia capturan diferencias finas entre posiciones adyacentes.

La intuición práctica es: imagina un reloj con múltiples manecillas. La manecilla de las horas da información gruesa (en qué parte del día estás), la de los minutos da información más fina, y la de los segundos da información muy precisa. Las codificaciones sinusoidales funcionan igual, pero con dimensiones del vector en lugar de manecillas.

Los modelos modernos han adoptado alternativas como las Rotary Position Embeddings (RoPE), que integran la información posicional de forma más elegante dentro del mecanismo de atención mismo, en lugar de añadirla externamente. Pero el principio es el mismo: el modelo necesita saber dónde está cada token en la secuencia.

---

## Multi-head attention: ver la misma frase con distintos ojos

Un único mecanismo de self-attention produce una única "vista" de las relaciones entre tokens. Pero las relaciones en el lenguaje son multidimensionales: dos palabras pueden estar relacionadas sintácticamente (sujeto-verbo), semánticamente (sinonimia), posicionalmente (adyacencia), y más. Un solo conjunto de pesos de atención tiene que comprometer entre todos estos tipos de relación a la vez.

La multi-head attention (atención multi-cabeza) resuelve esto corriendo la self-attention en paralelo varias veces, con diferentes proyecciones aprendidas. Si el modelo tiene $h$ cabezas (heads) de atención, cada cabeza $i$ tiene sus propias matrices de proyección $W_i^Q$, $W_i^K$, $W_i^V$ de dimensión reducida $d_k = d_{model}/h$. Cada cabeza produce su propio conjunto de outputs de atención, y al final todos se concatenan y se proyectan de vuelta a $d_{model}$:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \cdot W^O$$

donde $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

El resultado práctico es que distintas cabezas se especializan en distintos tipos de relaciones. En modelos analizados empíricamente, se ha observado que algunas cabezas se especializan en relaciones sintácticas (quien es el sujeto del verbo), otras en correferencia (a qué se refiere "él"), y otras en posiciones relativas (el siguiente token, el anterior). El modelo no es programado para hacer esto — emerge solo del entrenamiento.

Por qué importa para el fine-tuning, que es el tema central de este libro: cuando apliquemos técnicas como [[03-lora-adaptacion-de-bajo-rango|LoRA]] (que veremos en capítulos posteriores), estaremos modificando estas mismas matrices de proyección $W^Q$, $W^K$, $W^V$, $W^O$. Entender qué hace cada una es fundamental para tomar buenas decisiones sobre qué modificar y qué dejar intacto.

---

## El Transformer completo: juntando las piezas

Con self-attention y multi-head attention como bloques fundamentales, el Transformer añade algunos ingredientes más para completar la arquitectura:

**Feed-forward por posición.** Después de la atención, cada token pasa por una red neuronal feed-forward (totalmente conectada) que opera de forma independiente en cada posición. Esta red tiene dos capas con una activación no lineal en el medio (originalmente ReLU, hoy más frecuente GELU). Su función es procesar y transformar las representaciones que la atención ha mezclado entre tokens — si la atención mezcla información entre posiciones, el feed-forward procesa cada posición en profundidad.

**Normalización de capa (Layer Norm).** Para estabilizar el entrenamiento con redes tan profundas, se aplica normalización de capa antes o después de cada sub-bloque. Sin ella, los gradientes pueden explotar o desvanecerse a medida que el error se propaga hacia atrás por decenas de capas.

**Conexiones residuales.** Cada sub-bloque (atención + feed-forward) tiene una conexión directa que suma la entrada al output del sub-bloque. La idea es simple pero poderosa: si el bloque no aprende nada útil, el gradiente puede seguir fluyendo hacia atrás directamente sin pasar por él. Esto hace que la optimización sea mucho más estable y permite entrenar redes de 100+ capas.

Un bloque Transformer apila estos componentes: la entrada llega, pasa por multi-head self-attention, se suma con la conexión residual, se normaliza, pasa por el feed-forward, se suma de nuevo con otra conexión residual, y se normaliza. Ese bloque se repite $N$ veces (en GPT-3 eran 96 veces; en LLaMA 3 70B son 80). La profundidad es la que da capacidad al modelo.

---

## Tres arquitecturas, tres filosofías

Cuando se habla de "LLMs", la mayoría piensa en ChatGPT o Claude — modelos que generan texto. Pero el Transformer como arquitectura base admite tres configuraciones distintas, cada una adecuada para un tipo diferente de tarea.

### Encoder-only: para entender, no para generar

Los modelos encoder-only como BERT procesan toda la secuencia de entrada con self-attention bidireccional: cada token puede atender a todos los demás, tanto los que vienen antes como los que vienen después. Esto produce representaciones extremadamente ricas del significado de cada token en su contexto completo.

La palabra "banco" en "fui al banco a sacar dinero" tiene una representación diferente a "banco" en "me senté en el banco del parque" — y el modelo puede distinguirlas porque ve toda la frase a la vez. Esta capacidad hace que los modelos encoder-only sean ideales para clasificación de texto, extracción de entidades, análisis de sentimiento, y búsqueda semántica.

Lo que no pueden hacer es generar texto. Un encoder-only no tiene mecanismo para predecir el siguiente token — su objetivo de entrenamiento (el masked language modeling, donde se enmascaran tokens aleatorios y se predice qué debería haber allí) no está diseñado para generación secuencial.

### Encoder–decoder: para transformar secuencias

La arquitectura encoder–decoder es la del Transformer original, diseñada para traducción automática. El encoder procesa la secuencia de entrada con atención bidireccional y produce representaciones de alta calidad. El decoder genera la secuencia de salida token a token, usando dos mecanismos de atención:

- **Causal self-attention** (o masked self-attention): en el decoder, cuando se genera el token en la posición $t$, solo puede atender a los tokens generados anteriormente (posiciones 1 hasta $t-1$). Esto es lo que hace la generación autorregresiva — no puede "hacer trampa" mirando el futuro.
- **Cross-attention**: el decoder consulta las representaciones del encoder para obtener información sobre la entrada. Es aquí donde fluye la información de "lo que hay que traducir" hacia "cómo traducirlo".

Modelos como T5 y BART usan esta arquitectura. Son poderosos para tareas donde tanto la entrada como la salida son secuencias de longitud variable: traducción, resumen, reformulación, pregunta-respuesta.

### Decoder-only: la arquitectura que gobierna el mundo actual

Los modelos decoder-only eliminan el encoder por completo. Solo tienen un decoder con causal self-attention: cada token solo puede atender a los tokens que lo preceden. El objetivo de entrenamiento es simplemente predecir el siguiente token.

A primera vista parece una simplificación radical — ¿no es mejor tener un encoder que procese la entrada completa con atención bidireccional? La respuesta empírica es sorprendente: no necesariamente. Con suficientes parámetros y datos, los modelos decoder-only aprenden a hacer todo lo que los encoder-decoders hacen, y más. Y tienen una ventaja fundamental: un modelo entrenado para predecir el siguiente token puede resolver cualquier tarea si esa tarea se formula como completar una secuencia. Traducción, resumen, codificación, razonamiento matemático, respuesta a preguntas — todo puede formularse como "dado este texto, ¿cuál es la continuación natural?"

Esta flexibilidad, combinada con la simplicidad del objetivo de entrenamiento (que lo hace escalable a enormes cantidades de datos), explica por qué GPT, LLaMA, Qwen, Mistral, y prácticamente todos los LLMs de vanguardia actuales son decoder-only. Es la arquitectura que estudiaremos en el resto del libro.

> **Descripción visual:** Diagrama con tres subgrafos lado a lado, cada uno con orientación vertical interna. El subgrafo izquierdo (fondo azul claro, borde azul) representa "Encoder-only (BERT)" con tres rectángulos apilados: "Tokens entrada", "Atención bidireccional" y "Clasificar / entender". El subgrafo central (fondo amarillo pálido, borde dorado) representa "Encoder-Decoder (T5)" con cuatro rectángulos: "Secuencia entrada", "Encoder bidireccional", "Cross-Attention" y "Secuencia salida". El subgrafo derecho (fondo verde claro, borde verde) representa "Decoder-only (GPT/LLaMA)" con tres rectángulos: "Contexto previo", "Atención causal" y "Siguiente token". Flechas descendentes dentro de cada subgrafo. Fondo blanco, tipografía sans-serif, estilo comparativo.

---

## Leyes de escala: por qué más grande es más inteligente (hasta cierto punto)

Una de las observaciones más importantes de la última década en IA es que el rendimiento de los Transformers mejora de forma predecible y continua al aumentar tres magnitudes: el número de parámetros del modelo, la cantidad de datos de entrenamiento, y el cómputo total utilizado.

Esta relación se formaliza en lo que se conoce como leyes de escala (scaling laws). Publicadas originalmente por Kaplan et al. (2020) de OpenAI, muestran que la pérdida de un modelo (su error en predecir el siguiente token) sigue una relación de ley de potencias (power-law) con estas tres variables. Matemáticamente, si tienes $N$ parámetros, $D$ tokens de datos, y $C$ FLOPs de cómputo:

$$L \propto N^{-\alpha}, \quad L \propto D^{-\beta}, \quad L \propto C^{-\gamma}$$

donde $\alpha$, $\beta$, $\gamma$ son exponentes empíricos. La consecuencia práctica es dramática: si duplicas el número de parámetros manteniendo todo lo demás constante, puedes predecir aproximadamente cuánto va a mejorar el modelo. No es lineal, pero sí predecible.

Chinchilla (2022) refinó esta intuición: dado un presupuesto de cómputo fijo, la distribución óptima no es "el modelo más grande posible". Es una distribución balanceada entre parámetros y tokens de entrenamiento — aproximadamente 20 tokens por parámetro. Un modelo de 7B parámetros debería ver idealmente 140B tokens. Un modelo de 70B parámetros debería ver 1.4T tokens. Modelos que habían entrenado con pocos datos para su tamaño estaban "compute-suboptimal" — habían gastado cómputo de forma ineficiente.

Para el fine-tuning, las leyes de escala tienen una implicación directa: el modelo base que vamos a afinar ya parte de un punto muy capaz. El fine-tuning no "enseña" capacidades nuevas de cero — refina y direcciona capacidades que ya existen. Esa distinción cambia fundamentalmente cómo debemos pensar sobre el proceso.

---

## Pretraining: donde se forja la inteligencia

Llegamos al concepto central de este capítulo. El pretraining (preentrenamiento) es la primera y más fundamental fase del entrenamiento de un LLM. Es donde el modelo aprende el lenguaje desde cero — o, más precisamente, aprende a modelar la distribución estadística del lenguaje humano.

El objetivo es deceptivamente simple: dado un texto de entrada, predecir el siguiente token. Para un modelo decoder-only, esto se llama causal language modeling (CLM, modelado de lenguaje causal). Dado un fragmento de texto $x_1, x_2, ..., x_t$, el modelo aprende a predecir $x_{t+1}$.

La función de pérdida (loss function, la medida de cuán equivocado está el modelo) que se minimiza durante el pretraining es la entropía cruzada negativa (negative cross-entropy):

$$\mathcal{L}_{CLM} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_1, ..., x_{t-1}; \theta)$$

donde $T$ es el número de tokens en el batch, $P(x_t \mid x_1, ..., x_{t-1}; \theta)$ es la probabilidad que el modelo con parámetros $\theta$ asigna al token correcto $x_t$ dado el contexto anterior. Cuando el modelo asigna alta probabilidad al token correcto, el logaritmo es cercano a cero y la pérdida es baja. Cuando asigna baja probabilidad al token correcto (se equivoca), el logaritmo es muy negativo, y la pérdida sube.

Esto es aprendizaje auto-supervisado (self-supervised learning): no necesitas etiquetas elaboradas por humanos. La supervisión está en los propios datos — el token $t+1$ es la etiqueta del contexto $x_1, ..., x_t$. Eso es lo que hace el pretraining escalable: el internet entero es un conjunto de entrenamiento gratuito y enorme.

El resultado del pretraining es lo que se llama un modelo base (base model o foundation model). Este modelo ha visto cantidades astronómicas de texto — LLaMA 3 fue entrenado en 15 billones de tokens — y ha absorbido gramática, sintaxis, semántica, hechos del mundo, patrones de razonamiento, código, múltiples idiomas, y mucho más. Es extraordinariamente capaz de completar texto de forma coherente y factual.

Pero también es un modelo peculiar. Si le preguntas "¿Cuál es la capital de Francia?", es igual de probable que complete con "es París" que con "es una pregunta frecuente en exámenes de geografía" o con "¿Cuál es la capital de Alemania?". El modelo no tiene concepto de "responder preguntas" — solo tiene el concepto de "continuar texto de forma plausible". Le falta algo crucial: saber cómo comportarse.

---

## La pipeline completa: de texto crudo a asistente útil

El camino de un modelo base a un asistente como ChatGPT fue formalizado en 2022 por el paper de InstructGPT (OpenAI), y desde entonces se ha convertido en el estándar de la industria. La pipeline tiene tres etapas principales:

**Etapa 1: Pretraining.** Como hemos descrito. Un modelo decoder-only entrena sobre billones de tokens de texto crudo. El output es un modelo base — capaz, pero sin comportamiento definido.

**Etapa 2: [[02-supervised-finetuning|Supervised Fine-Tuning]] (SFT, ajuste fino supervisado).** Se toma el modelo base y se entrena sobre un dataset (conjunto de datos) de pares de instrucción-respuesta cuidadosamente elaborados por humanos. "¿Cuál es la capital de Francia?" → "La capital de Francia es París." El modelo aprende el formato de la interacción: qué significa una pregunta y cómo se ve una respuesta adecuada. Tras el SFT, el modelo ya responde como un asistente — pero sus preferencias sobre qué es una "buena" respuesta todavía no están bien calibradas.

**Etapa 3: Alignment mediante [[05-rlhf-alineacion-llms|RLHF]].** RLHF significa Reinforcement Learning from Human Feedback — Aprendizaje por Refuerzo con Feedback Humano. Dado que el término aparecerá repetidamente en el libro, conviene definirlo desde ya: es un proceso donde el modelo recibe señales de calidad sobre sus outputs (generadas por evaluadores humanos o por un modelo recompensa entrenado para ello), y ajusta sus pesos para generar outputs que reciben mejor valoración. El resultado es un modelo que no solo responde correctamente sino que responde de formas que los humanos encuentran útiles, seguras, y apropiadas.

> **Descripción visual:** Diagrama de flujo horizontal con siete nodos alternando entre estadios de datos (óvalos azul claro) y procesos de entrenamiento (rectángulos amarillo pálido). Los óvalos representan artefactos: "Texto crudo / Internet", "Modelo base / Foundation", "Modelo Instruido" y "Asistente útil". Los rectángulos representan fases: "Pretraining CLM", "Supervised Fine-Tuning" y "RLHF Alineación". Las flechas son grises con punta triangular, dirección izquierda a derecha. Fondo blanco, tipografía sans-serif, estilo minimalista.

Hay también una forma alternativa de ver este proceso que resulta útil: se puede dividir en solo dos fases. La primera fase es el pretraining, donde el modelo construye capacidades lingüísticas y de conocimiento del mundo. La segunda fase es el post-training, que agrupa el SFT y el RLHF — todo lo que ocurre después del pretraining para dar forma al comportamiento del modelo. Ambas perspectivas (tres etapas vs. dos fases) describen el mismo proceso; la diferencia es de énfasis conceptual.

En el resto de este libro nos concentramos casi exclusivamente en el post-training: cómo tomar un modelo base o un modelo SFT existente y adaptarlo a nuestras necesidades específicas. Pero sin entender de dónde viene ese modelo, sin saber qué aprendió durante el pretraining y por qué, las técnicas que vamos a estudiar — LoRA, QLoRA, DPO, GRPO — quedan sin sustento.

---

## Pretraining continuo: cuando el modelo base necesita aprender más

Con la teoría en claro, llegamos a la primera técnica práctica del libro: el Continued Pre-Training (CPT, o Pretraining Continuo). Es aquí donde la teoría se vuelve acción, y donde muchos proyectos reales de fine-tuning encuentran su primer obstáculo.

El escenario es frecuente: tienes un modelo base potente (por ejemplo, LLaMA 3 8B) entrenado sobre texto general de internet. Necesitas que ese modelo entienda un dominio especializado — vocabulario médico, documentación técnica interna, normativa legal, patrones matemáticos avanzados. El problema: esos conceptos estaban subrepresentados en el dataset original del modelo.

¿Por qué no simplemente hacer SFT directamente sobre ese dominio? Porque el SFT funciona mejor cuando el modelo base ya tiene representaciones internas del dominio. Si le pides al modelo que responda preguntas de jurisprudencia pero nunca ha procesado texto jurídico en profundidad, el SFT puede enseñarle el formato de las respuestas, pero no puede insertar el conocimiento que falta. El resultado son respuestas que suenan confiadas pero son inexactas — el temido fenómeno de las alucinaciones.

El CPT resuelve esto volviendo al mismo objetivo del pretraining original: predecir el siguiente token sobre texto crudo del dominio objetivo. No hay pares instrucción-respuesta, no hay etiquetas humanas — solo documentos del dominio, tokenizados y procesados con la misma pérdida CLM. La diferencia con el pretraining original es que partimos de pesos ya entrenados (no de pesos aleatorios), lo cual hace el proceso mucho más eficiente y rápido.

Las ventajas concretas del CPT son tres. Primero, el vocabulario especializado: los modelos de propósito general suelen tokenizar términos técnicos de forma ineficiente, dividiéndolos en sub-tokens que no capturan bien su significado. Tras el CPT, el modelo aprende las relaciones entre esos términos y su contexto. "Hipoteca variable indexada al Euribor" deja de ser una sucesión de tokens sin cohesión y pasa a tener representaciones propias del ecosistema financiero. Segundo, el conocimiento del mundo actualizado: si el modelo base fue entrenado en 2023 y el mundo ha cambiado, el CPT puede incorporar información reciente. Tercero, la mejor base para SFT: un modelo que ya comprende el dominio responde mejor a la instrucción — los gradientes del SFT tienen menos trabajo que hacer porque el "terreno" ya está preparado.

---

## Aprendizaje por currículum: el orden de los datos importa más de lo que crees

Una vez que decides hacer CPT, la pregunta inmediata es: ¿en qué orden presento los datos al modelo? La respuesta intuitiva sería "en cualquier orden aleatorio" — y esa respuesta sería incorrecta.

El curriculum learning (aprendizaje por currículum) es la práctica de organizar los datos de entrenamiento de menor a mayor dificultad, replicando cómo los maestros humanos estructuran la enseñanza. No enseñas cálculo diferencial antes de que el alumno sepa álgebra. Lo mismo aplica a los modelos.

El problema que el currículum resuelve es la saturación. Cuando un modelo en fase de pretraining o CPT recibe datos de máxima complejidad desde el primer momento, los gradientes — las señales de corrección que fluyen hacia atrás durante el entrenamiento — pueden volverse inestables. El modelo no tiene un punto de apoyo desde el que interpretar la complejidad, y los pesos convergen hacia soluciones subóptimas. En el peor caso, el proceso de entrenamiento colapsa completamente.

La implementación práctica del currículum para CPT tiene dos dimensiones:

**Ordenar por calidad de la señal.** Un dataset de matemáticas podría contener libros de texto universitarios, papers académicos, soluciones de problemas en foros especializados, y posts casuales en Reddit sobre matemáticas. Estos cuatro tipos de datos tienen relaciones señal-ruido muy distintas. El currículum sugiere empezar con los libros de texto (gold standard: datos limpios, bien estructurados, densos en conocimiento verdadero) y gradualmente introducir los foros y los posts casuales (long tail: datos ruidosos, a veces incorrectos, pero que aportan variedad de expresión). El modelo aprende primero la lógica matemática limpia y luego aprende a operar en la messy realidad de cómo la gente escribe sobre matemáticas.

**Escalar la longitud del contexto progresivamente.** Empezar con ventanas de contexto de 512 tokens y escalar gradualmente a 4.096, luego a 8.192, y finalmente a la ventana máxima del modelo. La razón es que las dependencias de corto alcance (sintaxis local, coherencia de frase) son más fáciles de aprender que las de largo alcance (coherencia de párrafo, argumentación que se desarrolla a lo largo de páginas). Un modelo que intenta aprender dependencias de 128K tokens desde el primer día tiene dificultades para establecer las relaciones locales sobre las que se apoyan las relaciones largas.

> **Descripción visual:** Diagrama con dos subgrafos horizontales apilados verticalmente. El subgrafo superior "Calidad de datos" muestra cuatro óvalos conectados de izquierda a derecha: "Libros de texto" (verde, máxima calidad), "Papers curados" (amarillo), "Foros espec." (rojo claro) y "Posts casuales" (rojo claro, mayor ruido). El subgrafo inferior "Longitud de contexto" muestra tres óvalos azules conectados: "512 tokens", "2048 tokens" y "8192 tokens", representando escala progresiva. Las flechas indican dirección de progresión en el entrenamiento. Fondo blanco, tipografía sans-serif, estilo limpio y didáctico.

El resultado de un currículum bien diseñado es convergencia más rápida, menor pérdida final, y modelos más robustos. Es una de esas optimizaciones que no requiere hardware adicional ni modificaciones de arquitectura — solo disciplina en la preparación de los datos.

---

## Destilación del conocimiento: cómo un modelo pequeño hereda la sabiduría de uno grande

Otra técnica que aparece frecuentemente en los pipelines modernos de pretraining y post-training es la destilación del conocimiento (knowledge distillation). Merece un lugar en este capítulo porque es la explicación detrás de un fenómeno que quizás hayas observado: modelos de 7B parámetros que en ciertas tareas superan a modelos de 70B no destilados.

La idea parte de una observación sobre el entrenamiento estándar. Cuando un modelo aprende a predecir el siguiente token, recibe una etiqueta dura (hard label): el token correcto es este, los demás son incorrectos. Pero esta supervisión binaria ignora información valiosa que está implícita en la distribución de probabilidades del modelo: el hecho de que "París" sea la respuesta correcta, pero "Lyon" tenga un 5% de probabilidad y "Madrid" tenga un 1%, dice algo sobre la estructura del espacio semántico que la etiqueta dura no captura.

A estas probabilidades distribuidas las llamamos soft labels (etiquetas suaves). Y la idea de la destilación es usar las soft labels de un modelo grande (el "profesor" o teacher) para entrenar un modelo más pequeño (el "alumno" o student). El alumno no aprende simplemente que la respuesta es "París" — aprende que "París" tiene 90% de probabilidad, que "Lyon" tiene 8%, y que "Londres" tiene 1%. Esa información estructural sobre las relaciones entre conceptos es lo que los investigadores llaman dark knowledge (conocimiento oscuro): el conocimiento que no está en las respuestas correctas, sino en la estructura de las respuestas incorrectas.

La función de pérdida del alumno combina dos términos:

$$\mathcal{L}_{distill} = (1 - \lambda) \cdot \mathcal{L}_{CE}(y, \hat{y}_{student}) + \lambda \cdot \mathcal{L}_{KL}(p_{teacher}^{(T)}, p_{student}^{(T)})$$

donde $\mathcal{L}_{CE}$ es la pérdida de entropía cruzada estándar contra la etiqueta dura, $\mathcal{L}_{KL}$ es la divergencia KL entre las distribuciones del profesor y el alumno (ambas "suavizadas" con temperatura $T > 1$ para hacer las soft labels más informativas), y $\lambda$ es un hiperparámetro que balancea cuánto pesa el conocimiento del profesor vs. la supervisión directa.

Una variante especialmente relevante para LLMs es la Chain-of-Thought Distillation: en lugar de destilar distribuciones de probabilidad sobre tokens, se usan las cadenas de razonamiento paso a paso del modelo profesor como datos de entrenamiento del alumno. Si el profesor tiene 70B parámetros y genera trazas de razonamiento matemático de alta calidad, el alumno de 7B puede aprender esos patrones de razonamiento directamente. Esto explica por qué modelos como DeepSeek-R1-Distill son capaces de razonar de forma sorprendente a pesar de su tamaño reducido.

---

## El muro de la memoria: por qué entrenar es más difícil de lo que parece

Aquí es donde la teoría choca con la física. Supón que tienes un modelo de 7B parámetros almacenados en precisión FP16 (punto flotante de 16 bits, 2 bytes por parámetro). El tamaño de los pesos es $7 \times 10^9 \times 2 = 14$ GB. Tienes una GPU con 16 GB de VRAM. ¿Caben? En inferencia, sí. En entrenamiento, no.

Durante el entrenamiento, el modelo necesita almacenar mucho más que los pesos:

- **Gradientes**: por cada parámetro, hay que guardar cuánto cambiar ese parámetro. Mismo tamaño que los pesos: otros 14 GB.
- **Estados del optimizador**: si usas Adam (el optimizador más común), hay que guardar dos momentos estadísticos por parámetro (la media y la varianza del gradiente). En FP32 (4 bytes), eso son $7 \times 10^9 \times 4 \times 2 = 56$ GB solo para el optimizador.
- **Activaciones**: durante el forward pass, el modelo calcula y guarda las activaciones intermedias de cada capa (los valores que produce antes de aplicar la siguiente operación). En un modelo grande con batches grandes, esto puede ser otros 20-30 GB.

Total: un modelo de 7B puede necesitar más de 100 GB de VRAM para entrenarse. Eso es más que una sola GPU H100 (80 GB). Este es el muro de la memoria, y todo el ecosistema de técnicas de eficiencia que estudiaremos en capítulos posteriores existe para derrumbarlo.

Hay tres palancas principales para reducir el consumo de memoria durante el entrenamiento:

**Precisión mixta (BF16).** El formato Brain Float 16 usa solo 16 bits por número, pero con un rango de magnitudes equivalente al de FP32 (32 bits). Esto es crucial porque FP32 estándar usa 8 bits para el exponente (el rango de magnitudes) y 23 para la mantisa (la precisión decimal). BF16 sacrifica precisión decimal (solo 7 bits de mantisa) pero conserva el rango (8 bits de exponente), lo que hace el entrenamiento estable. El resultado: los pesos del modelo pueden almacenarse en BF16, reduciendo a la mitad el tamaño. Los estados del optimizador suelen mantenerse en FP32 para estabilidad, pero los gradientes pueden calcularse en BF16. Una regla práctica: pasar de FP32 a BF16 en pesos y activaciones puede reducir el consumo a la mitad con impacto mínimo en calidad.

**Gradient accumulation (acumulación de gradiente).** Si no puedes entrenar con el batch size óptimo por limitaciones de memoria, puedes simular batches grandes dividiendo el batch en micro-batches más pequeños, calculando gradientes para cada uno, y acumulándolos (sumándolos) antes de hacer el paso de optimización. Si quieres un batch effective de 64 samples pero solo caben 8 en memoria, haces 8 pasos de forward+backward con 8 samples cada uno, acumulas los gradientes, y luego actualizas los pesos. El modelo "ve" efectivamente 64 samples por update, pero nunca más de 8 a la vez en VRAM.

**Activation checkpointing.** Durante el forward pass, el modelo calcula una cascada de activaciones intermedias (los outputs de cada capa antes de pasarlos a la siguiente). Para calcular los gradientes en el backward pass, necesitas esas activaciones. La solución estándar es guardarlas todas en memoria durante el forward pass — costoso. Activation checkpointing propone guardar solo un subconjunto (checkpoints) y recalcular el resto durante el backward pass cuando se necesitan. El trade-off es concreto: reduces el consumo de VRAM de activaciones en aproximadamente un 70-80%, a costa de aumentar el tiempo de cómputo en un ~25% por los recálculos. En muchos escenarios, ese es un intercambio que vale la pena.

> **Descripción visual:** Diagrama con dos subgrafos horizontales. El subgrafo izquierdo "Memoria necesaria — modelo 7B" tiene cuatro óvalos rojos (pesos 14 GB, gradientes 14 GB, optimizador Adam 56 GB, activaciones ~20 GB) con flechas que convergen en un óvalo gris neutro "Total ~100 GB". El subgrafo derecho "Técnicas de reducción" tiene tres óvalos verdes (BF16 ÷2 memoria, Grad. Accum. batch virtual, Act. Checkpt. -70% activ.) con flechas que convergen en un óvalo azul "GPU accesible". Fondo blanco, tipografía sans-serif, estilo problema-solución.

---

## Paralelismo en múltiples GPUs: el modelo como orquesta

Incluso con todas las optimizaciones de memoria anteriores, los modelos más grandes simplemente no caben en una sola GPU. La solución es distribuir el trabajo entre múltiples GPUs — lo que se llama entrenamiento paralelo. Hay cuatro estrategias principales:

**Data Parallelism (DP, paralelismo de datos).** La estrategia más simple: copias el modelo completo en cada GPU, pero cada GPU procesa una porción diferente del batch de datos. Al final de cada paso, las GPUs sincronizan sus gradientes — cada GPU comparte cuánto quería cambiar cada parámetro, y todos promedian. El resultado es que el modelo se actualiza como si un solo proceso hubiera procesado todo el batch. La limitación: si el modelo no cabe en una sola GPU, este enfoque no funciona.

**Tensor Parallelism (TP, paralelismo de tensores).** Cuando una sola capa del modelo es demasiado grande para una GPU, se divide la capa en fragmentos que van a distintas GPUs. Por ejemplo, una matriz de atención de 8.192 × 8.192 puede dividirse en dos matrices de 8.192 × 4.096, cada una en una GPU diferente. Las GPUs deben comunicarse constantemente durante cada forward pass — literalmente en cada operación de multiplicación matricial — por lo que esta estrategia requiere interconexiones de muy alta velocidad como NVLink (la tecnología de NVIDIA para comunicación GPU-GPU directa, con anchos de banda de cientos de GB/s). Sin NVLink, la comunicación se convierte en el cuello de botella.

**Pipeline Parallelism (PP, paralelismo de pipeline).** En lugar de dividir las capas horizontalmente (TP), se divide el modelo verticalmente por grupos de capas: GPU 0 procesa las capas 1-20, GPU 1 procesa las capas 21-40, etc. Los activations fluyen de una GPU a la siguiente como en una cadena de montaje. El problema clásico son las "burbujas" (bubbles): cuando GPU 1 espera los outputs de GPU 0 para empezar a procesar, está ociosa. Hay técnicas para reducir estas burbujas (como el micro-batching en pipeline), pero siempre habrá algo de ineficiencia inherente.

**Sequence Parallelism (SP, paralelismo de secuencia).** Cuando el contexto es extremadamente largo (pienso en modelos con ventanas de 128K tokens), las activaciones asociadas a esa secuencia pueden ser enormes. SP distribuye la secuencia entre GPUs: GPU 0 procesa los tokens 1-32K, GPU 1 procesa los 32K-64K, etc. Es complementario al TP y se usa frecuentemente en conjunto.

La solución estándar moderna que combina varias de estas estrategias es FSDP (Fully Sharded Data Parallelism, Paralelismo de Datos Completamente Fragmentado). En lugar de replicar el modelo completo en cada GPU (DP), FSDP fragmenta los pesos, gradientes, y estados del optimizador entre todas las GPUs. Cuando una GPU necesita una porción de los pesos para calcular, los recupera de las otras GPUs en ese momento. El resultado es la eficiencia de memoria del model parallelism con la escalabilidad del data parallelism. Frameworks como DeepSpeed y PyTorch FSDP implementan esta estrategia y son el estándar en proyectos serios de fine-tuning a gran escala.

---

## El lab: pretraining continuo en matemáticas con un SLM

Con la teoría clara, vamos a ensuciarnos las manos. El experimento de este capítulo ilustra CPT en su forma más directa: tomamos un Small Language Model (SLM, Modelo de Lenguaje Pequeño) preentrenado y lo sumergimos en un corpus de matemáticas para adaptar sus representaciones internas al razonamiento matemático.

La decisión de usar un SLM en lugar de un LLM grande es deliberada. Primero, hace el experimento reproducible en hardware accesible (una sola GPU). Segundo, los principios que aplican a 7B parámetros son exactamente los mismos que aplican a 70B — la única diferencia es la escala. Si entiendes el experimento con el modelo pequeño, entiendes los fundamentos.

El setup de referencia del experimento es:

- **Modelo base**: un SLM con arquitectura decoder-only preentrenado en texto general (la elección exacta puede variar, pero modelos como Qwen2.5 0.5B o Smollm2 360M son candidatos típicos para este tipo de experimento).
- **Dataset**: un corpus de matemáticas con datos de distinta calidad — desde libros de texto bien estructurados hasta derivaciones de foros, organizados en currículum de menor a mayor ruido.
- **Objetivo**: causal language modeling estándar (predecir el siguiente token).
- **Hardware típico**: una GPU con 24 GB de VRAM (por ejemplo, RTX 3090 o A10G).

Tres decisiones de implementación merecen atención especial:

**El currículum de datos.** El dataset no se mezcla aleatoriamente. Los primeros epochs (pasadas completas por el dataset; un epoch significa que el modelo ha visto cada ejemplo una vez) procesan los datos de mayor calidad — libros de texto, papers curados. Los epochs posteriores introducen datos más ruidosos. Esta estructura no requiere modificaciones de código complejas: basta con concatenar los archivos de datos en el orden correcto antes de comenzar el entrenamiento.

**La longitud de contexto progresiva.** El entrenamiento comienza con `max_seq_length=512`. Después de un número determinado de pasos, se escala a 1024, luego a 2048. Esto se implementa modificando el DataLoader entre fases de entrenamiento, o usando callbacks que ajustan el parámetro en caliente.

**Las métricas a vigilar.** La métrica primaria durante CPT es la perplexity (perplejidad) sobre el dominio objetivo. La perplejidad mide cuán "sorprendido" está el modelo por los datos — una perplejidad de 5 significa que el modelo, en promedio, considera 5 tokens igualmente probables en cada posición. Una perplejidad de 2 indica que el modelo está mucho más seguro. Durante el CPT, deberías ver la perplejidad en el conjunto de validación matemática descender progresivamente a lo largo de las horas de entrenamiento. Si la perplejidad en el dominio objetivo baja pero la perplejidad en texto general sube dramáticamente, estás sufriendo catastrophic forgetting (olvido catastrófico) — el modelo está perdiendo sus capacidades generales para ganar capacidades matemáticas. La proporción típica para evitar esto es mezclar un 5-10% de datos generales con los datos del dominio objetivo.

La señal de que el CPT funcionó no es solo la perplejidad baja — es la calidad de las respuestas en el SFT posterior. Un modelo que ha hecho CPT sobre matemáticas generará razonamientos matemáticos más coherentes cuando se instruccione después, con menos alucinaciones en los pasos intermedios. Ese efecto compuesto — CPT como preparación del terreno para SFT — es la razón por la que vale la pena invertir el tiempo en este paso.

---

## Qué viene después

Este capítulo ha construido los cimientos. Entendemos qué es un Transformer y por qué funciona, conocemos las tres arquitecturas y por qué los modelos decoder-only dominan, sabemos cómo un modelo base adquiere sus capacidades durante el pretraining, y hemos explorado las primeras técnicas prácticas: CPT, currículum de datos, destilación, y optimizaciones de memoria y paralelismo.

Lo que aún no hemos respondido es la pregunta central del libro: ¿cómo tomamos ese modelo base y lo convertimos en algo útil para una tarea específica sin el costo prohibitivo de reentrenarlo desde cero? El siguiente capítulo introduce el Supervised Fine-Tuning — la primera y más directa respuesta a esa pregunta — y comienza a explorar cómo los avances en eficiencia de memoria (en particular, [[04-qlora-cuantizacion-4bit|QLoRA]]) hacen posible el fine-tuning en hardware accesible.

---

## Tags

#arquitectura/transformer #concepto/atención #concepto/self-attention #técnica/pretraining #concepto/causal-language-modeling #concepto/modelo-base #técnica/knowledge-distillation #nivel/introducción #tipo/lección #estado/completo



---
capitulo: "02"
titulo: "Supervised Fine-Tuning: Cómo un Modelo Aprende a Conversar"
aliases:
  - "Capítulo 02"
  - "Cap 02"
  - "SFT"
  - "Supervised Fine-Tuning"
tema: "técnica-sft"
subtemas: [sft, chat-template, chain-of-thought]
dificultad: "introducción"
tipo: "lección"
estado: "completo"
conceptos_centrales:
  - supervised-fine-tuning
  - sft
  - chat-template
  - loss-masking
  - chain-of-thought
  - continued-pretraining
prerequisitos:
  - "[[01-fundamentos-transformers-y-pretraining]]"
relacionados:
  - "[[03-lora-adaptacion-de-bajo-rango]]"
  - "[[05-rlhf-alineacion-llms]]"
tags:
  - técnica/sft
  - concepto/chat-template
  - concepto/loss-masking
  - técnica/chain-of-thought
  - técnica/continued-pretraining
  - concepto/transformer
  - técnica/reinforcement-learning
  - nivel/introducción
  - tipo/lección
  - estado/completo
---

# Capítulo 2 — Supervised Fine-Tuning: Cómo un Modelo Aprende a Conversar

> Basado en "The Engineer's Guide to Supervised Finetuning" y "Supervised Finetuning for Reasoning Models (From Dataset to Deployment)", The Neural Maze, Finetuning Sessions · Lesson 2 & Lab 2.

Imagina que contratas a alguien que ha leído toda la biblioteca pública de tu ciudad. Sabe de física, de cocina, de historia medieval, de contratos de alquiler. Pero si le pides que atienda el teléfono de atención al cliente de tu empresa, se va a quedar mirándote. No porque no sepa cosas — sabe demasiadas — sino porque nunca aprendió el ritmo de una conversación estructurada: cuándo escuchar, cuándo responder, qué tono usar, cómo terminar un turno. Ese es exactamente el problema que resuelve el Supervised Fine-Tuning.

Un modelo que solo ha pasado por preentrenamiento es ese empleado enciclopédico: asombroso en conocimiento, inútil en protocolo. El SFT es el proceso de entrenamiento que le enseña el protocolo.

---

## De la predicción de texto a la conversación estructurada

Para entender qué hace el SFT, primero hay que entender qué hace el preentrenamiento y dónde se queda corto.

Durante el preentrenamiento — o más precisamente durante el Continued Pretraining (CPT, preentrenamiento continuado) — el modelo recibe un flujo masivo e ininterrumpido de texto. Libros, artículos de Wikipedia, hilos de Reddit, código fuente, foros de física cuántica, recetas de cocina: todo mezclado en una secuencia gigantesca. El objetivo del modelo en esta etapa es sencillo y brutal: dado el texto que has visto hasta ahora, predice el siguiente token.

Un token es la unidad mínima con la que trabaja el modelo. No es exactamente una palabra ni exactamente un carácter — es algo intermedio. La palabra "tokenización" se convierte, según el vocabulario del modelo, en algo como ["token", "ización"] o ["tok", "en", "iza", "ción"]. El proceso de convertir texto en estos fragmentos se llama tokenización. Todo lo que el modelo procesa — preguntas, respuestas, código, markdown — pasa primero por este filtro y se convierte en una secuencia de enteros que representan posiciones en el vocabulario.

Con suficiente texto y suficiente cómputo, el modelo aprende correlaciones estadísticas profundas. "Quantum" aparece frecuentemente junto a "physics" y "entanglement". "La solución a la ecuación diferencial" tiende a ir seguido de pasos algebraicos. El modelo no entiende estas relaciones en ningún sentido filosófico — simplemente las ha visto tantas veces que las ha internalizado como patrones predictivos. El resultado es un sistema muy bueno completando texto.

El problema es exactamente ese: solo sabe completar texto. No tiene ningún concepto de turno. Si le preguntas "¿Cuál es la capital de Francia?", puede que responda "París. La ciudad fue fundada por la tribu celta de los Parisios hacia el siglo III a.C. y..." — y siga generando texto indefinidamente. No sabe que debía parar tras "París". No sabe que había una pregunta, que hay alguien esperando, ni que su papel es el de asistente. Simplemente continúa el flujo.

Aquí entra el SFT. El Supervised Fine-Tuning (entrenamiento supervisado por ajuste fino) no añade conocimiento nuevo al modelo — la enciclopedia ya está dentro. Lo que hace es enseñarle estructura: quién habla cuándo, dónde empieza una respuesta, dónde termina, qué tono adoptar. Es la diferencia entre saber mucho y saber comportarse.

---

## El chat template: traducir conversaciones a tokens

Hay una pregunta que suena filosófica pero es completamente práctica: ¿dónde termina "tú" y empieza "yo" dentro de un [[01-fundamentos-transformers-y-pretraining|transformer]]?

A nivel de pesos y matrices, no hay ninguna distinción. Solo hay una secuencia de tokens fluyendo por capas de atención. El modelo no sabe que algunos de esos tokens son del usuario y otros son suyos — a menos que se lo enseñemos explícitamente.

Eso es lo que hace el chat template (plantilla de conversación). Es la capa de traducción entre la representación limpia que usamos en Python — una lista de diccionarios con campos `role` y `content` — y la cadena plana de texto con tokens especiales que el modelo realmente ve. Sin esta plantilla, todo colapsa en un único stream indiferenciado.

Cuando en Python escribes algo así:

```python
messages = [
    {"role": "system", "content": "Eres un asistente útil."},
    {"role": "user", "content": "¿Cuál es la capital de Francia?"},
    {"role": "assistant", "content": "La capital de Francia es París."}
]
```

El tokenizador, siguiendo su chat template, convierte eso en algo parecido a esto (usando el formato de Qwen3 como ejemplo):

```
<|im_start|>system
Eres un asistente útil.<|im_end|>
<|im_start|>user
¿Cuál es la capital de Francia?<|im_end|>
<|im_start|>assistant
La capital de Francia es París.<|im_end|>
```

Esos tokens especiales — `<|im_start|>`, `<|im_end|>`, las etiquetas de rol — no son lenguaje natural. Son señales estructurales. El modelo aprende durante el SFT a asociar `<|im_start|>assistant` con el estado de "ahora me toca a mí responder", y `<|im_end|>` con "aquí termino". Es análogo a las señales de tráfico: no forman parte del paisaje, pero sin ellas el tráfico se caotiza.

Esta plantilla vive dentro del archivo `tokenizer_config.json` del modelo, generalmente escrita en Jinja2, un lenguaje de plantillas. El hecho de que esté en Jinja2 no es trivial: significa que la plantilla puede incluir lógica condicional. Por ejemplo, el template puede omitir el bloque `<think>...</think>` si el usuario desactiva el razonamiento, o formatear los tool calls de forma diferente según el modelo.

> **Descripción visual:** Diagrama de flujo horizontal que se expande en abanico hacia la derecha. Un bloque azul ("Lista de mensajes Python") apunta a un bloque amarillo ("Tokenizador + Jinja2"), que a su vez apunta a otro bloque amarillo ("Tokens de estructura"). Desde este bloque central parten cuatro flechas hacia cuatro bloques verdes dispuestos verticalmente: "Zona Sistema", "Zona Usuario", "Zona Asistente" y "Zona Think / Herramientas". Las flechas son grises con punta triangular. Fondo blanco, bloques con esquinas redondeadas, tipografía sans-serif.

Aquí yace una trampa que atrapa a muchos equipos. Si entrenas un modelo con un chat template y luego lo despliegas con otro, la estructura que el modelo aprendió ya no coincide con los tokens que recibe en producción. El modelo empieza a comportarse de forma extraña: responde en el turno del usuario, ignora el token de fin de turno, genera respuestas interminables. Estos bugs son difíciles de diagnosticar porque no producen errores de Python — solo comportamiento ilógico. La solución es simple pero requiere disciplina: guardar siempre la plantilla junto al modelo y usarla de forma consistente en entrenamiento y en inferencia.

El chat template se vuelve especialmente crítico en escenarios de razonamiento. Modelos como Qwen3 o DeepSeek-R1 usan etiquetas `<think>` y `</think>` para delimitar un espacio de razonamiento interno. El modelo genera pensamiento dentro de esas etiquetas — como un borrador que el usuario no ve — y luego produce la respuesta final. Sin el template correcto, esas etiquetas no tienen semántica y el modelo no aprende a usarlas.

Los sistemas agénticos van un paso más allá. Cuando el modelo necesita llamar a herramientas externas, el template introduce tokens como `<|tool_call|>` y `<|tool_response|>`. Ahora la conversación no es solo texto — es texto con zonas arquitectónicas: zona de usuario, zona de asistente, zona de herramienta, zona de observación. El template es lo que hace legible toda esa estructura para el modelo.

---

## CPT vs. SFT: el buffet y el menú de degustación

La diferencia entre Continued Pretraining y Supervised Fine-Tuning no es solo el tipo de datos — es fundamentalmente distinta la forma en que el modelo aprende de esos datos.

El CPT es un buffet. El modelo ingiere texto en volúmenes masivos, maximizando la cantidad de conocimiento absorbido por FLOP (Floating-point Operation — operación de punto flotante, la unidad básica de cómputo que usan las GPUs). No importa si los textos son de géneros distintos o hablan de temas sin relación. El objetivo es la amplitud: que el modelo vea tantos patrones lingüísticos y factuales como sea posible. En el CPT, el modelo calcula una pérdida — es decir, una penalización por predecir incorrectamente el siguiente token — sobre absolutamente cada token de la secuencia.

El SFT es un menú de degustación. El orden importa, la presentación importa, la separación entre platos importa. Y aquí está la diferencia técnica más importante: el enmascaramiento de pérdida, o loss masking.

En SFT, no queremos que el modelo aprenda a predecir las palabras del usuario — queremos que aprenda a responder a esas palabras. Para lograrlo, durante el SFT solo calculamos la pérdida en los tokens del asistente, no en los del usuario ni del sistema. En PyTorch, la función de pérdida estándar para este problema es `CrossEntropyLoss`. Esta función tiene un parámetro `ignore_index` que, cuando se establece en `-100`, le dice al optimizador: "ignora este token, no lo uses para actualizar los pesos". Así, los tokens de usuario reciben la etiqueta `-100` y el modelo los procesa para entender contexto pero no intenta aprender a predecirlos. Solo los tokens del asistente generan gradiente.

> **Descripción visual:** Diagrama de flujo horizontal con dos caminos paralelos que confluyen en la derecha. Camino superior: un bloque rojo ("Tokens Sistema y Usuario") apunta con una flecha etiquetada "label = -100" a un bloque gris ("Sin gradiente / ignorados"). Camino inferior: un bloque verde ("Tokens Asistente") apunta con una flecha etiquetada "label = real" a un bloque azul ("Genera gradiente"), que a su vez apunta a un bloque morado ("Actualiza pesos del modelo"). Los bloques tienen esquinas redondeadas. El camino superior es visualmente apagado (gris), el inferior es vivo y progresivo. Fondo blanco, tipografía sans-serif.

La pérdida de entropía cruzada mide cuánto se equivoca el modelo. Si el token correcto era "París" y el modelo le asignó probabilidad 0.1 (10%), la pérdida es alta — el modelo debe ajustar sus pesos para aumentar esa probabilidad. Si el modelo ya asignaba 0.9 (90%), la pérdida es baja y el ajuste es mínimo. Este mecanismo, repetido millones de veces sobre millones de tokens del asistente, es lo que da forma al comportamiento conversacional del modelo.

> **Descripción visual:** Diagrama de flujo horizontal con dos subgrafos paralelos conectados por una flecha central. El subgrafo izquierdo (fondo azul oscuro, etiqueta "CPT — El Buffet") contiene tres bloques verticales azul medio: "Texto continuo mezclado", "Pérdida en todos los tokens" y "Conocimiento amplio", unidos por flechas descendentes. El subgrafo derecho (fondo verde oscuro, etiqueta "SFT — El Menú") replica la estructura con bloques verdes: "Pares pregunta-respuesta", "Pérdida solo en asistente" y "Comportamiento moldeado". Una flecha gris etiquetada "construye la base" une el subgrafo izquierdo con el derecho. Estilo limpio, fondo blanco, tipografía sans-serif.

### El Packing Paradox

En CPT, es habitual usar una técnica llamada packing (empaquetado): concatenar múltiples documentos en una sola ventana de contexto de 8k o 128k tokens para mantener las GPUs trabajando a plena capacidad. Si los documentos son cortos, en lugar de desperdiciar espacio los apilamos uno tras otro. Tiene todo el sentido: la GPU procesa bloques completos y el throughput se maximiza.

Pero en SFT, el packing ingenuo produce un problema sutil y devastador: la contaminación cruzada. Imagina que el final de la conversación A y el inicio de la conversación B quedan en el mismo bloque de contexto sin separación adecuada. El mecanismo de atención del transformer — que por diseño puede "mirar" cualquier token anterior en la secuencia — puede crear conexiones entre el final de A y el inicio de B. El modelo empieza a contaminar el contexto de un diálogo con el de otro, como si el asistente estuviera respondiendo una pregunta mezclando información de dos conversaciones completamente distintas.

La solución moderna combina dos tecnologías. La primera es Flash Attention 2, una implementación eficiente de la atención que soporta secuencias de longitud variable (Varlen sequences). La segunda es el tensor `cu_seqlens` (cumulative sequence lengths — longitudes de secuencia acumuladas). En lugar de rellenar las conversaciones cortas con ceros inútiles (padding) o concatenarlas sin separación, le pasamos a la GPU un tensor que le dice exactamente dónde termina cada conversación. La GPU procesa el batch completo — con múltiples conversaciones de diferentes longitudes — pero mantiene un cortafuegos rígido entre ellas. Ningún token de la conversación A puede "contaminar" la conversación B. Librerías como Hugging Face TRL y Unsloth implementan esto de forma transparente.

El resultado es lo mejor de ambos mundos: throughput cercano al packing puro, con integridad estructural de SFT.

---

## La taxonomía del entrenamiento: lo que SFT es y lo que no es

En la industria, los términos se confunden regularmente. Se habla de "modelos de instrucción" y "modelos de razonamiento" como si fueran categorías taxonómicas distintas, dos especies diferentes de LLM. No lo son.

SFT no es un tipo de modelo. Es un paso de entrenamiento. Una herramienta.

El principio subyacente es simple: el modelo aprende a mapear un tipo de entrada a un tipo de salida. Si los datos de entrenamiento contienen respuestas cortas y directas, el modelo aprende a producir respuestas cortas y directas. Si contienen largas cadenas de razonamiento explícito, aprende a producir largas cadenas de razonamiento. La diferencia entre un "modelo de instrucción" y un "modelo de razonamiento" no reside en un algoritmo secreto — reside en los datos con los que se hizo el SFT.

Una confusión relacionada, y más perniciosa, es creer que el Reinforcement Learning (RL — aprendizaje por refuerzo) es el ingrediente mágico que "hace pensar" a un modelo. Esa narrativa oculta cómo funciona el pipeline real.

El SFT es el que le da al modelo la estructura del razonamiento. Expone al sistema a ejemplos de cómo se descomponen los problemas, cómo fluye la lógica, cómo se estructura un argumento. En otras palabras, el SFT enseña al modelo qué aspecto tiene el razonamiento. El RL llega después. Su papel no es inventar el razonamiento desde cero — es reforzar el buen comportamiento y desincentivar los atajos. Empuja al modelo hacia respuestas más claras, más verídicas, más eficientes. Pero si el modelo nunca vio razonamiento estructurado durante el SFT, no hay nada que el RL pueda refinar.

DeepSeek-R1 es el ejemplo canónico. Antes de aplicar RL, el equipo ejecutó una fase de "cold start" de SFT sobre datos de alta calidad de Chain-of-Thought (CoT — cadena de pensamiento, es decir, razonamiento explícito paso a paso escrito en texto). Esa fase sembró el comportamiento de razonamiento secuencial. Solo después de establecida esa base, el RL entró a premiar la consistencia y penalizar la lógica débil. Sin el SFT previo, el RL no sabía qué recompensar.

La forma correcta de pensar en el pipeline de entrenamiento es como capas:

- CPT construye conocimiento amplio — la enciclopedia.
- SFT da forma al comportamiento — la persona conversacional y las reglas de interacción.
- RL refina la alineación — fomentando precisión, coherencia y preferencias humanas.

Cada capa construye sobre la anterior. Juntas, mueven al sistema desde patrones estadísticos crudos hacia comportamiento estructurado y alineado.

> **Descripción visual:** Diagrama de flujo horizontal con cinco bloques rectangulares redondeados conectados en línea por flechas grises con punta triangular. De izquierda a derecha: "Texto crudo / Libros, foros, código" (gris neutro), "CPT / Enciclopedia" (azul), "SFT / Comportamiento" (verde), "RL / Alineación" (amarillo oscuro), "Modelo listo para producción" (rojo-rosa). Cada bloque tiene dos líneas de texto: la primera en negrita indica la etapa, la segunda indica el resultado. Las flechas son de igual tamaño y el layout es perfectamente lineal. Fondo blanco, estilo minimalista.

---

## Curación de datos: la hipótesis LIMA y la calidad sobre la cantidad

Durante mucho tiempo, la ecuación era simple: más datos igual a mejor modelo. Datasets más grandes, mayor cobertura, más ejemplos — esa era la fórmula. Resultó estar equivocada, al menos para el SFT.

El paper LIMA (Less Is More for Alignment — Menos es Más para la Alineación, 2023) sacudió esa premisa. Los investigadores tomaron un [[01-fundamentos-transformers-y-pretraining|modelo base]] y lo ajustaron con apenas 1.000 ejemplos cuidadosamente curados. El resultado derrotó en benchmarks de comportamiento a modelos entrenados con decenas de miles de ejemplos de peor calidad. Mil frente a decenas de miles, y el más pequeño ganó.

¿Por qué? Porque el SFT es absorbente de maneras que van más allá del contenido factual. El modelo no solo aprende qué decir — aprende cómo decirlo. Y eso incluye los vicios. Si el dataset contiene razonamientos descuidados, respuestas vagas, o inconsistencias de formato, el modelo los internaliza como la norma. No distingue entre "este es un ejemplo mediocre" y "este es el estándar de comportamiento que debo seguir". Trata todo lo que ve como el patrón correcto.

Eso convierte la curación de datos en una tarea de composición, no de recolección. Un dataset de SFT fuerte es como una sinfonía bien orquestada: necesitas razonamiento matemático para afinar la lógica, código de alta calidad para reforzar la estructura, ejemplos conversacionales para moldear el tono y la persona, muestras con restricciones de seguridad para anclar el comportamiento dentro de límites aceptables. La proporción importa tanto como el contenido individual.

Por eso muchos labs han abandonado los grandes datasets de instrucciones extraídos de internet y han migrado hacia pipelines sintéticos. En lugar de recopilar lo que está disponible, generan ejemplos de alta calidad usando modelos "teacher" potentes — modelos más grandes que actúan como tutores — y luego filtran agresivamente. El objetivo no es cantidad sino densidad: que cada ejemplo esté cargado de estructura útil y claridad máxima.

El pipeline del lab de este capítulo ilustra esto bien. Partiendo de 20.000 transcripciones de audio del dataset YouTube Commons, se aplicó NVIDIA Nemotron-3-Nano-30B-A3B como teacher model para generar respuestas estructuradas de alta calidad — incluyendo trazas de razonamiento. La inferencia corrió en batch sobre vLLM en una sola H100 GPU durante unas 3 horas y 20 minutos. El resultado: un dataset sintético limpio, denso, con las dos variantes que se necesitaban para el experimento.

---

## SFT para razonamiento: entrenar el proceso, no solo la respuesta

El SFT tradicional tiene una arquitectura conceptual simple: entra una pregunta, sale una respuesta. El modelo aprende a mapear A directamente a C. Para muchas tareas, eso funciona perfectamente. Pero también incentiva los atajos. Si el modelo puede ir de A a C sin pasar por B, lo hará — incluso cuando debería razonar. Y esos atajos son exactamente donde aparecen las alucinaciones: el modelo genera una respuesta que suena plausible pero es factualmente incorrecta, porque nunca construyó el razonamiento que habría detectado el error.

El SFT orientado a razonamiento cambia el patrón. En lugar de entrenar el salto A → C, entrenamos el camino completo: A → B → C. Ese paso intermedio — B — es la traza de razonamiento. El desglose paso a paso. El pensamiento visible. Lo que se entrena no es la respuesta final, sino el proceso que conduce a ella.

Una traza de razonamiento bien construida se ve así. Dada la pregunta "¿Cuántos segundos hay en una semana?", en lugar de responder directamente "604.800", la traza mostraría:

```
<think>
Una semana tiene 7 días.
Un día tiene 24 horas.
Una hora tiene 60 minutos.
Un minuto tiene 60 segundos.
Entonces: 7 × 24 × 60 × 60 = 7 × 24 × 3.600 = 7 × 86.400 = 604.800.
</think>

Hay 604.800 segundos en una semana.
```

El modelo que se entrena con ejemplos así aprende algo sutil pero poderoso: antes de producir una respuesta, existe una fase dedicada a trabajar el problema. La etiqueta `<think>` crea un espacio protegido para ese proceso. El modelo internaliza que ese espacio no es opcional — es parte del protocolo.

La clave, y esto fue central en DeepSeek-R1, es que no puedes esperar que el RL invente este comportamiento desde cero. El espacio de búsqueda es demasiado grande. Sin ejemplos de razonamiento estructurado en el SFT, el modelo no tiene plantilla que optimizar. Deambula. El hábito de razonamiento tiene que sembrarse primero en el SFT, y solo entonces el RL puede refinarlo — recompensando derivaciones correctas y penalizando saltos injustificados.

La diferencia práctica es visible incluso en modelos pequeños. En el experimento de este capítulo, entrenamos dos versiones de Qwen3-0.6B con el mismo dataset pero columnas diferentes: una con las respuestas directas (`messages_no_thinking`) y otra con las trazas de razonamiento (`messages_thinking`). Al desplegar ambos modelos y hacerles la misma pregunta matemática, el modelo sin razonamiento produce una respuesta que puede ser correcta o no — pero no puedes saber por qué. El modelo con razonamiento muestra su trabajo: si se equivoca, puedes identificar exactamente en qué paso. Y si acierta, tienes confianza en que no fue por azar.

---

## Los detalles que importan: masking, shift-right y batching

El SFT parece conceptualmente simple, pero su implementación exige una precisión que el CPT no requiere. Tres problemas específicos aparecen sistemáticamente.

### El shift-right y la frontera usuario-asistente

Los modelos de lenguaje causal — Causal Language Models (CLM) — son aquellos que predicen el siguiente token basándose únicamente en los tokens anteriores, nunca en los futuros. Esto es lo que permite que la generación sea autoregresiva: el modelo genera un token, lo añade a la secuencia, y predice el siguiente. Es el paradigma de todos los LLMs modernos tipo GPT, Llama, Qwen.

Esta arquitectura tiene una consecuencia importante en el entrenamiento: el modelo siempre predice la posición siguiente, no la actual. Esto se llama setup shift-right (desplazamiento a la derecha). Si la secuencia de tokens de entrada es [A, B, C, D], los targets (lo que el modelo debe predecir) son [B, C, D, E]. Cada posición predice la siguiente.

El problema de la frontera surge aquí. El último token del prompt del usuario no es un token cualquiera — es el token que precede al primer token de la respuesta del asistente. Cuando el modelo predice lo que viene después de ese último token del usuario, está aprendiendo el inicio de la respuesta del asistente. Si el masking está desplazado en un token — si pones la etiqueta `-100` un token de más o de menos — el modelo puede aprender a iniciar respuestas de forma incorrecta. Los síntomas son sutiles: el modelo empieza sus respuestas de manera extraña, usa el token incorrecto de apertura, o no respeta el formato del rol de asistente.

La precisión aquí no es opcional. El DataCollator (el componente que prepara los batches de datos) debe construir el tensor de etiquetas colocando `-100` en exactamente los tokens correctos — sistema y usuario — y dejando los tokens del asistente con sus valores reales para que contribuyan a la pérdida.

### Gradiente spikes en la frontera

Hay otro efecto en esa frontera usuario-asistente que merece atención. Cuando el modelo pasa de procesar tokens ignorados (-100) a procesar tokens activos (los del asistente), se produce un salto abrupto en la señal de entrenamiento. Los primeros tokens de la respuesta del asistente — el token de apertura del rol, el primer token del contenido — de repente pasan de "no contribuir nada" a "contribuir completamente al gradiente". Este salto puede provocar picos de gradiente (gradient spikes) que desestabilizan el entrenamiento temprano.

La solución estándar es doble: usar un learning rate (tasa de aprendizaje — cuánto cambian los pesos en cada actualización) conservador, y aplicar un warm-up schedule (programación de calentamiento). El warm-up comienza con un learning rate muy bajo — quizás 10x menor que el objetivo — y lo sube gradualmente durante los primeros cientos de pasos. Esto le da al modelo tiempo para "sentir" la frontera antes de que los gradientes alcancen su magnitud completa. Sin warm-up, es común ver que las primeras iteraciones producen pérdidas erráticamente altas seguidas de colapsos parciales del modelo.

### Batching agrupado vs. packing puro

En CPT, el packing constante es la norma. Rellenas cada contexto hasta el límite y maximizas el throughput. En SFT, ya vimos que el packing puro es peligroso por la contaminación cruzada. Pero tampoco puedes simplemente no empaquetar nada — eso desperdiciaría la GPU con padding (tokens de relleno) en conversaciones cortas.

La solución intermedia es el grouped batching (batching agrupado): agrupar secuencias de longitud similar en el mismo batch. Si tienes conversaciones de 200, 210, 195 y 205 tokens, meterlas en el mismo batch significa que el padding que necesitas para igualar longitudes es mínimo. Sacrificas un poco de throughput versus packing puro, pero mantienes la integridad estructural — y evitas el caos de mezclar conversaciones. Con Flash Attention 2 y `cu_seqlens`, puedes ir un paso más lejos y eliminar el padding completamente incluso dentro del batch.

---

## SFT Agéntico: cuando el modelo aprende a usar herramientas

Hasta aquí, el SFT enseña al modelo a hablar. El SFT agéntico enseña al modelo a actuar.

En un sistema agéntico, el modelo no solo genera texto — genera acciones. Reconoce cuándo una tarea requiere información externa que no tiene, produce una llamada a una herramienta en el formato correcto, espera el resultado, y construye su respuesta final con esa información. Un asistente que puede consultar una API de precios en tiempo real, buscar en una base de datos vectorial, o ejecutar código en un sandbox es un sistema agéntico.

Para que el modelo aprenda este comportamiento, los datos de entrenamiento deben reflejarlo. Un dataset de SFT agéntico sigue un loop estructurado:

**Thought → Action → Action Input → Observation → Final Response**

Primero, el modelo razona sobre qué necesita hacer (Thought). Luego genera la llamada a la herramienta en formato JSON estricto (Action + Action Input). En ese punto, la generación se pausa — el sistema ejecuta realmente la herramienta — y el resultado se introduce en el contexto como Observation. Solo entonces el modelo produce la respuesta final.

> **Descripción visual:** Diagrama de secuencia con tres participantes dispuestos horizontalmente: "Usuario" (izquierda, bloque azul claro), "Modelo" (centro, bloque verde), "Herramienta" (derecha, bloque amarillo). Las líneas de vida descienden verticalmente. Una flecha sólida va de Usuario a Modelo ("Pregunta / tarea"), seguida de una nota flotante sobre Modelo ("Thought — razona qué necesita"). Luego una flecha sólida de Modelo a Herramienta ("Tool call JSON") y una flecha discontinua de vuelta ("Observation"). Una segunda nota sobre Modelo ("Procesa resultado") y finalmente una flecha discontinua de Modelo a Usuario ("Respuesta final"). Las notas son rectángulos amarillo pálido. Estilo técnico, fondo blanco, tipografía monospace para los labels de acciones.

Un ejemplo concreto con llamada a API de precios de acciones:

```
<think>
El usuario pregunta por el precio actual de Apple. 
No tengo esa información en mis parámetros (mi conocimiento tiene fecha de corte).
Necesito llamar a la herramienta get_stock_price.
</think>

<|tool_call|> get_stock_price {"ticker": "AAPL"} <|eot_id|>

[Sistema ejecuta la herramienta y obtiene: {"price": 189.43, "currency": "USD"}]

<|tool_response|> {"price": 189.43, "currency": "USD"} <|eot_id|>

El precio actual de Apple (AAPL) es de $189.43 USD.
```

El aspecto crítico aquí es la disciplina de formato. Si el JSON tiene un corchete mal cerrado, la herramienta falla. Si el nombre del campo es `ticker_symbol` en lugar de `ticker`, la API devuelve un error. El agente se rompe. Por eso en el SFT agéntico, la sintaxis tiene el mismo peso que el contenido. El modelo debe internalizar el schema exacto — no aproximarlo.

Esto explica por qué los datasets de SFT agéntico suelen ser sintéticos y extremadamente limpios. No puedes permitirte variabilidad en el formato de las llamadas a herramientas. Cada ejemplo debe mostrar exactamente el patrón correcto, sin excepciones. Un solo ejemplo malformado puede enseñar al modelo que "más o menos" está bien — y en producción, "más o menos" rompe el pipeline.

El cambio conceptual que produce el SFT agéntico es profundo. El modelo deja de ser un sistema cerrado de conocimiento y pasa a ser un componente de un sistema mayor. Sabe que no sabe el precio actual de las acciones de Apple — y sabe exactamente qué secuencia de tokens producir para solicitarlo. Eso convierte al LLM de una biblioteca estática en un agente que opera dentro de un entorno.

---

## Evaluación: más allá de la curva de pérdida

En el CPT, evaluar es sencillo. Observas la pérdida de validación — si baja, el modelo mejora. Si la pérdida de entrenamiento cae pero la de validación sube, estás sobreajustando. El mecanismo es directo.

En el SFT, una pérdida muy baja puede ser una señal de alarma. No una buena señal — una mala. ¿Por qué? Porque pérdida muy baja puede significar que el modelo memorizó el wording exacto del dataset. En lugar de aprender cómo responder bien, aprendió a reproducir las respuestas de entrenamiento. El resultado es lo que se llama coloquialmente un "loro pulido" (polished parrot): el modelo recita con fluidez pero no ha internalizado la intención detrás de las respuestas.

Puedes detectar esto comparando pérdida de entrenamiento versus validación. Pero la pérdida de validación tampoco te dice si el modelo realmente se comporta bien — solo si reproduce tokens similares a los de validación. Por eso el sector ha migrado a evaluaciones más ricas.

### LLM-as-a-judge

El paradigma actual es usar un modelo más potente como evaluador. Le das al juez el prompt original y la respuesta generada por el modelo evaluado, y le pides que puntúe en dimensiones como claridad, utilidad, seguimiento de instrucciones, y corrección factual. El juez puede ser GPT-4, Claude, o cualquier modelo suficientemente capaz. Esta aproximación captura calidad semántica que la pérdida no puede ver.

### Benchmarks de seguimiento de instrucciones

IFEval (Instruction Following Evaluation) es un benchmark que impone restricciones verificables mecánicamente: "responde en menos de 100 palabras", "no uses la letra Q", "tu respuesta debe contener exactamente 3 párrafos". Estas restricciones son binarias — se cumplen o no — y testean si el SFT reestructuró las prioridades del modelo. Un modelo que pasa IFEval no solo parece que sigue instrucciones; las sigue cuando se mide objetivamente.

### Benchmarks agénticos

Para sistemas agénticos, la barra es más alta. Un modelo puede generar respuestas hermosas a preguntas individuales pero fallar miserablemente en tareas de múltiples pasos. Los benchmarks más relevantes aquí son:

- **GAIA** (General AI Assistants): Tareas cotidianas de asistente que requieren uso de herramientas y multimodalidad. Por ejemplo: "Encuentra el precio de un vuelo Madrid-Tokyo para el próximo viernes con menos de una escala". Requiere navegación real, no solo conocimiento estático.

- **SWE-bench**: El modelo actúa como ingeniero de software. Recibe un issue real de GitHub y debe producir un parche de código que pase los tests unitarios del repositorio. Mide capacidad de razonar sobre código en contexto real.

- **WebShop / Mind2Web**: El modelo navega por interfaces web para completar tareas. "Compra este producto específico por menos de $50 en Amazon". Requiere coordinar navegación, lectura de UI, y toma de decisiones.

La métrica común en todos estos benchmarks no es la pérdida — es la tasa de éxito en la tarea. O lo hace o no lo hace.

---

## El lab: dos modelos, una verdad

Con la teoría clara, el lab de esta lección la vuelve tangible con un experimento concreto y ejecutable. La configuración es deliberadamente simple para que la comparación sea limpia: mismo modelo base (Qwen3-0.6B), mismo dataset, misma cantidad de pasos de entrenamiento (200 steps), mismo hardware (A10G GPU). La única variable es la columna de datos usada.

### El modelo y la elección del full fine-tuning

Qwen3-0.6B es un modelo de 600 millones de parámetros — pequeño para los estándares actuales, pero suficientemente capaz para demostrar el efecto del SFT sobre el razonamiento. La elección de full fine-tuning (actualización de todos los parámetros del modelo) es intencionalmente pedagógica: en producción, [[03-lora-adaptacion-de-bajo-rango|LoRA]] o QLoRA (que veremos en capítulos posteriores) son la norma porque son más eficientes. Pero el full fine-tuning expone el mecanismo sin capas de abstracción adicionales.

### El dataset sintético

El dataset proviene de un proceso de [[01-fundamentos-transformers-y-pretraining|destilación]] sobre YouTube Commons — 20.000 transcripciones de vídeos de YouTube, procesadas con NVIDIA Nemotron-3-Nano-30B-A3B como teacher model. El resultado tiene dos columnas:

- `messages_no_thinking`: pares pregunta-respuesta donde la respuesta es directa, sin razonamiento explícito.
- `messages_thinking`: los mismos pares, pero la respuesta incluye una traza `<think>...</think>` generada por el teacher model antes de la respuesta final.

El teacher model no solo generó respuestas — generó razonamientos. Esas trazas son el "cómo llegué aquí" que el modelo pequeño aprenderá a imitar.

### Los comandos de entrenamiento

El entrenamiento corre via Hugging Face Jobs, que aprovisiona hardware automáticamente. El script principal recibe como argumentos qué columna usar y cuántos pasos entrenar:

```bash
# Modelo sin razonamiento
hf jobs uv run --flavor a10g-small \
  -e COMET_PROJECT_NAME="finetuning-sessions-full-finetuning-no-thinking" \
  -s COMET_API_KEY="YOUR_COMET_API_KEY" \
  -s HF_TOKEN="YOUR_HF_TOKEN" \
  --timeout 3h main.py -- \
  --hub_model_id Qwen3-0.6B-Full-Finetuning-No-Thinking \
  --dataset_column messages_no_thinking \
  --max_steps 200

# Modelo con razonamiento
hf jobs uv run --flavor a10g-small \
  -e COMET_PROJECT_NAME="finetuning-sessions-full-finetuning-thinking" \
  -s COMET_API_KEY="YOUR_COMET_API_KEY" \
  -s HF_TOKEN="YOUR_HF_TOKEN" \
  --timeout 3h main.py -- \
  --hub_model_id Qwen3-0.6B-Full-Finetuning-Thinking \
  --dataset_column messages_thinking \
  --max_steps 200
```

El flag `--flavor a10g-small` especifica el tipo de GPU. El `--timeout 3h` es un límite de seguridad para no acumular costes si algo va mal. Los secrets (`-s`) se pasan de forma segura sin exponerlos en el comando.

### Qué vigilar en las curvas de entrenamiento

Las curvas de pérdida en Comet ML permiten diagnosticar el entrenamiento en tiempo real. Con 200 steps, el comportamiento esperado es:

- **Primeros 20-40 steps**: La pérdida cae rápidamente. El modelo está aprendiendo el formato básico — los chat tokens, el ritmo de turnos, cómo iniciar una respuesta. Este es el aprendizaje más rápido porque parte de cero en cuanto a estructura conversacional.

- **Steps 40-150**: La caída se ralentiza. El modelo está refinando contenido — no solo "responde de esta manera" sino "responde esta cosa concreta". La curva puede tener pequeñas oscilaciones; es normal.

- **Steps finales**: Si la pérdida de entrenamiento sigue cayendo pero la de validación se estanca o sube ligeramente, estás al borde del overfitting. Con 200 steps en un dataset de esta escala, generalmente no llegas al overfitting — pero es la señal a vigilar.

El modelo de razonamiento (`messages_thinking`) típicamente tiene una pérdida inicial más alta y una caída más lenta. Esto es esperable: las trazas `<think>` añaden tokens que el modelo debe aprender a generar, incrementando la complejidad del objetivo. Si las curvas de ambos modelos son idénticas, algo está mal — probablemente el masking no está diferenciando correctamente entre las dos columnas.

### Interpretando los resultados

Al desplegar ambos modelos y compararlos con la misma pregunta, las diferencias son inmediatas. El modelo sin razonamiento responde con confianza y brevedad. El modelo con razonamiento hace visible su proceso: puedes seguir cómo llega a la respuesta, identificar si el razonamiento es correcto o contiene un error, y entender por qué produce lo que produce.

Este segundo modelo no es más "inteligente" en el sentido de tener más conocimiento — ambos parten del mismo Qwen3-0.6B. Lo que tiene es un patrón de comportamiento diferente: ha internalizado que ante un problema, la acción correcta es trabajarlo antes de responderlo. Y eso, cuando el RL venga a refinar el comportamiento en capítulos posteriores, le dará una base sobre la cual optimizar.

---

## Guía de métricas para SFT

Al entrenar, estas son las señales que debes vigilar y sus umbrales orientativos:

| Métrica | Valor de referencia | Señal de alarma |
|---|---|---|
| Training loss (primeros steps) | Caída rápida en los primeros 50 steps | No cae o sube — problema de masking o learning rate |
| Training loss final | Debe estabilizarse, no llegar a 0 | Llega a 0 — overfitting severo |
| Val loss vs. train loss | Val loss ligeramente superior | Val loss sube mientras train cae — overfitting |
| Gradient norm | Estable, bajo (~1-3) | Picos de 10x o más — sube el gradient clipping |
| Learning rate (con warm-up) | Empieza en 10-20% del LR objetivo, sube gradualmente | LR completo desde el step 0 puede causar spikes iniciales |
| Eval qualitativa (LLM-as-judge) | Mejora vs. modelo base | Respuestas formulaicas o memorizadas |

Una nota sobre el gradient clipping (recorte de gradiente): es un mecanismo de seguridad que limita la magnitud máxima del gradiente. Si el gradiente es demasiado grande — por un batch atípico o por la frontera usuario-asistente que discutimos — puede modificar los pesos de forma tan drástica que destruya lo aprendido anteriormente. El gradient clipping recorta el vector de gradiente para que su norma no supere un umbral (típicamente 1.0). Si pones el umbral demasiado bajo (0.1), el aprendizaje se vuelve innecesariamente lento. Si lo pones muy alto (10.0), vuelves al problema original.

---

## El panorama completo

El SFT es el puente entre el conocimiento en bruto del preentrenamiento y el comportamiento útil que esperamos de un asistente. No añade hechos al modelo — reorienta cómo usa los que tiene. Le enseña el protocolo de la conversación, la estructura de la respuesta, el ritmo de los turnos, la disciplina del formato.

Pero hay algo más profundo. El SFT es también el punto donde el diseñador de datos tiene más control sobre la personalidad del sistema. La forma en que se estructura un dataset de SFT determina si el modelo es directo o elaborado, conciso o detallado, cauteloso o asertivo. Esas no son propiedades del algoritmo — son propiedades de los datos de entrenamiento. Y eso pone una responsabilidad inusual en manos del ingeniero que diseña el dataset.

En el próximo capítulo, veremos cómo LoRA y QLoRA permiten hacer este mismo proceso de forma dramáticamente más eficiente, sin necesidad de actualizar todos los parámetros del modelo. Eso abre el SFT a hardware convencional — y cambia completamente la economía del fine-tuning.

---

## Tags

#técnica/sft #concepto/chat-template #concepto/loss-masking #técnica/chain-of-thought #técnica/continued-pretraining #concepto/transformer #técnica/reinforcement-learning #nivel/introducción #tipo/lección #estado/completo



---
capitulo: "03"
titulo: "LoRA: Adaptación de Bajo Rango desde los Fundamentos"
aliases:
  - "Capítulo 3"
  - "Cap 3"
  - "LoRA"
  - "Low-Rank Adaptation"
tema: "técnica-peft"
subtemas: [lora, intrinsic-rank-hypothesis, olvido-catastrofico]
dificultad: "intermedio"
tipo: "lección"
estado: "completo"
conceptos_centrales:
  - lora
  - low-rank-adaptation
  - intrinsic-rank-hypothesis
  - factorización-de-bajo-rango
  - olvido-catastrófico
prerequisitos:
  - "[[01-fundamentos-transformers-y-pretraining]]"
relacionados:
  - "[[04-qlora-cuantizacion-4bit]]"
  - "[[07-finetuning-multimodal-vision-tts]]"
tags:
  - técnica/lora
  - técnica/low-rank-adaptation
  - concepto/intrinsic-rank-hypothesis
  - concepto/factorización-de-bajo-rango
  - concepto/olvido-catastrófico
  - modelo/transformer
  - técnica/qlora
  - nivel/intermedio
  - tipo/lección
  - estado/completo
---

# Capítulo 3 — LoRA: Adaptación de Bajo Rango desde los Fundamentos

> Basado en "Understanding LoRA from First Principles" y "Engineering LoRA for Real-World Finetuning" — The Neural Maze, Finetuning Sessions, Lecciones 3 y Lab 3.

Imagina que has pasado cinco años aprendiendo a hablar mandarín. Tu cerebro ha reorganizado millones de conexiones neuronales para almacenar vocabulario, gramática, entonación. Ahora alguien te pide que también aprendas a cocinar pasta italiana. ¿Tienes que olvidar el mandarín para aprender a cocinar? Por supuesto que no. Las nuevas habilidades se construyen sobre una base existente, activando circuitos específicos sin borrar los que ya funcionaban.

El fine-tuning completo de un LLM, en cambio, funciona más como si le pidieras a ese cerebro que rehaga todas sus conexiones desde cero cada vez que aprende algo nuevo. Es radical, potente, y devastadoramente caro. LoRA (Low-Rank Adaptation — adaptación de bajo rango) propone algo mucho más inteligente: aprender solo las correcciones necesarias, en el subespacio mínimo donde esas correcciones viven, sin tocar lo que ya funciona.

Este capítulo te lleva desde la intuición geométrica hasta la implementación práctica, pasando por la aritmética exacta de por qué LoRA hace posible el fine-tuning en hardware de consumo.

---

## Los pesos de un modelo son una biblioteca, no un archivo ejecutable

Antes de entender qué hace LoRA, necesitamos entender qué son esos pesos que pretende modificar.

Un modelo de lenguaje como Llama-3 o Mistral-7B es, en esencia, una composición de matrices de pesos. Cada capa del [[01-fundamentos-transformers-y-pretraining|Transformer]] contiene varias de estas matrices, y cada una transforma un vector de entrada en otro vector de salida. Esta transformación lineal es el vocabulario básico de todo lo que el modelo sabe hacer.

Para hacerlo concreto: en un modelo con dimensión de embedding $d = 4096$, la matriz de proyección de queries en la capa de atención tiene forma $\mathbb{R}^{4096 \times 4096}$, es decir, algo así como 16.7 millones de parámetros en una sola matriz. Un modelo de 7B parámetros tiene docenas de estas capas, cada una con múltiples proyecciones. Los números se acumulan rápido.

Cuando hacemos fine-tuning completo (Full Fine-Tuning o FFT, que exploramos en el capítulo anterior), dejamos que todos esos parámetros sean modificables. Si la matriz original es $W$, el proceso de entrenamiento aprende una corrección $\Delta W$, y la nueva matriz queda como:

$$W' = W + \Delta W$$

La actualización $\Delta W$ tiene exactamente las mismas dimensiones que $W$: 4096 × 4096 en nuestro ejemplo. Hay que calcular gradientes para cada uno de esos 16.7 millones de valores, almacenar los estados del optimizador (momento y varianza en Adam, que explicaremos más adelante), y hacer todo esto en paralelo para decenas de capas. El resultado es una factura de memoria que rompe cualquier tarjeta gráfica de consumo.

Pero hay algo más profundo que la memoria. El fine-tuning completo tiene otro problema estructural: le da al modelo demasiada libertad.

---

## El problema de la libertad irrestricta: olvido catastrófico

El olvido catastrófico (catastrophic forgetting) es uno de los fenómenos más estudiados y más frustrantes del aprendizaje automático. Ocurre cuando un modelo aprende una tarea nueva y, al hacerlo, sobreescribe los pesos que le permitían hacer bien la tarea anterior.

Piensa en esto con el ejemplo del mandarín: si el proceso de "aprender a cocinar pasta" consistiera en reescribir todas las conexiones neuronales del cerebro con igual probabilidad, habría un riesgo real de borrar la gramática del mandarín. En el fine-tuning completo, el gradiente no distingue entre "información útil para la nueva tarea" y "conocimiento general que llevé años adquiriendo". Simplemente minimiza la pérdida sobre el nuevo dataset, y si eso requiere mover pesos que codifican razonamiento general, los mueve.

En la práctica, esto se manifiesta de formas concretas: un modelo entrenado en diagnóstico médico puede empeorar en benchmarks de razonamiento lógico general. Uno ajustado para código puede volverse más rígido en lenguaje natural. El problema no es que los nuevos datos sean malos — es que la optimización no tiene ninguna restricción que preserve el conocimiento previo.

El fine-tuning completo tiene, en términos técnicos, demasiados grados de libertad. Puede mover cualquier parámetro en cualquier dirección. Sin una restricción estructural, nada garantiza que el update $\Delta W$ sea "quirúrgico" en lugar de "invasivo".

LoRA introduce exactamente esa restricción estructural.

---

## La hipótesis del rango intrínseco

En 2021, investigadores de Microsoft publicaron el paper "LoRA: Low-Rank Adaptation of Large Language Models" con una apuesta teórica audaz: la corrección $\Delta W$ que necesitas para adaptar un modelo a una nueva tarea no requiere ser una matriz densa y de rango completo. Puede vivir en un subespacio de mucho menor dimensión.

Esta idea se llama la **hipótesis del rango intrínseco** (intrinsic rank hypothesis). Antes de explicarla formalmente, necesitamos clarificar qué significa "rango" en este contexto.

El **rango de una matriz** es el número de filas (o columnas) linealmente independientes que contiene. Dicho de otra forma, es la dimensión del espacio que esa matriz puede "alcanzar". Una matriz de rango completo de tamaño $4096 \times 4096$ puede apuntar en 4096 direcciones independientes en el espacio de representaciones. Una matriz de rango 8 solo puede apuntar en 8 direcciones.

¿Por qué importa esto? Porque la hipótesis de LoRA dice que la adaptación relevante — el cambio que necesita el modelo para comportarse bien en la nueva tarea — solo ocurre en unas pocas de esas 4096 direcciones. El resto del espacio es "ruido" para la nueva tarea. No necesitas actualizar todas las dimensiones; solo necesitas actualizar el subespacio correcto.

La analogía más útil viene del mundo del audio. Cuando grabas una sala de conciertos, el micrófono capta miles de frecuencias. Pero si quieres aislar el violín, no necesitas remezclar todas las frecuencias — solo las que corresponden al registro del violín. El resto puedes dejarlo tal cual. LoRA hace lo mismo con el espacio de pesos: identifica el "registro" relevante para la nueva tarea y solo toca ahí.

---

## La matemática de LoRA: descomposición de bajo rango

Para implementar la hipótesis del rango intrínseco, LoRA recurre a un truco de álgebra lineal: la descomposición de bajo rango de una matriz.

En lugar de aprender directamente $\Delta W \in \mathbb{R}^{d \times d}$ (una matriz de rango potencialmente completo), LoRA la factoriza en el producto de dos matrices más pequeñas:

$$\Delta W = B \cdot A$$

donde:
- $A \in \mathbb{R}^{r \times d}$ es la **matriz de proyección descendente** (down-projection), que comprime el input a una representación de dimensión $r$.
- $B \in \mathbb{R}^{d \times r}$ es la **matriz de proyección ascendente** (up-projection), que expande esa representación comprimida de vuelta a la dimensión original.
- $r$ es el **rango** del adaptador (rank), y es el hiperparámetro central de LoRA. Típicamente $r \ll d$.

El número de parámetros entrenables pasa de $d \times d$ a $r \times d + d \times r = 2rd$. Para $d = 4096$ y $r = 8$, eso es $2 \times 8 \times 4096 = 65{,}536$ parámetros en lugar de $16{,}777{,}216$. Una reducción de 256 veces.

Con esta factorización, la salida de una capa modificada con LoRA queda como:

$$h = W x + \frac{\alpha}{r} B A x$$

donde $x$ es el vector de entrada, $W$ son los pesos congelados del modelo preentrenado, $BA$ es la corrección de bajo rango, y $\frac{\alpha}{r}$ es un factor de escala que controla la fuerza de la actualización. Explicaremos cada uno de estos componentes en detalle.

La clave estructural es que $W$ está **congelado** — sus pesos no cambian durante el fine-tuning. Solo $A$ y $B$ son entrenables. El [[01-fundamentos-transformers-y-pretraining|modelo base]] se preserva intacto, y la adaptación vive enteramente en las matrices pequeñas.

Esta separación no es solo un truco de eficiencia: es una decisión arquitectónica que tiene consecuencias profundas para la estabilidad del entrenamiento y para la prevención del olvido catastrófico. Al no tocar $W$, garantizas que el conocimiento preentrenado no puede ser sobrescrito accidentalmente.

> **Descripción visual:** Diagrama horizontal con tres subgrafos. A la izquierda, un bloque rojo etiquetado "ΔW / d × k parámetros" representa el enfoque de full finetuning. En el centro, dos bloques verdes ("Matriz B / d × r" y "Matriz A / r × k") convergen con flechas hacia un tercer bloque verde "ΔW = B · A / rango r", mostrando la factorización LoRA. Una flecha punteada de izquierda a centro indica la sustitución. Abajo, un subgrafo naranja compara los recuentos numéricos de parámetros de ambos enfoques. Fondo blanco, tipografía sans-serif, estilo limpio.

---

## La conexión con los autoencoders: por qué funciona la compresión

Para entender por qué una representación de rango bajo puede capturar actualizaciones útiles, es valioso hacer una parada en un concepto de arquitecturas más antiguas: el autoencoder.

Un autoencoder es una red neuronal que aprende a comprimir datos y luego reconstruirlos. Consiste en dos partes: un **encoder** que toma una entrada de alta dimensión (por ejemplo, una imagen de 784 píxeles) y la comprime a un espacio latente de baja dimensión (por ejemplo, 32 dimensiones), y un **decoder** que reconstruye la imagen original desde esa representación comprimida.

El hecho de que los autoencoders funcionen tan bien para compresión nos dice algo fundamental sobre la estructura de los datos del mundo real: los datos de alta dimensión tienden a vivir en subespacios de mucho menor dimensión. Una imagen de 784 píxeles puede ser representada con 32 números porque los píxeles no son independientes — están correlacionados por la estructura visual del mundo.

LoRA hace exactamente la misma apuesta, pero sobre el espacio de actualizaciones de pesos en lugar del espacio de datos. La hipótesis es que el cambio útil $\Delta W$ para adaptar un modelo a una nueva tarea no es un punto arbitrario en el espacio de matrices de $d \times d$ dimensiones — vive en un subespacio de mucho menor dimensión.

La intuición es: un LLM preentrenado ya aprendió representaciones ricas y generales del lenguaje. Adaptar ese modelo a, digamos, responder preguntas médicas, no requiere reordenar todo ese conocimiento. Requiere ajustar unas pocas "direcciones" en el espacio de representaciones — los conceptos y patrones específicos del dominio médico. Esas pocas direcciones son el subespacio de rango bajo que LoRA captura con sus matrices $A$ y $B$.

> **Descripción visual:** Diagrama horizontal con dos subgrafos en paralelo y un nodo central de conclusión. El subgrafo superior (bloques azules en cadena horizontal) muestra el pipeline del autoencoder: "Imagen 784px" → "Encoder" → "Espacio latente 32 dim" → "Decoder" → "Imagen reconstruida". El subgrafo inferior (bloques verdes) muestra la analogía LoRA: "Pesos W" → "ΔW densa" con flecha punteada hacia "B·A / bajo rango r". Una flecha punteada diagonal conecta el espacio latente con el bloque B·A indicando la inspiración. Ambos convergen en un bloque naranja central "La estructura útil vive en baja dimensionalidad". Fondo blanco, tipografía sans-serif.

---

## Inicialización: el truco que hace posible el arranque

La inicialización de las matrices de LoRA no es trivial. Si inicializaras $A$ y $B$ de forma aleatoria, el producto $BA$ empezaría con un valor no nulo, y la primera pasada del modelo sería diferente del modelo preentrenado. Eso significaría empezar el fine-tuning desde un punto de partida perturbado, lo que desestabiliza el entrenamiento.

LoRA resuelve esto con una decisión elegante: **$B$ se inicializa a cero**. Esto garantiza que en el momento $t=0$ de entrenamiento:

$$\Delta W = B \cdot A = 0 \cdot A = 0$$

Y por tanto la salida de la capa es idéntica a la del modelo preentrenado:

$$h = W x + \frac{\alpha}{r} \cdot 0 \cdot x = W x$$

El modelo arranca exactamente donde dejó el preentrenamiento. No hay perturbación inicial. A medida que el entrenamiento progresa, $B$ aprende valores no nulos y la corrección $\Delta W$ crece gradualmente desde cero hasta el nivel que la nueva tarea requiere.

¿Y $A$? Se inicializa con valores aleatorios de una distribución gaussiana. La razón: si ambas matrices se inicializaran a cero, los gradientes sobre $A$ serían también cero (ya que el gradiente de $B$ sobre $A$ es cero cuando $B = 0$), y $A$ nunca aprendería nada. Inicializar $A$ con ruido gaussiano garantiza que, cuando $B$ empiece a alejarse de cero, tenga direcciones en el espacio de input desde las cuales construir la corrección.

---

## El factor de escala $\alpha$: desacoplando capacidad de magnitud

La fórmula completa del update de LoRA es:

$$h = W x + \frac{\alpha}{r} B A x$$

El parámetro $\alpha$ (alpha) es un escalar que controla la magnitud de la corrección relativa al modelo base. Para entender por qué es necesario, considera lo que pasa sin él.

Sin $\alpha$, si decides aumentar el rango de 8 a 32 para capturar actualizaciones más complejas, el producto $BA$ cambia de escala automáticamente porque ahora tienes más dimensiones contribuyendo. Tendrías que reajustar el learning rate para compensar. Eso es inconveniente: cada vez que cambias el rango, debes volver a tunear la tasa de aprendizaje.

El factor $\alpha/r$ desacopla la selección de rango de la magnitud del update. Cuando $\alpha = r$, el escalar $\alpha/r = 1$ y el update no está amplificado ni reducido. Cuando $\alpha = 2r$, el escalar es 2 y el update tiene el doble de impacto sobre el modelo base. Esto permite cambiar $r$ sin que la magnitud efectiva del update cambie, siempre que mantengas la relación $\alpha/r$ constante.

Un ejemplo numérico: supón que tienes $r = 8$ y $\alpha = 16$ (la heurística $\alpha = 2r$). Entonces $\alpha/r = 2$. Si subes a $r = 32$ y mantienes $\alpha = 64$ (siempre $\alpha = 2r$), el escalar sigue siendo 2. El update tiene la misma magnitud pero más capacidad expresiva. Si en cambio mantuvieras $\alpha = 16$ con $r = 32$, el escalar sería $16/32 = 0.5$ — el update tendría la mitad de impacto, y el modelo aprendería más despacio de lo que podría.

La regla práctica: **mantén $\alpha = r$ o $\alpha = 2r$**. La primera es la elección más conservadora y estable; la segunda es más agresiva y funciona bien cuando el dominio objetivo está lejos del preentrenamiento.

> **Descripción visual:** Diagrama de flujo vertical con ocho nodos. En la cima, un bloque azul "Inicio del entrenamiento" se ramifica hacia dos bloques púrpura en paralelo: "B inicializado a cero" y "A inicialización gaussiana". Ambos convergen en un bloque naranja "ΔW = B·A = 0 / Modelo se comporta como el base". La cadena continúa verticalmente con bloques naranjas para "Entrenamiento progresivo", "Factor de escalado α/r" y "ΔW efectivo = (α/r)·B·A", terminando en un bloque verde "Adaptación gradual sin perturbaciones bruscas". Flechas rectas descendentes. Fondo blanco, estilo limpio.

---

## La aritmética de la memoria: por qué LoRA cambia las reglas del juego

Con la teoría clara, hagamos la aritmética exacta del ahorro de memoria. Los números son el argumento más convincente a favor de LoRA, y vale la pena tenerlos internalizados.

En el fine-tuning completo con el optimizador Adam — el estándar de facto para LLMs — necesitas almacenar en GPU las siguientes cantidades por cada parámetro del modelo:

- **El peso en sí**: si trabajas en precisión simple FP32 (4 bytes por parámetro), o en BF16 (2 bytes) para ahorrar memoria.
- **El gradiente**: del mismo tamaño que el peso. En FP32: 4 bytes.
- **El momento de primer orden de Adam** ($m_t$, la media móvil de los gradientes): 4 bytes por parámetro en FP32.
- **El momento de segundo orden de Adam** ($v_t$, la media móvil de los gradientes cuadrados): 4 bytes por parámetro en FP32.

En total, en FP32 completo: $4 + 4 + 4 + 4 = 16$ bytes por parámetro. Para un modelo de 7 mil millones de parámetros:

$$16 \text{ bytes} \times 7 \times 10^9 = 112 \text{ GB}$$

Ciento doce gigabytes solo para los estados del modelo. Una NVIDIA A100 tiene 80 GB. Una RTX 4090 tiene 24 GB. El fine-tuning completo de un 7B queda completamente fuera del alcance del hardware de consumo.

LoRA rompe esta barrera con un cambio estructural: **congela el 99% de los parámetros**. Los pesos congelados no necesitan gradientes ni estados del optimizador. Solo ocupan memoria estática. Esto cambia la contabilidad dramáticamente:

**Base model (congelado) en BF16:** $2 \text{ bytes} \times 7 \times 10^9 = 14$ GB. Esto no cambia durante el entrenamiento.

**Adaptadores LoRA (entrenables) en FP32:** Si LoRA añade aproximadamente el 1% de parámetros extra — unos 70 millones de parámetros — y los entrena con Adam en FP32: $16 \text{ bytes} \times 70 \times 10^6 = 1.12$ GB.

**Total de estados del modelo: $14 + 1.12 \approx 15.12$ GB.**

Pasamos de 112 GB a 15 GB — una reducción de 7x. De requerir un clúster de datacenter a caber en una sola A100 con margen para el batch de activaciones.

Y esto solo cuenta los estados del modelo. En la práctica, también hay memoria para activaciones (que escala con la longitud de secuencia y el tamaño del batch), pero los estados del modelo son el cuello de botella dominante para modelos grandes, y LoRA lo elimina casi completamente.

> **Descripción visual:** Diagrama de flujo vertical encuadrado en un subgrafo "VRAM necesario". En la parte superior, dos bloques azules ("Pesos originales W" y "Corrección ΔW") apuntan hacia abajo a un bloque verde "Pesos actualizados W'=W+ΔW". Un bloque naranja "Gradientes de ΔW" apunta hacia el bloque ΔW. Dos bloques rojos ("Momento 1 de Adam / media" y "Momento 2 de Adam / varianza") también apuntan hacia ΔW. El subgrafo con borde gris encierra todos los bloques excepto el resultado final, enfatizando lo que debe residir en VRAM. Fondo blanco, flechas rectas, tipografía sans-serif.

---

## Los hiperparámetros de LoRA: el espacio de decisiones

LoRA reduce drásticamente el número de parámetros entrenables, pero no elimina la necesidad de tomar decisiones. De hecho, porque el espacio de adaptación es ahora más pequeño y más deliberado, las decisiones de configuración son más críticas. Una mala elección de rango o de alpha puede hacer que el adaptador no aprenda nada útil — o que aprenda pero destruya el comportamiento base.

Estos son los parámetros que debes entender y tunear:

### Rango ($r$)

El rango es el parámetro más importante de LoRA. Define cuántas dimensiones independientes tiene el adaptador para expresar la corrección. Piénsalo como el "vocabulario" del adaptador: un rango mayor le da más palabras para describir el cambio necesario, pero también le da más oportunidades de memorizar ruido o de sobreajustarse.

Con $r = 4$: el adaptador puede moverse en 4 direcciones ortogonales del espacio de pesos. Útil para tareas de adaptación de estilo o seguimiento de instrucciones simples donde el dominio objetivo no es radicalmente distinto del preentrenamiento.

Con $r = 8$ o $r = 16$: el rango recomendado por defecto para la mayoría de tareas de instruction-tuning. Suficiente capacidad para capturar patrones de dominio moderadamente especializados sin riesgo alto de sobreajuste.

Con $r = 64$, $r = 128$, o $r = 256$: necesario para dominios técnicos complejos como código, matemáticas, o razonamiento formal, donde la distancia entre el preentrenamiento y la tarea objetivo es grande. A estos rangos, LoRA empieza a aproximarse al poder expresivo del fine-tuning completo, pero el costo de memoria y el riesgo de sobreajuste también crecen.

Un error común: subir el rango sin datos suficientes. Si tienes 1.000 ejemplos de entrenamiento y usas $r = 128$, el adaptador tiene tanta capacidad que simplemente memoriza el dataset de entrenamiento sin generalizar. Como regla empírica, más datos permiten rangos más altos.

### Alpha ($\alpha$) y la regla de proporcionalidad

Como explicamos antes, $\alpha$ controla la magnitud del update relativa al modelo base a través del factor de escala $\alpha/r$.

La práctica recomendada, respaldada por múltiples estudios empíricos, es mantener $\alpha = r$ o $\alpha = 2r$. Así el factor de escala es siempre 1 o 2, independientemente del rango que elijas.

Hay una advertencia crítica que el artículo fuente menciona y merece subrayarse: si mantienes $\alpha$ fijo mientras aumentas $r$, el factor $\alpha/r$ decrece. Por ejemplo, con $\alpha = 8$ fijo y $r = 64$, el factor es $8/64 = 0.125$ — el update tiene apenas el 12.5% de la magnitud que tendrías con $\alpha = r$. El modelo subutiliza la capacidad del adaptador y puede sufrir olvido catastrófico porque las correcciones son demasiado débiles para superar el "ruido" del preentrenamiento.

### Tasa de aprendizaje (learning rate): la regla del factor 10x

Aquí viene una de las contradicciones más contraintuitivas de LoRA: aunque estás entrenando muchos menos parámetros, necesitas una **tasa de aprendizaje más alta** que en fine-tuning completo, no más baja.

¿Por qué? Recuerda que $B$ se inicializa a cero. En los primeros pasos del entrenamiento, el update $\Delta W = BA$ es prácticamente cero — el adaptador no tiene casi ningún impacto sobre las salidas del modelo. El gradiente sobre $B$ es proporcional a $Ax$, que en los primeros pasos es pequeño porque $A$ está aleatoriamente inicializada pero $B$ no ha "dado señal" todavía al modelo.

Para que el adaptador empiece a aprender de forma significativa desde esa situación de "update cero", necesitas tasas de aprendizaje más agresivas que en FFT. La regla práctica: si el learning rate óptimo para FFT es $2 \times 10^{-5}$, el learning rate óptimo para LoRA estará cerca de $2 \times 10^{-4}$ — exactamente un orden de magnitud mayor.

Dicho esto, esta regla no es universal. Depende de la arquitectura, del scheduler de learning rate, y del tamaño del dataset. Úsala como punto de partida, no como verdad absoluta.

### Módulos objetivo (target modules): dónde aplicas los adaptadores

En el artículo de teoría mencionamos que LoRA puede aplicarse a cualquier matriz lineal del Transformer. Pero ¿a cuáles deberías aplicarlo? Esta decisión tiene más impacto del que parece.

Las implementaciones tempranas de LoRA aplicaban los adaptadores solo a las matrices de queries ($W_q$) y values ($W_v$) de la atención. Era una elección razonable dada la teoría original, pero los estudios empíricos posteriores demostraron que esta configuración "attention-only" subestimaba el potencial de LoRA.

La razón: las capas MLP (Multi-Layer Perceptron) de cada bloque Transformer contienen la mayoría de los parámetros del modelo y son donde se almacena la mayor parte del conocimiento factual. En un Transformer decoder moderno, el bloque MLP típicamente tiene tres proyecciones:

- `gate_proj`: determina qué información "pasa" a través de la función de activación.
- `up_proj`: expande la representación a una dimensión mayor (típicamente 4x el tamaño del embedding).
- `down_proj`: comprime de vuelta a la dimensión original.

Si no aplicas LoRA a estas capas, el adaptador puede ajustar cómo el modelo procesa información (a través de la atención), pero no puede modificar qué conocimiento el modelo considera relevante. Para especialización de dominio — diagnóstico médico, código especializado, razonamiento matemático — necesitas modificar ambas partes.

La configuración recomendada para maximizar rendimiento es aplicar LoRA a **todos los módulos lineales principales**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. El costo de memoria adicional es modesto (más módulos = más parámetros de adaptador, pero todos siguen siendo mucho menores que los pesos congelados), y el ganancia de rendimiento suele ser sustancial.

> **Descripción visual:** Diagrama de árbol vertical. En la raíz, un bloque azul "Configuración LoRA" se ramifica en cuatro nodos púrpura: "Rank r", "Alpha α", "Learning Rate" y "Target Modules". Cada uno de los tres primeros se expande en dos hojas, y "Target Modules" en tres. Las hojas verdes indican opciones conservadoras (r bajo, α=r, solo q/v), las rojas indican opciones agresivas (r alto, α=2r, todos los módulos), y las naranjas opciones intermedias. Flechas descendentes rectas. Fondo blanco, tipografía sans-serif.

---

## Dentro del bloque de atención: qué modifica cada proyección

Para entender por qué los módulos objetivo importan tanto, es útil ver exactamente qué hace cada proyección en el mecanismo de atención.

El mecanismo de **self-attention** (auto-atención), que es el núcleo del Transformer, funciona proyectando cada vector de token en tres espacios distintos: el espacio de queries, el espacio de keys, y el espacio de values.

**Queries ($W_q$):** La proyección de query transforma cada token en una "pregunta" — una representación de qué información está buscando ese token en el resto de la secuencia. Si aplicas LoRA a $W_q$, le estás dando al adaptador la capacidad de modificar qué tipo de relaciones busca el modelo.

**Keys ($W_k$):** La proyección de key transforma cada token en una "etiqueta" — una representación de qué información contiene ese token. Los attention scores se calculan como el producto escalar entre queries y keys. Modificar $W_k$ cambia cómo cada token "se anuncia" a los demás.

**Values ($W_v$):** La proyección de value determina qué información se pasa efectivamente hacia adelante una vez que los pesos de atención están calculados. Es la "carga útil" que el modelo transporta: puedes pensar en queries y keys como el mecanismo de enrutamiento, y en values como el contenido que se enruta.

**Output ($W_o$):** Después de computar la atención con múltiples cabezas, los resultados de cada cabeza se concatenan y se proyectan con $W_o$ de vuelta a la dimensión del modelo. Esta proyección controla cómo se mezclan las diferentes perspectivas de las distintas cabezas de atención.

Para un modelo de 7B con dimensión de embedding $d = 4096$ y 32 capas, cada una de estas matrices de proyección tiene $4096 \times 4096 = 16.7M$ parámetros. Hay 4 por capa (q, k, v, o) y 32 capas: son $4 \times 32 \times 16.7M \approx 2.1B$ parámetros solo en las proyecciones de atención. Añade los MLP y llegas al total de 7B.

Cuando defines `target_modules = ["q_proj", "v_proj"]`, lo que realmente estás haciendo es insertar un adaptador LoRA en paralelo a cada una de esas matrices en cada capa:

$$h_{attn} = W_q x + \frac{\alpha}{r} B_q A_q x$$

Los pesos originales de $W_q$ siguen haciendo su trabajo. El adaptador $B_q A_q$ añade una corrección de bajo rango encima.

> **Descripción visual:** Diagrama de flujo vertical con subgrafos anidados. Un bloque verde "Entrada x" apunta hacia abajo a una capa con dos subgrafos internos. El subgrafo de "Bloque de Atención" contiene cuatro bloques azules (Q, K, V, O proj + LoRA) donde Q, K, V convergen en O. El subgrafo "Bloque MLP" contiene tres bloques naranjas (Gate, Up, Down proj + LoRA). A la derecha, un subgrafo púrpura muestra la ecuación del forward pass: "W·x (pesos congelados)" y "(α/r)·B·A·x (adaptador LoRA)" convergen en "Salida = W·x + (α/r)·B·A·x". Al final, un bloque verde "Salida". Fondo blanco, tipografía sans-serif.

---

## Tabla de referencia: cuándo usar qué configuración

| Escenario | Rango ($r$) | Alpha ($\alpha$) | Módulos objetivo | LR recomendado |
|---|---|---|---|---|
| Ajuste de estilo / tono | 4–8 | $r$ o $2r$ | q, v | $1 \times 10^{-4}$ |
| Instruction tuning general | 8–16 | $r$ o $2r$ | q, k, v, o | $2 \times 10^{-4}$ |
| Especialización de dominio moderada | 16–32 | $2r$ | q, k, v, o, gate, up, down | $2 \times 10^{-4}$ |
| Dominio técnico complejo (código, math) | 64–128 | $2r$ | Todos los lineales | $1 \times 10^{-4}$ |
| Proximidad a FFT (alta capacidad) | 256+ | $2r$ | Todos los lineales | $5 \times 10^{-5}$ |

Nota sobre el learning rate en rangos altos: a $r = 256$, el adaptador tiene tanta capacidad que puede sobreajustarse con learning rates agresivos. Por eso el LR baja a $5 \times 10^{-5}$, más cerca del territorio de FFT.

---

## Multi-LoRA: cuando los adaptadores se convierten en infraestructura

Hasta aquí hemos discutido LoRA como una herramienta de entrenamiento. Pero sus implicaciones van mucho más allá del lab de fine-tuning — llegan hasta la arquitectura de sistemas de producción.

Porque los adaptadores LoRA son pequeños — típicamente entre 5 y 500 MB dependiendo del rango y el número de módulos objetivo, frente a los 14+ GB del modelo base — abren una posibilidad que el fine-tuning completo hace inviable: tener **múltiples adaptadores especializados sobre una única copia del modelo base**.

Imagina una empresa que necesita un asistente de IA para tres departamentos: legal, ingeniería, y marketing. Con fine-tuning completo, tendrían que mantener tres copias completas del modelo (3 × 14 GB = 42 GB solo de pesos). Con LoRA, mantienen una sola copia del modelo base (14 GB) más tres adaptadores pequeños (~50 MB cada uno). El ahorro de almacenamiento es masivo, pero el ahorro más importante es en VRAM de inferencia.

### Serving multi-tenant con LoRA

Sistemas de serving modernos como **Punica** y **S-LoRA** explotan esta propiedad para servir docenas de adaptadores distintos desde una sola GPU. El modelo base está cargado en VRAM. Cuando llega una petición para el asistente legal, se carga el adaptador legal y se aplica durante el forward pass. Cuando llega una petición para el asistente de ingeniería, se usa el adaptador de ingeniería — potencialmente en el mismo batch.

¿Cómo es posible procesar peticiones con diferentes adaptadores en el mismo batch? Aquí entra el truco de álgebra lineal. En un batch de $N$ tokens, la salida de la capa adaptada es:

$$H = W X + \frac{\alpha}{r} B A X$$

Donde $X$ es la matriz de inputs del batch. Si diferentes elementos del batch usan diferentes adaptadores $(B_1, A_1)$, $(B_2, A_2)$, etc., la operación $BAX$ puede computarse como una suma ponderada de productos de bajo rango — una operación que las GPU modernas ejecutan de forma extremadamente eficiente gracias a sus unidades de multiplicación matricial.

El resultado: en lugar de tener que serializar las peticiones (primero todas las de legal, luego todas las de ingeniería), el sistema puede mezclar peticiones de diferentes tenants en un solo forward pass, maximizando la utilización de la GPU.

### Entrenamiento concurrente con Multi-LoRA

La idea va más allá del serving. Sistemas como **mLoRA** y **LoRAFusion** permiten entrenar múltiples adaptadores distintos en paralelo, compartiendo el mismo modelo base congelado en GPU.

Esto resuelve un problema práctico que los equipos de ML enfrentan constantemente: la variación en la longitud de las secuencias de entrenamiento. En un dataset real, algunas secuencias tienen 128 tokens y otras tienen 4096. En entrenamiento distribuido tradicional, esto crea desequilibrios: la GPU que procesa las secuencias cortas termina primero y espera a la que procesa las largas. Ese tiempo de espera se llama "burbuja de pipeline" y puede representar un 20-30% de ineficiencia.

Multi-LoRA resuelve esto tratando el batch como un problema de bin-packing: si el batch actual de un job tiene secuencias cortas que dejan espacio en la GPU, el scheduler puede rellenar ese espacio con secuencias de otro job de fine-tuning. Los dos adaptadores se entrenan simultáneamente, compartiendo el modelo base. La GPU siempre está al máximo de utilización.

El resultado es que el throughput total (tokens procesados por segundo) crece sustancialmente sin necesitar hardware adicional — simplemente mejor gestión del tiempo de cómputo.

> **Descripción visual:** Diagrama de flujo vertical con forma de árbol invertido. En la cima, un bloque azul grande "Modelo Base congelado / 7B parámetros en VRAM / compartido por todos" se ramifica hacia tres bloques púrpura en paralelo: "Adaptador LoRA A / Soporte técnico (~50 MB)", "Adaptador LoRA B / Generación de código (~50 MB)" y "Adaptador LoRA C / Redacción comercial (~50 MB)". Cada adaptador apunta hacia bloques naranjas de peticiones individuales dentro de un subgrafo "Batch de inferencia". Los cuatro bloques naranjas convergen en un bloque verde final "Una sola pasada forward / Adaptadores intercambiados por petición". Fondo blanco, tipografía sans-serif, estilo limpio.

---

## El lab: ingeniería de LoRA en un experimento real

Con la teoría sólida, es hora de ver cómo estas decisiones se traducen a un experimento de fine-tuning concreto. El setup del lab combina las ideas anteriores en una configuración práctica reproducible.

### Setup del experimento

El modelo base elegido es un Llama o Mistral de 7B, que ofrece el equilibrio justo para experimentar: suficientemente grande para que el ahorro de LoRA sea dramáticamente visible, suficientemente pequeño para que quepa en hardware razonable con BF16.

La configuración de LoRA de referencia para el lab:
- **Rango $r = 16$**: suficiente para instruction-tuning general, sin exceso de capacidad que invite al sobreajuste.
- **Alpha $\alpha = 32$** (la heurística $\alpha = 2r$): escala de update moderadamente agresiva.
- **Módulos objetivo**: todos los lineales principales (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Learning rate $= 2 \times 10^{-4}$**: el orden de magnitud esperado para LoRA.
- **Modelo base en BF16**: ahorro de memoria sin pérdida de calidad significativa.

El propósito del lab es variar estas configuraciones y medir el impacto. Por ejemplo:
- ¿Qué pasa si pasamos de `target_modules = ["q_proj", "v_proj"]` a todos los lineales? ¿Cuánto mejora la métrica de evaluación? ¿Cuánta memoria extra consume?
- ¿Cómo cambia la curva de training loss cuando duplicamos el rango de 16 a 32?
- ¿El learning rate de $2 \times 10^{-4}$ converge de forma estable, o hay que aplicar gradient clipping?

### Métricas a vigilar durante el entrenamiento

El entrenamiento con LoRA tiene sus propias señales de advertencia que debes monitorizar. Aquí están las más importantes, con valores de referencia:

**Training loss:** Debe bajar de forma suave y monótona (con oscilaciones normales del batch). Una bajada demasiado rápida al principio — de 2.5 a 0.3 en los primeros 100 pasos — suele indicar sobreajuste al dataset de entrenamiento, especialmente si el dataset es pequeño. El rango normal de convergencia depende de la tarea, pero para instruction-tuning típico esperas bajar de ~2.5 a ~0.8–1.2 al final del entrenamiento.

**Validation loss (pérdida en validación):** El indicador de generalización. Si training loss sigue bajando pero validation loss empieza a subir, el adaptador está memorizando el dataset de entrenamiento. Considera reducir el rango, añadir dropout sobre los adaptadores, o reducir el número de épocas.

**Gradient norm:** La norma del gradiente sobre los parámetros LoRA debe ser estable. Si ves spikes — valores de 50, 100, o más — en los primeros pasos, es señal de que el learning rate es demasiado alto. El gradient clipping (típicamente en `max_norm = 1.0`) es tu primera línea de defensa.

**VRAM utilizada:** Con la configuración descrita ($r = 16$, todos los lineales, modelo 7B en BF16), el consumo de VRAM debería estar en el rango de 18–22 GB incluyendo activaciones para batch size de 4 y longitud de secuencia de 2048. Si ves números muy diferentes, revisa si el modelo base está realmente cargado en BF16 o si alguna capa quedó en FP32.

**Tokens por segundo (throughput):** Con LoRA, el overhead computacional respecto al forward pass del modelo base es mínimo — estás añadiendo dos multiplicaciones matriciales pequeñas por capa. Si el throughput de entrenamiento es significativamente menor que el esperado para un forward+backward pass del modelo base, revisa si el overhead viene del dataloading o de operaciones en CPU.

> **Descripción visual:** Diagrama de árbol vertical con raíz azul "Métricas a vigilar durante el entrenamiento LoRA". Se ramifica en cuatro nodos púrpura de segundo nivel: "Training Loss", "Gradient Norm", "Ratio parámetros entrenables" y "Eval Loss". Cada uno se expande en dos o tres hojas. Las hojas verdes indican comportamiento normal o saludable. Las hojas rojas indican señales de advertencia o error. Flechas descendentes rectas. Fondo blanco, tipografía sans-serif, estilo diagnóstico de semáforo verde/rojo.

### Interpretación de resultados: cuándo es "suficientemente bueno"

Una pregunta que surge siempre en el lab: ¿cómo sé si mi adaptador LoRA ha aprendido lo suficiente? No hay una respuesta universal, pero hay indicadores prácticos:

Si la tarea de evaluación es **generación con prompts similares al entrenamiento**: evalúa con una muestra representativa del dominio objetivo y mide con métricas automáticas (ROUGE, perplexity sobre el dominio) o evaluación humana. Un adaptador entrenado correctamente debería mostrar mejoras claras sobre el modelo base en el dominio objetivo y degradación mínima en benchmarks generales.

Si la tarea es **instruction-following** con una métrica clara (accuracy en una tarea de clasificación, F1 en extracción de información): traza la curva de la métrica de validación a lo largo del entrenamiento. El punto donde la mejora marginal por paso adicional es menor que el ruido de medición es una señal natural de parada.

El fine-tuning completo suele establecer el "techo" de rendimiento que LoRA puede aspirar a alcanzar. En benchmarks estándar, LoRA configurado correctamente (todos los lineales, rango adecuado) suele alcanzar el 90–95% del rendimiento del fine-tuning completo con una fracción del cómputo.

---

## Resumen de comprensión: el mapa conceptual de LoRA

Antes de cerrar, vale la pena conectar todos los conceptos en una cadena de razonamiento coherente que puedas reproducir sin consultar notas.

Los LLMs codifican su conocimiento en matrices de pesos de alta dimensión. Adaptar esos modelos a nuevas tareas mediante fine-tuning completo requiere actualizar todas esas matrices — un proceso que consume cientos de GB de VRAM y arriesga destruir el conocimiento previo a través del olvido catastrófico.

LoRA parte de la hipótesis del rango intrínseco: el cambio útil $\Delta W$ para cualquier nueva tarea vive en un subespacio de mucho menor dimensión que el espacio completo de la matriz. Esto es análogo a cómo un autoencoder demuestra que los datos de alta dimensión pueden comprimirse en representaciones latentes de baja dimensión sin perder su estructura esencial.

Para explotar esta hipótesis, LoRA factoriza $\Delta W = BA$ donde $A$ y $B$ son matrices de rango $r \ll d$. Solo $A$ y $B$ son entrenables; los pesos originales $W$ están congelados. Esto reduce los parámetros entrenables en órdenes de magnitud, colapsando el consumo de VRAM de estados del modelo de 112 GB a ~15 GB para un modelo de 7B.

La estabilidad del entrenamiento se garantiza inicializando $B = 0$, lo que hace que el update empiece en cero y crezca gradualmente. El factor de escala $\alpha/r$ desacopla la capacidad (rango) de la magnitud (alpha), permitiendo cambiar $r$ sin reajustar el learning rate.

En producción, el pequeño tamaño de los adaptadores (megabytes vs gigabytes) abre el paradigma Multi-LoRA: múltiples adaptadores especializados sobre un único modelo base en GPU, permitiendo serving multi-tenant eficiente y entrenamiento concurrente de varios fine-tuning jobs.

Los hiperparámetros clave — rango, alpha, módulos objetivo, learning rate — no son detalles de implementación sino decisiones de arquitectura con consecuencias medibles en rendimiento, memoria, y estabilidad. Entender la mecánica detrás de cada uno, como hemos hecho en este capítulo, es lo que separa el uso de LoRA como caja negra del uso de LoRA como herramienta de ingeniería.

En el siguiente capítulo, exploraremos [[04-qlora-cuantizacion-4bit|QLoRA]] — la extensión de LoRA que introduce cuantización del modelo base para reducir aún más el footprint de memoria, llevando el fine-tuning de modelos de 70B al rango de posibilidad en hardware de consumo.

---

## Tags

#técnica/lora #técnica/low-rank-adaptation #concepto/intrinsic-rank-hypothesis #concepto/factorización-de-bajo-rango #concepto/olvido-catastrófico #modelo/transformer #técnica/qlora #nivel/intermedio #tipo/lección #estado/completo



---
capitulo: 4
titulo: "QLoRA: Cómo la Cuantización a 4 Bits Rompe el Muro de la VRAM"
aliases:
  - "Capítulo 4"
  - "Cap 4"
  - "QLoRA"
  - "Cuantización 4 bits"
tema: "técnica-peft"
subtemas: [qlora, cuantizacion-4bit, nf4]
dificultad: "intermedio"
tipo: "lección"
estado: "completo"
conceptos_centrales:
  - qlora
  - cuantización-4bit
  - nf4
  - double-quantization
  - paged-optimizer
prerequisitos:
  - "[[01-fundamentos-transformers-y-pretraining]]"
  - "[[03-lora-adaptacion-de-bajo-rango]]"
relacionados:
  - "[[02-supervised-finetuning]]"
  - "[[07-finetuning-multimodal-vision-tts]]"
tags:
  - técnica/qlora
  - técnica/cuantizacion-4bit
  - concepto/nf4
  - técnica/double-quantization
  - técnica/paged-optimizer
  - herramienta/unsloth
  - herramienta/bitsandbytes
  - nivel/intermedio
  - tipo/lección
  - estado/completo
---

# Capítulo 4 — QLoRA: Cómo la Cuantización a 4 Bits Rompe el Muro de la VRAM

> Basado en "QLoRA Explained - How 4 Bit Quantization Unlocks Frontier Models" y "Engineering QLoRA for memory-efficient LLM Finetuning" — The Neural Maze, Finetuning Sessions Lección 4.

Hay un momento que todo practicante de fine-tuning conoce bien: abres el terminal, lanzas el script que tanto has preparado, y treinta segundos después aparece en pantalla un mensaje rojo que dice `CUDA out of memory`. El modelo no cabe en la GPU. No porque el código esté mal, sino porque la física de la memoria de vídeo tiene sus propias reglas, y tú acabas de chocar contra ellas.

El capítulo anterior resolvió el problema de *cuántos parámetros entrenamos* mediante [[03-lora-adaptacion-de-bajo-rango|LoRA]] (Low-Rank Adaptation — adaptación de rango bajo), que reemplaza actualizaciones completas de matrices de millones de elementos por matrices de rango bajo entrenables. Pero LoRA tiene un límite: aunque no entrenes los pesos originales, esos pesos siguen viviendo en la GPU. Un modelo de 70B parámetros en precisión de 16 bits ocupa 140 GB de VRAM antes de que hayas procesado un solo token. En una H100, la GPU más potente del mercado, solo tienes 80 GB. El muro no ha desaparecido; simplemente se ha desplazado.

QLoRA — Quantized LoRA, es decir, LoRA sobre pesos cuantizados — ataca ese muro desde un ángulo diferente: no reduce lo que entrenamos, sino que repiensa *cómo almacenamos lo que no entrenamos*. El resultado es un modelo congelado a 4 bits de precisión, combinado con adaptadores LoRA en alta precisión para el aprendizaje. En la práctica, esto significa poder hacer fine-tuning de un modelo de 70B parámetros en una GPU de 48 GB, o de un modelo de 7B en una tarjeta de consumo de 24 GB que cuesta menos de mil euros.

Este capítulo explica por qué eso es posible sin que el modelo pierda la cabeza, y cómo poner todo eso en práctica con código real.

---

## El coste fijo de entrenar: por qué 7B parámetros no son 14 GB

Antes de entender la solución, hay que entender en profundidad el problema. Y el problema tiene capas que no son evidentes a primera vista.

Cuando dices "tengo un modelo de 7 mil millones de parámetros en FP16", la cuenta más obvia es: 7.000.000.000 parámetros × 2 bytes por parámetro (que es lo que ocupa un número en FP16, o "Float de 16 bits") = 14 GB. Hasta aquí, todo bien. Pero esa cifra solo cubre los pesos del modelo — lo que el modelo *sabe*. Para que el modelo *aprenda*, necesitas tres cosas adicionales.

La primera es el gradiente. Durante el entrenamiento, cada peso del modelo tiene asociada una señal de error que le dice en qué dirección moverse para reducir la pérdida. Ese gradiente tiene exactamente el mismo tamaño que el peso: 2 bytes por parámetro, otros 14 GB. Hasta ahora vas por 28 GB.

La segunda y tercera son los estados del optimizador. El optimizador estándar en la industria para fine-tuning es AdamW — una variante de Adam (Adaptive Moment Estimation — estimación de momento adaptativo) con regularización de peso desacoplada. AdamW no actualiza los pesos usando solo el gradiente crudo del momento actual; mantiene dos memorias históricas por cada parámetro:

- El **primer momento** (también llamado momentum o $m_t$): un promedio exponencial de los gradientes pasados. Sirve para que el optimizador "recuerde" la dirección en que venía moviéndose y no cambie de rumbo bruscamente con cada batch. Ocupa 4 bytes por parámetro (en FP32, precisión de 32 bits).
- El **segundo momento** (también llamado varianza o $v_t$): un promedio exponencial del cuadrado de los gradientes pasados. Sirve para ajustar el tamaño del paso de forma diferente para cada parámetro — parámetros con gradientes grandes y variables reciben pasos pequeños; parámetros con gradientes pequeños y estables reciben pasos más grandes. También 4 bytes por parámetro en FP32.

Además, AdamW mantiene una **copia maestra de los pesos en FP32** (4 bytes por parámetro), incluso cuando el modelo trabaja en FP16, porque la acumulación de actualizaciones pequeñas en FP16 introduce errores de redondeo que pueden desestabilizar el entrenamiento a largo plazo.

El desglose completo por parámetro queda así:

| Tensor | Precisión | Bytes/parámetro |
|---|---|---|
| Pesos del modelo | FP16 | 2 |
| Gradientes | FP16 | 2 |
| Copia maestra (pesos) | FP32 | 4 |
| Primer momento AdamW | FP32 | 4 |
| Segundo momento AdamW | FP32 | 4 |
| **Total** | | **16** |

> **Descripción visual:** Diagrama de flujo horizontal con cinco bloques de origen convergiendo en un bloque destino. Los bloques de la izquierda están coloreados en degradado: rojo (Pesos FP16), naranja (Gradientes), azul (tres tensores del optimizador — Copia maestra, Momento 1, Momento 2). Todos ellos apuntan con flechas grises hacia un bloque verde a la derecha etiquetado "Total: 16 bytes/param — 112 GB para 7B". Fondo blanco, tipografía sans-serif, estilo limpio y técnico.

16 bytes por parámetro. Para un modelo de 7B: 7.000.000.000 × 16 = 112 GB. Para un modelo de 70B: 1,12 TB. No terabytes de disco: terabytes de VRAM, la memoria ultrarrápida que vive directamente dentro de la GPU.

Eso es lo que se llama el **coste fijo** del entrenamiento — en inglés, *model states*. Es el "depósito de entrada" que hay que pagar antes de procesar un solo ejemplo de entrenamiento.

Y esto crea un problema que se retroalimenta. Lo que sobra después de pagar el coste fijo es lo que la literatura llama *residual states* — la memoria dinámica disponible para los estados intermedios de la red durante el paso hacia adelante (*activations*) y para el KV Cache durante la generación. Cada gigabyte que se traga el coste fijo es un gigabyte que no está disponible para el batch size ni para el contexto.

El batch size — el número de ejemplos que se procesan en paralelo en cada paso de actualización — tiene un impacto directo en la calidad del entrenamiento. Con un batch size de 1, el gradiente que calculas en cada paso es el gradiente de un solo ejemplo, que puede ser muy ruidoso (atípico respecto al dataset completo). Con un batch size de 32, el gradiente es el promedio de 32 ejemplos, mucho más estable y representativo. Dicho de otro modo: trabajar con batch size bajo no solo es lento; introduce ruido en las actualizaciones que puede obligarte a hacer más pasos para alcanzar la misma calidad, consumiendo más tiempo y compute.

QLoRA ataca precisamente el coste fijo de los pesos. Si en lugar de almacenar cada peso en 2 bytes (FP16) lo almacenas en 0,5 bytes (4 bits), reduces el peso del modelo en un 75%. Para el modelo de 7B: pasas de 14 GB solo en pesos a 3,5 GB. Esa diferencia — 10,5 GB liberados — se puede reinvertir en un batch size mayor, en contextos más largos, o en modelos directamente más grandes que de otra forma no cabrían.

---

## El KV Cache: la segunda pared de memoria que nadie te cuenta

Hay un segundo tipo de memoria que crece de forma silenciosa y que puede asfixiarte incluso cuando el modelo ya cabe: el KV Cache.

Para entenderlo, hay que recordar cómo funciona la atención en un Transformer. Cuando el modelo genera un nuevo token, necesita "mirar" todos los tokens anteriores de la secuencia para calcular la atención — es decir, para decidir qué información de contexto es relevante para el siguiente paso. Si tienes una secuencia de 4.000 tokens y estás generando el token número 4.001, el modelo necesita considerar los 4.000 anteriores.

Sin optimización, recalcular todas las representaciones internas de esos 4.000 tokens en cada paso sería computacionalmente insostenible — la complejidad es $O(n^2)$ en la longitud de la secuencia. La solución es el **KV Cache** (Key-Value Cache — caché de claves y valores): almacenar en VRAM los tensores intermedios de claves ($K$) y valores ($V$) de la capa de atención para cada token ya procesado, de modo que solo haya que calcular los del nuevo token.

Es una optimización brillante para la velocidad, pero tiene un coste de memoria que escala linealmente con cuatro factores: longitud de la secuencia, tamaño del batch, número de capas del modelo, y dimensión de las cabezas de atención. Para un modelo de 70B con 80 capas y cabezas de 128 dimensiones, una ventana de contexto de 4.000 tokens puede consumir fácilmente 2,5 GB de VRAM por solicitud activa.

En inferencia esto se convierte en un problema de concurrencia. Si tu modelo de 70B, cuantizado a 4 bits, ocupa 35 GB, y cada usuario activo necesita 2,5 GB para su contexto, en una GPU de 80 GB tienes espacio para: (80 - 35) / 2,5 = 18 usuarios concurrentes antes de quedarte sin VRAM. Si el modelo no estuviera cuantizado y ocupara 140 GB... directamente no cabría en ninguna GPU individual.

Pero el KV Cache no solo es un problema de capacidad. También es un problema de **ancho de banda de memoria**. Durante la inferencia, la GPU pasa la mayor parte del tiempo moviendo datos desde la HBM (High Bandwidth Memory — la VRAM de alta velocidad) hacia los núcleos de cómputo, no haciendo las operaciones matemáticas en sí. Este cuello de botella se llama ser *memory-bound* (limitado por memoria). Cuando cuantizas el modelo a 4 bits, mueves 4 veces menos datos por peso en cada paso, lo que alivia ese cuello de botella y se traduce directamente en más tokens generados por segundo y menor latencia para el usuario final.

La combinación de pesos cuantizados a 4 bits con arquitecturas modernas de atención como GQA (Grouped Query Attention — atención por grupos de consultas, que reduce el número de cabezas K y V) produce un cambio cualitativo: se pasa de un mundo donde el cuello de botella es el *tamaño del modelo* a un mundo donde el cuello de botella es el *tamaño del contexto*. Eso abre la puerta a aplicaciones de contexto largo — como sistemas RAG (Retrieval-Augmented Generation — generación aumentada por recuperación) que necesitan leer documentos de cientos de páginas en un solo prompt.

---

## De FP32 a 4 bits: la geometría de la precisión

Para comprender cómo es posible comprimir pesos a 4 bits sin destruir el modelo, hay que entender qué significa precisión en el contexto de los números de punto flotante y cómo los distintos formatos hacen sus concesiones.

Un número en **FP32** (Float de 32 bits, también llamado single precision) utiliza 32 bits organizados en tres campos:

- 1 bit de signo (positivo o negativo)
- 8 bits de exponente (controla el rango, es decir, qué órdenes de magnitud puede representar)
- 23 bits de mantisa (controla la precisión, es decir, cuántos decimales significativos tiene el número)

Con 23 bits de mantisa, FP32 puede representar aproximadamente 7 dígitos decimales de precisión y cubrir magnitudes desde $10^{-38}$ hasta $10^{38}$. Es como una regla con marcas cada nanómetro — perfecta para simulaciones físicas donde un error de redondeo puede comprometer el resultado, pero excesiva para los pesos de una red neuronal que van a ser ajustados iterativamente de todas formas.

**BF16** (Brain Float 16 — float de 16 bits del cerebro, desarrollado por Google Brain) fue el primer gran paso hacia la eficiencia. Sus 16 bits se distribuyen como: 1 de signo, 8 de exponente, 7 de mantisa. La clave es que mantiene los mismos 8 bits de exponente que FP32, conservando el mismo rango dinámico. Sacrifica precisión (de 23 bits de mantisa a 7), pero los ingenieros de Google observaron que las redes neuronales son más sensibles al orden de magnitud de un peso que a su valor exacto — lo que importa es saber si algo vale "alrededor de 0.3" más que si vale exactamente "0.2998742". BF16 fue adoptado masivamente porque permitía entrenar sin cambiar la convergencia de los modelos, simplemente usando la mitad de la memoria.

> **Descripción visual:** Diagrama de flujo horizontal con cuatro bloques rectangulares en cadena de izquierda a derecha. El primero (gris) representa FP32 con 4 bytes/param. El segundo (azul) BF16 con 2 bytes. El tercero (naranja) INT8 con 1 byte. El cuarto (verde) NF4 con 0.5 bytes. Las flechas entre bloques llevan la etiqueta "÷2 memoria". Los colores van aclarándose hacia la derecha, reflejando la reducción progresiva de bits. Fondo blanco, estilo técnico minimalista.

**INT8** (Integer de 8 bits) marcó la primera transición desde números de coma flotante hacia enteros. Con 8 bits, solo tienes 256 valores posibles (de -128 a 127 si usas representación con signo). El proceso de asignar pesos continuos a esos 256 slots discretos se llama **cuantización**, y requiere **calibración**: elegir los valores mínimo y máximo del rango de pesos que vas a mapear, de modo que los pesos extremos no queden fuera de los 256 slots disponibles. Si calibras mal y el rango es demasiado estrecho, los pesos que sobresalen se "recortan" (clipping) y pierden información; si el rango es demasiado amplio, los 256 slots se distribuyen en un espacio grande y la resolución entre slots adyacentes se hace grosera. INT8 funcionó bien para inferencia a gran escala en centros de datos, pero resultó frágil para modelos generativos modernos.

**El salto a 4 bits** es donde la geometría cambia de forma dramática. Con 4 bits solo tienes $2^4 = 16$ valores posibles para representar un peso. Imagina que quieres pintar un retrato fotorrealista pero solo tienes 16 colores en tu paleta — y además no puedes elegir qué 16 colores. Si esos 16 colores están distribuidos de forma uniforme entre el negro absoluto y el blanco absoluto, vas a tener problemas para capturar los tonos medios donde están todos los detalles.

Eso es exactamente lo que pasa con la cuantización lineal de 4 bits: los 16 slots se distribuyen a intervalos iguales por el rango de los pesos, ignorando que la distribución de los pesos no es uniforme.

Para entender por qué importa, hay que fijarse en cómo se distribuyen realmente los pesos de una red neuronal entrenada. Invariablemente, siguen una distribución **gaussiana** (campana de Gauss) centrada en cero. La gran mayoría de los pesos son valores pequeños, cercanos a cero — digamos entre -0.5 y 0.5. Solo unos pocos pesos tienen valores grandes, como -2.0 o +1.8. Si distribuyes tus 16 slots de forma uniforme entre -2.0 y 2.0, con intervalos de 0.25, estás dedicando la misma resolución a la zona poco poblada de los extremos que a la zona densamente poblada del centro. Es un desperdicio de información.

---

## NF4: los bits en el lugar correcto

La solución que propone QLoRA se llama **NF4** (NormalFloat 4-bit — punto flotante normal de 4 bits). Es un tipo de dato diseñado específicamente para la forma en que los pesos de redes neuronales se distribuyen.

La idea central es la siguiente: en lugar de espaciar los 16 slots a intervalos iguales, los colocamos donde realmente viven los datos. Para una distribución gaussiana estándar, esto equivale a definir los 16 valores de NF4 como los **cuantiles** de esa distribución — los puntos que dividen la distribución en 17 zonas de igual probabilidad. En términos concretos:

- Si el 10% más pequeño de los pesos cae entre -2.0 y -1.2, hay un slot de NF4 en ese rango que cubre todo ese 10%.
- Si el 10% siguiente va de -1.2 a -0.85, otro slot cubre ese intervalo.
- Y así sucesivamente, acumulando slots en la zona central donde hay más pesos.

El resultado es que cada slot de NF4 cubre aproximadamente el mismo *volumen de datos* aunque no el mismo *rango de valores*. La resolución es mayor donde más la necesitas (cerca del cero) y menor donde menos la necesitas (en los extremos). Este diseño, fundamentado en la teoría de la información — específicamente en el principio de minimizar la entropía cuantizada — logra que la cuantización NF4 introduzca mucho menos error que INT4 lineal con los mismos 4 bits.

El proceso concreto de cuantización NF4 tiene dos fases. Primero, se normaliza un bloque de pesos (típicamente 64 pesos consecutivos) dividiéndolos por el valor absoluto máximo del bloque, de modo que todos caigan en el rango $[-1, 1]$. Segundo, ese valor normalizado se mapea al índice NF4 más cercano de los 16 predefinidos, que se almacena como un entero de 4 bits. En total, se almacena el índice (4 bits) más una constante de escala por bloque (16 bits en FP16): el overhead de la constante de escala es pequeño — por cada 64 pesos de 4 bits usamos 64 × 4 bits + 1 × 16 bits = 272 bits, frente a los 64 × 16 bits = 1024 bits en FP16. Eso es una reducción de 3.76x en almacenamiento, prácticamente el 4x teórico.

Para deshacer la cuantización (dequantización) cuando el modelo necesita hacer una operación, el proceso es inverso: se toma el índice de 4 bits, se busca el valor NF4 correspondiente en la tabla de 16 entradas, y se multiplica por la constante de escala del bloque. Este proceso ocurre en cada capa durante el paso hacia adelante, y es lo suficientemente rápido como para que el overhead sea mínimo en hardware moderno.

> **Descripción visual:** Diagrama de flujo horizontal con seis bloques en cadena. El primero (morado) representa el bloque de 64 pesos continuos. Los siguientes dos (azules) son las fases de proceso: normalización y mapeo al cuantil NF4. El cuarto (naranja) representa el almacenamiento comprimido: índice de 4 bits más escala FP16. El quinto (azul) es la dequantización que ocurre en cada forward pass. El sexto (verde) es el tensor BF16 resultante listo para cómputo. Las flechas son lineales de izquierda a derecha. Fondo blanco, tipografía sans-serif, estilo técnico.

---

## Double Quantization: cuantizando las constantes de escala

Hay un detalle de ingeniería adicional en QLoRA que parece un truco de magia de prestidigitación pero tiene un impacto real: la **Double Quantization** (cuantización doble).

Recuerda que cada bloque de 64 pesos necesita almacenar su propia constante de escala en FP16 (16 bits). Para un modelo de 7B parámetros con bloques de 64, necesitas almacenar: 7.000.000.000 / 64 = ~109.375.000 constantes de escala, cada una en FP16. Eso son 109.375.000 × 2 bytes = ~218 MB solo en constantes de escala.

La double quantization observa que estas constantes de escala también son números continuos y también tienen su propia distribución — son básicamente los "rangos locales" de cada bloque de pesos, y tienden a ser razonablemente uniformes. Así que las trata como un segundo nivel de datos y las cuantiza a su vez: agrupa bloques de constantes de escala (típicamente 256 constantes por superbloque) y cuantiza esas constantes de escala de FP16 a FP8, manteniendo una única constante de superescala por superbloque en FP32.

El ahorro de bits por parámetro que consigue la double quantization es: en el sistema sin double quantization, la constante de escala FP16 cuesta 16 bits / 64 parámetros = 0.25 bits por parámetro. Con double quantization (constante FP8 en bloques de 256 dentro de un superbloque con constante FP32): la constante FP8 cuesta 8/64 = 0.125 bits por parámetro, más la constante FP32 del superbloque que cuesta 32 / (256 × 64) ≈ 0.002 bits por parámetro. El ahorro total es de 0.25 - 0.127 ≈ 0.123 bits por parámetro, que los autores de QLoRA reportan como aproximadamente 0.37 bits por parámetro considerando todo el overhead del sistema.

¿Por qué importa ese ahorro aparentemente minúsculo? Para un modelo de 70B parámetros: 70.000.000.000 × 0.37 bits / 8 bits por byte ≈ 3.24 GB. Tres gigabytes que aparecen de la nada — espacio suficiente para aumentar el batch size de 4 a 12, o para extender el contexto de 2K a 6K tokens sin tocar nada más del setup.

> **Descripción visual:** Diagrama de flujo horizontal con cinco bloques en cadena. Los dos primeros (azules) representan los pesos NF4 y la escala FP16 por bloque. Los dos siguientes (naranjas) representan el superbloque de 256 escalas y la escala FP8 resultante. El último bloque (verde) muestra el ahorro total de aproximadamente 0.37 bits por parámetro, equivalente a 3 GB en un modelo de 70B. Las flechas son lineales de izquierda a derecha. Fondo blanco, estilo técnico minimalista.

---

## Paged Optimizers: el seguro contra el OOM

Incluso con todo el modelo en 4 bits, hay un momento durante el entrenamiento donde la VRAM puede desbordarse de forma inesperada: el backpropagation sobre los adaptadores LoRA. Durante el paso hacia atrás, los gradientes de los adaptadores se acumulan, y en ciertos puntos del batch — especialmente con secuencias largas o ejemplos con activaciones extremas — puede producirse un **gradient spike**, un pico temporal de uso de memoria que supera lo que la GPU tiene disponible. El resultado es un error `CUDA out of memory` que aborta el entrenamiento, potencialmente después de horas de ejecución.

La solución de QLoRA a este problema se llama **Paged Optimizer** (optimizador con paginación), y usa una función del hardware NVIDIA llamada Unified Memory (memoria unificada). Normalmente, la VRAM de la GPU y la RAM del sistema son espacios de memoria completamente separados — el código que corre en la GPU no puede acceder directamente a la RAM del sistema, y viceversa. Unified Memory rompe esa barrera creando un espacio de direcciones virtual compartido: el sistema operativo puede mover páginas de memoria automáticamente entre la RAM del sistema y la VRAM de la GPU según las necesidades.

El Paged Optimizer aprovecha esto para los **estados del optimizador** — los primeros y segundos momentos de AdamW. En condiciones normales, esos estados viven en VRAM para poder participar en la actualización de pesos. Con la paginación activada, cuando la GPU detecta que se está acercando al límite de VRAM (por ejemplo, durante un gradient spike), puede "desalojar" los estados del optimizador hacia la RAM del sistema — que en un servidor moderno puede ser de 128 GB o más — y traerlos de vuelta cuando los necesita para el próximo step de actualización.

La penalización de latencia por este movimiento de datos existe, pero es aceptable: el movimiento ocurre entre steps de actualización, no en medio del cálculo del gradiente, y solo cuando hay presión de memoria. El 95% del tiempo los estados del optimizador están en VRAM como de costumbre; la paginación solo se activa como válvula de seguridad en los momentos críticos. La alternativa — que el run se aborte con OOM — es obviamente peor.

En la práctica, el Paged Optimizer transforma el comportamiento del entrenamiento de "todo o nada" a "siempre completa el run". Puedes configurar batch sizes más agresivos sabiendo que si el memoria se desborda en algún ejemplo extremo, el sistema se recupera automáticamente en lugar de crashear.

> **Descripción visual:** Diagrama de flujo horizontal con ramificación. El flujo principal va de izquierda a derecha: "Step de entrenamiento" (azul) hacia "Estados AdamW en VRAM" (azul) hacia un rombo de decisión naranja "Presión de memoria". Desde el rombo salen dos caminos: el superior (etiquetado "normal") lleva directamente al bloque verde "Actualización pesos LoRA"; el inferior (etiquetado "pico OOM") lleva a dos bloques rojos en secuencia — "Offload a RAM" y "Retorno a VRAM" — que luego convergen en el bloque verde. Fondo blanco, tipografía sans-serif, estilo técnico.

---

## El mapa completo de QLoRA: tres piezas, una arquitectura

Con todos los componentes definidos, conviene ver cómo encajan en un pipeline coherente. QLoRA combina tres innovaciones:

**Primera pieza — Modelo base congelado en NF4.** El [[01-fundamentos-transformers-y-pretraining|modelo pre-entrenado]] (digamos, Qwen3-7B con 7.000 millones de parámetros) se carga y se cuantiza a NF4. Sus pesos no se actualizan durante el fine-tuning — están congelados. Pero la cuantización no es una operación destructiva irreversible: en cada paso hacia adelante, los bloques de pesos se dequantizan a BF16 para hacer los cálculos, y ese tensor temporal se descarta al terminar el paso. El modelo "piensa" en BF16, pero "duerme" en NF4.

**Segunda pieza — Adaptadores LoRA en BF16.** Sobre las capas proyección de atención del modelo congelado (Q, K, V, O y opcionalmente las capas de feedforward), se inyectan matrices LoRA de bajo rango en precisión completa BF16. Estas matrices sí se actualizan durante el entrenamiento. Para un modelo de 7B con rango $r = 16$ y 32 capas Transformer, el total de parámetros entrenables es del orden de 20-40 millones — menos del 0.6% del modelo original, lo que permite que los gradientes sean pequeños, limpios y computacionalmente baratos.

La interacción entre el modelo congelado en NF4 y los adaptadores en BF16 funciona así en cada capa: la entrada $x$ pasa por el peso cuantizado dequantizado $W_0$ (que produce la salida "base"), y también pasa por los adaptadores LoRA $BA$ (que producen el "delta" de la tarea). Ambas contribuciones se suman:

$$h = W_0 x + \frac{\alpha}{r} BAx$$

donde $\alpha$ es el hiperparámetro lora_alpha que escala la contribución del adaptador, $r$ es el rango de los adaptadores, $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times k}$ son las matrices LoRA. Esta suma ocurre en BF16, con el gradiente fluyendo solo a través del término $BA$ — el modelo congelado $W_0$ no recibe gradientes.

**Tercera pieza — Paged Optimizer con AdamW en 8 bits.** Los estados del optimizador para los adaptadores LoRA se mantienen en 8 bits (no en 32 bits como en AdamW estándar) gracias a la librería `bitsandbytes`. Esto reduce el overhead del optimizador de 8 bytes por parámetro (4 para el primer momento + 4 para el segundo en FP32) a 2 bytes (1 + 1 en INT8). Para 30 millones de parámetros entrenables, la diferencia es: (8 - 2) bytes × 30.000.000 = 180 MB — no enorme, pero tampoco despreciable. La paginación por encima de esto actúa como red de seguridad.

El resultado del sistema completo: un modelo de 7B que en entrenamiento estándar con AdamW en FP16 necesitaría ~112 GB de VRAM, con QLoRA necesita aproximadamente 10-12 GB, incluyendo activaciones y KV Cache de training. Eso entra cómodamente en una GPU de 24 GB como la RTX 4090.

> **Descripción visual:** Diagrama de flujo horizontal con bifurcación y convergencia. Un bloque morado a la izquierda ("Modelo base 7B") se divide en dos ramas paralelas: la superior gris ("Pesos congelados NF4") y la inferior azul ("Adaptadores LoRA BF16"). Ambas ramas convergen en un bloque naranja central ("Salida combinada h = W₀x + BAx"). De ahí fluye hacia un bloque azul ("Paged Optimizer AdamW 8 bits") y finalmente al bloque verde resultado ("Fine-tuning 7B en GPU 24 GB"). Las flechas son limpias y lineales. Fondo blanco, tipografía sans-serif, estilo técnico arquitectónico.

---

## La evolución del hardware: de Turing a Blackwell

QLoRA es software, pero su impacto real depende de con qué hardware interactúa. La historia de la cuantización en GPUs NVIDIA es también la historia de cómo el silicon fue adaptándose para convertir lo que era un truco de software en una operación nativa.

La arquitectura **Turing** (2018, GPUs T4) fue la primera en introducir Tensor Cores dedicados a la multiplicación de matrices — aceleradores de hardware especializados que multiplican matrices de 4×4 en una sola operación en lugar de hacerlo elemento por elemento. Turing estableció FP16 como el formato estándar de entrenamiento. La cuantización a 4 bits en Turing era un truco de software: los pesos se almacenaban en 4 bits, pero para operar se dequantizaban a FP16, haciendo las operaciones en ese formato. El ahorro era en almacenamiento y ancho de banda, no en velocidad de cómputo.

**Ampere** (2020, A100) resolvió el problema de estabilidad de BF16 e introdujo soporte nativo para enteros de 8 bits (INT8) en sus Tensor Cores. Un A100 puede hacer multiplicaciones de matrices directamente en INT8, sin dequantizar. También introdujo el concepto de Transformer Engine, una capa hardware-software que monitoriza la magnitud de los tensores en tiempo real y elige automáticamente entre FP16 y BF16 por capa. Para 4 bits, sin embargo, Ampere todavía usaba emulación por software.

**Hopper** (2022, H100) dio el siguiente salto con soporte nativo para FP8 — números de punto flotante de 8 bits con diferentes configuraciones de exponente y mantisa (E4M3 y E5M2). El Transformer Engine en Hopper gestiona FP8 de forma transparente: elige qué capas usar en FP8 y cuáles en BF16, ajustando factores de escala para prevenir desbordamientos. La velocidad de throughput de FP8 en H100 dobla la de BF16. Para 4 bits, Hopper todavía recurría a emulación en INT8 — los pesos de 4 bits se desempaquetaban a INT8 antes de operar.

**Blackwell** (2024, B200/GB200) es la primera arquitectura con soporte nativo para 4 bits en hardware: sus Tensor Cores de quinta generación pueden operar directamente en NVFP4 sin desempaquetar ni dequantizar. El formato NVFP4 que usa Blackwell — que discutimos en la sección anterior — usa bloques de 16 pesos con una escala compartida de 8 bits en formato E4M3, permitiendo factores de escala fraccionarios (1.5x, 2.25x, etc.) en lugar de los saltos discretos de potencias de dos que usaba MXFP4. Esto le permite "abrazar" la distribución real de los datos con mucha más precisión.

El resultado práctico: Blackwell puede ejecutar inferencia en NVFP4 a hasta 15 PetaFLOPS por GPU — 30x más throughput que las arquitecturas Pascal de apenas ocho años antes. Y más importante, lo hace sin necesitar la dequantización que introducía latencia adicional. El hardware ahora "piensa" en el mismo idioma de 4 bits en que almacena los pesos.

Para el practicante, esto cambia el cálculo de dónde invertir. En una H100, la cuantización a 4 bits beneficia principalmente la huella de memoria pero no la velocidad de cómputo (que todavía opera en INT8 emulado). En una B200, la cuantización a 4 bits beneficia simultáneamente la memoria *y* el throughput — de modo que modelos que antes necesitaban multi-GPU para alcanzar latencias aceptables pueden ahora correr en una sola GPU Blackwell.

---

## De la teoría al terminal: el lab de QLoRA con Unsloth

Con la arquitectura de QLoRA completamente interiorizada, es el momento de ver cómo se materializa en código. El lab usa la librería **Unsloth** — una librería Python de código abierto que reimplementa las capas críticas de Transformers con kernels CUDA optimizados a mano, consiguiendo 2-3x de velocidad adicional sobre una implementación estándar de Hugging Face + PEFT. Unsloth abstrae los detalles de NF4, double quantization y paged optimizers en una API que se parece mucho a Hugging Face Transformers, lo que facilita enormemente la adopción.

El punto de entrada es `FastLanguageModel.from_pretrained`, que en la práctica hace tres cosas: descarga el modelo desde Hugging Face Hub, lo cuantiza a NF4 usando la librería `bitsandbytes` (la implementación de referencia de cuantización en 4 bits para PyTorch), y lo carga en VRAM en ese formato comprimido. El parámetro que activa todo esto es `load_in_4bit=True`:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-0.6B",
    max_seq_length = 2048,
    load_in_4bit = True,      # Activa NF4 + double quantization
    dtype = None,             # Unsloth elige BF16 automáticamente si el hardware lo soporta
)
```

La diferencia entre `load_in_4bit=True` y `load_in_4bit=False` parece trivial en el código, pero en VRAM es una reducción del 75% en el peso del modelo. Para Qwen3-0.6B (600 millones de parámetros), la diferencia es de ~1.2 GB a ~0.3 GB — en este caso prácticamente irrelevante. Pero para Qwen3-7B: de ~14 GB a ~3.5 GB. Para Qwen3-72B: de ~144 GB — imposible en una sola GPU — a ~36 GB, que cabe en una A100 o H100 con margen para activaciones y contexto.

Después de cargar el modelo base, se inyectan los adaptadores LoRA mediante `get_peft_model`:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                   # Rango de los adaptadores: balance entre capacidad y eficiencia
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],  # Qué capas adaptar
    lora_alpha = 16,           # Factor de escala alpha (suele igualarse a r)
    lora_dropout = 0,          # Unsloth recomienda 0 para máxima velocidad
    bias = "none",             # No adaptar sesgos
    use_gradient_checkpointing = "unsloth",  # Técnica para reducir memoria de activaciones
)
```

El parámetro `target_modules` merece atención. En este ejemplo se adaptan no solo las proyecciones de atención (q, k, v, o) sino también las capas de feedforward de Qwen3 (gate, up, down). Añadir las capas feedforward al fine-tuning LoRA aumenta la capacidad del adaptador — más parámetros entrenables, más capacidad de adaptar comportamiento — a cambio de ligeramente más uso de VRAM y computo. Para tareas de instrucción siguiendo un formato estructurado, las capas de atención suelen ser suficientes. Para tareas que requieren cambiar el "vocabulario" de respuesta del modelo de forma significativa — por ejemplo, entrenarlo en un dominio técnico muy específico — incluir las capas feedforward puede ser importante.

El `use_gradient_checkpointing = "unsloth"` es una técnica que merece una explicación aparte. Durante el paso hacia adelante de una red neuronal, todas las activaciones intermedias (los tensores que produce cada capa para ser usados por las capas siguientes durante el backpropagation) se almacenan en VRAM. En modelos grandes con secuencias largas, estas activaciones pueden consumir tanto como los pesos del modelo. El gradient checkpointing resuelve esto guardando solo un subconjunto de activaciones y recalculando el resto durante el backpropagation cuando se necesitan. El coste es aproximadamente un 33% más de tiempo de cómputo a cambio de reducir la memoria de activaciones drásticamente. La versión de Unsloth tiene optimizaciones adicionales que reducen ese overhead.

---

## Elegir el hardware: la decisión arquitectónica que nadie documenta bien

Cuando se lanza un job de entrenamiento en Hugging Face (o en cualquier plataforma cloud), elegir la GPU no es solo una decisión de precio — es una decisión arquitectónica que afecta qué puede y qué no puede hacer tu entrenamiento. Vale la pena entender qué implica cada opción.

**T4 (16 GB VRAM, Turing, ~$0.5/hora en HF Jobs):** La GPU que NVIDIA ofrece para inferencia de bajo coste. Tiene Tensor Cores de primera generación (FP16 nativo) pero carece de soporte para BF16 hardware y tiene un ancho de banda de memoria limitado comparado con GPUs más modernas. Con QLoRA, puedes hacer fine-tuning de modelos hasta ~1B parámetros cómodamente, ~3B con ajustes agresivos. Ideal para verificar que tu pipeline y tu código funcionan antes de gastar dinero en hardware más caro. No la uses para producción si el modelo tiene más de 3B.

**A10G (24 GB VRAM, Ampere, ~$1.5/hora en HF Jobs):** El punto óptimo para QLoRA en la nube. 24 GB de VRAM con la eficiencia de Ampere y BF16 nativo. Con QLoRA puedes hacer fine-tuning de modelos de 7B cómodamente y de 13B con contextos moderados. Esta es la GPU donde QLoRA "brilla" en el sentido original del artículo: la diferencia entre poder y no poder entrenar un 7B es exactamente el salto de FP16 (14 GB solo en pesos) a NF4 (~3.5 GB en pesos). El A10G es también el hardware de referencia para equipos que quieren iterar rápido sin comprometer calidad.

**A100 (40 GB o 80 GB VRAM, Ampere, ~$3-6/hora en HF Jobs):** El caballo de trabajo de la era post-GPT-3. 80 GB de VRAM con ancho de banda de 2 TB/s. Sin cuantización puedes entrenar modelos de hasta ~30B; con QLoRA, modelos de hasta ~70B. El Transformer Engine de Hopper no está disponible aquí (A100 es Ampere), pero la cantidad bruta de VRAM hace que muchas limitaciones de batch size desaparezcan. Úsalo cuando necesites contextos muy largos (>16K tokens) o batch sizes grandes para estabilidad del gradiente.

**H100 (80 GB VRAM, Hopper, ~$6-8/hora en HF Jobs):** El estándar de oro actual para entrenamiento a escala. Añade FP8 nativo sobre A100, lo que con el Transformer Engine permite throughput casi doble. Para QLoRA específicamente, la diferencia respecto al A100 es menos dramática (ambos dequantizan a BF16 para operar), pero el mayor ancho de banda de memoria de H100 (3.35 TB/s vs 2 TB/s del A100) reduce el tiempo que el modelo pasa esperando datos. Úsalo para runs de producción donde el tiempo de entrenamiento es importante o cuando necesitas el máximo contexto posible.

La estrategia práctica es simple pero frecuentemente ignorada: **empieza siempre en el hardware más pequeño que permita correr el modelo**. Lanza 50 steps en A10G para verificar que la pérdida baja correctamente y que no hay bugs en el pipeline. Solo cuando el experimento está validado, escala a hardware más caro para el run completo. Un run completo en H100 que falla porque el learning rate está mal es dinero tirado; ese fallo se habría detectado en A10G por una décima parte del coste.

---

## Los adaptadores LoRA como unidad de despliegue

El concepto final del lab que merece atención explícita es la naturaleza de lo que produce un fine-tuning con QLoRA: un adaptador, no un modelo.

Cuando termina el entrenamiento y ejecutas `model.push_to_hub("mi-adaptador")`, lo que sube al repositorio de Hugging Face no son los ~3.5 GB del modelo cuantizado — ese es el modelo base, que ya está en Hugging Face y cualquiera puede descargarlo. Lo que sube son solo los pesos de los adaptadores LoRA: las matrices $A$ y $B$ para cada capa objetivo. Para un modelo de 7B con rango 16 y cubriendo las 7 capas de proyección más feedforward en 32 capas Transformer, eso son aproximadamente 80-100 MB. No gigabytes: megabytes.

Esta separación entre **base** (el conocimiento general) y **adaptador** (la habilidad específica) tiene implicaciones prácticas importantes. En producción, puedes mantener un único modelo base en memoria y cargar adaptadores distintos para distintos usuarios o distintas tareas. Un servidor con un Qwen3-7B en NF4 y tres adaptadores (uno para soporte técnico, otro para generación de código, otro para resumen de documentos) ocupa: ~3.5 GB de modelo + 3 × 0.1 GB de adaptadores ≈ 3.8 GB. Puedes hacer inferencia con cualquiera de las tres especialidades cambiando solo cuál adaptador está activo, sin recargar el modelo base.

Esta arquitectura también significa que el resultado de tu trabajo de fine-tuning es extremadamente portable. Un adaptador de 100 MB puede enviarse por email, guardarse en un repositorio git, o desplegarse en un edge device. El receptor solo necesita el modelo base (que puede descargar de Hugging Face Hub) para reconstruir el modelo completo. En términos de distribución y actualización de modelos, es un cambio de paradigma comparable al paso de distribuir aplicaciones compiladas a distribuir solo los patches.

---

## Qué vigilar durante el entrenamiento: las métricas que importan

Un entrenamiento de QLoRA en marcha produce una corriente de datos que puede ser abrumadora si no sabes qué buscar. Estas son las señales críticas y lo que dicen:

**Pérdida de entrenamiento (training loss).** Debe descender de forma monótona en las primeras épocas. El patrón típico tiene un "codo" o caída pronunciada en los primeros 10-20% de los steps (el modelo adapta rápidamente su comportamiento a la distribución del dataset), seguido de una caída más gradual. Si la pérdida oscila violentamente o sube y baja sin dirección clara después de los primeros steps, sospecha del learning rate — probablemente demasiado alto para la precisión cuantizada del modelo base.

Los **gradient spikes** son picos abruptos en la pérdida seguidos de recuperación. En QLoRA son más frecuentes que en entrenamiento en precisión completa porque la dequantización introduce un ruido adicional en el gradiente. El gradient clipping — limitar la norma L2 del vector de gradiente a un valor máximo como 1.0 — ayuda a contener estos spikes sin eliminar la señal del gradiente. Si los spikes son frecuentes e intensos, considera reducir el learning rate en un factor de 2-5x.

**GPU Utilization vs. GPU Memory.** Si la memoria es una línea plana cercana al 95-100% pero la utilización de los núcleos de cómputo es baja (digamos, 40-60%), estás limitado por el batch size — el modelo espera que lleguen más datos pero no tiene dónde ponerlos. Con QLoRA, la respuesta correcta es aumentar el `per_device_train_batch_size` o el `gradient_accumulation_steps` (que simula un batch size mayor acumulando gradientes de varios mini-batches antes de hacer el step del optimizador).

**Norma del gradiente (gradient norm).** Debe ser estable en el rango de 0.1 a 10. Valores consistentemente por encima de 100 sugieren que el learning rate es demasiado alto o que la tarea de adaptación es demasiado difícil para el rango LoRA elegido. Valores consistentemente cercanos a 0 sugieren que los adaptadores no están aprendiendo nada útil — puede que la tarea esté ya bien cubierta por el modelo base, o que el dataset sea demasiado pequeño para producir una señal clara.

**Pérdida de validación (validation loss).** Si tu dataset tiene un split de validación, la pérdida de validación debe seguir a la de entrenamiento de cerca. Si la pérdida de entrenamiento sigue bajando pero la de validación se estabiliza o sube, estás sobreajustando el adaptador al dataset de entrenamiento. En QLoRA, el sobreajuste es relativamente común con datasets pequeños (<1000 ejemplos) porque los adaptadores tienen suficiente capacidad para memorizar los ejemplos en lugar de generalizar. Las soluciones incluyen: reducir el rango $r$, aumentar el dropout en los adaptadores, o conseguir más datos.

---

## El ecosistema de cuantización más allá de NF4

Para terminar de situar NF4 en el mapa, conviene mencionar brevemente los otros formatos de cuantización que encontrarás en la práctica, porque la elección entre ellos tiene consecuencias concretas.

**GGUF** (formato desarrollado por llama.cpp) es el estándar de facto para inferencia local en CPU y GPUs de consumo. Los modelos en GGUF están cuantizados a nivel de peso y se dequantizan a FP16 o BF16 para computar. Hay múltiples niveles: Q4_K_M, Q5_K_M, Q8_0, etc. La ventaja es la portabilidad extrema — funciona en Mac, Linux, Windows, y en hardware sin soporte CUDA. La desventaja respecto a NF4 es que está diseñado para inferencia, no para entrenamiento; no puedes hacer fine-tuning con un modelo en GGUF.

**GPTQ** (Generative Pre-Trained Transformer Quantization) usa optimización de segundo orden para minimizar el error de cuantización capa por capa. En lugar de quantizar cada peso independientemente, ajusta los pesos no cuantizados de una capa para compensar el error introducido por los pesos cuantizados. Es más preciso que la cuantización lineal pero más lento de aplicar (el proceso de cuantización puede tardar horas para modelos grandes). Diseñado para inferencia.

**AWQ** (Activation-Aware Weight Quantization) identifica el ~1% de pesos "salientes" — aquellos que interactúan con activaciones de alta magnitud y que por tanto tienen mayor impacto en el output del modelo — y les aplica un factor de escala que los protege de la pérdida de información al cuantizar. El resto de los pesos se cuantizan normalmente. El resultado es que el modelo mantiene su "inteligencia" en los puntos más críticos aunque su footprint global sea pequeño. AWQ supera a GPTQ en calidad con tasas de compresión similares, pero como GPTQ, está orientado a inferencia.

**NF4** (el formato de QLoRA) es el único de esta lista diseñado específicamente para *fine-tuning*: está integrado con el flujo de gradientes de PyTorch y permite que los adaptadores LoRA reciban gradientes correctos mientras el modelo base permanece cuantizado. Para inferencia post-fine-tuning, lo habitual es exportar el modelo mergado a GGUF o AWQ para mayor compatibilidad y velocidad.

La regla de oro es esta: usa NF4 para entrenar, y luego convierte a GGUF/AWQ/GPTQ para servir, dependiendo de tu hardware de inferencia.

---

## El resultado: el muro de la VRAM como puerta de entrada

El muro de la VRAM que parecía infranqueable al inicio de este capítulo no ha desaparecido — la física del hardware no cambia por decreto. Lo que ha cambiado es que ahora tenemos las herramientas matemáticas e ingenieriles para negociar con ese muro de forma inteligente.

QLoRA demuestra que la restricción de memoria, cuando se aborda con rigor, se convierte en una palanca de innovación. NF4 prueba que no necesitas más bits — solo necesitas que tus bits estén en el lugar correcto, donde viven los datos. La double quantization muestra que aplicar la misma lógica un nivel más arriba recupera memoria adicional sin coste de calidad. El Paged Optimizer convierte los crashes de OOM en pausas gestionadas. Y los adaptadores LoRA en BF16 garantizan que todo el aprendizaje ocurre en alta precisión aunque el modelo base duerma comprimido.

El resultado práctico es que un modelo de 7B — con la capacidad de razonar, generar código, resumir documentos o mantener conversaciones complejas — ahora puede entrenarse con datos propios en una GPU de 24 GB que cuesta menos que un mes de suscripción a un servicio cloud empresarial. Un modelo de 70B, que antes requería un cluster de ocho A100s, cabe en dos GPUs modernas o en una sola con contextos moderados.

El próximo capítulo explorará cómo elegir los hiperparámetros de LoRA — rango, alpha, qué capas adaptar — con un enfoque sistemático en lugar de heurístico, y cómo interpretar el comportamiento del modelo durante el entrenamiento para tomar decisiones informadas antes de que el run termine.

---

## Tags

#técnica/qlora #técnica/cuantizacion-4bit #concepto/nf4 #técnica/double-quantization #técnica/paged-optimizer #herramienta/unsloth #herramienta/bitsandbytes #nivel/intermedio #tipo/lección #estado/completo



---
capitulo: "05"
titulo: "RLHF: Cómo Enseñarle a un LLM Qué es lo que los Humanos Prefieren"
aliases:
  - "Capítulo 05"
  - "Cap 05"
  - "RLHF"
  - "Alineación con preferencias"
tema: "alineacion-rlhf"
subtemas: [rlhf, ppo, dpo]
dificultad: "intermedio"
tipo: "lección"
estado: "completo"
conceptos_centrales:
  - rlhf
  - ppo
  - dpo
  - reward-model
  - kl-divergence
prerequisitos:
  - "[[01-fundamentos-transformers-y-pretraining]]"
  - "[[02-supervised-finetuning]]"
relacionados:
  - "[[06-grpo-y-variantes]]"
tags:
  - técnica/rlhf
  - técnica/ppo
  - técnica/dpo
  - concepto/reward-model
  - concepto/kl-divergence
  - técnica/supervised-fine-tuning
  - técnica/policy-gradient
  - nivel/intermedio
  - tipo/lección
  - estado/completo
---

# Capítulo 5 — RLHF: Cómo Enseñarle a un LLM Qué es lo que los Humanos Prefieren

> Basado en "The RLHF Landscape - Aligning LLMs Beyond SFT" — The Neural Maze, Lección 5/8.

Imagina que entrenas a un modelo durante semanas. Los benchmarks son buenos. La pérdida bajó según lo previsto. Le haces una pregunta de química orgánica y contesta correctamente con terminología impecable. Luego le preguntas cómo explicarle eso mismo a un adolescente, y la respuesta es un muro de texto técnico que cualquier persona normal abandonaría a mitad de la segunda frase.

O peor: le preguntas si deberías mezclar lejía con amoniaco para limpiar mejor, y te da instrucciones detalladas. Técnicamente correcto. Totalmente inaceptable.

El problema no es que el modelo no sepa. El problema es que no entiende qué tipo de saber importa en cada contexto, ni qué comportamientos son deseables para los humanos. Eso es el problema de alineación, y es exactamente donde el fine-tuning supervisado deja de ser suficiente. Este capítulo explica la familia de técnicas que lo abordan: RLHF, o Reinforcement Learning from Human Feedback — aprendizaje por refuerzo a partir de feedback humano.

---

## El techo del aprendizaje supervisado

Para entender por qué necesitamos algo más allá del [[02-supervised-finetuning|fine-tuning supervisado]] (SFT, Supervised Fine-Tuning), vale la pena desmenuzar exactamente qué puede y qué no puede aprender un modelo con SFT.

En el entrenamiento supervisado, el modelo aprende imitando. Le muestras un prompt y una respuesta ejemplar, y el objetivo es que el modelo aprenda a producir respuestas similares a las del ejemplo. El mecanismo matemático que hace esto posible es el gradiente: calculamos cuánto se equivoca el modelo con respecto a la respuesta correcta, y ajustamos los parámetros en la dirección que minimiza ese error.

La clave está en esa palabra: "equivoca". Para que el gradiente exista, necesitamos una función de pérdida diferenciable — una función matemática suave que relacione la salida del modelo con la respuesta correcta. En SFT esa función es la entropía cruzada: mide cuánta probabilidad asigna el modelo a los tokens correctos.

Pero ahora considera un escenario diferente. Un evaluador humano compara la respuesta A y la respuesta B del modelo, y dice simplemente: "Prefiero A". No hay una fórmula matemática que conecte ese juicio subjetivo con los parámetros del modelo. El evaluador humano es, desde la perspectiva del modelo, una caja negra: entra una respuesta, sale una preferencia, pero no hay ninguna función suave ni diferenciable en medio.

Aquí está el muro. SFT puede enseñarle al modelo a imitar ejemplos buenos. Pero no puede enseñarle a optimizar para preferencias humanas que no se pueden expresar como una función de pérdida diferenciable.

La solución viene de un campo completamente distinto: el aprendizaje por refuerzo.

---

## Una introducción honesta al aprendizaje por refuerzo

El aprendizaje por refuerzo (RL, Reinforcement Learning) tiene fama de ser difícil. En parte es merecida — puede ser notoriamente inestable — pero la intuición central es sorprendentemente simple. Y la mejor manera de entenderla es con una analogía que probablemente ya conoces.

Imagina que estás enseñándole a un perro a sentarse. Dices "¡Siéntate!" y esperas. Si el perro se sienta, le das un premio. Si no, no pasa nada y lo intentas de nuevo. Después de muchas repeticiones, el perro aprende que la acción de sentarse al escuchar esa orden le genera un resultado positivo. No le explicaste la anatomía del movimiento. No le mostraste videos de perros sentándose. Solo conectaste una acción con una consecuencia.

El aprendizaje por refuerzo funciona exactamente así, pero con un agente computacional en lugar del perro. Hay cinco componentes fundamentales que definen cualquier sistema de RL:

**Agente.** El aprendiz que toma decisiones. En nuestro caso, el propio modelo de lenguaje.

**Entorno.** El mundo con el que interactúa el agente. El entorno observa las acciones del agente y devuelve feedback. En videojuegos, el entorno es el juego en sí. Para LLMs, el entorno es más abstracto — lo exploraremos en un momento.

**Acción.** Lo que el agente puede hacer en cada paso. Un jugador de ajedrez mueve una pieza. Un robot mueve un brazo. Un modelo de lenguaje... elige el siguiente token.

**Recompensa.** Una señal escalar — un número — que le dice al agente qué tan bien lo hizo. Las recompensas positivas refuerzan las acciones buenas; las negativas las desincentivan. Esta señal de recompensa es lo que hace que el RL sea fundamentalmente diferente del aprendizaje supervisado: no necesitas decirle al agente qué hacer exactamente, solo evaluarle qué tan bien lo hizo.

**Política.** La estrategia que usa el agente para elegir acciones dado el estado actual del entorno. Es la función que queremos optimizar. Al principio puede ser aleatoria o muy básica. A través del entrenamiento, la política aprende a escoger las acciones que maximizan la recompensa acumulada.

El bucle de entrenamiento en RL tiene esta forma: el agente observa su estado actual, elige una acción siguiendo su política actual, recibe una recompensa del entorno, y actualiza su política para favorecer las acciones que trajeron mejores recompensas. Luego repite. Muchas veces.

La diferencia crucial con SFT es que no hay un dataset fijo de "respuestas correctas". El agente genera su propia experiencia a través de la exploración, y la calidad de esa experiencia depende de la política actual. Esto crea un bucle de retroalimentación que puede ser extraordinariamente potente, pero también inestable si no se maneja con cuidado — un tema al que volveremos repetidamente en este capítulo.

> **Descripción visual:** Diagrama de flujo circular horizontal con seis bloques de colores distintos conectados en ciclo. "Estado actual" en azul, "Política" en púrpura, "Acción" en naranja, "Entorno" en rojo, "Recompensa" en verde, "Actualizar política" en verde azulado. Las flechas forman un ciclo continuo que enfatiza la naturaleza iterativa del aprendizaje por refuerzo. Cada bloque tiene dos líneas de texto: el nombre del componente en negrita y una descripción en cursiva debajo. Fondo blanco, estilo minimalista.

### Por qué el RL puede aprender lo que SFT no puede

Volvamos al problema del gradiente. Cuando el evaluador humano dice "prefiero A", no tenemos una función diferenciable. Pero el RL tiene una solución elegante: los [[06-grpo-y-variantes|métodos de gradiente de política]] (policy gradient methods) no necesitan diferenciar a través del proceso que genera la recompensa. En cambio, usan la recompensa como un peso para ajustar los gradientes de la política misma.

El mecanismo conceptual es simple: las acciones que llevaron a recompensas altas se vuelven más probables en el futuro; las que llevaron a recompensas bajas, menos probables. No necesitamos saber cómo el evaluador llegó a su juicio. Solo necesitamos ese juicio — el número de recompensa — para ajustar la política en la dirección correcta.

Esto nos permite entrenar modelos con señales de feedback completamente arbitrarias, incluyendo preferencias humanas subjetivas. Esa es la apertura que RLHF explota.

---

## RLHF en el contexto del pipeline completo de LLMs

Antes de profundizar en los algoritmos específicos, necesitamos situar RLHF en el contexto del pipeline completo de entrenamiento de un LLM moderno. Porque RLHF no sustituye a las etapas anteriores — las complementa.

**Etapa 1: Preentrenamiento.** El modelo se entrena en cantidades masivas de texto con un objetivo autosupervisado: predecir el siguiente token. Esta etapa le da al modelo fluidez en el lenguaje, conocimiento del mundo, y capacidades de razonamiento básicas. El resultado es algo así como un motor de autocompletado extremadamente sofisticado. Puede generar texto coherente y factualmente correcto, pero no tiene ningún concepto de ser útil. No sabe qué comportamientos son deseables y cuáles no.

**Etapa 2: Fine-tuning supervisado (SFT).** El modelo preentrenado se ajusta sobre demostraciones curadas de comportamiento de asistente deseable. Esto le enseña a seguir instrucciones, producir estructuras específicas de respuesta, y adoptar un estilo útil. Es efectivo, pero limitado: solo puede enseñar mediante ejemplos. No puede optimizar para preferencias subjetivas que son difíciles de codificar en un dataset.

**Etapa 3: RLHF.** Aquí pasamos de "mostrarle al modelo qué decir" a "enseñarle qué es lo que los humanos prefieren". En lugar de proporcionar respuestas ideales, proporcionamos juicios comparativos. El modelo aprende a optimizar directamente para esas preferencias.

Esta tercera etapa es lo que transforma un generador de texto competente en un asistente alineado — un modelo que no solo sabe cómo hablar, sino que entiende qué tipo de respuestas los humanos realmente valoran.

> **Descripción visual:** Diagrama de flujo horizontal con tres bloques principales conectados por flechas de izquierda a derecha. El bloque "Pretraining" es azul intenso, "SFT" es naranja, "RLHF" es verde. De cada bloque cuelga una nota gris con fondo claro que describe lo que aporta esa etapa. Las flechas punteadas conectan cada etapa con su nota descriptiva. Estilo limpio, fondo blanco, tipografía sans-serif. La progresión visual de azul a naranja a verde refuerza la idea de etapas sucesivas hacia la alineación.

### Cómo el RL se mapea sobre la generación de texto

Hay una simetría elegante entre el framework de RL y la generación de texto en LLMs que vale la pena hacer explícita:

- **Política** → El modelo de lenguaje. Sus parámetros definen una distribución de probabilidad sobre el vocabulario de tokens dado un contexto. Esto es exactamente lo que es una política en RL: una función que mapea estados a distribuciones sobre acciones.

- **Estado** → La secuencia de texto actual: el prompt más los tokens generados hasta el momento. Cada vez que el modelo genera un token, el estado se actualiza.

- **Acción** → Elegir el siguiente token. Con vocabularios típicos de 32.000 a 128.000 tokens, el espacio de acciones es enorme.

- **Recompensa** → Una puntuación de calidad para la respuesta completa generada. Crucialmente, esta recompensa llega solo al final del episodio — después de que el modelo ha generado el último token de su respuesta.

- **Episodio** → Una generación completa: desde el primer token de la respuesta hasta el token de fin de secuencia.

Este mapeo tiene una implicación importante: a diferencia del ajedrez, donde el agente recibe feedback después de cada movimiento, en la generación de texto el modelo produce potencialmente cientos de tokens antes de recibir cualquier señal de recompensa. Esto hace que el problema del crédito — ¿qué tokens específicamente causaron que la respuesta fuera buena o mala? — sea particularmente difícil.

---

## El proceso RLHF paso a paso

Ahora que tenemos el framework claro, veamos cómo funciona RLHF en la práctica. El proceso tiene tres pasos que se ejecutan en secuencia.

### Paso 1: Recolectar preferencias humanas

Para un conjunto de prompts, el modelo SFT genera múltiples respuestas candidatas. Anotadores humanos las comparan en pares: "¿Es la respuesta A o la respuesta B mejor?"

Por ejemplo: el prompt es "Explica cómo funciona la fotosíntesis a un niño de 10 años". El modelo genera cuatro respuestas diferentes. Los anotadores comparan A vs B, A vs C, B vs D, y así sucesivamente, marcando cuál prefieren en cada par.

El resultado es un dataset de preferencias: pares (respuesta_ganadora, respuesta_perdedora) para cada prompt. Este dataset captura el juicio humano de forma escalable — una vez recogido, lo podemos usar muchas veces sin necesitar un humano para cada evaluación futura.

La calidad de este dataset es crítica. Si los anotadores tienen criterios inconsistentes — unos valoran la concisión, otros la exhaustividad, sin directrices claras — el ruido en los datos se propagará a todo lo que viene después. Volvemos a este punto más adelante.

### Paso 2: Entrenar un modelo de recompensa

Con el dataset de preferencias en mano, entrenamos un modelo separado — el reward model o modelo de recompensa — para predecir qué respuestas los humanos prefieren. Este modelo toma como entrada un prompt y una respuesta, y produce como salida un número escalar: una puntuación de calidad.

El modelo de recompensa aprende a asignar puntuaciones más altas a las respuestas que los humanos marcaron como preferidas, y puntuaciones más bajas a las rechazadas. Una vez entrenado, se convierte en un proxy escalable del juicio humano: puede evaluar miles de respuestas por segundo sin necesitar un humano para cada una.

La arquitectura típica parte del mismo modelo base (o uno similar), añade una cabeza de regresión lineal sobre el último token, y se entrena con una pérdida de ranking que maximiza la diferencia de puntuación entre la respuesta ganadora y la perdedora para cada par.

Una advertencia importante: el modelo de recompensa no es infalible. Solo ha visto el dataset de preferencias, que es finito y recogido bajo condiciones específicas. Si el modelo de lenguaje que estamos entrenando aprende a explotar los puntos ciegos del modelo de recompensa — produciendo respuestas que puntúan alto pero no son genuinamente buenas — el entrenamiento se descarrila. Por eso necesitamos el siguiente mecanismo.

### Paso 3: Optimizar la política con RL

Con el modelo de recompensa listo, comenzamos el entrenamiento de RL. El flujo es el siguiente:

1. El modelo de lenguaje (la política) recibe un prompt del dataset de entrenamiento.
2. Genera una respuesta completa de forma autorregresiva, token a token.
3. El modelo de recompensa puntúa esa respuesta.
4. Un algoritmo de RL usa esa puntuación para actualizar los parámetros de la política, haciéndola más probable de generar respuestas similares si la puntuación fue alta, y menos probable si fue baja.
5. Repetir.

El detalle crítico: en cada paso de entrenamiento, hay una penalización por divergencia KL — una medida de cuánto se ha alejado la política actual del modelo SFT original.

> **Descripción visual:** Diagrama de flujo horizontal dividido en tres subgrafos con fondo de colores suaves. El subgrafo izquierdo (azul claro) representa la recolección de preferencias humanas, con cuatro bloques conectados en secuencia. El subgrafo central (naranja claro) muestra el entrenamiento del Reward Model con dos bloques. El subgrafo derecho (verde claro) muestra el bucle de optimización RL con cuatro bloques en ciclo. Las conexiones entre subgrafos son flechas horizontales gruesas. Estilo profesional con texto blanco sobre bloques de colores sólidos.

Sin esta penalización, el modelo aprende rápidamente a "hackear" el modelo de recompensa: descubre que ciertas frases o estructuras obtienen puntuaciones altas independientemente de si la respuesta es genuinamente útil, y las repite hasta producir outputs degenerados. Con la penalización KL, el modelo tiene un incentivo para quedarse cerca de su baseline SFT, lo que limita este tipo de exploits.

La divergencia KL (Kullback-Leibler divergence) es una medida estadística de cuán diferente es una distribución de probabilidad de otra. En este contexto, mide cuánto ha cambiado la distribución de tokens que genera el modelo respecto al modelo SFT de referencia. Si la KL divergencia es alta, el modelo se ha alejado mucho de su baseline — señal de que está explotando el modelo de recompensa en lugar de mejorar genuinamente. Profundizamos en esto al final del capítulo.

---

## PPO: El algoritmo que entrenó a los primeros asistentes

PPO, o Proximal Policy Optimization (Optimización de Política Proximal), es el algoritmo de RL que impulsó InstructGPT — el trabajo que popularizó RLHF para LLMs y que fue el precursor directo de ChatGPT. Si hay un algoritmo que debes entender en este espacio, es este.

### El problema que PPO viene a resolver: REINFORCE y la varianza

Para entender PPO, necesitamos empezar un paso antes, con REINFORCE — el algoritmo de gradiente de política más simple que existe y el ancestro conceptual de PPO.

La idea de REINFORCE es directa: ejecuta la política actual durante un episodio completo, observa la recompensa total que obtuvo, y ajusta los parámetros para que las acciones que llevaron a esa recompensa sean más o menos probables en el futuro. El gradiente que guía esta actualización es:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau) \right]$$

Donde:
- $\theta$ son los parámetros de la política (los pesos del modelo de lenguaje).
- $\pi_\theta(a_t | s_t)$ es la probabilidad que la política actual asigna a la acción $a_t$ — en nuestro caso, la probabilidad de generar el token $a_t$ dado el contexto $s_t$.
- $R(\tau)$ es la recompensa total del episodio completo $\tau$ — la puntuación del modelo de recompensa para la respuesta entera.
- $\nabla_\theta \log \pi_\theta(a_t | s_t)$ es el gradiente del log-probability de la acción: la dirección en el espacio de parámetros que aumenta la probabilidad de esa acción.

En lenguaje llano: para cada token que el modelo generó en esa respuesta, calcula cuánto cambiarían los parámetros para hacer ese token más probable, y escala ese cambio por la recompensa total de la respuesta. Si la respuesta fue buena (recompensa alta), todos los tokens de esa respuesta se vuelven más probables. Si fue mala, menos probables.

El problema es severo: la recompensa $R(\tau)$ es una señal muy ruidosa. Imagina que el modelo generó una respuesta de 200 tokens y obtuvo una recompensa de 0.7 (en una escala de 0 a 1). El algoritmo sube la probabilidad de todos y cada uno de esos 200 tokens — pero ¿cuáles específicamente contribuyeron a la buena puntuación? Quizás los primeros 100 tokens fueron brillantes y los últimos 100 mediocres. REINFORCE no puede distinguirlo: trata toda la respuesta como igualmente responsable del resultado.

Este problema se llama alta varianza en el estimador del gradiente. Formalmente, la varianza es el grado en que las estimaciones del gradiente difieren de un batch de entrenamiento a otro. Cuando la varianza es alta, los pasos de gradiente apuntan en direcciones inconsistentes de una iteración a la siguiente, y el entrenamiento se vuelve errático e inestable.

Para un LLM generando respuestas de cientos de tokens, esta varianza puede ser catastrófica. Necesitamos algo más sofisticado.

### La función de ventaja: midiendo la calidad relativa

PPO introduce una mejora conceptual crucial: en lugar de ponderar cada acción por la recompensa total del episodio, la pondera por la **ventaja** — cuánto mejor fue esa acción específica comparada con lo que el modelo esperaba en ese punto.

La ventaja se define formalmente como:

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

Donde:
- $Q(s_t, a_t)$ es el valor Q — el retorno esperado de tomar la acción $a_t$ en el estado $s_t$ y continuar siguiendo la política. En términos de LLMs: "si genero este token aquí, ¿qué puntuación total espero obtener?".
- $V(s_t)$ es la función de valor — el retorno esperado desde el estado $s_t$ independientemente de qué acción se tome. Es la "línea de base" del modelo: "en este contexto, ¿qué puntuación espero obtener en promedio?".
- $A(s_t, a_t)$ es la ventaja: si es positiva, esta acción es mejor de lo esperado; si es negativa, es peor.

Pongamos un ejemplo concreto para que esto sea tangible. El modelo está generando una explicación de la fotosíntesis y ha llegado al punto donde necesita decidir el siguiente token. La línea de base del modelo para ese estado (función de valor) es 0.6 — en promedio, desde este punto, espera una puntuación de 0.6. Si genera la palabra "luz" como siguiente token y eventualmente obtiene una puntuación de 0.8, la ventaja de ese token es $0.8 - 0.6 = +0.2$: fue mejor de lo esperado. Si genera "complicado" y la puntuación final es 0.4, la ventaja es $0.4 - 0.6 = -0.2$: peor de lo esperado.

Usar la ventaja en lugar de la recompensa bruta reduce la varianza dramáticamente porque estamos midiendo calidad relativa, no absoluta. Dos respuestas que obtienen puntuaciones de 0.8 y 0.6 respectivamente pueden tener la misma estructura de ventajas internas si las expectativas del modelo eran distintas en cada caso.

En la práctica, la ventaja se estima usando una técnica llamada Estimación de Ventaja Generalizada (GAE, Generalized Advantage Estimation), que combina estimaciones de ventaja a corto plazo y largo plazo para balancear sesgo y varianza. No entraremos en los detalles matemáticos de GAE aquí, pero el concepto clave es que nos da una estimación más estable de qué tan buena fue cada acción individual dentro de un episodio.

Para calcular la función de valor $V(s_t)$, PPO mantiene una red separada llamada **el crítico** (critic). El crítico toma el estado actual como entrada y predice el retorno esperado. Es básicamente un modelo que aprende a evaluar "qué tan buena es la situación en este punto de la generación". Volveremos al crítico cuando hablemos del setup de cuatro modelos de PPO.

### El objetivo surrogate recortado: el seguro de velocidad de PPO

La ventaja resuelve el problema de la varianza. Pero PPO introduce una segunda innovación igualmente importante: el **objetivo surrogate recortado** (clipped surrogate objective). Este es el mecanismo que hace que PPO sea "proximal" — que mantenga las actualizaciones dentro de una región segura.

Para entender por qué es necesario, considera este escenario: el modelo ve un batch de ejemplos donde generar una cierta frase produce una ventaja positiva muy alta. Sin ningún límite, el gradiente podría actualizar los pesos agresivamente para hacer esa frase mucho más probable — quizás triplicando o cuadruplicando su probabilidad en una sola iteración. Pero el modelo de recompensa solo vio esa frase en el contexto de los ejemplos de entrenamiento actuales. Si el modelo la hace muchísimo más probable de forma generalizada, probablemente empiece a usarla en contextos donde no es apropiada, destruyendo el comportamiento aprendido previamente.

PPO limita cuánto puede cambiar la política en un solo paso de entrenamiento. Lo hace a través del **ratio de probabilidad** entre la política nueva (después del update) y la política antigua (antes del update):

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

Este ratio mide cuánto ha cambiado la probabilidad de una acción específica. Si $r_t = 1$, la probabilidad no cambió. Si $r_t = 2$, se duplicó. Si $r_t = 0.5$, se redujo a la mitad.

El objetivo PPO es:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \right) \right]$$

Donde:
- $A_t$ es la ventaja estimada en el paso $t$.
- $\varepsilon$ (epsilon) es el parámetro de recorte, típicamente $\varepsilon = 0.2$.
- $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ recorta el ratio para que nunca sea menor que $0.8$ ni mayor que $1.2$.
- $\min(\cdot, \cdot)$ toma el menor de los dos términos, lo que crea un techo conservador.

Trabajemos esto con números reales para que la mecánica quede clara.

**Escenario 1 — ventaja positiva, update excesivo.** El modelo quería aumentar la probabilidad de un token. Antes del update, $\pi_{\theta_{\text{old}}}(a_t|s_t) = 0.10$. Después de calcular el gradiente, $\pi_\theta(a_t|s_t) = 0.25$. El ratio es $r_t = 0.25 / 0.10 = 2.5$. La ventaja es $A_t = +0.8$.

Sin recorte, el término sería $2.5 \times 0.8 = 2.0$.

Con recorte, el ratio se limita a $1.2$ (ya que $\varepsilon = 0.2$), dando $1.2 \times 0.8 = 0.96$.

El objetivo toma el mínimo de $2.0$ y $0.96$, que es $0.96$. El gradiente se calcula sobre este valor recortado, no sobre el valor sin recortar. El resultado: el update se produce, pero de forma mucho más moderada.

**Escenario 2 — ventaja negativa, update excesivo en la otra dirección.** El modelo quería reducir la probabilidad de un token malo. Antes: $0.30$. Después: $0.05$. Ratio: $r_t = 0.05 / 0.30 = 0.167$. Ventaja: $A_t = -0.6$.

Sin recorte: $0.167 \times (-0.6) = -0.1$.

Con recorte: el ratio se limita a $0.8$ (el límite inferior), dando $0.8 \times (-0.6) = -0.48$.

El objetivo toma el mínimo de $-0.1$ y $-0.48$, que es $-0.48$. Esto también recorta, pero en la dirección contraria.

> **Descripción visual:** Diagrama de flujo horizontal con bifurcación central. El bloque de entrada "Ratio rt" es azul. El rombo de decisión "¿Fuera del rango?" es naranja con dos salidas: la rama inferior verde ("Objetivo sin recortar", update normal) y la rama superior roja ("Objetivo recortado", update excesivo que se frena). Ambas ramas convergen en un bloque verde azulado "min de ambos" que selecciona el más conservador. La salida final "Trust region / Política estable" es verde azulado. Las flechas muestran claramente cómo el clipping evita actualizaciones destructivas. Estilo técnico con colores semáforo (verde=seguro, rojo=peligroso).

El efecto neto es que PPO crea una **trust region** — una región de confianza — alrededor de la política actual. Los updates que llevarían la política fuera de esa región (más allá del factor $1 \pm \varepsilon$) se recortan. El gradiente queda efectivamente a cero para esos updates, lo que significa que la política no puede hacer cambios catastróficos en un solo batch.

¿Qué pasa si eliges $\varepsilon$ demasiado pequeño, como $0.05$? El modelo aprende con extrema lentitud. Cada iteración solo puede mover la política una cantidad ínfima, y necesitas diez veces más pasos para converger. ¿Y si lo pones en $0.5$? Vuelves al problema original: updates tan grandes que el entrenamiento se desestabiliza. El valor de $0.2$ es el consenso empírico — ofrece un buen equilibrio entre velocidad de convergencia y estabilidad, y ha funcionado bien en una gran variedad de experimentos.

### El setup de cuatro modelos: por qué PPO es caro

Aquí llegamos a la parte que hace que los equipos con presupuestos limitados se pongan nerviosos. Un entrenamiento PPO completo para RLHF requiere cuatro modelos simultáneos en memoria:

**Actor (la política).** El modelo de lenguaje que estamos optimizando. Este es el que actualiza sus pesos durante el entrenamiento. Para un modelo de 7B parámetros en precisión BFloat16, esto ocupa aproximadamente 14 GB de VRAM.

**Crítico (la red de valor).** Un modelo separado que estima la función de valor $V(s_t)$ — el retorno esperado desde cada estado. Típicamente tiene la misma arquitectura que el actor, aunque a veces se usa una versión más pequeña. En la práctica, muchos frameworks inicializan el crítico desde el mismo checkpoint que el actor y lo entrenan en paralelo. Otro bloque de ~14 GB.

**Modelo de recompensa.** El modelo entrenado en el paso 2 sobre las preferencias humanas. Permanece congelado durante el entrenamiento RL — sus pesos no se actualizan. Su único rol es puntuar las respuestas del actor. Otros ~14 GB (asumiendo mismo tamaño).

**Modelo de referencia.** Una copia congelada del modelo SFT. Se usa exclusivamente para calcular la penalización por divergencia KL: en cada paso, calculamos la KL entre la distribución del actor actual y la del modelo de referencia, y añadimos esa penalización al objetivo para evitar que el actor derive demasiado. Otros ~14 GB.

Total para un modelo de 7B: aproximadamente 56 GB de VRAM solo para los pesos. A eso hay que añadir los gradientes, los estados del optimizador, las activaciones durante el forward pass, los tokens generados... En la práctica, para un modelo de 7B necesitas uno o varios nodos A100 de 80 GB.

Para un modelo de 70B, la matemática escala brutalmente: cuatro modelos de ~140 GB cada uno. Necesitas un clúster de varias GPUs solo para tener los pesos en memoria, más toda la infraestructura de generación autorregresiva en cada paso de entrenamiento.

> **Descripción visual:** Diagrama de flujo con dos subgrafos diferenciados. El subgrafo izquierdo azul ("Modelos actualizables") contiene el Actor y el Crítico en azul intenso. El subgrafo derecho gris ("Modelos congelados") contiene el Reward Model y el Modelo de referencia en gris. El flujo parte del Prompt en verde azulado, pasa por el Actor que genera una respuesta, y esa respuesta es evaluada por los tres modelos de soporte. Las evaluaciones convergen en el cálculo de ventaja (naranja) que actualiza el Actor. Las flechas muestran el ciclo de entrenamiento on-policy. Leyenda visual: azul = activo, gris = congelado.

Esto no significa que PPO sea impracticable. Funciona, y funciona bien. Pero el costo de infraestructura es significativo, y es la razón principal por la que DPO — que presentamos a continuación — ha ganado tanto terreno.

### Cuándo usar PPO

PPO brilla cuando necesitas máxima calidad de alineación y tienes el compute para soportarlo. Su naturaleza on-policy — el hecho de que los datos de entrenamiento siempre reflejan el comportamiento actual del modelo — crea un bucle de retroalimentación ajustado y autocorrector. El modelo explora, recibe feedback, ajusta, explora de nuevo. Con suficientes iteraciones, puede mejorar más allá de lo que cualquier dataset estático de preferencias puede capturar.

Si estás entrenando un modelo frontier donde la calidad de alineación es la prioridad absoluta, PPO es la elección probada. Si estás trabajando con un presupuesto de compute limitado o necesitas iteraciones rápidas, sigue leyendo.

---

## DPO: Alineación sin el bucle de RL

DPO (Direct Preference Optimization — Optimización Directa de Preferencias) llegó en 2023 con una pregunta provocadora: ¿y si pudiéramos obtener la alineación de RLHF sin necesitar RL en absoluto?

La respuesta resultó ser sí. Y cambió la conversación práctica en el campo.

### La intuición matemática detrás de DPO

Recuerda el objetivo que PPO está optimizando: maximizar la recompensa esperada (según el modelo de recompensa) mientras se mantiene cerca del modelo SFT de referencia (a través de la penalización KL). Formalmente:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) \right] - \beta \cdot \mathbb{KL}\left[ \pi(y|x) \, \| \, \pi_{\text{ref}}(y|x) \right]$$

Donde $r(x, y)$ es la puntuación del modelo de recompensa para la respuesta $y$ al prompt $x$, y $\beta$ es un coeficiente que controla cuánto peso damos a la penalización KL.

Los autores de DPO se preguntaron: ¿podemos resolver este problema analíticamente? Es decir, ¿podemos encontrar la política óptima $\pi^*$ en forma cerrada, sin necesitar un proceso iterativo de RL?

La respuesta es sí. La política óptima tiene la forma:

$$\pi^*(y|x) = \frac{1}{Z(x)} \cdot \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x,y)}{\beta}\right)$$

Donde $Z(x)$ es una constante de normalización.

Lo que esto dice: la política óptima es la política de referencia, reescalada por la exponencial de la recompensa dividida por $\beta$. Las respuestas con alta recompensa reciben más peso; las de baja recompensa, menos. El coeficiente $\beta$ controla cuánto se amplifica esa diferencia.

Hasta aquí parece un resultado teórico interesante pero no directamente útil, porque aún necesitamos el modelo de recompensa $r(x,y)$ para evaluar esa fórmula. Pero los autores fueron un paso más lejos: invirtieron la relación. Si sabemos cómo es la política óptima, podemos expresar la función de recompensa en términos de la política:

$$r(x,y) = \beta \cdot \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \cdot \log Z(x)$$

Y dado que los humanos usan las recompensas para expresar preferencias, podemos sustituir este modelo de recompensa implícito directamente en el objetivo de preferencias. El resultado es la función de pérdida DPO:

$$L_{DPO}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \cdot \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Donde:
- $x$ es el prompt.
- $y_w$ es la respuesta ganadora (preferred response) — la que el humano prefirió.
- $y_l$ es la respuesta perdedora (rejected response) — la que el humano rechazó.
- $\pi_\theta$ es la política que estamos entrenando (el modelo cuyo pesos actualizamos).
- $\pi_{\text{ref}}$ es la política de referencia congelada (el modelo SFT original, cuyos pesos no cambian).
- $\beta$ es un coeficiente (típicamente entre 0.1 y 0.5) que controla cuánto puede desviarse la política entrenada de la referencia. Un $\beta$ alto mantiene el modelo cerca del baseline; un $\beta$ bajo le da más libertad.
- $\sigma$ es la función sigmoide, que mapea cualquier número real a un valor entre 0 y 1.

### Desglose de la pérdida DPO con números concretos

Para que esta fórmula no sea solo símbolos, trabajémosla con un ejemplo. Tienes el prompt "Explica qué es una neurona". La respuesta ganadora $y_w$ es "Una neurona es la célula básica del sistema nervioso. Recibe señales de otras neuronas y, si son suficientemente fuertes, genera un impulso eléctrico que transmite a sus vecinas." La respuesta perdedora $y_l$ es "Las neuronas son unidades computacionales biológicas que procesan señales a través de mecanismos electroquímicos transmembrana."

Supón que el modelo de referencia $\pi_{\text{ref}}$ asigna log-probabilidades de $-15$ a $y_w$ y $-18$ a $y_l$ (los modelos base tienden a preferir levemente el lenguaje más técnico). El modelo que estamos entrenando $\pi_\theta$ actualmente asigna $-14$ a $y_w$ y $-19$ a $y_l$.

Los log-ratios (diferencias entre log-probabilidades de modelo entrenado y referencia) son:

$$\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} = -14 - (-15) = +1.0$$

$$\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} = -19 - (-18) = -1.0$$

Con $\beta = 0.3$, el argumento de la sigmoide es:

$$\beta \cdot (1.0 - (-1.0)) = 0.3 \cdot 2.0 = 0.6$$

Y $\sigma(0.6) \approx 0.646$. La pérdida para este ejemplo es $-\log(0.646) \approx 0.437$.

Si el modelo estuviera perfectamente alineado — asignando mucho más log-probability a $y_w$ que a $y_l$ — el argumento de la sigmoide sería grande y positivo, $\sigma$ estaría cerca de 1, y $-\log(\sigma)$ estaría cerca de 0: pérdida mínima. El gradiente empuja al modelo en exactamente esa dirección: aumentar el log-ratio de $y_w$ relativo a la referencia, y disminuir el de $y_l$.

En lenguaje llano: para cada par de preferencias, haz que la respuesta ganadora sea más probable que la referencia, y la perdedora menos probable. Eso es DPO.

### Por qué esto es revolucionario en la práctica

El impacto práctico de esta reformulación es enorme. Compara el setup de PPO con el de DPO:

| Aspecto | PPO | DPO |
|---|---|---|
| Modelos en memoria | 4 (actor, crítico, reward model, referencia) | 2 (política + referencia) |
| Generación durante entrenamiento | Sí, en cada paso | No |
| Velocidad de entrenamiento | Lenta (similar a inferencia repetida) | Rápida (similar a SFT) |
| Infraestructura necesaria | Alta | Media |
| Hiperparámetros críticos | Muchos ($\varepsilon$, coef. KL, tasa de aprendizaje del crítico...) | Pocos (principalmente $\beta$) |
| Tipo de datos necesarios | Prompts (genera durante entrenamiento) | Dataset de preferencias estático |

> **Descripción visual:** Diagrama comparativo lado a lado con dos subgrafos de colores contrastantes. El subgrafo izquierdo azul ("PPO — On-policy") muestra cinco bloques conectados con un ciclo de retroalimentación entre los pasos 3 y 5, enfatizando la naturaleza iterativa. El subgrafo derecho verde ("DPO — Off-policy") muestra cuatro bloques en secuencia lineal sin bucle, comunicando la simplicidad del flujo. Las etiquetas en las cajas son cortas y descriptivas. El contraste azul/verde y la presencia/ausencia del ciclo son el mensaje visual clave. Fondo blanco, tipografía sans-serif, estilo técnico limpio.

DPO no requiere generar texto durante el entrenamiento — simplemente calcula las log-probabilidades de respuestas que ya existen en el dataset. Esto hace que el entrenamiento sea órdenes de magnitud más rápido: se parece más a un fine-tuning supervisado que a un bucle de RL completo.

Para un ingeniero con un par de GPUs A100, DPO es la diferencia entre poder experimentar con alineación de modelos y no poder hacerlo en absoluto.

### Las limitaciones que no debes ignorar

DPO no es un almuerzo gratis. Tiene limitaciones reales que importan en producción.

**Es un método off-policy.** DPO aprende de un dataset de preferencias estático — recolectado en algún punto del pasado, con alguna versión del modelo. Si la distribución del modelo que estás entrenando se desvía significativamente de la distribución con la que se recolectaron las preferencias, la calidad del aprendizaje se degrada. El modelo que generó $y_w$ y $y_l$ puede ser muy diferente al modelo que estás entrenando ahora, y DPO no tiene mecanismo para corregir esto.

Contrasta esto con PPO: al ser on-policy, siempre genera respuestas con la política actual, obtiene feedback sobre esas respuestas, y aprende directamente de esa experiencia. El dataset de entrenamiento se actualiza en cada iteración.

**Overfitting más rápido.** Estudios empíricos han encontrado que DPO tiende a sobreajustarse al dataset de preferencias más rápido que PPO. Cuando el modelo ha memorizado los patrones del dataset, empieza a optimizar para los tokens específicos de las respuestas ganadoras en lugar de generalizar las preferencias humanas subyacentes.

**Techo de calidad inferior.** En benchmarks de alineación difíciles — tareas que requieren que el modelo explore más allá de lo que el dataset de preferencias captura — PPO consistentemente supera a DPO cuando el compute no es un factor limitante. DPO cambia algo de calidad máxima de alineación por una enorme reducción en costo y complejidad.

La elección entre PPO y DPO no es sobre cuál es "mejor" en abstracto. Es sobre qué trade-offs son aceptables dado tu contexto.

---

## El ecosistema más amplio: otros algoritmos que debes conocer

PPO y DPO son los protagonistas, pero el espacio RLHF está activo y nuevas variantes aparecen con frecuencia. Aquí está un tour de los más relevantes:

**REINFORCE / Gradientes de política básicos.** El ancestro de PPO que ya describimos. Simple de implementar y útil para construir intuición. Raramente usado directamente en producción para LLMs debido a su alta varianza, pero reaparece en variantes modernas como GRPO (que cubriremos en el siguiente capítulo).

**IPO (Identity Preference Optimization).** Una variante de DPO con una función de pérdida modificada que es más robusta al overfitting. IPO añade una regularización explícita que penaliza cuando el ratio entre la respuesta ganadora y la perdedora crece demasiado. Si estás usando DPO y ves que el modelo sobreajusta tu dataset de preferencias rápidamente, IPO es el primer lugar donde mirar.

**KTO (Kahneman-Tversky Optimization).** Un enfoque interesante que rompe con el paradigma de comparaciones en pares. En lugar de necesitar pares (buena respuesta, mala respuesta) para el mismo prompt, KTO solo necesita saber si una respuesta individual es "buena" o "mala" — sin necesitar una comparación directa. Esto simplifica enormemente la recolección de datos: cualquier conjunto de respuestas etiquetadas con thumbs up / thumbs down sirve. La contrapartida es que la señal de aprendizaje es más débil, ya que no hay contexto comparativo.

**Rejection Sampling / Best-of-N.** Genera N respuestas candidatas para cada prompt, puntúalas con el modelo de recompensa, y fine-tunea el modelo sobre las mejores. Conceptualmente simple y sorprendentemente efectivo. No requiere RL en absoluto. La desventaja es que escala linealmente con N — generar 10 respuestas por prompt es 10 veces más caro que generar 1. Útil como complemento a PPO o como paso intermedio cuando aún no tienes un pipeline de RL completo.

**RLAIF (RL from AI Feedback).** En lugar de anotadores humanos, usa un LLM más grande como juez para generar las preferencias. Reduce dramáticamente el costo y el tiempo de recolección de datos. El riesgo es que el modelo juez tiene sus propios sesgos, que se propagan al modelo que estás entrenando. En la práctica, RLAIF funciona sorprendentemente bien para muchas tareas, y la combinación de RLHF (para calibrar el juez inicial) + RLAIF (para escalar) es cada vez más común.

---

## La KL divergencia: el guardián silencioso de la alineación

Hemos mencionado la penalización por divergencia KL varias veces. Merece una explicación más profunda porque es posiblemente el mecanismo más subestimado del proceso RLHF.

La divergencia KL entre dos distribuciones de probabilidad $P$ y $Q$ se define como:

$$\mathbb{KL}[P \| Q] = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Intuitivamente, mide cuánta información se pierde cuando usamos $Q$ para aproximar $P$. Si $P = Q$, la KL es cero. Mientras más diferentes sean, mayor la KL.

En RLHF, la penalización KL mide cuánto ha cambiado la distribución del modelo de lenguaje respecto al modelo SFT de referencia. En cada paso de entrenamiento, el objetivo incluye:

$$\text{objetivo total} = \mathbb{E}[r(x,y)] - \beta \cdot \mathbb{KL}[\pi_\theta(y|x) \, \| \, \pi_{\text{ref}}(y|x)]$$

El primer término quiere maximizar la recompensa. El segundo término penaliza el alejamiento del modelo de referencia. El coeficiente $\beta$ controla el equilibrio entre estos dos objetivos.

¿Por qué es esto tan importante? Considera lo que pasa sin esta penalización. El modelo de recompensa, aunque fue entrenado cuidadosamente, es imperfecto. Solo vio un subset de posibles respuestas durante su entrenamiento, y hay inevitablemente regiones del espacio de respuestas que no exploró. Un modelo de lenguaje optimizando únicamente por recompensa aprenderá a explotar esas regiones — encontrará patrones de texto que obtienen puntuaciones altas del modelo de recompensa pero que son inútiles o incluso perjudiciales para usuarios reales.

Los ejemplos documentados en la literatura son ilustrativos: modelos que aprenden a generar listas de bullet points vacías que el modelo de recompensa puntúa bien por su apariencia de estructura, o modelos que generan respuestas extremadamente largas porque el modelo de recompensa fue entrenado con sesgo hacia la longitud. Sin la penalización KL, el modelo optimiza el proxy (el modelo de recompensa) en lugar del objetivo real (ser genuinamente útil).

La penalización KL actúa como un ancla. Dice: "puedes mejorar respecto al modelo SFT, pero no puedes alejarte tanto que empieces a generar comportamientos completamente distintos a lo que el modelo base haría". Mantiene al modelo dentro de una región del espacio de comportamientos que fue validada durante el preentrenamiento y el SFT.

El valor de $\beta$ importa enormemente. Un $\beta$ alto (por ejemplo, 1.0) mantiene el modelo muy cerca del baseline — aprende despacio y la mejora de alineación es modesta. Un $\beta$ bajo (por ejemplo, 0.01) le da al modelo mucha más libertad para optimizar la recompensa — puede mejorar rápidamente, pero también puede desestabilizarse y explotar el modelo de recompensa con más facilidad. En DPO, los valores típicos de $\beta$ están entre 0.1 y 0.5. En PPO, el coeficiente KL suele ser aún más pequeño porque hay otros mecanismos de estabilización (el clipping del surrogate objective).

---

## Guía práctica: decisiones que importan en el entrenamiento RLHF

Antes de cerrar el capítulo, vale la pena repasar los aspectos prácticos que cortan transversalmente todos los algoritmos y que nadie te cuenta hasta que cometes los errores.

### La calidad de los datos de preferencia es el factor que más importa

Es fácil quedar hipnotizado por la elegancia de PPO o la eficiencia de DPO y olvidar que ambos son completamente dependientes de la calidad de los datos de preferencia que los alimentan.

Un dataset de preferencias ruidoso produce un modelo de recompensa ruidoso, y un modelo de recompensa ruidoso produce una política mal alineada — independientemente de si usas PPO, DPO, o cualquier otra cosa. El garbage in, garbage out nunca ha sido más verdad que aquí.

Los problemas más comunes en datasets de preferencias:

**Inconsistencia entre anotadores.** Si un anotador valora la concisión y otro valora la exhaustividad, y no hay directrices claras que los alineen, el modelo de recompensa aprenderá señales contradictorias. Invertir en guías de anotación detalladas y en métricas de acuerdo entre anotadores (como el coeficiente kappa de Cohen) vale más que añadir más datos ruidosos.

**Sesgo de posición.** Los anotadores humanos tienen tendencia a preferir la primera respuesta que ven, o la más larga, o la más confiada en tono — independientemente del contenido real. Rotar el orden de las respuestas en las comparaciones y contrabalancear el diseño del experimento mitiga esto.

**Prompts mal seleccionados.** Si todos los prompts de entrenamiento son similares, el modelo aprendió preferencias muy localizadas. La diversidad de prompts — en dominio, dificultad, estilo de pregunta — determina la generalización de la alineación resultante.

### Métricas para vigilar durante el entrenamiento RLHF

Si estás ejecutando PPO, hay señales clave que indican si el entrenamiento va bien o está a punto de descarrilarse:

**Recompensa media.** Debe aumentar gradualmente. Si se estanca, el modelo no está aprendiendo. Si aumenta demasiado rápido, probablemente está explotando el modelo de recompensa.

**Divergencia KL respecto al modelo de referencia.** Debe mantenerse dentro de un rango controlado. Si crece sin control, el modelo está derivando demasiado. Ajusta el coeficiente $\beta$ al alza o revisa el $\varepsilon$ del clipping.

**Ratio de clipping (PPO).** El porcentaje de actualizaciones que fueron recortadas por el mecanismo de clipping. Si es demasiado alto (>30%), el modelo intenta hacer updates muy grandes consistentemente — considera reducir la tasa de aprendizaje. Si es demasiado bajo (<5%), el clipping no está siendo necesario y puedes ser más agresivo.

**Entropía de la política.** La entropía mide cuán diversas son las distribuciones de tokens que genera el modelo. Si la entropía colapsa, el modelo se ha vuelto demasiado determinístico — está generando siempre los mismos tipos de respuestas. Un poco de entropía es deseable para mantener diversidad en las respuestas.

Para DPO, el indicador principal es la **accuracy en el dataset de preferencias**: ¿con qué frecuencia el modelo asigna mayor log-probability a $y_w$ que a $y_l$? Debe crecer durante el entrenamiento y estabilizarse. Si llega al 100% muy rápido, probablemente hay overfitting — considera añadir regularización o usar IPO en su lugar.

### El modelo de recompensa no es el objetivo final

Para los métodos que usan un modelo de recompensa explícito (PPO, rejection sampling), es fundamental recordar que el modelo de recompensa es un proxy — una aproximación imperfecta del objetivo real. Optimizar el modelo de recompensa no es lo mismo que optimizar la alineación real.

Las áreas de investigación activa incluyen ensembles de modelos de recompensa (usar varios modelos de recompensa y promediar sus puntuaciones para reducir la explotación de cualquier modelo individual), modelos de recompensa basados en procesos (que puntúan los pasos intermedios del razonamiento, no solo la respuesta final), y refinamiento iterativo del modelo de recompensa (reentrenarlo periódicamente con preferencias actualizadas sobre las respuestas del modelo actualmente entrenado).

---

## De la teoría al mapa de decisiones

Con todo este contexto, la decisión práctica entre PPO y DPO — y cuándo considerar las alternativas — se simplifica bastante.

Elige PPO cuando:
- Tienes acceso a un clúster de GPUs con suficiente VRAM para los cuatro modelos.
- La calidad de alineación es la métrica crítica y no puedes permitirte sacrificarla.
- Tu tarea requiere que el modelo explore respuestas que no están representadas en ningún dataset de preferencias estático.
- Estás entrenando un modelo frontier que va a servir a millones de usuarios.

Elige DPO cuando:
- Tienes presupuesto de compute limitado o quieres iteraciones rápidas.
- Ya dispones de un dataset de preferencias de buena calidad.
- Estás empezando a experimentar con alineación y necesitas un punto de partida manejable.
- El modelo es relativamente pequeño (7B-13B) y el gap de calidad frente a PPO es aceptable.

Un patrón que muchos equipos adoptan en producción es un enfoque escalonado: primero DPO para una alineación rápida y económica, luego PPO si la evaluación muestra que el modelo no ha alcanzado el nivel de calidad requerido. Esto optimiza tanto el tiempo de iteración como el uso de recursos.

---

## Cierre: la alineación es un pipeline, no un algoritmo

RLHF ha pasado de ser una técnica de investigación de nicho a convertirse en un componente estándar del pipeline de producción de LLMs. Pero el mayor error que puedes cometer al aplicarlo es tratarlo como un paso único con una solución técnica única.

La alineación es un pipeline con múltiples pasos interdependientes: recolección de preferencias humanas cuidadosa, entrenamiento de un modelo de recompensa que generalice bien, y optimización de la política con el algoritmo correcto para tu contexto. La calidad de cada paso determina el techo de calidad del siguiente.

PPO ha demostrado ser la elección probada para máxima calidad de alineación, con su naturaleza on-policy creando un bucle de retroalimentación que se ajusta continuamente al comportamiento actual del modelo. DPO ha hecho la alineación fuerte accesible a un espectro mucho más amplio de practicantes, colapsando la complejidad del RL en un flujo de trabajo supervisado. Y las variantes más recientes — IPO, KTO, RLAIF — continúan expandiendo el toolkit disponible.

En el siguiente capítulo, exploraremos GRPO (Group Relative Policy Optimization), un enfoque más reciente que replantea cómo comparamos y puntuamos las salidas del modelo durante el entrenamiento — y que está ganando terreno rápidamente como alternativa a PPO en escenarios donde el razonamiento estructurado importa.

---

## Tags

#técnica/rlhf #técnica/ppo #técnica/dpo #concepto/reward-model #concepto/kl-divergence #técnica/supervised-fine-tuning #técnica/policy-gradient #nivel/intermedio #tipo/lección #estado/completo



---
capitulo: 6
titulo: "GRPO: Optimización de Políticas sin Crítico"
aliases:
  - "Capítulo 6"
  - "Cap 6"
  - "GRPO"
  - "Group Relative Policy Optimization"
tema: "técnica-rl"
subtemas: [grpo, policy-gradient, importance-sampling]
dificultad: "intermedio"
tipo: "lección"
estado: "completo"
conceptos_centrales:
  - grpo
  - group-relative-policy-optimization
  - dapo
  - gspo
  - dr-grpo
  - importance-sampling
  - policy-gradient
prerequisitos:
  - "[[05-rlhf-alineacion-llms]]"
relacionados: []
tags:
  - técnica/grpo
  - técnica/policy-gradient
  - concepto/importance-sampling
  - técnica/dapo
  - técnica/gspo
  - técnica/dr-grpo
  - concepto/reward-model
  - técnica/ppo
  - nivel/intermedio
  - tipo/lección
  - estado/completo
---

# Capítulo 6 — GRPO: Optimización de Políticas sin Crítico

> Basado en "The RL Algorithm Behind DeepSeek's Reasoning Models", The Neural Maze, Lección 6/8.

Imagina que has decidido entrenar un modelo de razonamiento matemático con reinforcement learning. Tienes la intuición correcta, el dataset correcto y, si has seguido los capítulos anteriores, probablemente PPO en mente como el algoritmo de referencia. Entonces enciendes el experimento, monitoreas el uso de memoria y ves que tus GPUs están al 95% de VRAM... antes de haber procesado un solo batch de entrenamiento. El culpable no es el modelo de lenguaje. Es el modelo de valor, ese segundo modelo neuronal del tamaño del primero que PPO necesita para funcionar. Y aquí es donde la historia de GRPO comienza.

---

## El cuello de botella de PPO

Para entender por qué GRPO existe, primero hay que entender exactamente qué dolor resuelve. [[05-rlhf-alineacion-llms|PPO]] — Proximal Policy Optimization — es el algoritmo estándar de RL online para alinear LLMs, y en teoría funciona muy bien. En la práctica, entrenar con PPO a escala requiere mantener cuatro modelos simultáneamente en memoria:

1. El **modelo de política activo** (actor): el LLM que estamos optimizando.
2. El **modelo de referencia** (reference model): una copia congelada del modelo base que actúa como ancla para evitar que el modelo se aleje demasiado de su comportamiento original.
3. El **[[05-rlhf-alineacion-llms|modelo de recompensa]]** (reward model): una red entrenada separadamente para puntuar la calidad de las respuestas.
4. El **modelo de valor** (critic): una red neuronal que estima, para cada estado del proceso de generación, cuánta recompensa futura podemos esperar a partir de ese punto.

El modelo de política necesita gradientes activos, así que ocupa toda su memoria de activaciones durante el forward y backward pass. El modelo de referencia y el modelo de recompensa necesitan al menos sus pesos cargados para hacer inferencia. Pero es el modelo de valor — el crítico — el que realmente rompe el presupuesto: es típicamente una red del mismo orden de magnitud que el modelo de política, y necesita sus propios gradientes para actualizarse en paralelo.

En números concretos: si estás entrenando un modelo de 7B parámetros en bfloat16, eso son aproximadamente 14 GB solo para los pesos. Con gradientes y estados del optimizador (Adam usa dos momentos adicionales por parámetro), el actor necesita en torno a 56 GB. El crítico añade otros 14 GB de pesos más su propio overhead de entrenamiento. El modelo de referencia son otros 14 GB. El reward model, otros 14 GB. Estamos hablando de más de 100 GB de VRAM para un modelo de solo 7B parámetros — y eso asumiendo que lo distribuyes perfectamente entre GPUs A100 de 80 GB cada una.

Pero el problema no es solo de memoria. Hay una fricción arquitectónica más profunda en usar un crítico con LLMs. En RL clásico — piensa en un agente aprendiendo a jugar un videojuego — el agente recibe retroalimentación en cada paso: mueve el personaje a la izquierda, gana 10 puntos; cae en un hoyo, pierde una vida. La función de valor aprende a estimar el retorno futuro porque tiene un flujo denso de señal: miles de transiciones por segundo, cada una con su recompensa inmediata.

El post-entrenamiento de LLMs no funciona así. El modelo genera una cadena de tokens — digamos, 500 tokens de razonamiento matemático — y solo al final recibe una señal binaria: ¿la respuesta final fue correcta o incorrecta? Eso significa que el crítico debe aprender a asignar una estimación de valor a cada uno de esos 500 tokens intermedios, a pesar de que ninguno de ellos tiene una recompensa propia. El token número 237 ("por lo tanto") no tiene inherentemente una recompensa: su "valor" depende de si contribuye a una respuesta correcta 263 tokens más adelante. Entrenar un crítico para hacer esa extrapolación con precisión, a lo largo de secuencias de razonamiento largas y complejas, es tanto difícil como dispendioso.

Frente a este problema, los investigadores de DeepSeek se hicieron una pregunta que parece radical a primera vista: ¿y si eliminamos el crítico por completo?

---

## GRPO: La idea del grupo como línea base

Group Relative Policy Optimization — GRPO, pronunciado "grupo R-P-O" — es el algoritmo que DeepSeek desarrolló para responder exactamente a esa pregunta. El nombre lo resume todo: en lugar de un crítico que estime el valor absoluto de cada estado, GRPO usa un grupo de respuestas generadas para el mismo prompt y las evalúa relativamente entre sí.

Antes de entrar en la mecánica, es útil entender qué es una **ventaja** (advantage) en el contexto de RL para LLMs. La ventaja de una respuesta mide cuánto mejor (o peor) fue esa respuesta comparada con lo que esperábamos. Si el crítico de PPO estima que el valor esperado de un estado es 0.6 y la recompensa real fue 0.9, la ventaja es +0.3: el resultado fue mejor de lo esperado, así que reforzamos ese comportamiento. Si la recompensa fue 0.3, la ventaja es -0.3: fue peor de lo esperado, así que lo inhibimos. La ventaja, en otras palabras, es la señal que le dice al modelo "esto fue buena o mala idea relativo a lo que anticipabas".

GRPO calcula la ventaja sin un crítico. ¿Cómo? En lugar de una sola respuesta por prompt, genera un **grupo** de $G$ respuestas (típicamente entre 4 y 8) usando la política actual. Todas parten del mismo prompt. Cada una se evalúa con el mismo mecanismo de recompensa — un modelo de recompensa entrenado, o una función de verificación directa como "¿es numéricamente correcta la respuesta?". El resultado es un conjunto de puntuaciones $\{r_1, r_2, \ldots, r_G\}$.

La ventaja de la respuesta $i$ se calcula así:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$$

Donde $\mu_G$ es la media de las recompensas del grupo y $\sigma_G$ es su desviación estándar. Esto es, sencillamente, una normalización z-score aplicada a las puntuaciones del grupo.

Vamos a ponerle números concretos. Supón que GRPO genera cuatro respuestas a un problema de matemáticas y les asigna estas recompensas:

| Respuesta | Recompensa $r_i$ |
|-----------|-----------------|
| A         | 1.0 (correcta)  |
| B         | 0.0 (incorrecta)|
| C         | 1.0 (correcta)  |
| D         | 0.5 (parcialmente correcta)|

La media del grupo es $\mu_G = (1.0 + 0.0 + 1.0 + 0.5) / 4 = 0.625$. La desviación estándar, calculando la varianza $(0.375^2 + 0.625^2 + 0.375^2 + 0.125^2)/4 \approx 0.148$, da $\sigma_G \approx 0.385$.

Las ventajas quedan:

- Respuesta A: $(1.0 - 0.625) / 0.385 \approx +0.97$
- Respuesta B: $(0.0 - 0.625) / 0.385 \approx -1.62$
- Respuesta C: $(1.0 - 0.625) / 0.385 \approx +0.97$
- Respuesta D: $(0.5 - 0.625) / 0.385 \approx -0.32$

El resultado es intuitivo: las respuestas correctas reciben ventaja positiva y el modelo aprende a imitarlas. Las incorrectas reciben ventaja negativa y el modelo aprende a evitarlas. La respuesta parcialmente correcta recibe una penalización leve. Todo esto sin que ningún modelo de valor haya intervenido: el grupo mismo es la línea base.

> **Descripción visual:** Diagrama de flujo horizontal. A la izquierda, un rectángulo violeta claro etiquetado "Prompt de entrada" del que parten cuatro flechas hacia cuatro bloques de respuestas dispuestos verticalmente en el centro: dos rectángulos verdes (Respuesta A y C, reward 1.0), uno rojo (Respuesta B, reward 0.0) y uno amarillo (Respuesta D, reward 0.5). Todas las respuestas convergen con flechas hacia un bloque azul claro "Ventaja relativa / z-score del grupo", que a su vez conecta hacia un bloque violeta final "Actualizar política". Fondo blanco, tipografía sans-serif, estilo técnico minimalista. Es como grading on a curve en un examen universitario — no necesitas una rúbrica absoluta si puedes comparar a los estudiantes entre sí.

### El objetivo de entrenamiento de GRPO

Con las ventajas calculadas, GRPO actualiza el modelo minimizando un objetivo que combina dos elementos: el objetivo de política recortado (heredado de PPO) y una penalización KL explícita.

La política en este contexto significa el propio LLM: una función parametrizada por sus pesos $\theta$ que, dado un prompt $x$, produce una distribución de probabilidad sobre el siguiente token. Escribimos esta política como $\pi_\theta$. La política antigua (la que generó las respuestas del grupo, antes de la actualización) la llamamos $\pi_{\theta_{old}}$, y el modelo de referencia original (el punto de partida, que actúa como ancla) lo llamamos $\pi_{ref}$.

El **ratio de importancia** para un token específico $t$ de la respuesta $i$ mide cuánto más (o menos) probable es ese token bajo la política nueva comparada con la antigua:

$$\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid x, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid x, o_{i,<t})}$$

Si $\rho = 1.0$, las dos políticas asignan exactamente la misma probabilidad a ese token. Si $\rho = 2.0$, la política nueva lo considera dos veces más probable. Este ratio es la corrección que permite reutilizar las respuestas generadas anteriormente sin sesgar el gradiente.

El objetivo de política de GRPO, que queremos maximizar, es:

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t}\hat{A}_i,\; \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

Esta ecuación tiene tres piezas que vale la pena diseccionar una por una.

**Primera pieza: el objetivo recortado.** La expresión $\min(\rho_{i,t}\hat{A}_i, \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i)$ limita cuánto puede cambiar el ratio de importancia en una sola actualización. Con $\varepsilon = 0.2$ (el valor estándar), el ratio solo puede moverse dentro del rango $[0.8, 1.2]$. Si el modelo quería triplicar la probabilidad de un token (ratio = 3.0), GRPO solo permite que llegue a 1.2 en esta iteración. Esto evita que un solo batch destruya semanas de entrenamiento previo.

Volvamos al ejemplo numérico: supón que la respuesta A tenía un token con probabilidad 20% bajo $\pi_{\theta_{old}}$ y el gradiente quiere subirla a 80% bajo $\pi_\theta$. El ratio sería $0.80/0.20 = 4.0$. Con el clipping a 1.2, el gradiente efectivo se calcula como si el ratio fuera solo 1.2, no 4.0. El modelo aprende en esa dirección, pero da un paso pequeño y seguro.

**Segunda pieza: la normalización por longitud.** El término $1/|o_i|$ promedia el objetivo sobre el número de tokens de la respuesta $i$. Esto parece razonable — respuestas más largas tienen más tokens, así que tiene sentido dividir — pero introduce un problema que exploraremos en profundidad en la siguiente sección.

**Tercera pieza: la penalización KL.** $D_{KL}(\pi_\theta \| \pi_{ref})$ mide cuánto se ha alejado la política actual del modelo de referencia original. La [[05-rlhf-alineacion-llms|divergencia KL]] (Kullback-Leibler) entre dos distribuciones $P$ y $Q$ se define como:

$$D_{KL}(P \| Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$$

En términos intuitivos: si el modelo original asignaba 40% de probabilidad a una respuesta y el modelo actualizado le asigna 80%, la KL captura ese desplazamiento. Un valor de KL cercano a cero significa que el modelo apenas ha cambiado; un valor grande significa que ha derivado significativamente de su comportamiento original.

El parámetro $\beta$ controla cuánto pesamos esta penalización. Con $\beta = 0.0$, el modelo puede derivar libremente — optimizará la recompensa sin importar cuánto cambie. Con $\beta$ muy alto, el modelo apenas se mueve, preservando su comportamiento pero sin aprender mucho. En la práctica, $\beta$ suele estar entre 0.01 y 0.1.

La diferencia clave de GRPO respecto a PPO aquí es arquitectónica: en PPO, la penalización KL se incorpora directamente dentro del cálculo de recompensa, de modo que las ventajas ya llevan contaminación del término de regularización. GRPO la separa limpiamente: primero calcula ventajas puras basadas en el rendimiento relativo del grupo, y luego aplica la KL como un término independiente en la función de pérdida. El resultado es una señal de aprendizaje más limpia: el modelo sabe con claridad qué parte del gradiente viene de "fuiste mejor que tus compañeros de grupo" y qué parte viene de "te estás alejando demasiado de tu comportamiento original".

### El impacto en recursos: de cluster a escritorio

El beneficio práctico de eliminar el crítico es difícil de exagerar. Con PPO entrenando un modelo de 7B, necesitas aproximadamente:

- Actor (pesos + gradientes + optimizador): ~56 GB
- Crítico (pesos + gradientes + optimizador): ~56 GB  
- Referencia (solo inferencia): ~14 GB
- Reward model (solo inferencia): ~14 GB
- Total: ~140 GB de VRAM

Con GRPO, el crítico desaparece:

- Actor: ~56 GB
- Referencia: ~14 GB
- Reward model: ~14 GB
- Total: ~84 GB de VRAM

> **Descripción visual:** Diagrama horizontal con dos bloques agrupados, PPO a la izquierda y GRPO a la derecha, conectados por una flecha con etiqueta que indica la reducción del 40 % de VRAM. El bloque PPO contiene cuatro rectángulos azul claro (Actor, Crítico, Referencia, Reward Model). El bloque GRPO tiene tres rectángulos azul claro y un óvalo con borde rojo discontinuo marcado "Crítico eliminado". Fondo blanco, tipografía sans-serif, estilo limpio y técnico.

Una reducción de aproximadamente el 40%. Pero la historia no termina ahí: si además usas una función de verificación directa en lugar de un reward model (por ejemplo, comparar la respuesta numérica con la solución correcta), el reward model también desaparece y el total cae a ~70 GB. Para un modelo de 1.5B parámetros, que es el tamaño que DeepSeek usó en varios de sus experimentos públicos, esto cabe holgadamente en una GPU de 16 GB de VRAM — el hardware que puede comprar un desarrollador individual.

Los informes de DeepSeek estimaron que el costo total de entrenamiento con GRPO es aproximadamente un dieciocho-avo del costo equivalente con RL tradicional. Eso no es una mejora marginal: es la diferencia entre un experimento que cuesta $100.000 en compute y uno que cuesta $5.500.

---

## Las grietas en el barniz: limitaciones de GRPO base

GRPO es elegante y eficiente, pero su diseño contiene tres fragilidades que se vuelven problemáticas a medida que los modelos escalan. Para apreciarlas bien, hay que entender primero el mecanismo de **importance sampling** que subyace al algoritmo.

### El problema del importance sampling acumulado

Generar respuestas completas de un LLM grande es caro. Si tuviéramos que generar respuestas nuevas con la política actualizada en cada paso de gradiente, el entrenamiento sería prohibitivamente lento. La solución estándar en RL para LLMs — tanto en PPO como en GRPO — es **importance sampling**: generamos un lote de respuestas con la política en un momento $t$, y luego hacemos múltiples pasos de gradiente sobre esas mismas respuestas, corrigiendo el sesgo con el ratio $\rho_{i,t}$ que definimos antes.

Mientras la política no cambie demasiado entre la generación y la actualización, el ratio es cercano a 1.0 y la corrección es precisa. El problema surge cuando el modelo ha aprendido lo suficiente como para que su política actual difiera significativamente de la política con la que se generaron las respuestas. En ese punto, algunos ratios se disparan muy por encima de 1.0 (el modelo actual habría generado ese token con mucha más probabilidad) o colapsan hacia 0.0 (casi nunca lo generaría ahora). Esto introduce ruido masivo en los gradientes.

Ahora viene la parte que hace a GRPO especialmente vulnerable: el ratio de importancia de una **secuencia completa** es el producto de los ratios individuales de cada token. Si una respuesta tiene 500 tokens, y cada token tiene un ratio de, digamos, 1.1 (un desvío aparentemente pequeño del 10%), el ratio de secuencia es $1.1^{500} \approx 1.45 \times 10^{20}$. Un número astronómico generado por pequeñas imprecisiones en cada token.

En la práctica no llega a esos extremos porque el clipping lo limita, pero la varianza se acumula. Y el problema es peor en arquitecturas **MoE** — Mixture of Experts — como los modelos DeepSeek. En una MoE, cada token es procesado por un subconjunto de "expertos" seleccionados dinámicamente por un router. Si la política vieja enrutó un token al experto 3 y 7, pero la política nueva lo enruta al experto 1 y 5 (expertos completamente distintos), el ratio de ese token puede ser extremo simplemente porque los dos modelos tienen arquitecturas de activación diferentes para esa entrada. GRPO hereda toda esta volatilidad sin ningún mecanismo para atenuarla sistemáticamente.

### El sesgo de longitud: verbosidad como estrategia de supervivencia

El segundo problema es más sutil y tiene consecuencias comportamentales directas y observables. La función de objetivo de GRPO promedia la pérdida por el número de tokens de cada respuesta ($1/|o_i|$). La intención es justa: no queremos que una respuesta larga domine el gradiente solo porque tiene más tokens. Pero esta normalización crea un incentivo perverso para las respuestas incorrectas.

Pensemos en dos respuestas incorrectas a un problema de matemáticas:

- **Respuesta corta incorrecta** (50 tokens): "La respuesta es 42." — Recompensa: 0.
- **Respuesta larga incorrecta** (2000 tokens): Un extenso desarrollo con cálculos erróneos que concluye "por lo tanto, la respuesta es 42." — Recompensa: 0.

Ambas reciben la misma penalización en recompensa absoluta. Pero con la normalización por longitud, el gradiente de penalización para cada token de la respuesta corta es $\text{penalización}/50$, mientras que para cada token de la respuesta larga es $\text{penalización}/2000$. La respuesta larga recibe una penalización por token 40 veces menor.

El modelo lo aprende gradualmente: cuando no sabe la respuesta — cuando todas sus respuestas en el grupo van a ser incorrectas — la estrategia óptima según el objetivo matemático es ser lo más verboso posible. No resuelve el problema, pero minimiza la penalización por token. Con el tiempo, los modelos entrenados con GRPO puro tienden a generar respuestas largas y ramificadas cuando están inseguros, no porque eso les ayude a razonar mejor, sino porque el gradiente los ha entrenado a hacerlo.

Este fenómeno se ha documentado empíricamente: modelos entrenados extensamente con GRPO a veces generan razonamientos de varios miles de tokens para problemas simples, llenando espacio con reformulaciones del problema, casos especiales innecesarios y comprobaciones redundantes. Es verbosidad aprendida como escudo.

### El sesgo de dificultad: el modelo aprende de lo que ya sabe

El tercer problema ocurre en los extremos del espectro de dificultad. Recuerda que la ventaja se normaliza dividiendo por la desviación estándar del grupo:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$$

Considera dos casos problemáticos:

**Caso 1 — Pregunta trivial:** El modelo genera ocho respuestas al mismo problema fácil y las ocho son correctas. Todas reciben recompensa 1.0. La media es 1.0, la desviación estándar es 0. División por cero — técnicamente undefined, en la práctica una pequeña constante de estabilidad hace que el gradiente sea enorme. El modelo recibe señal de gradiente gigantesca de un problema donde ya lo sabe todo.

**Caso 2 — Pregunta imposible:** El modelo genera ocho respuestas a un problema extremadamente difícil y las ocho fallan. Todas reciben recompensa 0.0. Media 0.0, desviación estándar 0. Mismo problema: gradiente artificial. El modelo recibe una penalización fuerte de preguntas que todavía no puede resolver.

**Caso 3 — Pregunta de dificultad media:** Cuatro respuestas correctas, cuatro incorrectas. Media 0.5, desviación estándar ~0.5. Las ventajas son $\pm 1.0$ — bien calibradas. El modelo aprende de un problema donde hay señal genuina.

El resultado: el entrenamiento de GRPO se sesga hacia los extremos de dificultad, que son precisamente donde menos se puede aprender. Las preguntas demasiado fáciles no enseñan nada nuevo. Las demasiado difíciles tampoco. El rango de aprendizaje real está en las preguntas de dificultad media — y es exactamente allí donde la normalización produce señales más moderadas y el optimizador presta menos atención.

En la zona pedagógicamente óptima, la señal es débil. En las zonas triviales e imposibles, la señal es fuerte. Es el equivalente a un sistema de calificaciones que ignora a los estudiantes medios y se obsesiona con los que ya sacaron 10 y los que sacaron 0.

> **Descripción visual:** Diagrama de flujo horizontal con tres columnas. La columna izquierda tiene un rectángulo azul claro etiquetado "GRPO base". La columna central tiene tres rectángulos amarillo dorado (las tres causas: ratios acumulados, sesgo de longitud, sesgo de dificultad). La columna derecha tiene tres rectángulos rojo claro (los tres efectos: varianza explosiva, divagar, sin aprendizaje en extremos). Las flechas van de izquierda a derecha con puntas triangulares grises. Fondo blanco, tipografía sans-serif, estilo técnico diagnóstico.

---

## Las variantes que reparan GRPO

Reconocidos estos problemas, la comunidad investigadora respondió con rapidez. En el período 2024-2025 emergieron tres variantes principales que abordan las limitaciones descritas de maneras complementarias. Cada una se puede entender como un parche quirúrgico a un problema específico.

### DAPO: precisión en la señal de gradiente

Dynamic Advantage Policy Optimization — DAPO — es la variante más comprehensiva, y ataca los tres problemas con cuatro intervenciones distintas.

**Intervención 1: Token-Level Gradient Loss.** DAPO corrige el sesgo de longitud cambiando cómo se promedian los gradientes. En lugar de promediar por respuesta y luego por grupo, DAPO promedia directamente sobre todos los tokens de todas las respuestas del grupo. La diferencia parece sutil pero tiene consecuencias importantes.

En GRPO, una respuesta de 2000 tokens y una de 50 tokens contribuyen igual al gradiente del grupo (cada una cuenta como "una respuesta"). Dentro de cada respuesta, el gradiente se diluye entre sus tokens. En DAPO, todos los tokens del batch contribuyen por igual, independientemente de qué respuesta los generó. Una respuesta de 2000 tokens tiene 40 veces más tokens que una de 50, y todos esos tokens participan directamente en el gradiente sin que la longitud los diluya artificialmente.

El efecto práctico: las cadenas de razonamiento largas y correctas reciben un gradiente de refuerzo proporcional a su longitud — cuanto más razonamiento útil produjiste, más te refuerzo. Y las cadenas largas incorrectas reciben una penalización igualmente proporcional, eliminando el escudo que la longitud proporcionaba en GRPO.

**Intervención 2: Overlong Reward Shaping.** Para que el modelo no compense el loss token-level volviéndose aún más verboso de otras formas, DAPO añade una penalización suave progresiva por longitud excesiva. Define un umbral $L_{max}$ — digamos, 2048 tokens — y cualquier respuesta que lo supere sin haber llegado a una conclusión correcta recibe una penalización creciente. La penalización no es un cliff (corte abrupto), sino una rampa: cuanto más largo, peor la recompensa ajustada, de forma continua. El modelo aprende que la verbosidad sin resultado tiene un costo explícito.

**Intervención 3: Clip-Higher, clipping asimétrico.** GRPO y PPO usan clipping simétrico: el ratio de importancia no puede subir más de $1 + \varepsilon$ ni bajar más de $1 - \varepsilon$. DAPO observa que este clipping simétrico es innecesariamente conservador en una dirección: cuando queremos subir la probabilidad de un token que el modelo actualmente asigna con baja probabilidad (porque la respuesta fue buena pero el token era infrecuente), el límite superior $1 + \varepsilon$ frena ese aprendizaje demasiado pronto.

DAPO implementa clipping asimétrico: el límite inferior se mantiene en $1 - \varepsilon$ (para no castigar demasiado agresivamente tokens de respuestas malas), pero el límite superior se eleva a $1 + \varepsilon_{high}$ donde $\varepsilon_{high} > \varepsilon$. Tokens en respuestas buenas con probabilidades bajas tienen más margen para crecer hacia probabilidades altas, acelerando el aprendizaje de secuencias raras pero correctas.

**Intervención 4: Dynamic Sampling.** Esta es quizá la más elegante. DAPO añade una restricción al proceso de generación de grupos: cada grupo evaluado debe contener al menos una respuesta correcta y al menos una incorrecta. Si el modelo genera un grupo donde todas las respuestas son correctas (o todas incorrectas), ese grupo se descarta y se reemplaza.

¿Por qué? Porque un grupo uniformemente correcto o incorrecto produce ventajas todas iguales a cero (media igual a todas las recompensas, desviación estándar cero, ventaja cero). Gradiente cero. Compute desperdiciado completamente. Dynamic Sampling garantiza que cada evaluación de grupo produce al menos algún gradiente útil. Además, estructuralmente asegura que el modelo siempre esté aprendiendo de comparaciones que tienen señal — lo que es bueno comparado con lo que es malo, sin casos degenerados.

### GSPO: corrección al nivel matemático fundamental

Group Sequence Policy Optimization — GSPO — toma un ángulo diferente. En lugar de añadir correcciones encima de GRPO, identifica el error matemático de base y lo corrige directamente.

El error es la discordancia de granularidad: la recompensa se asigna al nivel de **secuencia** (¿fue correcta la respuesta completa?) pero el importance sampling se aplica al nivel de **token** (¿cuánto cambió la probabilidad de cada token individual?). Combinar una señal de nivel de secuencia con correcciones de nivel de token es como medir el rendimiento de un equipo de fútbol por el número total de pases individuales correctos, en lugar de por si ganaron el partido.

GSPO resuelve esto elevando todo al nivel de secuencia. En lugar de un ratio de importancia por token, define un ratio de importancia por secuencia:

$$\rho_i^{seq} = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log\frac{\pi_\theta(o_{i,t})}{\pi_{\theta_{old}}(o_{i,t})}\right)$$

Este es la **media geométrica** de los ratios individuales de tokens, expresada como exponencial de la media de los logaritmos. La media geométrica tiene una propiedad estadística fundamental: es mucho más robusta a valores extremos que la media aritmética. Si un token tiene un ratio de 10.0 y los otros 499 tienen ratio de 1.0, la media aritmética del producto sería explosiva; la media geométrica sería $(10.0 \times 1.0^{499})^{1/500} \approx 1.005$ — casi sin perturbación.

Esta única corrección matemática tiene un efecto cascada sobre la estabilidad del entrenamiento. Las varianzas en los gradientes caen dramáticamente porque ya no se acumulan multiplicativamente token a token. El clipping del ratio ahora opera sobre una cantidad que mide el cambio de la secuencia completa, no de tokens individuales, lo que lo hace mucho más interpretable y predecible.

El impacto más llamativo de GSPO es en arquitecturas MoE, como DeepSeek-V3 o los modelos Mixtral. En estas arquitecturas, el problema del importance sampling se agravaba porque diferentes tokens se enrutaban a diferentes expertos, y si la política nueva tomaba decisiones de enrutamiento distintas a las de la política vieja, los ratios individuales explotaban aunque la secuencia completa fuera razonablemente similar.

GSPO, al evaluar el ratio de la secuencia completa, es agnóstico a las decisiones de enrutamiento internas: solo mide si la probabilidad total de la secuencia cambió, no cuáles expertos procesaron cada token. Esto elimina la necesidad de una técnica llamada "Routing Replay" — un workaround costoso que congelaba las rutas de expertos durante el entrenamiento para estabilizar el importance sampling en MoE. Con GSPO, la estabilización ocurre matemáticamente, sin overhead adicional.

### Dr. GRPO: la corrección minimalista

Dr. GRPO — nombre que en inglés juega con "GRPO Done Right" — adopta la filosofía opuesta a DAPO: en lugar de añadir mecanismos, elimina los que crean sesgo.

Los investigadores hicieron un análisis de la función objetivo de GRPO y encontraron que dos términos de normalización que parecen inocuos son en realidad la fuente directa del sesgo de longitud y del sesgo de dificultad:

1. La normalización por longitud de secuencia ($1/|o_i|$): introduce el sesgo de longitud.
2. La normalización por desviación estándar del grupo ($1/\sigma_G$): introduce el sesgo de dificultad.

La solución de Dr. GRPO es simplemente quitarlos. La función objetivo queda:

$$\mathcal{J}_{DrGRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t}(r_i - \mu_G),\; \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)(r_i - \mu_G)\right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

La ventaja ahora es simplemente $r_i - \mu_G$ — la desviación de la recompensa respecto a la media del grupo, sin dividir por la desviación estándar. Y el promedio es sobre todos los tokens de todas las respuestas, sin ponderar por la longitud de cada respuesta.

Volvamos al ejemplo de antes para ver qué cambia en el caso del sesgo de dificultad:

- **Pregunta trivial** (todos correctos, $r_i = 1.0$): $r_i - \mu_G = 1.0 - 1.0 = 0.0$ para todas las respuestas. Gradiente cero. Sin señal espuria.
- **Pregunta imposible** (todos incorrectos, $r_i = 0.0$): $r_i - \mu_G = 0.0 - 0.0 = 0.0$ para todas. Gradiente cero. Sin señal espuria.
- **Pregunta de dificultad media**: La mitad correctas (1.0), mitad incorrectas (0.0). Media 0.5. Ventajas: $\pm 0.5$. Gradiente bien calibrado.

Dr. GRPO produce automáticamente gradiente cero cuando no hay nada que aprender, y gradiente proporcional a la variabilidad real dentro del grupo cuando sí hay señal. Es un estimador matemáticamente insesgado de la ventaja, en contraste con la versión normalizada de GRPO.

El resultado empírico es notable para ser tan simple: los modelos entrenados con Dr. GRPO dejan de desarrollar el comportamiento de verbosidad defensiva, porque la penalización por token ya no se diluye con la longitud. La longitud promedio de respuestas incorrectas cae significativamente — el modelo aprende que ser largo y equivocado no tiene ventaja sobre ser corto y equivocado.

> **Descripción visual:** Diagrama horizontal con dos subgrafos apilados verticalmente unidos por una flecha de transformación. El subgrafo superior "GRPO original" muestra una cadena de tres rectángulos: el primero azul claro (Ventaja bruta), el segundo y tercero rojo claro (divisiones que introducen sesgo). El subgrafo inferior "Dr. GRPO" muestra solo dos rectángulos verdes (Ventaja bruta y Gradiente directo). La flecha de transformación está etiquetada "Eliminar ÷ longitud y ÷ std". Fondo blanco, tipografía sans-serif, estilo comparativo limpio.

---

## Comparando las variantes: cuándo usar cada una

Con tres variantes encima de la mesa, la pregunta práctica es cuál elegir para un proyecto dado. Aquí hay un mapa de decisión:

| Situación | Recomendación | Razón |
|-----------|---------------|-------|
| Primer experimento, modelo ≤ 3B | GRPO base | Simple de implementar; los problemas de escala no son críticos aún |
| Modelo > 7B, training largo | DAPO | Las cuatro intervenciones abordan todos los problemas conocidos de GRPO |
| Arquitectura MoE | GSPO | Elimina el problema de routing volatility sin overhead adicional |
| Diagnóstico claro de verbosidad excesiva | Dr. GRPO | Solución quirúrgica, mínimo cambio de código |
| Máximo control sobre hiperparámetros | DAPO | Más knobs, pero también más decisiones que tomar |

> **Descripción visual:** Árbol de decisión vertical. En la cima, un óvalo azul claro "Elegir variante GRPO". Debajo, cuatro rombos amarillo dorado con preguntas de diagnóstico encadenadas hacia abajo. Cada rombo tiene una rama "Sí" que lleva a un rectángulo verde con el nombre de la variante recomendada (GRPO Base, GSPO, Dr. GRPO, DAPO). La rama "No" final del último rombo lleva a un rectángulo violeta "DAPO + Dr. GRPO". Flechas etiquetadas Sí/No. Fondo blanco, tipografía sans-serif, estilo de diagrama de decisión técnico.

Una nota práctica: DAPO y Dr. GRPO no son mutuamente excluyentes. Puedes implementar el token-level gradient loss de DAPO (que es esencialmente lo mismo que eliminar la normalización por longitud de Dr. GRPO) junto con el dynamic sampling de DAPO. Muchos equipos terminan con una variante híbrida que toma lo mejor de cada aproximación.

---

## El legado de GRPO: democratización del RL para LLMs

Es difícil sobreestimar el impacto que tuvo GRPO en el campo. Antes de su publicación, el RL online para LLMs era prerrogativa de organizaciones con clústeres de cientos de GPUs. La implementación de referencia de PPO para modelos de 70B requería decenas de nodos A100 funcionando en paralelo con infraestructura de networking especializada.

GRPO desplazó el umbral hacia abajo por un factor de diez o más. Los equipos que trabajan con modelos de 1.5B a 7B pueden ejecutar experimentos completos de RL en una o dos GPUs de gama alta. Las universidades sin acceso a compute clouds industriales pueden investigar post-entrenamiento con RL. Los desarrolladores individuales pueden iterar sobre ideas en sus propias máquinas.

Este cambio de accesibilidad tiene consecuencias directas sobre el ritmo de la investigación: cuando más equipos pueden experimentar, más rápido se descubren problemas como el sesgo de longitud y el sesgo de dificultad, y más rápido emergen correcciones como DAPO, GSPO y Dr. GRPO. La historia de GRPO es la historia de una idea simple — sustituir el crítico por el grupo — que abrió una puerta que resultó llevar a toda una habitación de investigación activa.

Las variantes que hemos explorado no son el punto final de esta historia. Son la generación actual de una línea de investigación que seguirá evolucionando mientras los modelos de razonamiento sigan escalando. Lo que el próximo capítulo traerá es la pregunta que estas variantes aún no responden completamente: ¿cómo diseñamos las funciones de recompensa que alimentan a GRPO y sus variantes? ¿Cuándo es suficiente una señal binaria de corrección/incorrección, y cuándo necesitamos recompensas más matizadas? Esa es la pieza que completa el puzzle.

---

## Tags

#técnica/grpo #técnica/policy-gradient #concepto/importance-sampling #técnica/dapo #técnica/gspo #técnica/dr-grpo #concepto/reward-model #técnica/ppo #nivel/intermedio #tipo/lección #estado/completo



---
capitulo: "07"
titulo: "Más Allá del Texto: Fine-tuning Multimodal para Visión y Síntesis de Voz"
aliases:
  - "Capítulo 7"
  - "Cap 7"
  - "Fine-tuning Multimodal"
  - "VLM y TTS"
tema: "multimodal"
subtemas: [vision-language-model, text-to-speech, neural-audio-codec]
dificultad: "intermedio"
tipo: "lección"
estado: "completo"
conceptos_centrales:
  - vision-language-model
  - vlm
  - fine-tuning-multimodal
  - vision-transformer
  - vit
  - text-to-speech
  - tts
  - neural-audio-codec
  - snac
prerequisitos:
  - "[[02-supervised-finetuning]]"
  - "[[03-lora-adaptacion-de-bajo-rango]]"
  - "[[04-qlora-cuantizacion-4bit]]"
relacionados:
  - "[[01-fundamentos-transformers-y-pretraining]]"
tags:
  - modelo/vision-language-model
  - técnica/fine-tuning-multimodal
  - modelo/vision-transformer
  - técnica/text-to-speech
  - modelo/neural-audio-codec
  - técnica/lora
  - concepto/olvido-catastrófico
  - nivel/intermedio
  - tipo/lección
  - estado/completo
---

# Capítulo 7 — Más Allá del Texto: Fine-tuning Multimodal para Visión y Síntesis de Voz

> Basado en "Beyond Text: A Guide to Vision & TTS Finetuning" y "The Builder's Guide to Multimodal Finetuning (Vision + TTS)", The Neural Maze, Lección y Lab 7/8.

Durante seis capítulos hemos estado haciendo lo mismo: tomar un modelo que convierte texto en texto, y hacer que lo haga mejor. Esa es la columna vertebral del fine-tuning moderno, y las técnicas que has aprendido — [[03-lora-adaptacion-de-bajo-rango|LoRA]], [[04-qlora-cuantizacion-4bit|QLoRA]], [[02-supervised-finetuning|SFT]], RLHF — todas viven cómodamente en ese mundo. Pero hay un problema: el mundo real no es texto.

Un radiólogo no diagnostica leyendo un párrafo; diagnostica mirando una radiografía de tórax. Un podcaster no escribe su contenido — lo habla. Y un cliente que llama al soporte de una empresa no quiere recibir un JSON como respuesta; quiere escuchar una voz humana que le resuelva el problema. Si nos quedamos en el dominio texto-a-texto, estamos dejando encima de la mesa algunas de las aplicaciones más valiosas del fine-tuning.

Lo que hace que este capítulo sea especialmente satisfactorio es el siguiente hecho: las técnicas que ya dominas se aplican casi sin cambios a los modelos multimodales. Las arquitecturas cambian en cómo codifican o decodifican señales no textuales, pero el bucle de entrenamiento es el mismo. Adjuntas adaptadores LoRA al backbone transformer, preparas un dataset, y lanzas el entrenamiento. La mecánica es familiar. Lo que cambia es el rango de problemas que puedes resolver.

En este capítulo exploraremos dos direcciones multimodales: el fine-tuning de modelos de visión-lenguaje (VLM — Vision-Language Models, modelos que entienden imágenes y generan texto) y el fine-tuning de síntesis de voz (TTS — Text-to-Speech, sistemas que convierten texto en habla). Para cada uno construiremos la intuición de la arquitectura, entenderemos por qué funcionan las técnicas que usamos, y luego pasaremos al laboratorio concreto: fine-tuning de Qwen3-VL sobre conversión de escritura a mano en LaTeX, y fine-tuning de Orpheus-TTS para clonar una voz específica.

---

## Por qué importa el fine-tuning multimodal

Antes de entrar en arquitecturas, vale la pena detenerse en el "para qué". No toda tarea justifica la complejidad adicional de trabajar con múltiples modalidades. Pero hay categorías de problemas donde un modelo de texto, por más grande y capaz que sea, simplemente no puede llegar — porque la información que necesita no existe en formato texto.

Considera el caso de la inspección de calidad industrial. Una cámara en una línea de producción captura imágenes de piezas manufacturadas. El defecto que interesa detectar — una microfisura en una cerámica, un desalineamiento de 0.3mm, una burbuja de aire bajo la pintura — no es algo que se pueda describir en un prompt y esperar que el modelo "deduzca". El modelo tiene que ver la imagen. Un VLM de propósito general preentrenado en imágenes de internet sabe que las fisuras existen, pero no sabe qué aspecto tiene una "fisura hairline en cerámica mate bajo luz de fábrica". Para eso necesitas fine-tuning con cientos de ejemplos etiquetados de tu dominio específico.

El mismo razonamiento aplica a la medicina. Los VLMs de propósito general cometen errores en terminología clínica y pasan por alto hallazgos sutiles en imágenes médicas — no porque sean modelos mediocres, sino porque la distribución de imágenes médicas de alta especificidad está subrepresentada en sus datos de preentrenamiento. Un modelo finetuneado sobre pares imagen-informe de un dominio concreto (radiografías de tórax de una red hospitalaria, fondo de ojo para retinopatía diabética, histología de biopsias de piel) aprende exactamente la terminología, los formatos de informe, y los patrones visuales que importan en ese contexto.

Para el monitoreo agrícola, la situación es análoga: las enfermedades de cultivos costaron aproximadamente 220 mil millones de dólares en pérdidas globales solo en 2022. La detección temprana requiere identificar diferencias visuales sutiles — un hongo en etapa inicial se parece mucho a un déficit de nitrógeno, pero las implicaciones de tratamiento son completamente distintas. Un VLM finetuneado sobre imágenes etiquetadas de una región y cultivo específico aprende esas diferencias. Un agricultor puede sacar el teléfono, fotografiar una hoja, y recibir un diagnóstico diferencial en segundos.

En el dominio de la voz, el fine-tuning TTS resuelve un problema que la clonación de voz zero-shot no puede: la profundidad de personalización. Un sistema TTS base puede escuchar 10 segundos de tu voz e imitarte superficialmente — captura el timbre general, pero no tus patrones de énfasis, tu forma de pausar antes de una idea importante, tus vocales regionales. Es como pedirle a alguien que te imite después de escucharte en una llamada de elevador. El fine-tuning, en cambio, es como pasar semanas estudiando tus grabaciones hasta poder reproducir cada matiz de tu delivery. Con apenas 30 minutos de audio limpio de un solo hablante, se puede finetunear un modelo TTS para producir resultados que un oyente no distingue de la voz original.

---

## Parte I — Visión: cómo un modelo aprende a ver

### La arquitectura de los modelos de visión-lenguaje

Todo modelo de visión-lenguaje moderno — Qwen3-VL, Llama 3.2 Vision, Gemma 3, PaliGemma — comparte la misma estructura de tres etapas. La variedad entre ellos está en los detalles de cada etapa, no en la estructura. Entender esta arquitectura te permitirá razonar sobre cualquier VLM que encuentres, no solo los que usamos en el lab.

**Etapa 1: El codificador de visión (vision encoder).**

El codificador de visión es el componente que transforma una imagen en una representación que el modelo puede procesar. Piensa en él como el "sistema visual" del modelo: convierte píxeles en algo semántico.

En casi todos los VLMs modernos, este componente es un Vision Transformer o ViT (pronunciado "vit"). El ViT aplica al dominio visual la misma arquitectura transformer que ya conoces del procesamiento de texto. Funciona así: la imagen se divide en pequeños cuadrados llamados parches (patches) — típicamente de 14×14 o 16×16 píxeles. Cada parche se proyecta linealmente en un vector de alta dimensión (un embedding), exactamente como un token de texto se proyecta en su embedding. La secuencia de todos estos embeddings de parches se procesa con capas de [[01-fundamentos-transformers-y-pretraining|auto-atención]] (self-attention), igual que en un transformer de texto.

Hagamos los números concretos para que esto sea tangible. Una imagen de 448×448 píxeles con parches de 14×14 produce $\frac{448}{14} \times \frac{448}{14} = 32 \times 32 = 1024$ parches. Eso es 1024 tokens visuales, cada uno de dimensión $d_v$ (la dimensión oculta del codificador de visión, típicamente 1024 o 1280). El codificador procesa esta secuencia de 1024 vectores y produce 1024 vectores de características visuales enriquecidos con información contextual — un vector por parche, pero donde cada vector ya "sabe" lo que hay en los parches vecinos gracias a la auto-atención.

> **Descripción visual:** Diagrama de flujo horizontal con cinco bloques conectados por flechas. El primer bloque (azul oscuro) dice "Imagen 448x448 px". El segundo (azul) dice "Division en parches 14x14". El tercero (naranja) resalta el numero resultante "1024 parches = 1024 tokens". El cuarto (azul) dice "Auto-atencion del ViT". El quinto (naranja) dice "256 tokens visuales finales" tras la compresion. Las flechas son grises. Estilo minimalista, fondo blanco.

La magia del ViT es que fue preentrenado sobre enormes colecciones de pares imagen-texto (modelos como CLIP o SigLIP usaron cientos de millones de pares). Durante ese preentrenamiento, el ViT aprendió a codificar información visual en sus vectores de manera que sea semánticamente significativa. Un parche que muestra ojos humanos produce un vector diferente al de un parche que muestra árboles. El codificador ya sabe diferenciar esos conceptos visuales — y eso es exactamente por qué, como veremos, no necesitas entrenarlo de nuevo.

La salida del codificador se puede expresar formalmente:

$$\mathbf{V} = \text{ViT}_{\text{frozen}}(\mathbf{I}) \in \mathbb{R}^{N_v \times d_v}$$

donde $\mathbf{I}$ es la imagen de entrada, $N_v$ es el número de tokens visuales (1024 en nuestro ejemplo), y $d_v$ es la dimensión de embedding del codificador. El subíndice "frozen" es fundamental — volveremos a él.

**Etapa 2: La capa de proyección (projector).**

Aquí surge un problema: los 1024 vectores visuales de dimensión $d_v$ viven en un espacio matemático diferente al de los embeddings de texto que el LLM espera. Es como tener dos idiomas que no comparten vocabulario. La capa de proyección es el traductor entre esos dos idiomas.

En la práctica, esta capa es a menudo un MLP (Multi-Layer Perceptron — red neuronal de capas densas) de dos o tres capas, o en algunos modelos una capa de cross-attention. Lo que hace es transformar cada vector visual de dimensión $d_v$ a un vector de dimensión $d$, donde $d$ es la dimensión oculta del LLM. Si el LLM es un modelo de 7B parámetros con dimensión $d = 4096$, y el ViT produce vectores de $d_v = 1024$, el projector aprende una transformación lineal (más no-linealidad) que mapea $1024 \rightarrow 4096$.

$$\mathbf{V'} = \text{Projector}(\mathbf{V}) \in \mathbb{R}^{N'_v \times d}$$

Muchos modelos también aplican compresión aquí: agrupan parches adyacentes para reducir el número de tokens. Por ejemplo, agrupar parches en grupos de 2×2 reduce $N_v$ a $N'_v = N_v / 4$. Si teníamos 1024 tokens visuales, pasamos a 256. Esto importa porque cada token visual extra añade costo cuadrático en la atención del LLM — y una imagen de alta resolución sin compresión puede producir tantos tokens que la secuencia se vuelve prohibitiva. Con compresión 4×, los 1024 tokens se reducen a 256, equivalente a 256 palabras de texto en el contexto del LLM.

**Etapa 3: El decodificador LLM.**

Este es exactamente el transformer autoregresivo que has estado finetuneando durante todo el libro. Recibe una secuencia concatenada: primero los tokens visuales proyectados, luego los tokens de texto del prompt del usuario. Genera una respuesta de texto token a token, como siempre.

$$\hat{y} = \text{LLM}_{\theta + \Delta\theta}([\mathbf{V'}; \mathbf{h}_t])$$

donde $\mathbf{h}_t$ son los embeddings de los tokens de texto del prompt, $\theta$ son los pesos originales del LLM (congelados), y $\Delta\theta$ son los adaptadores LoRA que entrenamos. La pérdida se calcula únicamente sobre los tokens de texto generados — no sobre los tokens visuales, que actúan como contexto de entrada, no como objetivo de predicción:

$$\mathcal{L} = -\sum_{i} \log P_{\theta + \Delta\theta}(y_i \mid \mathbf{V'}, \mathbf{h}_t, y_{<i})$$

Esta fórmula es idéntica a la pérdida de SFT estándar que vimos en los capítulos anteriores. La única diferencia es que el contexto ahora incluye $\mathbf{V'}$, los tokens visuales proyectados. El mecanismo de entrenamiento es el mismo.

> **Descripción visual:** Diagrama de flujo horizontal con dos ramas de entrada que convergen en el LLM. La rama superior muestra la imagen pasando por un bloque azul claro (ViT congelado, indicado con candado), luego un bloque amarillo de tokens visuales, luego un bloque verde (projector entrenable), y tokens proyectados. La rama inferior muestra el texto de prompt pasando a embeddings de texto amarillos. Ambas ramas se unen en un nodo gris de concatenación, que alimenta al LLM (verde, entrenable con LoRA), que produce la respuesta (rosa). Estilo limpio, fondo blanco, tipografía sans-serif, flechas grises con punta sólida.

### Por qué el codificador de visión permanece congelado

Esta es la decisión de diseño más importante del fine-tuning de VLMs, y merece una explicación profunda porque en la superficie puede parecer contraintuitiva: si quieres que el modelo "vea mejor" en tu dominio, ¿no deberías entrenar también el componente que hace la visión?

La respuesta corta es no, y aquí está el razonamiento largo.

El codificador de visión fue preentrenado con cientos de millones (o miles de millones) de pares imagen-texto. Durante ese proceso aprendió representaciones visuales extraordinariamente ricas: sabe qué es una radiografía vs una fotografía de paisaje, sabe que una laceración en una pieza industrial es diferente a una grieta natural en una roca, sabe distinguir la hoja sana de una hoja con hongos. Esas representaciones son lo que hace que el modelo sea valioso en primer lugar.

Si entrenas el codificador de visión con tu pequeño dataset de dominio (digamos 500 imágenes de fracturas cerámicas), corres el riesgo de sobreescribir esas representaciones ricas con algo muy específico y mucho menos generalizable. El proceso de aprender "fractura cerámica" puede destruir lo que el modelo sabía sobre "fractura de metal" o "arañazo superficial". Esto es lo que los investigadores llaman [[03-lora-adaptacion-de-bajo-rango|olvido catastrófico]] (catastrophic forgetting), y es especialmente devastador para componentes que tomaron miles de millones de pares de entrenamiento para alcanzar su calidad actual.

El objetivo del fine-tuning de dominio no es cambiar cómo el modelo ve las imágenes — es cambiar cómo el LLM interpreta y describe lo que ve. Un modelo base puede mirar una radiografía de tórax y saber que hay estructuras anatómicas, zonas de mayor y menor densidad, y anomalías. Lo que no sabe es nombrar esas anomalías con la terminología correcta de radiología, estructurar el informe según el protocolo hospitalario, o priorizar los hallazgos según su urgencia clínica. Eso es conocimiento del LLM, no del codificador visual.

Entonces la estrategia es:
- Codificador de visión: congelado. No recibe gradientes, no se actualiza.
- Capa de proyección: puede ser entrenable o no, dependiendo del modelo y la tarea.
- LLM decodificador: recibe adaptadores LoRA en sus capas de atención, exactamente como en fine-tuning de texto.

El resultado práctico es impresionante: el costo de entrenamiento de un VLM con esta estrategia es esencialmente igual al de un LLM de texto del mismo tamaño. El codificador añade costo en inferencia (tienes que codificar la imagen), pero durante el entrenamiento está congelado — no hay gradientes que propagar por él, no hay memoria adicional para sus activaciones intermedias en el backward pass.

### Cuándo sí conviene entrenar el codificador de visión

Hay un caso donde vale la pena experimentar con entrenar (parcialmente) el codificador: cuando la distribución visual de tu tarea es genuinamente diferente a la de los datos de preentrenamiento. La escritura a mano matemática es un buen ejemplo — los corpus de preentrenamiento de visión están dominados por fotografías naturalistas, diagramas digitales y texto tipografiado. La escritura a mano con notación matemática (integrales, sumatorias, letras griegas a mano) es una distribución visual bastante distinta.

En esos casos, finetunear también el codificador — con una tasa de aprendizaje muy baja, para no destruir las representaciones existentes — puede dar mejoras adicionales. La regla práctica: empieza siempre con el codificador congelado. Si los resultados son insuficientes después de un ciclo de entrenamiento completo sobre el decodificador, entonces experimenta con descongelar las últimas capas del codificador con lr ~10x más pequeña.

### Consideraciones prácticas que el artículo menciona pero no explica

**La trampa de la resolución.** Cada pixel importa en términos computacionales. Una imagen de 224×224 con parches de 14×14 produce $(224/14)^2 = 256$ tokens visuales. Una imagen de 448×448 produce 1024 tokens — cuatro veces más. Una imagen de 896×896 produciría 4096 tokens solo de la imagen, más los tokens del prompt y la respuesta. Eso es un contexto enorme. El costo de atención es cuadrático en la longitud de secuencia, así que duplicar la resolución cuadruplica los tokens visuales, lo que puede octuplica (o más) el uso de VRAM según la implementación.

La mayoría de los VLMs modernos soportan resolución dinámica — el modelo acepta imágenes de cualquier tamaño y las procesa eficientemente. Pero durante el fine-tuning, debes decidir un rango de resolución razonable. La recomendación del artículo de 300-1000px es sensata: por debajo de 300px pierdes detalles importantes en tareas como OCR o inspección industrial; por encima de 1000px en una GPU de consumo puedes quedarte sin VRAM en el primer batch. Empieza en el punto medio del rango (600-700px) y ajusta según lo que permite tu hardware.

**La trampa del aspect ratio variable.** Si tus imágenes tienen formas muy diferentes — algunas cuadradas, otras panorámicas, otras verticales — el número de tokens visuales por muestra varía enormemente. Meter en un mismo batch una imagen de 256 tokens y otra de 1024 tokens es ineficiente: hay que rellenar la imagen pequeña con tokens de padding hasta alcanzar la longitud de la grande. Cuando hay mucha varianza, estás desperdiciando cómputo en padding. Estrategia práctica: agrupa las imágenes por aspect ratio similar al crear los batches, o simplemente normaliza todas las imágenes a un mismo tamaño durante el preprocesamiento.

**La trampa del olvido de dominio general.** Si finetuneas un VLM exclusivamente sobre imágenes de radiología, el modelo puede "olvidar" cómo responder preguntas visuales generales — si le preguntas qué hay en una fotografía de un gato, puede responder de forma extraña o degenerada. Esto ocurre porque el fine-tuning sobreajusta la distribución de activaciones del LLM hacia los patrones de los informes médicos, y los pesos originales que manejaban la variedad visual general se alejan de su estado preentrenado.

La solución estándar es mezclar datos: 80% de imágenes de tu dominio específico, 20% de un dataset VQA general (Visual Question Answering — preguntas y respuestas sobre imágenes de temática general). Esto actúa como un "ancla" que mantiene las capacidades generales mientras el modelo aprende el dominio nuevo. No es una regla fija — con datasets muy pequeños (bajo 1000 ejemplos) la mezcla puede dilur demasiado la señal de dominio. Experimenta con ratios 90/10 y 80/20.

---

## Parte II — Voz: cómo un modelo aprende a hablar

### El problema que TTS moderno resolvió

La síntesis de voz tradicional (TTS en su forma clásica) era una cadena de etapas independientes, cada una con su propio modelo, sus propios hiperparámetros, y sus propios modos de fallo:

1. **Normalización de texto:** convertir "Sr." en "señor", "25/12" en "veinticinco de diciembre", expandir abreviaciones.
2. **Conversión a fonemas:** descomponer el texto en unidades de sonido (fonemas). El inglés tiene ~44 fonemas, el español ~24. Esta etapa requiere conocimiento lingüístico extenso y maneja mal las excepciones.
3. **Modelado de prosodia:** decidir el ritmo, el tono, el énfasis. Una pregunta sube al final. Una afirmación descends. Una exclamación tiene energía alta. Modelar esto requería reglas artesanales o modelos estadísticos separados.
4. **Síntesis de forma de onda:** convertir la representación fonética con prosodia en audio real, típicamente a través de una representación intermedia llamada mel-spectrogram — una representación visual del audio que muestra cómo varía la energía en diferentes frecuencias a lo largo del tiempo.

Cada etapa propagaba sus errores a la siguiente. Un error en la normalización (convertir mal una abreviatura) producía un error en los fonemas, que producía prosodia incorrecta, que producía audio defectuoso. El sistema completo era frágil y requería ingeniería especializada para mantener.

El TTS moderno basado en LLMs tira todo eso a la basura con una idea conceptualmente simple pero poderosa: tratar el audio como otro idioma más.

En lugar de predecir el siguiente token de texto, el LLM predice el siguiente token de audio. La arquitectura es el mismo transformer autoregresivo que ya conoces. El único ingrediente nuevo es un codec de audio neuronal — esencialmente, un "tokenizador para el sonido" — que convierte formas de onda de audio continuas en secuencias de números enteros discretos, y que puede reconstruir el audio a partir de esos números.

### El codec de audio neuronal: tokenizando el sonido

El codec es la pieza clave que hace posible el TTS basado en LLMs. Entenderlo en profundidad es esencial para entender por qué el fine-tuning funciona como funciona.

Un codec de audio neuronal tiene dos componentes principales:

**El encoder del codec** transforma una forma de onda de audio crudo (una secuencia de muestras numéricas que representan la presión del aire a lo largo del tiempo) en una secuencia de códigos enteros discretos. Es análogo a un tokenizador de texto: así como un tokenizador convierte "hola mundo" en `[28120, 49445]`, el encoder del codec convierte 1 segundo de audio en, digamos, `[2341, 891, 3027, ...]`. Estos enteros se toman de un codebook aprendido durante el entrenamiento del codec — un vocabulario de fragmentos de audio.

**El decoder del codec** hace el proceso inverso: toma los códigos enteros y reconstruye la forma de onda de audio. La reconstrucción no es perfecta (es compresión con pérdida, como MP3), pero los codecs modernos logran calidad notable incluso a tasas de bits bajas.

La innovación clave de codecs modernos como SNAC (Multi-Scale Neural Audio Codec), EnCodec, o DAC es su estructura jerárquica de múltiples capas de cuantización. Esto requiere explicación detallada porque es central para entender el formato de los datos de entrenamiento.

Imagina que quieres describir una canción a alguien que nunca la ha escuchado. Podrías hacerlo en tres niveles de detalle:
- **Nivel grueso:** "Es una balada lenta en mi menor, con voz masculina grave, tempo de 70 BPM."
- **Nivel medio:** "En el compás 3 hay un crescendo, la vocal 'a' en 'amor' se estira dos tiempos, hay vibrato al final de cada frase."
- **Nivel fino:** "La voz tiene una calidad ligeramente ronca, con un pequeño suspiro antes del coro, las consonantes sibilantes tienen un ligero siseo característico de esa persona."

El codec SNAC que usa Orpheus-TTS opera exactamente con esta filosofía en tres capas:

**Capa 1 (gruesa, ~12 Hz):** Captura la estructura global — ritmo, prosodia general, identidad del hablante. Produce 1 token por frame temporal. Si el audio dura 1 segundo, esta capa produce ~12 tokens.

**Capa 2 (media, ~24 Hz):** Captura patrones fonéticos y contornos de entonación. Corre el doble de rápido que la capa 1, produciendo 2 tokens por frame de la capa 1. En 1 segundo: ~24 tokens.

**Capa 3 (fina, ~48 Hz):** Captura la textura acústica — respiraciones, crujidos vocales, calidad tímbrica sutil. Corre cuatro veces más rápido que la capa 1, produciendo 4 tokens por frame de la capa 1. En 1 segundo: ~48 tokens.

> **Descripcion visual:** Diagrama de flujo horizontal con forma de abanico. A la izquierda, un bloque amarillo "Forma de onda audio crudo 24 kHz" apunta al encoder SNAC (azul). Desde el encoder salen tres flechas paralelas hacia tres bloques morados verticales que representan las capas jerarquicas: Capa 1 (ritmo global, 12 Hz), Capa 2 (contorno fonetico, 24 Hz), Capa 3 (textura acustica, 48 Hz). Las tres capas convergen en un bloque verde "Interleaving 7 tokens por frame". De ahi salen dos bloques rojos: Decoder SNAC y Audio reconstruido. Estilo limpio, fondo blanco, flechas con punta triangular.

En total, 1 segundo de audio se representa con $12 + 24 + 48 = 84$ tokens (aproximadamente 83 según la implementación exacta). Esto es manejable para un LLM. Un texto de 200 palabras (~266 tokens) corresponde a, digamos, 4 segundos de habla (~332 tokens de audio). La proporción es razonable.

Para que el LLM pueda procesar esta estructura jerárquica como una secuencia plana (el LLM solo sabe de secuencias), los tokens de las tres capas se entrelazan (interleave) en un patrón específico. Por cada frame temporal, la secuencia es:

$$[c^1_t, \; c^2_{2t}, \; c^3_{4t}, \; c^3_{4t+1}, \; c^2_{2t+1}, \; c^3_{4t+2}, \; c^3_{4t+3}]$$

donde $c^l_i$ es el token $i$-ésimo de la capa $l$. Esto produce 7 tokens por frame. Para 1 segundo de audio (12 frames a 12 Hz): $12 \times 7 = 84$ tokens. Para 10 segundos: 840 tokens. Este número importa mucho cuando calculamos los límites de generación, como veremos.

> **Descripcion visual:** Diagrama de flujo horizontal. A la izquierda, un recuadro contenedor gris "1 frame temporal (1/12 seg)" agrupa verticalmente 7 bloques de colores: uno azul (Token capa 1, ritmo global), dos verdes (Tokens capa 2a y 2b, fonetica), y cuatro rosas (Tokens capa 3a-3d, textura). El contenedor apunta a un bloque amarillo "Secuencia plana, 7 tokens por frame, 84 tokens por seg", que a su vez apunta a un bloque amarillo "LLM predice cada token en orden". Estilo limpio, fondo blanco, tipografia sans-serif.

### Cómo el LLM aprende a generar habla

Con el codec en mano, el pipeline de entrenamiento es conceptualmente sencillo.

**Preparación de datos:** dado un dataset de pares (texto, audio), se pasa cada audio por el encoder del codec para obtener las secuencias de tokens. Se extiende el vocabulario del LLM para incluir los IDs de tokens de audio — si el LLM original tiene 128,266 tokens de texto, los tokens de audio se asignan a partir del ID 128,267 en adelante, sin colisión.

La extensión de vocabulario funciona así. Cada capa del codec tiene un codebook de 4,096 entradas (es decir, 4,096 posibles valores). Con 3 capas, necesitamos $3 \times 4{,}096 = 12{,}288$ IDs adicionales, pero en la práctica se añaden offsets por capa para que el modelo pueda distinguir a qué capa pertenece cada token:

$$\text{ID\_en\_vocabulario} = c^l_i + V_\text{text} + (l-1) \times 4096$$

donde $V_\text{text} = 128{,}266$ es el tamaño del vocabulario de texto original y $l \in \{1,2,3\}$ es la capa del codec. Así, un token de valor 2341 de la capa 1 se mapea a $2341 + 128266 + 0 = 130607$. El mismo valor 2341 de la capa 2 se mapea a $2341 + 128266 + 4096 = 134703$. Son IDs distintos, por lo que el LLM los trata como tokens distintos — lo cual es correcto, porque representan cantidades diferentes.

**Entrenamiento:** el LLM aprende a predecir los tokens de audio dados los tokens de texto, con pérdida estándar de next-token prediction:

$$\mathcal{L}_\text{TTS} = -\sum_{i=1}^{|s|} \log P_{\theta + \Delta\theta}(s_i \mid t_1, \ldots, t_{|t|}, s_1, \ldots, s_{i-1})$$

donde $t = (t_1, \ldots, t_{|t|})$ son los tokens de texto de entrada y $s = (s_1, \ldots, s_{|s|})$ es la secuencia interleaved de tokens de audio. Los $\Delta\theta$ son, nuevamente, los adaptadores LoRA.

**Inferencia:** en tiempo de inferencia, el LLM recibe los tokens de texto y genera autoregresivamente los tokens de audio, uno por uno. Cuando termina de generar (llega a un token especial de fin de habla), los tokens de audio generados se pasan al decoder del codec para reconstruir la forma de onda.

> **Descripcion visual:** Diagrama de flujo horizontal con dos ramas de entrada que convergen en el LLM. La rama superior muestra "Texto de entrada" (amarillo) apuntando directamente al LLM Orpheus (azul). La rama inferior muestra "Audio de entrenamiento" (amarillo) pasando por el Encoder SNAC (morado) produciendo "Tokens de audio interleaved" (morado) que alimentan tambien al LLM. El LLM genera "Tokens de audio generados" (morado), que pasan por el Decoder SNAC (morado) para producir "Forma de onda reconstruida 24 kHz" (verde). Estilo limpio, fondo blanco, flechas grises con punta solida.

El LLM no "sabe" que está generando habla. Desde su perspectiva, está prediciendo el siguiente token en una secuencia, como siempre. Lo que hace que el sistema funcione es que el espacio de tokens de audio es lo suficientemente estructurado — gracias al entrenamiento del codec — para que el LLM pueda aprender patrones significativos en él.

### Por qué el fine-tuning supera a la clonación zero-shot

Los modelos TTS base modernos pueden hacer clonación de voz zero-shot: le das 10 segundos de audio de una voz, y el modelo intenta generar nueva habla con esa voz. El resultado es... aceptable. Captura el timbre general, las frecuencias básicas de la voz. Pero falla en los detalles que hacen reconocible a una persona: sus patrones de énfasis específicos, cómo sube el tono en las preguntas retóricas, cuánto tiempo pausa antes de una idea importante, sus vocales regionales, su tendencia a acelerar en los fragmentos emocionantes.

La clonación zero-shot es como contratar a un imitador que te escuchó hablar durante 10 segundos en un elevador. El fine-tuning es como contratar a alguien que pasó semanas estudiando cada uno de tus videos y podcasts.

En términos más técnicos: la clonación zero-shot funciona condicionando la generación en una representación del audio de referencia (el embedding de estilo de voz). Esa representación captura información estadística gruesa — frecuencia fundamental media, timbre global, ritmo general. El fine-tuning, en cambio, modifica los pesos del modelo para que el espacio de generación completo del LLM esté sesgado hacia los patrones de esa voz específica. No es un condicionamiento externo; es conocimiento internalizado.

La cantidad de audio necesaria para fine-tuning de voz es sorprendentemente pequeña: 30 minutos de audio limpio de un solo hablante suele ser suficiente para alta calidad. Lo que más importa es la calidad sobre la cantidad: condiciones de grabación consistentes, mínimo ruido de fondo, transcripciones precisas. 30 minutos de audio de alta calidad supera a 5 horas de audio con ruido de fondo y transcripciones incorrectas.

### Consideraciones prácticas de TTS que requieren atención especial

**El límite de duración por tokens.** Esto es crítico para el despliegue en producción y raramente se menciona en los tutoriales. Si tu codec produce 83 tokens por segundo de audio, y la longitud máxima de contexto del LLM es 2048 tokens (parámetro `max_seq_length` en el entrenamiento), entonces la duración máxima de audio generada es:

$$\text{duración máxima} = \frac{2048 \text{ tokens}}{83 \text{ tokens/segundo}} \approx 24.7 \text{ segundos}$$

Para Orpheus-TTS con max_seq_length = 2048, esto significa que no puedes generar más de ~24 segundos de audio de una vez. Para textos más largos — un artículo de 500 palabras, que hablaría durante 3-4 minutos — necesitas dividir el texto en fragmentos y generar el audio por partes. Esto introduce el problema de la consistencia entre fragmentos: la voz no debe cambiar de carácter entre el fragmento 1 y el fragmento 5. Orpheus maneja esto razonablemente bien porque la voz está internalizada en los pesos, no solo condicionada por un embedding externo.

**El repetition penalty: el control de calidad más importante en TTS.** Los modelos TTS basados en LLMs tienen una tendencia peculiar: sin un mecanismo corrector, pueden entrar en bucles de repetición, generando la misma sílaba o el mismo patrón fonético indefinidamente. Esto ocurre porque en el espacio de tokens de audio, las secuencias repetitivas son estadísticamente comunes (piensa en un sonido largo sostenido — es una repetición de frames similares) y el LLM puede quedar "atrapado" en un mínimo local de ese tipo.

El `repetition_penalty` — que típicamente multiplica por $(1/p)^\gamma$ la probabilidad de tokens que ya aparecieron en la secuencia generada, reduciendo su probabilidad de ser seleccionados — es la solución estándar. En generación de texto, un `repetition_penalty` de 1.1 es una precaución leve. En TTS, es prácticamente obligatorio con valores $\geq 1.1$. Sin él, o con valores demasiado bajos, el modelo puede producir minutos de "ta-ta-ta-ta-ta..." en lugar de habla. Con valores demasiado altos (>1.3), el modelo evita demasiado los patrones repetitivos y puede producir sonidos extraños o degradados, ya que cierto grado de repetición es natural en el habla continua.

**Tamaño del modelo vs calidad de audio.** En el dominio del texto, más parámetros generalmente implica mejor rendimiento — un modelo de 70B supera a uno de 7B en casi todas las benchmarks de razonamiento y conocimiento. En TTS, la relación es más compleja. La calidad del audio percibida por un oyente humano satura con tamaños relativamente pequeños de modelo, mientras que la latencia (el tiempo que tarda el modelo en generar el audio) escala directamente con el tamaño. Para aplicaciones de tiempo real — call centers, asistentes de voz — una latencia de 200ms marca la diferencia entre una experiencia fluida y una conversación entrecortada. Orpheus con 3B parámetros puede lograr ~200ms de latencia para streaming en hardware razonable. Un modelo de 70B necesitaría hardware mucho más especializado para acercarse a eso. Para TTS, el sweet spot práctico son modelos de 1-3B parámetros.

---

## Los modelos: Qwen3-VL y Orpheus-TTS

### Qwen3-VL (8B) — El modelo de visión

Qwen3-VL es el VLM más reciente de la serie Qwen de Alibaba. Existe en variantes densas de 2B, 4B, 8B y 32B parámetros, además de variantes Mixture-of-Experts (MoE — arquitecturas donde solo una fracción de los parámetros se activa por token, permitiendo modelos muy grandes con costo computacional menor). Para el lab usamos la versión densa de 8B, que ofrece una buena relación entre capacidad y entrenabilidad.

Varias características de Qwen3-VL merecen explicación detallada porque afectan directamente cómo lo finetuneas:

**Ventana de contexto de 256K tokens.** Los LLMs estándar tienen ventanas de contexto de 4K a 128K tokens. Qwen3-VL soporta hasta 256K tokens de forma nativa, extensible a 1M. En la práctica para fine-tuning de imágenes individuales esto no cambia mucho — las imágenes raras veces superan los 2K tokens después de compresión. Donde importa es para el procesamiento de documentos multipágina (puedes pasar un PDF completo como secuencia de imágenes de páginas) y para razonamiento sobre video (múltiples frames del mismo video).

**DeepStack: fusión de características multi-escala.** El Qwen2.5-VL anterior usaba únicamente la salida de la última capa del ViT. El problema con esto es que las últimas capas del ViT capturan semántica de alto nivel ("hay una cara humana") pero pierden detalles de bajo nivel ("la letra está mal formada", "hay una microgrieta"). Qwen3-VL introduce DeepStack, que fusiona características de múltiples niveles del ViT — capas intermedias (que retienen más información espacial y de bajo nivel) con la capa final (que tiene la semántica más rica). Esto mejora significativamente las tareas que requieren atención a detalles finos, como OCR de texto pequeño, detección de defectos industriales, o lectura de escritura a mano.

**Interleaved-MRoPE — posiciones en tres dimensiones.** Los transformers estándar usan embeddings posicionales unidimensionales: cada token tiene una posición 1, 2, 3, ... en la secuencia. Pero las imágenes y los videos tienen estructura espacial bidimensional (fila, columna) y los videos añaden la dimensión temporal. MRoPE (Multimodal Rotary Position Embedding) es una extensión de los embeddings posicionales rotatorios (RoPE) que asigna posiciones en tres dimensiones: tiempo, altura y anchura. Esto permite al modelo razonar sobre la posición espacial de objetos en la imagen ("el texto en la esquina superior izquierda") y la posición temporal en videos ("el marcador aparece a los 2:34"). La versión interleaved de Qwen3-VL asigna frecuencias posicionales completas en cada dimensión, lo que mejora la comprensión de secuencias visuales largas.

**Modo de razonamiento (thinking mode).** Las ediciones Instruct y Thinking de Qwen3-VL pueden activar o desactivar el razonamiento en cadena (chain-of-thought) explícito. Para tareas de OCR simple, el thinking mode añade overhead sin beneficio visible. Para problemas matemáticos visuales complejos ("resuelve la ecuación en esta imagen paso a paso"), el thinking mode puede ser la diferencia entre una respuesta correcta y una incorrecta.

### Orpheus-TTS (3B) — El modelo de voz

Orpheus-TTS, construido por Canopy Labs, es un sistema TTS open-source basado en el backbone Llama 3B. Si has finetuneado Llama para generación de texto, ya entiendes el 90% de la arquitectura de Orpheus. La diferencia es que fue entrenado para predecir tokens de audio en lugar de tokens de texto.

Usa el codec SNAC (Multi-Scale Neural Audio Codec) a 24kHz — 24,000 muestras de audio por segundo, que es calidad de telefonía HD (mejor que la telefonía tradicional de 8kHz, similar a lo que escuchas en una videollamada de buena calidad). Con las 3 capas del codec a 12/24/48 Hz, produce aproximadamente 83 tokens por segundo de audio.

Una característica especialmente interesante son las **emotive tags** — etiquetas de expresividad inline. El modelo reconoce etiquetas como `<laugh>`, `<sigh>`, `<chuckle>`, `<gasp>` dentro del texto, y las usa para guiar el estilo de generación. "Esto es increíble `<laugh>` no puedo creerlo" generará una risa natural integrada en el habla. Esto es posible porque esas etiquetas forman parte del vocabulario del modelo preentrenado — fueron incluidas en los datos de entrenamiento de Orpheus como secuencias especiales asociadas con los patrones de audio correspondientes.

Las 8 voces preset (tara, leah, jess, leo, dan, mia, zac, zoe) se seleccionan prefijando el prompt con el nombre de la voz: `"tara: Hola, soy Tara"`. El fine-tuning puede añadir una voz nueva al repertorio del modelo, o afinar una de las voces existentes hacia una personalización específica.

---

## El laboratorio, parte 1 — Escritura a mano en LaTeX con Qwen3-VL

Con la teoría de VLMs bien entendida, pasemos a construirlo con las manos. La tarea: finetunear Qwen3-VL 8B para convertir fotografías de fórmulas matemáticas escritas a mano en código LaTeX limpio y compilable. El modelo base puede describir una imagen en términos generales, pero no sabe producir LaTeX preciso desde notación manuscrita — eso requiere aprender el formato específico de salida y mejorar su capacidad de leer escritura no tipografiada.

El entorno es Google Colab con una GPU T4 gratuita. Usamos Unsloth, la librería de optimización de fine-tuning que hemos estado usando en capítulos anteriores, con su clase `FastVisionModel` especializada para VLMs.

### Paso 1: Carga del modelo en 4 bits

```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)
```

La diferencia respecto a un modelo de texto es mínima: `FastVisionModel` en lugar de `FastLanguageModel`. Unsloth maneja internamente la carga del codificador de visión, la capa de proyección, y el LLM backbone como un único objeto coherente. La cuantización a 4 bits (QLoRA) aplica al LLM y potencialmente a la proyección, pero el codificador de visión se mantiene con mayor precisión — degradar la precisión del codificador afecta directamente la calidad de las características visuales extraídas.

Una implicación práctica: aunque cargamos en 4 bits, el modelo Qwen3-VL 8B ocupa alrededor de 5-6 GB de VRAM en la T4 (que tiene 16 GB). El resto del VRAM queda para activaciones, gradientes y los adaptadores LoRA.

### Paso 2: Añadir adaptadores LoRA con control granular

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,  # ¿Entrenar el ViT?
    finetune_language_layers   = True,  # ¿Entrenar el LLM?
    finetune_attention_modules = True,  # ¿Entrenar atención?
    finetune_mlp_modules       = True,  # ¿Entrenar capas MLP?
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)
```

Unsloth expone cuatro interruptores independientes para decidir qué se entrena. Esto es más granular que en el fine-tuning de texto estándar. Algunas guías de diseño:

- `finetune_language_layers = True` siempre. El LLM es el que aprende el formato de salida.
- `finetune_vision_layers`: para la mayoría de tareas, empieza con `False`. Para escritura a mano (distribución visual inusual), prueba `True` y compara los resultados.
- `finetune_attention_modules` y `finetune_mlp_modules`: si activas el entrenamiento de alguna parte, entrenar ambos (atención y MLP) suele dar mejores resultados que entrenar solo uno.

El rango `r = 16` y `lora_alpha = 16` (lo que hace que el factor de escala $\alpha/r = 1.0$) es un buen punto de partida para tareas de visión de complejidad media. Para tareas que requieren mayor capacidad de adaptación (dominios muy alejados de los datos de preentrenamiento), prueba `r = 32` o `r = 64`.

### Paso 3: Preparar el dataset

```python
from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")
```

El dataset LaTeX_OCR contiene pares de imágenes de fórmulas manuscritas con su representación en LaTeX. La estructura de cada muestra es simple: una imagen PIL y una cadena de texto LaTeX.

El paso crítico es convertir cada muestra al formato de conversación que los VLMs esperan:

```python
instruction = "Write the LaTeX representation for this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
```

La estructura `{"type": "image"}` es el único elemento nuevo respecto al formato de conversación de texto que ya conoces de SFT. El tokenizador de Qwen3-VL sabe cómo procesar este formato: convierte la imagen en tokens visuales usando el codificador de visión, los proyecta a la dimensión del LLM, y los concatena con los tokens de texto del prompt antes de pasarlos al LLM.

Una nota sobre el proceso de tokenización: cuando el tokenizador procesa la imagen, primero la redimensiona para que encaje dentro del presupuesto de tokens configurado (controlado por `max_pixels` o parámetros similares en el modelo), luego la pasa por el ViT, y finalmente proyecta los tokens visuales. Este proceso ocurre dentro de la función `tokenizer()` — no tienes que manejarlo manualmente.

### Paso 4: Baseline antes del entrenamiento

Antes de entrenar, siempre es útil documentar el comportamiento base. Esto sirve tanto para verificar que el setup es correcto como para medir el impacto real del fine-tuning:

```python
FastVisionModel.for_inference(model)

image = dataset[2]["image"]
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Write the LaTeX representation for this image."}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
_ = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)
```

El modelo base intentará producir algo LaTeX-like, pero típicamente cometerá errores: usará `\frac` donde debería haber `\int`, omitirá subíndices, o producirá LaTeX sintácticamente inválido. Esto es exactamente lo esperado — el modelo sabe que "esta imagen contiene una fórmula matemática" gracias al ViT, pero no sabe el idioma específico de LaTeX para expresarla.

`FastVisionModel.for_inference()` llama internamente es importante: activa el modo de inferencia de Unsloth, que desactiva el cálculo de gradientes y optimiza la generación. Luego de entrenamiento necesitas llamar `FastVisionModel.for_training()` para volver al modo de entrenamiento.

### Paso 5: Entrenamiento con SFTTrainer

```python
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        # Requeridos para fine-tuning de visión — no omitir:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)
trainer_stats = trainer.train()
```

Tres diferencias respecto al SFTTrainer estándar de texto merecen atención:

`UnslothVisionDataCollator` — este collator personalizado maneja el proceso de agrupar muestras de diferentes tamaños en un batch. En el fine-tuning de texto, el collator simplemente rellena con padding tokens. Para visión, también necesita procesar las imágenes (pasarlas por el ViT, proyectar los tokens) para cada muestra del batch. El collator estándar de HuggingFace no sabe cómo hacer esto; el de Unsloth sí.

Los tres campos adicionales de `SFTConfig` — `remove_unused_columns=False`, `dataset_text_field=""`, `dataset_kwargs={"skip_prepare_dataset": True}` — son necesarios porque el SFTTrainer está diseñado para text-only y hace algunas suposiciones sobre el formato del dataset que no se cumplen cuando hay imágenes. `remove_unused_columns=False` evita que el Trainer descarte la columna de imágenes (que no reconoce como "texto"). `skip_prepare_dataset=True` le dice al Trainer que no intente tokenizar el dataset en el preprocessing — queremos que la tokenización ocurra dentro del collator, que es cuando tenemos la imagen disponible.

`max_steps = 30` para el demo vs `num_train_epochs = 1` para producción. En 30 pasos con batch size efectivo de 8 (2 × 4 gradient accumulation), el modelo ve 240 ejemplos — suficiente para ver mejoras claras en un demo de 5 minutos en T4. Para un dataset completo de miles de pares, un epoch completo puede tomar horas y producir resultados mucho más robustos.

### Paso 6: Evaluación y guardado

Después del entrenamiento, el modelo produce LaTeX limpio y compilable para las mismas imágenes que antes describía imprecisamente. La diferencia visual es dramática: donde antes generaba `x + y = z`, ahora genera `\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = 0` — sintácticamente correcto, semánticamente preciso, directamente compilable.

Para guardar y desplegar:

```python
# Guardar adaptadores LoRA (opción mínima, para HuggingFace Hub)
model.save_pretrained("qwen_lora")
tokenizer.save_pretrained("qwen_lora")

# Exportar a GGUF para despliegue local con llama.cpp u Ollama
model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method="q4_k_m")
```

La opción GGUF es especialmente útil para despliegue local: permite ejecutar el modelo en CPU (con degradación de velocidad) o con aceleración GPU parcial en hardware de consumo, sin necesidad de la infraestructura completa de HuggingFace + PyTorch.

---

## El laboratorio, parte 2 — Clonación de voz con Orpheus-TTS

Ahora cambiamos de modalidad completamente. Finetuneamos Orpheus-TTS sobre un dataset de voz para que el modelo aprenda a generar habla en esa voz específica, con sus características únicas de timbre, ritmo y expresividad.

### Paso 1: Carga del modelo en precisión completa

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)
```

Nótese el regreso a `FastLanguageModel` — porque Orpheus es un Llama 3B estándar bajo el capó. Y crucialmente, `load_in_4bit = False`. Este detalle importa: la calidad del audio de salida de un modelo TTS es perceptiblemente sensible a la cuantización del modelo. Mientras que para generación de texto la diferencia entre 4-bit y full-precision puede ser imperceptible en tareas de alto nivel, para TTS la cuantización a 4 bits puede introducir artefactos audibles — pequeñas distorsiones, pérdida de claridad en ciertas frecuencias, o inconsistencias en la prosodia. Si tu GPU tiene suficiente VRAM (el modelo de 3B ocupa ~12 GB en full bfloat16), usa precisión completa.

### Paso 2: Adaptadores LoRA con rango alto

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

El rango `r = 64` es significativamente mayor que el `r = 16` del ejemplo de visión. ¿Por qué? El fine-tuning de voz necesita capturar patrones acústicos sutiles — timbre, microritmo, calidad vocal — que son inherentemente de alta dimensionalidad. Con `r = 16`, los adaptadores LoRA tienen matrices de bajo rango que pueden no tener suficiente capacidad expresiva para capturar esos patrones. Con `r = 64`, los adaptadores pueden representar transformaciones más ricas en el espacio de activaciones del LLM.

La relación entre `r` y el número de parámetros adicionales es lineal: con `r = 64` versus `r = 16`, tienes 4 veces más parámetros en los adaptadores. Para Llama 3B con los módulos especificados, `r = 64` añade aproximadamente 20-30M de parámetros entrenables — manejable en VRAM y suficiente para capturar la personalidad vocal de un hablante.

### Paso 3: El proceso crítico de tokenización de audio

Esta es la parte más técnica del lab y donde más valor añade entender el mecanismo en detalle.

```python
from datasets import load_dataset
dataset = load_dataset("MrDragonFox/Elise", split="train")
```

El dataset MrDragonFox/Elise contiene grabaciones de audio de un único hablante (la voz "Elise") con sus transcripciones. Es un dataset diseñado para fine-tuning TTS: audio consistente, buenas condiciones de grabación, transcripciones precisas.

La siguiente función es el corazón del pipeline de datos:

```python
from snac import SNAC
import torchaudio.transforms as T

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")

def tokenise_audio(waveform):
    # 1. Convertir a tensor PyTorch
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    
    # 2. Resamplear a 24kHz (el codec SNAC opera a 24kHz)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    
    # 3. Codificar con SNAC para obtener las 3 capas de tokens
    waveform = waveform.unsqueeze(0).to("cuda")
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    # codes[0]: capa 1 (12Hz), shape [1, 1, T_1]
    # codes[1]: capa 2 (24Hz), shape [1, 1, T_2]  con T_2 = 2*T_1
    # codes[2]: capa 3 (48Hz), shape [1, 1, T_3]  con T_3 = 4*T_1
    
    # 4. Interleave: 7 tokens por frame, con offsets por capa
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)           # capa 1: offset base
        all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)  # capa 2a: offset+4096
        all_codes.append(codes[2][0][4*i].item() + 128266 + 2*4096)   # capa 3a
        all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + 3*4096) # capa 3b
        all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + 4*4096) # capa 2b
        all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + 5*4096) # capa 3c
        all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + 6*4096) # capa 3d
    
    return all_codes
```

Veamos el interleaving con números concretos para que la lógica sea clara. Supongamos que el primer frame produce los siguientes tokens raw de cada capa:
- Capa 1, frame 0: valor 2341
- Capa 2, frames 0-1: valores 891, 1203
- Capa 3, frames 0-3: valores 3027, 415, 2890, 1077

El interleaving produce la secuencia de 7 tokens:
1. $2341 + 128266 = 130607$ (capa 1)
2. $891 + 128266 + 4096 = 133253$ (capa 2, frame 0)
3. $3027 + 128266 + 8192 = 139485$ (capa 3, frame 0)
4. $415 + 128266 + 12288 = 140969$ (capa 3, frame 1)
5. $1203 + 128266 + 16384 = 145853$ (capa 2, frame 1)
6. $2890 + 128266 + 20480 = 151636$ (capa 3, frame 2)
7. $1077 + 128266 + 24576 = 153919$ (capa 3, frame 3)

Cada uno de estos 7 números es un ID único en el vocabulario extendido del modelo. El LLM aprende a predecirlos en este orden, y la estructura del interleaving garantiza que cuando predice el token de capa 3 (que aporta detalle acústico fino), ya ha generado el token de capa 1 (que establece el contexto rítmico global) para ese mismo frame temporal.

### Paso 4: Formato de las secuencias de entrenamiento

```python
def create_input_ids(example):
    text_prompt = example["text"]
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    text_ids.append(end_of_text)

    input_ids = (
        [start_of_human]
        + text_ids
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)
    return example
```

La estructura de la secuencia de entrenamiento es un protocolo de conversación especial: `[SOH]` (Start Of Human) marca el inicio del texto de entrada, `[EOH]` su final. Luego `[SOA]` (Start Of AI), `[SOS]` (Start Of Speech), la secuencia de tokens de audio, `[EOS]` (End Of Speech), y `[EOA]` (End Of AI).

Observa que `labels = input_ids` — el modelo aprende a predecir tanto los tokens de texto como los tokens de audio. Pero en la práctica, el aprendizaje significativo ocurre sobre los tokens de audio: el modelo ya sabe encodear texto (no necesita aprender eso de nuevo), pero necesita aprender qué secuencias de audio corresponden a qué texto para esta voz específica.

El paso de deduplicación de frames es una limpieza de datos inteligente:

```python
def remove_duplicate_frames(example):
    vals = example["codes_list"]
    result = vals[:7]  # Siempre mantener el primer frame
    for i in range(7, len(vals), 7):
        current_first = vals[i]       # Token de capa 1 del frame actual
        previous_first = result[-7]   # Token de capa 1 del frame anterior
        if current_first != previous_first:
            result.extend(vals[i:i+7])
    example["codes_list"] = result
    return example
```

El token de capa 1 (el más grueso) captura la estructura rítmica global. Si dos frames consecutivos tienen el mismo token de capa 1, significa que son "cuasi-silencio" o una consonante sostenida sin movimiento fonético significativo. Remover estas duplicaciones elimina las secciones mudas o repetitivas del audio, lo que produce datos de entrenamiento más limpios y eficientes.

### Paso 5: Entrenamiento con el Trainer estándar

```python
from transformers import TrainingArguments, Trainer

trainer = Trainer(
    model = model,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
trainer_stats = trainer.train()
```

Usamos el `Trainer` estándar en lugar de `SFTTrainer`. La razón: `SFTTrainer` está diseñado para recibir texto crudo y tokenizarlo internamente. Nosotros ya hemos hecho la tokenización compleja (codec SNAC + interleaving + offsets) en el paso anterior, y el resultado está en `input_ids`. El `SFTTrainer` intentaría rehacer esa tokenización y rompería el formato. El `Trainer` base acepta directamente `input_ids` precomputados.

`per_device_train_batch_size = 1` refleja que las secuencias de TTS son largas. Un clip de audio de 10 segundos produce ~830 tokens, más el texto (~50-100 tokens) más los tokens especiales. Una secuencia de entrenamiento puede tener 900-1000 tokens. Con batch size de 1 y gradient_accumulation de 4, el batch efectivo es 4, lo que permite entrenar de forma estable sin quedarse sin VRAM.

`max_steps = 60` — más pasos que en el ejemplo de visión porque los patrones acústicos son más sutiles y requieren más ejemplos para consolidarse. Para un fine-tuning de producción sobre 30 minutos de audio (aproximadamente 1000-2000 clips dependiendo de la longitud media de cada frase), un epoch completo puede ser suficiente.

### Paso 6: Inferencia — escuchando los resultados

```python
FastLanguageModel.for_inference(model)
snac_model.to("cpu")  # Liberar GPU para la generación del LLM

prompts = [
    "Hey there my name is Elise, <giggles> and I'm a speech generation model.",
]
```

La secuencia de inferencia es el proceso inverso al entrenamiento: el LLM genera autoregresivamente los tokens de audio, que luego se pasan al decoder del SNAC para reconstruir la forma de onda.

`snac_model.to("cpu")` libera VRAM de GPU durante la fase de generación del LLM. La GPU se necesita para la generación autoregresiva (que es donde está el cuello de botella computacional). Luego, cuando los tokens de audio ya están generados, se mueve SNAC de vuelta a GPU (o se usa CPU) para el decoding, que es más barato computacionalmente.

La etiqueta `<giggles>` en el prompt es un ejemplo de las emotive tags de Orpheus. El modelo finetuneado, habiendo aprendido los patrones de audio de la voz Elise, intentará generar esa risa en el estilo y timbre específicos de Elise — no como una risa genérica, sino integrada con la calidad vocal específica aprendida.

La diferencia audible entre el modelo base y el finetuneado es el indicador más honesto del éxito. El modelo base genera voces con los estilos genéricos de sus 8 voces preset. El modelo finetuneado genera en el timbre, el ritmo, y la personalidad de la voz de Elise del dataset de entrenamiento.

### Paso 7: Guardado para producción

```python
# Adaptadores LoRA (ligeros, para iterar rápido)
model.save_pretrained("orpheus_lora")
tokenizer.save_pretrained("orpheus_lora")

# Modelo merged en 16 bits (para despliegue con máxima calidad)
model.save_pretrained_merged("orpheus_finetune_16bit", tokenizer, save_method="merged_16bit")
```

Para TTS el modelo merged en 16 bits es a menudo la elección correcta para producción. La calidad del audio es perceptiblemente mejor que con adaptadores LoRA en 4 bits. El tamaño es mayor (3B × 2 bytes × float16 ≈ 6 GB), pero la diferencia en calidad de audio vale el costo de almacenamiento en la mayoría de aplicaciones.

---

## Qué aprendimos y qué sigue

La conclusión central de este capítulo es la misma que motivó el diseño de los sistemas que hemos estudiado: las modalidades cambian, la mecánica no. El fine-tuning de un VLM o un sistema TTS es conceptualmente idéntico al fine-tuning de texto que hemos estado haciendo. Los adaptadores LoRA van sobre las capas de atención del transformer backbone. Los datos se formatean como conversaciones (usuario-asistente o texto-audio). El entrenamiento usa pérdida de next-token prediction. El optimizador es AdamW.

Lo que varía son los pasos de preprocesamiento — y esos pasos reflejan la física del problema: las imágenes tienen estructura espacial que el codificador de visión traduce a tokens; el audio tiene estructura temporal jerárquica que el codec neuronal traduce a secuencias discretas. Pero una vez esos preprocesadores hacen su trabajo, el LLM ve exactamente lo mismo que siempre: una secuencia de tokens, y la tarea de predecir el siguiente.

Lo que esto habilita en términos de aplicaciones es notable. Con visión, puedes construir sistemas que lean documentos, inspeccionen productos industriales, analicen imágenes médicas, o conviertan cualquier información visual en texto estructurado. Con TTS, puedes construir asistentes de voz personalizados, sistemas de narración automática, call centers con voces consistentes, o content engines que generen audio en cualquier voz.

> **Descripcion visual:** Diagrama de flujo horizontal dividido en tres columnas paralelas representando subgrafos. La columna izquierda (tonos azules) muestra "Fine-tuning de Texto" con tres bloques verticales simples: tokens de texto, LLM con LoRA, texto generado. La columna central (tonos verdes) muestra "Fine-tuning de Vision" con cinco bloques: imagen mas texto, ViT congelado, projector que entrena, LLM con LoRA, texto generado. La columna derecha (tonos rojos) muestra "Fine-tuning TTS" con cinco bloques: texto mas audio, codec encoder, LLM con LoRA, codec decoder, audio generado. Estilo comparativo, fondo blanco, tipografia sans-serif, cada columna enmarcada con borde gris punteado.

| Dimensión | Fine-tuning de texto | Fine-tuning de visión | Fine-tuning TTS |
|---|---|---|---|
| Backbone | Transformer LLM | Transformer LLM | Transformer LLM |
| Adaptadores | LoRA sobre atención | LoRA sobre atención (y opcionalmente ViT) | LoRA sobre atención |
| Input extra | — | ViT + projector | Codec neuronal (encoder) |
| Output extra | — | — | Codec neuronal (decoder) |
| Formato de datos | Conversaciones texto | Conversaciones texto + imagen | Texto + tokens de audio |
| Librería de carga | FastLanguageModel | FastVisionModel | FastLanguageModel |
| Cuantización 4-bit | Sí | Sí | Con precaución (calidad) |
| Rango LoRA típico | r=8-32 | r=16-32 | r=32-64 |
| Costo principal | LLM forward/backward | LLM forward/backward | LLM forward/backward |

El denominador común de esa tabla es que el costo principal — en tiempo, en VRAM, en complejidad — siempre es el LLM forward y backward pass. Los componentes adicionales (codificadores, decoders, codecs) son relativamente baratos una vez que los entiendes. Y LoRA sigue siendo la herramienta de elección porque permite entrenar adaptadores compactos sobre un backbone congelado, exactamente lo que queremos en todos los casos.

---

## Tags

#modelo/vision-language-model #técnica/fine-tuning-multimodal #modelo/vision-transformer #técnica/text-to-speech #modelo/neural-audio-codec #técnica/lora #concepto/olvido-catastrófico #nivel/intermedio #tipo/lección #estado/completo




---
title: "Inferencia LLM a Escala: de un Request a Miles de Usuarios"
aliases:
  - "Capítulo 8"
  - "Cap 8"
  - "Inferencia a Escala"
  - "LLM Inference at Scale"
capitulo: 8
slug: 08-inferencia-llms-a-escala
tema: inferencia-y-serving
dificultad: "avanzado"
prerequisitos:
  - "[[01-fundamentos-transformers-y-pretraining]]"
  - "[[04-qlora-cuantizacion-4bit]]"
  - "[[07-finetuning-multimodal-vision-tts]]"
relacionados:
  - "[[02-supervised-finetuning]]"
  - "[[03-lora-adaptacion-de-bajo-rango]]"
conceptos_centrales:
  - prefill
  - decode
  - kv-cache
  - ttft
  - tpot
  - static-batching
  - continuous-batching
  - ragged-batching
  - chunked-prefill
  - prefill-decode-disaggregation
  - paged-attention
  - vllm
  - kvcached
  - ollama
  - llama-cpp
  - sglang
  - glm-ocr
  - compute-bound
  - memory-bandwidth-bound
  - block-manager
tags:
  - inferencia
  - serving
  - vllm
  - paged-attention
  - kv-cache
  - prefill
  - decode
  - chunked-prefill
  - disaggregation
  - ollama
  - llama-cpp
  - glm-ocr
  - multimodal
  - nivel/avanzado
  - tipo/lab
  - estado/completo
---

# Capítulo 8 — Inferencia LLM a Escala: de un Request a Miles de Usuarios

> Basado en "A Practical Guide to LLM Inference at Scale" y "Run the World's Best OCR on Your Own Laptop", The Neural Maze, Lección y Lab 8/8.

Siete capítulos han girado en torno a la misma pregunta: ¿cómo hacer que un modelo aprenda mejor? Arquitecturas, [[03-lora-adaptacion-de-bajo-rango|LoRA]], [[04-qlora-cuantizacion-4bit|QLoRA]], [[05-rlhf-alineacion-llms|RLHF]], GRPO, fine-tuning multimodal. Todo ese trabajo tiene un único destino: un modelo entrenado que alguien va a usar. Y es en ese momento — cuando el modelo pasa de archivo de pesos a servicio real con usuarios reales — cuando un conjunto completamente distinto de problemas aparece en escena.

El fine-tuning puede estar perfecto. El modelo puede alcanzar state-of-the-art en tu benchmark. Y aun así, si el sistema de inferencia está mal configurado, cada usuario espera 30 segundos por el primer token, la GPU está al 12% de utilización, y el coste de servir mil usuarios concurrentes supera lo que cobras. Eso no es un problema de modelo. Es un problema de infraestructura de inferencia.

Lo que hace fascinante este problema es que, en el fondo, es una historia de dos fases que se odian la una a la otra. Una quiere toda la potencia de cálculo de la GPU. La otra necesita todo el ancho de banda de memoria. Y las dos compiten en el mismo hardware, al mismo tiempo, por cada request que llega. Toda la historia de la inferencia LLM moderna es el intento de negociar esa tensión de forma cada vez más elegante — y cada solución introduce un nuevo problema que la siguiente solución tiene que resolver.

En este capítulo trazaremos esa cadena completa: desde cómo funciona un único request, pasando por cinco generaciones de estrategias de batching y scheduling, hasta sistemas de producción como vLLM y kvcached. Y lo remataremoos ensuciándonos las manos con GLM-OCR, un modelo de OCR de 0.9B parámetros que alcanza el estado del arte en comprensión de documentos complejos, usándolo localmente con Ollama y analizando exactamente por qué el tiempo hasta el primer token es de casi tres minutos mientras que la generación tarda solo nueve segundos — una demostración perfecta de los conceptos teóricos en acción.

---

## El ciclo de vida de un request: prefill, decode y KV cache

Para entender por qué la inferencia a escala es difícil, hay que entender primero qué le ocurre a un único request desde que llega hasta que termina. Hay dos fases, y son computacionalmente tan distintas que en sistemas avanzados se ejecutan en hardware físicamente separado.

### La fase de prefill: leer el libro completo de una vez

Cuando un usuario envía un prompt — digamos, un documento de 2.000 tokens para resumir — el modelo no puede generar la respuesta sin antes procesar toda esa entrada. La fase de **prefill** (también llamada *prompt processing* o *context ingestion*) es precisamente eso: el modelo hace un único forward pass sobre todos los tokens del prompt, calculando sus representaciones internas en paralelo.

La palabra clave es "en paralelo". El [[01-fundamentos-transformers-y-pretraining|transformer]] procesa los 2.000 tokens simultáneamente mediante multiplicaciones de matrices masivas. Eso es lo que hace a esta fase **compute-bound** (limitada por capacidad de cómputo): la GPU está ejecutando FLOPs a máxima velocidad, y el cuello de botella es el número de operaciones aritméticas que puede hacer por segundo.

La métrica de rendimiento que mide esta fase es el **TTFT (Time to First Token)**, es decir, cuánto tiempo tarda el usuario en recibir el primer token de respuesta desde que envió su prompt. Para aplicaciones de chat en tiempo real, un TTFT por encima de 1-2 segundos empieza a sentirse lento. Para agentes que procesan documentos largos en pipeline, puede tolerarse un TTFT mayor.

Al final de la fase de prefill, el modelo ha generado el primer token. Pero además — y esto es crucial — ha calculado y almacenado en memoria GPU algo llamado el **KV cache**.

### El KV cache: la optimización que desplaza el cuello de botella

Durante el forward pass de la fase de prefill, cada capa del transformer computa matrices de consultas (Q), claves (K) y valores (V) para cada token. El mecanismo de atención necesita las matrices K y V de todos los tokens anteriores para calcular la atención del token actual. Sin caché, habría que recalcular K y V de los 2.000 tokens de entrada para cada nuevo token generado — un coste cuadrático que haría la inferencia prohibitivamente lenta.

La solución es guardar en memoria las matrices K y V calculadas durante el prefill. A esto se le llama **KV cache**. Con él, generar cada nuevo token solo requiere calcular Q, K y V para ese único token nuevo, y combinarlos con las K y V ya almacenadas. El coste por token pasa de cuadrático a lineal.

¿Cuánta memoria ocupa esto? La fórmula es:

$$\text{Tamaño KV cache} = 2 \times n_\text{layers} \times n_\text{heads} \times d_\text{head} \times L \times \text{bytes\_por\_elemento}$$

Donde:
- El factor $2$ aparece porque guardamos tanto K como V (dos matrices separadas).
- $n_\text{layers}$ es el número de capas transformer del modelo.
- $n_\text{heads}$ es el número de cabezas de atención por capa.
- $d_\text{head}$ es la dimensión de cada cabeza, normalmente $d_\text{model} / n_\text{heads}$.
- $L$ es la longitud de la secuencia (tokens de entrada más tokens generados hasta el momento).
- $\text{bytes\_por\_elemento}$ es 2 bytes para FP16/BF16, 4 bytes para FP32.

Pongamos números concretos para un modelo de 66B parámetros con arquitectura típica: supongamos $n_\text{layers} = 80$, $d_\text{model} = 8.192$, y cabezas de 64 dimensiones cada una (lo que implica $n_\text{heads} = 128$). Para una secuencia de 512 tokens en FP16:

$$\text{Tamaño KV cache} = 2 \times 80 \times 128 \times 64 \times 512 \times 2 \text{ bytes}$$

$$= 2 \times 80 \times 8.192 \times 512 \times 2 \text{ bytes}$$

$$= 2 \times 80 \times 8.388.608 \text{ bytes}$$

$$= 2 \times 671.088.640 \text{ bytes} \approx 1.34 \text{ GB}$$

Más de un gigabyte por request de 512 tokens en un único modelo grande. Si sirves 100 requests concurrentes con contextos de 2.000 tokens, el KV cache solo ocupa más de 20 GB de VRAM — incluso antes de contar los propios pesos del modelo. Esta es la razón por la que la memoria de la GPU se convierte en el recurso más escaso de la inferencia a escala, y la razón por la que toda la ingeniería de los últimos años se ha centrado en gestionarla con más inteligencia.

### La fase de decode: escribir la carta palabra a palabra

Una vez que el prefill ha terminado y el primer token está generado, comienza la fase de **decode** (también llamada *autoregressive generation* o *generation phase*). Aquí el modelo genera los tokens de la respuesta uno a uno, de forma secuencial e inevitable: no puedes generar el token 5 sin haber generado el token 4, porque cada nuevo token se convierte en parte del contexto del siguiente.

En cada paso de decode, el modelo procesa exactamente un token — el último generado — y produce el siguiente. El KV cache almacena el contexto de todos los tokens anteriores, así que la carga computacional por paso es pequeña. Pero hay un problema fundamental: aunque el cómputo es mínimo, el modelo todavía tiene que cargar sus parámetros completos (decenas o cientos de gigabytes) desde la **HBM (High-Bandwidth Memory, la VRAM de la GPU)** al **SRAM (la memoria on-chip ultra-rápida de los núcleos de cálculo)** en cada paso.

Cargar pesos masivos para hacer muy poco trabajo con ellos es el problema definitorio de la fase de decode: la GPU está **memory-bandwidth bound** (limitada por el ancho de banda de memoria). Sus miles de núcleos de cálculo están infrautilizados porque la operación dominante no es calcular — es mover bytes. El cómputo termina en microsegundos, pero espera a que los datos lleguen desde la HBM durante milisegundos.

La métrica que mide esta fase es el **TPOT (Time Per Output Token)**, el tiempo entre tokens consecutivos. También se llama *inter-token latency* (latencia entre tokens). Para una experiencia de streaming fluida, valores por debajo de 50ms por token son razonables; por encima de 100ms empieza a sentirse entrecortado.

La analogía que fija estos conceptos: el **prefill es como leer un libro completo de una sola vez** — consumes toda la información en paralelo, es cognitivamente intenso, y tarda en función de cuántas páginas tiene. El **decode es como escribir una carta a mano, palabra por palabra** — en cada momento solo escribes una palabra, pero tienes que ir al archivador a buscar tu vocabulario completo antes de cada una. El trabajo por palabra es mínimo, pero el tiempo de acceso al archivador domina el proceso.

*Figura 8.1 — Flujo prefill → decode: el prompt se procesa en paralelo (compute-bound), se construye el KV cache, y luego el decode itera token a token leyendo ese cache (memory-bandwidth-bound).*

> **Descripción visual:** Diagrama de flujo horizontal izquierda a derecha. Un bloque naranja "Prompt 2000 tokens" apunta a un bloque verde "Prefill forward pass paralelo". Desde prefill salen dos flechas: una a un bloque naranja "KV Cache almacenado en HBM" y otra a un bloque rojo "Primer token". Ambos convergen en un bloque azul "Decode token a token", que apunta a un bloque rojo "Token N" con una flecha de retorno etiquetada "siguiente paso". Fondo blanco, tipografía sans-serif, estilo limpio.

---

## Batching estático: el autobús que no sale hasta que todos bajan

Con las dos fases claras, el problema de escala empieza a tomar forma. Una GPU de alto rendimiento puede ejecutar decenas de miles de FLOPs por nanosegundo. Servir requests uno a uno desperdicia casi toda esa capacidad. La solución natural es procesar varios requests a la vez: **batching**.

La forma más simple de batching es el **static batching** (batching estático). El motor de inferencia espera hasta tener un número fijo de requests — digamos, un batch de 8 — y los procesa todos a la vez como una única operación de tensor. Al procesar un batch, los pesos del modelo se cargan una sola vez desde HBM y se usan para calcular los 8 requests simultáneamente. En términos de utilización de GPU, esto es mucho mejor que procesar cada request por separado.

Hay un requisito técnico que complica el static batching: las operaciones de tensor estándar trabajan con tensores rectangulares. Si los 8 prompts de un batch tienen longitudes distintas (50 tokens, 340 tokens, 120 tokens...), no puedes apilarlos en una matriz sin igualar sus longitudes. La solución clásica es **padding**: añadir tokens especiales al final de los prompts más cortos hasta que todos tengan la misma longitud que el más largo. Si el prompt más largo tiene 340 tokens y el más corto tiene 50, el prompt corto lleva 290 tokens de relleno que no contienen información pero sí consumen cómputo.

El coste de este padding es cuadrático en el desequilibrio: un batch con un outlier muy largo pagará padding masivo en todos los demás.

Pero el padding es el problema menor. El problema mayor del static batching es su política de finalización: **el batch entero termina cuando termina el último request**. Imagina un batch de 8 requests donde 7 generan respuestas cortas de 50 tokens pero uno necesita 500 tokens. Los 7 requests cortos terminan rápido, pero sus slots en la GPU quedan completamente ociosos esperando a que el octavo termine. La GPU sigue cargando los pesos completos del modelo en cada iteración de decode, pero la mayor parte del trabajo se desperdicia en slots que ya no tienen nada que generar.

La analogía es exacta: **el static batching es el autobús que no puede salir hasta que todos los pasajeros lleguen, y no puede dejar subir a nadie nuevo hasta que el último pasajero del viaje anterior se baje**. En horas pico, la consecuencia es un autobús que sale casi vacío cada vez, o que tiene a la mitad de los pasajeros esperando en la parada durante un viaje entero.

En producción, esto se traduce en:

- **Infrautilización severa de GPU**: si la distribución de longitudes de salida tiene varianza alta (y siempre la tiene en producción), una fracción grande del tiempo de GPU se gasta en padding o en slots ociosos esperando al request más largo.
- **Latencia adicional para nuevos requests**: un usuario que llega justo después de que empieza el batch tiene que esperar a que el batch entero termine antes de ser procesado. En el peor caso, espera tanto tiempo como el request más largo del batch anterior.

Hay una variante llamada **dynamic batching** que mejora ligeramente el static batching: en lugar de esperar un número fijo de requests, espera un tiempo fijo (digamos 100ms) y agrupa los que lleguen en ese ventana. Esto balancea mejor el throughput y la latencia, pero no resuelve el problema fundamental: los requests cortos siguen esperando a que terminen los largos.

*Figura 8.2 — Static vs. continuous batching: en static los slots ociosos (rojo) esperan al request más largo; en continuous el slot se libera de inmediato y entra el siguiente request.*

> **Descripción visual:** Diagrama de flujo horizontal con dos subgrafos. Izquierda, "Static Batching": tres bloques naranja de requests apuntan a dos bloques rojos "IDLE esperando" que convergen en "Fin batch". Derecha, "Continuous Batching": dos bloques naranja apuntan a un bloque verde "Slot libre al instante" y a "Nuevo req entra ya". Una flecha etiquetada "GPU idle = desperdicio" conecta los dos subgrafos. Fondo blanco, estilo limpio.

---

## Continuous batching: la cinta transportadora que nunca para

La solución definitiva a las limitaciones del static batching es el **continuous batching**, también conocido como **iteration-level scheduling** (scheduling a nivel de iteración). El nombre "iteration-level" captura la idea clave: en lugar de tomar decisiones de scheduling a nivel de batch (al principio y al final), las decisiones se toman en cada iteración de decode.

La mecánica es la siguiente. En cada paso de generación, el scheduler inspecciona el estado del batch:

1. ¿Algún request acaba de generar un token de fin de secuencia (EOS)? Si es así, ese request ha terminado. Su slot se libera inmediatamente.
2. ¿Hay requests esperando en la cola? Si hay un slot libre, el siguiente request de la cola se inserta en el batch ahora mismo.

No se espera a que el batch entero termine. No hay fronteras rígidas entre batches. El batch es un conjunto vivo de requests en distintos estados de generación, que se actualiza en cada iteración.

La analogía correcta es la **cinta transportadora**: en una fábrica con cinta transportadora, cuando una pieza sale de la línea, inmediatamente entra la siguiente. No hay esperas entre ciclos. El throughput depende de la velocidad de la cinta, no del tiempo que tarda la pieza más lenta.

Esto plantea un problema técnico: si los requests del batch están en iteraciones distintas (uno en el token 3, otro en el token 47, otro recién insertado empezando el prefill), sus secuencias tienen longitudes distintas en cada momento. El padding rectangular ya no es factible — el derroche sería enorme si tuvieras que igualar longitudes en cada iteración con requests en estados tan dispares.

La solución es el **ragged batching** (batching irregular o jagged). En lugar de apilar los tokens en una matriz rectangular con padding, los tokens de todos los requests se concatenan en una única secuencia larga y plana. Un request con 23 tokens y otro con 147 tokens simplemente se pegan uno detrás del otro, formando una secuencia de 170 tokens.

Para que la atención funcione correctamente — es decir, para que los tokens del primer request no "vean" los tokens del segundo request en el mecanismo de atención — el sistema usa **attention masks** (máscaras de atención) precisas que permiten la atención dentro de cada request pero la bloquean entre requests distintos. Este uso de masks no es nuevo; lo viste en el capítulo 2 durante el [[02-supervised-finetuning|SFT]]. Aquí simplemente se usa de forma más agresiva para permitir mezclar secuencias de longitudes completamente arbitrarias.

El continuous batching también introduce una nueva complejidad: ¿cómo mezclar la fase de prefill (compute-bound) de los nuevos requests con la fase de decode (memory-bandwidth-bound) de los requests en curso? Los frameworks modernos como vLLM implementan schedulers que alternan prefill y decode de forma inteligente — por ejemplo, no insertar el prefill de un prompt de 8.000 tokens si eso va a bloquear el decode de 32 requests en curso durante 500ms. Esta tensión entre prefill y decode dentro del continuous batching es precisamente el problema que resuelve la siguiente técnica.

---

## Chunked prefill: trocear el problema para repartir el dolor

El continuous batching resuelve el problema de los slots ociosos, pero descubre un problema nuevo: la **prefill-decode interference** (interferencia prefill-decode).

Cuando llega un nuevo request con un prompt muy largo — digamos, un documento de 8.000 tokens para analizar —, el sistema tiene que ejecutar su prefill antes de empezar a generar respuesta. Ese prefill de 8.000 tokens es una operación masiva, compute-bound, que puede tardar cientos de milisegundos. Si este prefill entra en el batch junto con requests en fase de decode, esos requests de decode tienen que esperar a que termine el prefill para avanzar al siguiente token. Su TPOT se dispara — el usuario que estaba recibiendo respuesta fluida de repente ve una pausa larga y antinatural.

El **chunked prefill** (prefill en trozos) ataca este problema dividiéndolo. En lugar de procesar el prompt largo de 8.000 tokens en un único forward pass monolítico, el sistema lo divide en **chunks** (fragmentos) de tamaño fijo — por ejemplo, 512 tokens cada uno. El scheduler procesa un chunk de prefill en cada iteración, intercalado con los pasos de decode de los otros requests.

El mecanismo se apoya en el KV cache. En el primer chunk, el modelo procesa los tokens 1-512 del prompt y guarda sus estados KV. En el segundo chunk, el modelo procesa los tokens 513-1024 y puede reutilizar (añadir al) KV cache ya calculado para el primer chunk, actualizando la attention mask para reflejar el contexto acumulado. Así sucesivamente hasta completar el prompt.

¿Por qué funciona esto? Porque el scheduler puede ahora "presupuestar" tokens por iteración. Si impone un límite de, digamos, 2.048 tokens por iteración, y hay 32 requests en decode (32 tokens) más un prefill pendiente, puede meter 32 tokens de decode más 512 tokens de prefill (= 544 tokens en total, bien dentro del presupuesto) en la misma iteración. Los requests de decode avanzan sin pausas significativas. El prefill progresa en segundo plano.

Pero el chunked prefill no es gratis. Introduce dos trade-offs importantes:

**Trade-off 1: TTFT aumenta para el request en prefill.** El usuario cuyo prompt se está procesando en chunks tiene que esperar más iteraciones antes de recibir su primer token, porque el prefill se reparte entre múltiples iteraciones compartidas. Con prefill monolítico, ese usuario esperaría, pongamos, 300ms (todo el prefill de golpe) y luego recibiría el primer token. Con chunked prefill, el prefill se distribuye en 16 chunks de 50 tokens cada uno, intercalados con decode de otros requests, y el usuario podría esperar 800ms para el primer token — más tiempo total, pero sin bloquear a nadie más.

**Trade-off 2: overhead cuadrático de recarga del KV cache.** Para procesar el chunk $k$, el motor tiene que cargar desde HBM al SRAM el KV cache de todos los chunks anteriores 1 a $k-1$. Este coste de carga crece cuadráticamente con el número de chunks, porque cada nuevo chunk carga más contexto. Con contextos muy largos y chunks muy pequeños, este overhead puede dominar el coste total del prefill.

En la práctica, el tamaño del chunk es un hiperparámetro crítico. Chunks muy pequeños (128 tokens) minimizan la interferencia en el TPOT de los requests de decode pero maximizan el overhead de recarga y aumentan el TTFT del request en prefill. Chunks grandes (2.048 tokens) tienen el efecto contrario. Los frameworks de producción típicamente usan valores entre 512 y 2.048 tokens y los exponen como parámetro configurable.

*Figura 8.3 — Chunked prefill: el prompt de 8000 tokens se divide en chunks que se intercalan iteración a iteración con los pasos de decode de requests en curso, sin bloquearlos.*

> **Descripción visual:** Diagrama de flujo horizontal. Un bloque naranja grande "Prompt largo 8000 tokens" diverge en tres bloques naranja (Chunk 1, 2, 3). Un bloque azul "Requests en decode activo" y los chunks convergen en tres bloques verdes de iteraciones encadenados de izquierda a derecha. El último bloque verde apunta a un bloque rojo "Primer token del prompt largo". Fondo blanco, estilo limpio.

---

## Disaggregación prefill-decode: separar físicamente lo que no puede coexistir

El chunked prefill suaviza la interferencia prefill-decode, pero no la elimina. A escala de miles de usuarios concurrentes, incluso una interferencia pequeña se amplifica hasta hacerse intolerable — y la causa fundamental sigue siendo la misma: prefill y decode tienen necesidades radicalmente distintas del hardware, pero comparten los mismos chips.

El prefill es compute-bound: quiere alta densidad de cómputo por unidad de tiempo. Se beneficia de intra-operator parallelism: dividir la multiplicación de matrices de atención entre muchos núcleos de GPU usando tensor parallelism. El decode es memory-bandwidth-bound: quiere alta velocidad de lectura desde HBM. Se beneficia de tener más requests en el batch simultáneamente para amortizar la carga de pesos entre más secuencias.

Estas necesidades no solo son distintas — son conflictivas. Optimizar la partición de GPU para prefill (más tensor parallelism, más núcleos) la hace peor para decode, y viceversa.

La solución arquitectónica de última generación es la **prefill-decode disaggregation** (disaggregación prefill-decode): asignar las dos fases a GPUs físicamente distintas.

En un sistema disaggregado:
- Las **prefill instances** son GPUs dedicadas exclusivamente a procesar prompts. Están configuradas con alta paralelización para maximizar el throughput de prefill.
- Las **decode instances** son GPUs dedicadas exclusivamente a la generación token a token. Están configuradas para maximizar el tamaño del batch y la eficiencia de memoria.

El flujo de un request es: llega a una prefill instance, que procesa el prompt completo y genera el primer token. Una vez que el prefill termina, la prefill instance transfiere el KV cache generado y el primer token a una decode instance, que toma el control y genera el resto de la respuesta.

Esta separación elimina totalmente la interferencia: ningún prefill puede bloquear ningún decode, porque viven en GPUs distintas. Además, permite escalar cada fase de forma independiente: si los prompts de tu servicio son muy largos (documentos, código), necesitas más prefill instances; si las respuestas son muy largas, necesitas más decode instances. Esta flexibilidad de escalado independiente es muy difícil de conseguir con arquitecturas colocalizadas.

### El cuello de botella de la transferencia: 90 Gbps no es suficiente

La disaggregación introduce un nuevo problema de primera magnitud: **transferir el KV cache** entre las prefill instances y las decode instances a través de la red.

Calculemos el volumen de datos involucrado. En la sección anterior ya vimos que servir un request de 512 tokens en un modelo de 66B parámetros genera más de 1 GB de KV cache. Ahora imaginemos que estamos sirviendo 100 requests por segundo. El volumen de KV cache que hay que transferir es:

$$\text{Ancho de banda requerido} = 100 \text{ req/s} \times 1.34 \text{ GB/req} = 134 \text{ GB/s}$$

Una red InfiniBand HDR moderna ofrece aproximadamente 200 Gbps = 25 GB/s. Incluso eso es insuficiente para este escenario. ¿Y una red Ethernet de 100 Gbps = 12.5 GB/s? La transferencia de KV cache se convierte en el cuello de botella primario, reemplazando exactamente el problema que la disaggregación vino a resolver.

El umbral citado en la literatura de producción — al menos 90 Gbps solo para hacer invisible el overhead de transferencia en workloads moderados — ya supera lo que muchos clusters de datacenter tienen en conectividad GPU-a-GPU a través de la red. Con 90 Gbps = 11.25 GB/s, y nuestro cálculo de 134 GB/s para 100 req/s con prompts de 512 tokens, queda claro que 90 Gbps es el umbral mínimo para workloads ligeros, no el número cómodo para producción a escala.

### Topology-aware placement: colocar las instancias donde el ancho de banda es gratuito

La solución práctica es la **topology-aware placement** (ubicación consciente de la topología de red). La idea es simple: si el ancho de banda de red inter-nodo no es suficiente para la transferencia de KV cache, coloca las prefill instances y decode instances correspondientes en el mismo nodo físico, de modo que la transferencia pueda usar las conexiones intra-nodo.

En nodos equipados con GPUs NVIDIA modernas, las GPUs dentro del mismo nodo están conectadas mediante **NVLink**, el bus de interconexión propietario de NVIDIA. Las versiones actuales de NVLink ofrecen hasta 600 GB/s de ancho de banda bidireccional entre GPUs del mismo nodo. Con 600 GB/s disponibles, transferir 1.34 GB de KV cache tarda aproximadamente 2.2 milisegundos — tiempo que el usuario no percibe como latencia.

Comparación clara:

| Interconexión | Ancho de banda | Tiempo transferencia (1.34 GB KV cache) |
|---|---|---|
| Ethernet 100G | ~12.5 GB/s | ~107 ms |
| InfiniBand HDR | ~25 GB/s | ~54 ms |
| NVLink (NVIDIA Hopper) | ~600 GB/s | ~2.2 ms |

Solo NVLink hace la transferencia invisible al usuario. Esto convierte la disaggregación en una técnica viable solo en clusters con NVLink intra-nodo, o en clusters con InfiniBand de alta generación (NDR o XDR, con ~50-100 GB/s) donde el overhead de transferencia puede amortizarse con bufferización inteligente.

Cuando ni NVLink ni InfiniBand de alta velocidad están disponibles, el sistema recurre a la colocación forzada prefill-decode en el mismo nodo, degradando graciosamente a un régimen semi-disaggregado donde parte del aislamiento se mantiene pero las limitaciones de ancho de banda son más relajadas.

*Figura 8.4 — Disaggregación prefill-decode: GPU-P procesa el prompt y transfiere el KV cache vía NVLink/InfiniBand (cuello de botella, en rojo) a GPU-D, que genera el resto de la respuesta de forma aislada.*

> **Descripción visual:** Diagrama de flujo horizontal. Un bloque naranja "Request prompt" apunta a un bloque azul "GPU-P Prefill instance". Desde GPU-P salen dos flechas: a un bloque naranja "KV Cache ~1.34 GB" y a un bloque rojo "Primer token". Ambos convergen en un rombo rojo "Transferencia NVLink / IB" (cuello de botella). El rombo apunta a un bloque azul "GPU-D Decode instance" que termina en un bloque rojo "Respuesta completa". Fondo blanco, estilo limpio.

---

## vLLM y PagedAttention: la memoria virtual de los LLMs

Todo lo que hemos discutido hasta aquí asume implícitamente que la memoria de GPU está disponible para los requests que la necesitan. Pero hay un problema de gestión de memoria que, en los frameworks de inferencia tradicionales, descarta de forma silenciosa entre el 60% y el 80% de la VRAM disponible.

El problema es la **fragmentación de memoria**. Los frameworks tradicionales asignan, para cada request, un bloque contiguo de memoria GPU igual al KV cache máximo posible del request — que es el KV cache para la longitud de contexto máxima del modelo. Si el modelo tiene un contexto máximo de 32.768 tokens y el KV cache por token es 0.5 MB (para simplificar), eso es 16 GB reservados para ese request, aunque el usuario solo vaya a generar 200 tokens de respuesta.

El problema no es que se reserve ese espacio — es que se reserva como bloque contiguo y no se puede compartir ni reutilizar hasta que el request termina. Con 24 GB de VRAM total, este esquema permite servir exactamente 1 request activo, con 8 GB desperdigados en fragmentos internos inutilizables. Eso es el 80% de fragmentación que citan los papers originales de vLLM.

**vLLM** (Virtual Large Language Model inference engine) es el framework de producción más ampliamente adoptado para servir LLMs, y su innovación central es **PagedAttention**: un nuevo mecanismo de gestión de memoria para el KV cache inspirado directamente en el sistema de memoria virtual de los sistemas operativos.

### La analogía del sistema operativo

En un SO moderno, la memoria de los procesos no es contigue en la RAM física. El SO mantiene una tabla de páginas que mapea páginas virtuales (lo que el proceso cree que tiene) a páginas físicas (dónde están realmente en la RAM). Esto permite:
- Que la memoria de un proceso sea no contigua físicamente.
- Que se asigne solo la memoria que realmente se usa (demanda paginada).
- Que páginas inactivas se desplacen al disco (swap) y vuelvan cuando se necesiten.
- Que múltiples procesos compartan páginas de memoria de solo lectura (páginas compartidas).

PagedAttention aplica exactamente la misma abstracción al KV cache de los LLMs.

### Cómo funciona PagedAttention

En lugar de asignar un bloque contiguo de memoria para el KV cache completo de un request, PagedAttention divide el KV cache en **bloques de tamaño fijo** (típicamente 16 o 32 tokens cada uno). Estos bloques no tienen que estar contiguos en memoria. El **Block Manager** (gestor de bloques) de vLLM mantiene una tabla de correspondencia entre:
- **Logical blocks** (bloques lógicos): el KV cache tal como el request lo "ve", numerado secuencialmente desde el principio del contexto.
- **Physical blocks** (bloques físicos): dónde están realmente esos datos en la VRAM.

Al inicio de un request, vLLM no reserva ningún bloque físico. Conforme el request genera tokens y el KV cache crece, el Block Manager asigna bloques físicos bajo demanda (*just-in-time allocation*). Cuando el request termina, los bloques físicos se marcan como libres instantáneamente y están disponibles para el siguiente request.

¿Qué fragmentación queda? Solo la del último bloque parcialmente lleno de cada request — en promedio, medio bloque. Si los bloques son de 32 tokens, la fragmentación máxima por request es de 16 tokens de KV cache. En términos de VRAM, eso es menos del 4% de fragmentación total, comparado con el 60-80% del esquema naive.

*Figura 8.5 — PagedAttention: los bloques lógicos del KV cache (visión del request) se mapean mediante tabla de páginas a páginas físicas no contiguas en VRAM, eliminando fragmentación.*

> **Descripción visual:** Diagrama de flujo horizontal con dos subgrafos. Izquierda, "Vista lógica del request": tres bloques naranja encadenados (Bloque lógico 0, 1, 2). Derecha, "VRAM física no contigua": tres bloques azules desordenados (Página física 3, 7, 1). Flechas diagonales etiquetadas "tabla de páginas" conectan cada bloque lógico con su página física correspondiente. Fondo blanco, estilo limpio.

### El impacto de PagedAttention en throughput

La reducción de fragmentación del 80% al 4% libera una cantidad enorme de VRAM. Esa VRAM recuperada se usa para mantener más requests activos simultáneamente — es decir, aumenta el batch size efectivo del motor. Un batch más grande en decode significa que cada carga de pesos desde HBM se amortiza entre más sequences, mejorando la utilización de GPU.

El resultado empírico es dramático: vLLM consigue entre **23x y 24x más throughput** que sistemas de static batching como Hugging Face Transformers inference naive, medido en tokens generados por segundo. Esto no es magia — es el efecto compuesto de:
1. Continuous batching (elimina slots ociosos).
2. PagedAttention (elimina fragmentación, permite batches más grandes).
3. Un scheduler preemptivo que puede pausar, reanudar y reordenar requests para maximizar la utilización.

Simultáneamente, la latencia p50 baja, porque el sistema puede manejar workloads bursting sin que los nuevos requests esperen a que un batch completo termine.

PagedAttention también permite implementar dos características avanzadas que hubieran sido imposibles con gestión de memoria contigua:

**Copy-on-Write para beam search**: en beam search, el modelo explora múltiples hipótesis de respuesta en paralelo. Hasta que una hipótesis diverge de las demás, comparten el mismo KV cache. Con PagedAttention, las hipótesis pueden literalmente compartir los mismos bloques físicos hasta el punto de divergencia, y solo copiarlos cuando es necesario escribir en ellos. Esto ahorra una fracción significativa de la memoria del KV cache en inferencia con beam search.

**Prefix caching**: si múltiples requests comparten el mismo prefijo de prompt (por ejemplo, todos los usuarios de un chatbot comparten el system prompt), los bloques del KV cache correspondientes al prefijo común pueden almacenarse una sola vez y compartirse entre todos los requests. El prefill de ese prefijo se hace una vez; los demás requests solo hacen el prefill de su parte privada.

---

## Elastic resource allocation: kvcached y la compartición de GPU sin fronteras

PagedAttention resuelve el problema de la fragmentación dentro de un único motor de inferencia sirviendo un único modelo. Pero en producción, un cluster de GPUs suele ejecutar múltiples modelos simultáneamente: un modelo de routing, un modelo de reranking, varios modelos de inferencia principal, modelos de visión. La pregunta es: ¿cómo compartir la VRAM del GPU entre todos ellos de forma eficiente?

**MIG (Multi-Instance GPU)** es la solución a nivel de hardware que NVIDIA ofrece. MIG permite dividir físicamente una GPU en particiones aisladas — por ejemplo, una A100 de 80 GB puede dividirse en hasta 7 particiones de ~10 GB cada una. Cada partición es invisible a las demás y puede ejecutar un modelo distinto. La ventaja es el aislamiento perfecto. La desventaja es la rigidez: las particiones son fijas en tamaño, no pueden redimensionarse sin reiniciar la GPU, y la memoria no utilizada en una partición no puede prestarse a otra que la necesite.

**kvcached** es un proyecto open-source que ataca el mismo problema desde el software, con mucha más flexibilidad. Su insight fundamental es que la VRAM de la GPU, como la RAM de un servidor, puede gestionarse con las mismas abstracciones que los SOs llevan décadas usando para gestionar memoria: direccionamiento virtual, asignación bajo demanda, y reclamación dinámica.

### El mecanismo de kvcached

Cuando un motor de inferencia como vLLM arranca con kvcached, en lugar de reservar físicamente toda la VRAM que necesitará para el KV cache al inicio (lo que haría por defecto), solo reserva el **espacio de direcciones virtuales** en la GPU. La memoria física se asigna estrictamente bajo demanda, bloque a bloque, a medida que los requests realmente necesitan espacio de KV cache.

kvcached implementa un sistema de dos niveles:
- **Fast path**: para bloques de KV cache que caben en VRAM, la asignación es directa y rápida (microsegundos).
- **Slow path**: cuando la presión de memoria es alta, kvcached puede desalojar bloques de KV cache menos usados a memoria de CPU (con latencia de milisegundos) o incluso a almacenamiento NVMe, y recuperarlos cuando sea necesario.

Este mecanismo tiene consecuencias importantes para varios escenarios de producción:

**Multi-LLM serving**: si tienes tres modelos compartiendo la misma GPU, y el Modelo A está recibiendo una ráfaga de requests mientras B y C están ociosos, kvcached permite que A use el 90% de la VRAM disponible durante esa ráfaga, y que la memoria se redistribuya automáticamente cuando B y C vuelvan a recibir tráfico. No hay particiones fijas — la memoria fluye hacia donde se necesita.

**Serverless LLM**: en arquitecturas serverless, los modelos solo están activos mientras sirven requests. Con gestión de memoria estática, mantener un modelo cargado pero ocioso reserva VRAM que ningún request está usando. Con kvcached, un modelo ocioso libera su KV cache y retiene solo los pesos del modelo (que deben estar cargados para poder responder rápido). Esta separación de "pesos siempre cargados, KV cache solo cuando hay requests" reduce el coste de cold start y permite una densidad mucho mayor de modelos por GPU.

**Colocación con otros workloads**: un GPU que sirve inferencia de LLMs también puede ejecutar trabajos de fine-tuning de baja prioridad, procesamiento de visión, o entrenamiento, usando la VRAM que la inferencia no está usando en ese momento. Con gestión de memoria estática, esto era imposible sin MIG. Con kvcached, es automático.

### El impacto en TTFT: 2x-28x de mejora

El beneficio medido más llamativo de kvcached es en TTFT bajo carga concurrente. ¿Por qué afecta kvcached al TTFT?

Con gestión de memoria estática, cuando la cola de requests crece — por ejemplo, durante un pico de tráfico — el motor no puede aceptar nuevos requests hasta que haya VRAM libre para su KV cache. Los requests esperan en cola hasta que un request activo termina y libera su bloque de VRAM. Si todos los bloques están ocupados, la cola crece y el TTFT se dispara.

Con kvcached, el motor puede aceptar nuevos requests usando memoria virtual que todavía no tiene backing físico. Conforme los requests en curso terminan y liberan VRAM física, esa memoria se asigna a los requests entrantes. En workloads con alta concurrencia y distribución de duraciones de request variable (exactamente lo que se ve en producción), esto se traduce en tiempos de espera en cola mucho menores.

El rango de mejora de **2x a 28x** refleja la varianza de los workloads de producción: un servicio con requests de duración muy variable (muchos cortos y pocos muy largos) se beneficia más, porque la gestión estática desperdicia más VRAM esperando a que los requests largos terminen. Un servicio con duraciones uniformes (todos los requests generan respuestas de longitud similar) se beneficia menos, porque la fragmentación dinámica es menor.

---

## Ecosistema de frameworks: Ollama, llama.cpp, vLLM y SGLang

Antes de pasar al laboratorio, conviene mapear el ecosistema de herramientas disponibles. No todas las situaciones requieren vLLM; la elección depende del hardware, la escala, y el caso de uso.

### llama.cpp: el motor de inferencia sin GPU

**llama.cpp** es un motor de inferencia escrito en C++ por Georgi Gerganov que implementa la inferencia de LLMs de forma eficiente en CPUs de consumo (y GPUs a través de backends opcionales como CUDA o Metal). Su filosofía es maximizar la eficiencia en hardware no especializado.

El nombre engaña — llama.cpp no solo ejecuta LLaMA. Soporta docenas de arquitecturas modernas, y su formato nativo de pesos es **GGUF** (Georgi's GPT-Unified Format), un formato binario único que encapsula los pesos del modelo junto con todos los metadatos necesarios para la inferencia (arquitectura, vocabulario, configuración de cuantización). Antes de GGUF existía GGML; si ves archivos con extensión `.bin` de modelos llama.cpp antiguos, son formato GGML.

La clave de la eficiencia de llama.cpp en CPU es la **cuantización**. Los pesos del modelo, normalmente almacenados en FP16 (2 bytes por parámetro), se comprimen a representaciones de menor precisión:

- **Q4_K_M**: cuantización de 4 bits con agrupación y matrices de escala separadas. Es el punto óptimo para la mayoría de los casos — reduce el tamaño del modelo en ~4x respecto a FP16 con una degradación mínima de calidad en la mayoría de las tareas.
- **Q2_K**: cuantización de 2 bits. Comprime en ~8x respecto a FP16, pero con degradación de calidad más visible en tareas de razonamiento complejo. Útil cuando la memoria es extremadamente limitada.
- **Q8_0**: cuantización de 8 bits, casi sin pérdida de calidad. Comprime en ~2x respecto a FP16. Para producción con GPUs dedicadas donde la calidad importa más que el tamaño.

La nomenclatura es importante: el número denota los bits por parámetro; la letra y sufijo siguientes indican el esquema de agrupación (K = agrupamiento por bloques con factores de escala cuantizados también, M = variante mixta). Más detalles de precisión se cuantifican a mayor número de bits.

¿Qué se pierde con cuantización agresiva? La respuesta varía por tarea. Para OCR y reconocimiento de documentos, la degradación de Q4 es prácticamente invisible — la tarea es suficientemente determinista. Para razonamiento matemático profundo o generación de código complejo, Q2 puede introducir errores que Q4 evitaría.

### Ollama: la interfaz humana de llama.cpp

**Ollama** es una herramienta que envuelve llama.cpp en una interfaz amigable para desarrolladores. La relación entre ambos es análoga a la que existe entre Git y GitHub Desktop: llama.cpp es el motor potente que hace el trabajo real; Ollama añade la capa de UX que lo hace accesible sin necesidad de compilar código C++ ni manejar argumentos de línea de comandos complejos.

Ollama gestiona:
- **Descarga de modelos**: `ollama pull llama3.2` descarga y cachea el modelo automáticamente.
- **Gestión del daemon**: mantiene el servidor de inferencia en background.
- **API REST**: expone un endpoint en `localhost:11434` con una API compatible con las convenciones de OpenAI.
- **Modelfiles**: permite crear versiones personalizadas de modelos con parámetros específicos (contexto, temperatura, system prompt), guardadas como una "receta" que puede reproducirse.

La limitación de Ollama es que está diseñado para uso local y no para producción a escala. No implementa continuous batching (es single-request por defecto), no tiene PagedAttention, y no soporta multi-GPU de forma nativa. Para producción con múltiples usuarios concurrentes, el salto a vLLM o SGLang es necesario.

### vLLM: producción en GPU

Ya describimos los mecanismos de vLLM en detalle. Su perfil de uso es: tienes acceso a una o más GPUs NVIDIA, quieres servir uno o varios modelos a múltiples usuarios concurrentes, y la eficiencia importa. vLLM expone una API compatible con OpenAI (`/v1/chat/completions`), soporta LoRA adapters en tiempo real, y puede servir modelos de hasta cientos de GB con tensor parallelism multi-GPU.

### SGLang: alto rendimiento para workflows complejos

**SGLang** (Structured Generation Language) es un framework de inferencia de alto rendimiento con soporte de primera clase para **structured outputs** (salidas estructuradas: JSON schemas, gramáticas formales), **multi-turn conversations**, y **workflows complejos** que mezclan múltiples llamadas al modelo. En benchmarks de throughput puro, SGLang iguala o supera a vLLM en muchos escenarios, especialmente con contextos largos y outputs estructurados.

Para el caso de uso de OCR a escala (múltiples documentos, extracción estructurada de datos), SGLang es una alternativa legítima a vLLM con ventajas específicas en la gestión de outputs JSON.

| Framework | Mejor para | Continuous batching | PagedAttention | GPU requerida |
|---|---|---|---|---|
| llama.cpp | Dev local, CPU-only | No | No | Opcional |
| Ollama | Prototipado, uso personal | No | No | Opcional |
| vLLM | Producción multi-usuario | Sí | Sí | Sí |
| SGLang | Producción, outputs estructurados | Sí | Sí | Sí |

---

## Lab: GLM-OCR con Ollama — de documento a Markdown en local

Con la teoría en mano, es hora de ensuciarnos las manos. El modelo que usaremos en el lab es **GLM-OCR**, desarrollado por Z.ai, un [[07-finetuning-multimodal-vision-tts|modelo de visión-lenguaje (VLM)]] de solo 0.9 billones de parámetros que en el momento de su lanzamiento alcanza un score de 94.62 en OmniDocBench V1.5 — el benchmark estándar para evaluación de comprensión de documentos complejos — situándose en el primer puesto del ranking, por encima de modelos propietarios y modelos de mucho mayor tamaño.

¿Cómo puede un modelo de 0.9B superar a modelos diez veces más grandes? La respuesta está en su especialización y en su arquitectura de dos etapas.

### GLM-OCR: arquitectura y ventajas

GLM-OCR está basado en la familia GLM (General Language Model) con un encoder de visión (GLM-V), pero su desempeño en OCR no proviene solo de la arquitectura base. Dos innovaciones en el entrenamiento son responsables:

**Multi-Token Prediction (MTP) loss**: en lugar de entrenar el modelo para predecir solo el siguiente token, MTP entrena simultáneamente para predecir los siguientes $k$ tokens. Para OCR, esto resulta en una generación más fluida del texto reconocido — el modelo aprende la estructura del documento a nivel de frase, no solo de carácter. Por ejemplo, si está reconociendo "Total $755", haberle entrenado para predecir múltiples tokens a la vez le ayuda a entender que el "$" debe ir seguido de un número y no de una palabra aleatoria.

**Full-task reinforcement learning**: el modelo se entrena con RL sobre la tarea completa de OCR — no solo sobre la pérdida token a token, sino sobre métricas de calidad del documento final (¿se preservó la estructura de la tabla? ¿Las fórmulas están bien formadas?). Esto ajusta el modelo para optimizar el resultado final que importa al usuario, no solo la probabilidad de cada token individual.

El segundo elemento clave es la integración con **PP-DocLayout-V3** de PaddlePaddle, un modelo de detección de layout que analiza primero la estructura del documento (dónde están los títulos, tablas, fórmulas, columnas de texto, imágenes incrustadas) antes de pasársela al decoder de OCR. Esta detección de layout permite manejar documentos complejos con layouts no lineales — una columna de tablas seguida de texto a dos columnas, código fuente intercalado con fórmulas matemáticas — que romperían la lectura secuencial simple.

### Paso 0: identificar el hardware

El rendimiento en CPU depende críticamente del número de **núcleos físicos**. Los procesadores modernos usan hyperthreading (SMT) para presentar el doble de núcleos lógicos que físicos, pero llama.cpp bajo el capó de Ollama se beneficia de núcleos físicos reales — el hyperthreading añade poco en cargas de inferencia LLM porque el cuello de botella es el acceso a memoria, no la ejecución de instrucciones.

```bash
# Linux
lscpu | grep -E '^CPU\(s\):|Core\(s\) per socket|Thread\(s\) per core'

# macOS
sysctl -n hw.physicalcpu hw.logicalcpu
```

```powershell
# Windows PowerShell
Get-WmiObject -Class Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
```

La regla para `num_thread` es: iguala el número de núcleos físicos, no los lógicos. Si tienes un procesador de 8 núcleos con hyperthreading (16 núcleos lógicos), usa `num_thread 8`. Usar 16 típicamente degrada el rendimiento porque los hilos compiten por los mismos recursos de memoria.

### Paso 1: lanzar el contenedor Ollama

Usamos Docker para aislar Ollama y evitar conflictos de versiones. El volumen `ollama_storage` es crucial: los modelos GGUF de GLM-OCR pesan entre 2 y 4 GB, y sin un volumen persistente tendrías que descargarlos de nuevo cada vez que borrases el contenedor.

```bash
docker run -d \
  --name ollama-server \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama
```

Desglose de flags:
- `-d`: detached mode, el contenedor corre en background.
- `--name ollama-server`: nombre fijo para referenciar el contenedor en comandos posteriores.
- `-v ollama_storage:/root/.ollama`: monta un volumen Docker nombrado en el directorio donde Ollama guarda los modelos. El volumen persiste aunque el contenedor se borre.
- `-p 11434:11434`: expone el puerto 11434 del contenedor (donde escucha la API de Ollama) en el mismo puerto del host.

### Paso 2: descargar el modelo

```bash
docker exec -it ollama-server ollama pull glm-ocr
```

Esto descarga el modelo GLM-OCR desde el Ollama model hub, en su formato GGUF con cuantización por defecto. `docker exec -it` ejecuta el comando dentro del contenedor ya corriendo (`-it` = interactive + pseudo-TTY, necesario para ver el progreso de la descarga).

### Paso 3: el Modelfile y por qué importa

El problema con la configuración por defecto de Ollama para modelos de visión es que el contexto estándar (2.048 o 4.096 tokens) es insuficiente para procesar imágenes de documentos. Una imagen de factura a 1024px se tokeniza como cientos de tokens visuales antes de que el decoder empiece a generar el texto. Sin suficiente contexto, el modelo trunca la imagen y pierde información.

Un **Modelfile** es la forma de Ollama de declarar una versión personalizada de un modelo. Es análogo a un Dockerfile para Docker: especifica la base y las modificaciones.

```bash
# Entramos al contenedor
docker exec -it ollama-server bash

# Creamos el Modelfile
cat <<EOF > GLM-Config
FROM glm-ocr

# Context and hardware
PARAMETER num_ctx 16384
PARAMETER num_thread 6

# Generation parameters (hardcoded for OCR)
PARAMETER num_predict 8192
PARAMETER temperature 0
PARAMETER top_p 0.00001
PARAMETER top_k 1
PARAMETER repeat_penalty 1.1
EOF

# Creamos la versión personalizada
ollama create glm-ocr-optimized -f GLM-Config

exit
```

Explicación de cada parámetro:

**`num_ctx 16384`**: el tamaño de la ventana de contexto en tokens. Para una imagen de 1024×1024 px, el encoder visual de GLM-OCR genera típicamente entre 1.000 y 4.000 tokens visuales dependiendo de la complejidad de la imagen. Con el contexto por defecto de 2.048, una imagen de tamaño moderado consumiría casi todo el contexto antes de que el modelo genere una sola palabra. Con 16.384, hay espacio más que suficiente tanto para la imagen como para una respuesta larga. El coste es memoria: un contexto de 16.384 tokens ocupa bastante más VRAM/RAM que uno de 2.048 — en una máquina con 8 GB de RAM esto puede ser ajustado.

**`num_thread 6`**: el número de hilos CPU para la inferencia. Aquí se asume una máquina con 6 núcleos físicos (o Performance Cores en un Intel híbrido). Adaptar según tu hardware siguiendo el Paso 0.

**`num_predict 8192`**: el número máximo de tokens de salida. Para un documento denso con tablas y código, la salida Markdown puede ser larga — 8.192 tokens garantiza que el modelo puede generar la transcripción completa sin truncar.

**`temperature 0` + `top_p 0.00001` + `top_k 1`**: esta combinación implementa **greedy decoding** — el modelo siempre elige el token más probable. Para OCR, no queremos creatividad ni diversidad: queremos la transcripción más probable de lo que hay en la imagen. Cualquier temperatura mayor a 0 introduciría varianza aleatoria que podría alterar palabras, números o signos de puntuación.

**`repeat_penalty 1.1`**: penaliza ligeramente repetir los mismos tokens. Sin esto, el modelo puede quedarse en bucle repitiendo líneas o secuencias de caracteres, especialmente en documentos con patrones repetitivos (tablas, listas).

Estos parámetros se hardcodean en el Modelfile, no se pasan en cada request, porque para esta tarea específica los parámetros óptimos son siempre los mismos. Para un chatbot de propósito general no harías esto — querrías flexibilidad en temperatura para distintos tipos de request.

### Paso 4: enviar imágenes a la API

Las imágenes deben enviarse como strings Base64. **Base64** es un esquema de codificación que convierte datos binarios (como los bytes de una imagen JPEG) en una cadena de caracteres ASCII imprimibles. Esto permite incluir la imagen directamente en el body JSON de la request HTTP sin necesidad de archivos adjuntos o URLs externas.

```python
import requests
import base64
import ollama
import time
from io import BytesIO
from PIL import Image

# --- Configuración ---
IMAGE_URL = "https://marketplace.canva.com/EAE92Pl9bfg/6/0/1131w/canva-black-and-gray-minimal-freelancer-invoice-wPpAXSlmfF4.jpg"
MODEL_NAME = "glm-ocr-optimized"
MAX_DIMENSION = 1024  # Redimensiona al borde más largo a 1024px


def get_optimized_image_b64(url: str) -> str:
    """
    Descarga la imagen, la redimensiona si es mayor de MAX_DIMENSION,
    y la codifica en Base64 JPEG.
    
    Usamos JPEG (no PNG) porque:
    - El OCR no necesita transparencia (que PNG preserva pero JPEG no).
    - JPEG comprime mucho más que PNG para imágenes de documentos típicos.
    - Menos bytes de Base64 = request más pequeña = menos tiempo de parsing.
    """
    print("Descargando imagen...")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    orig_w, orig_h = img.size
    print(f"Tamaño original: {orig_w}x{orig_h}")
    
    # Solo redimensionar si la imagen supera MAX_DIMENSION en algún eje
    if max(img.size) > MAX_DIMENSION:
        # thumbnail respeta el aspect ratio automáticamente
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
        print(f"Redimensionado a: {img.width}x{img.height}")
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def run_ocr():
    image_b64 = get_optimized_image_b64(IMAGE_URL)
    
    print("Enviando a Ollama (esperando primer token)...")
    start_time = time.time()
    first_token = True
    
    # Stream: recibimos tokens conforme se generan
    # Esto permite medir el TTFT con precisión
    stream = ollama.generate(
        model=MODEL_NAME,
        prompt="Text recognition:",  # Prompt estándar de GLM-OCR
        images=[image_b64],
        stream=True
    )
    
    for chunk in stream:
        if first_token:
            ttft = time.time() - start_time
            print(f"\nTTFT: {ttft:.2f}s\n")
            first_token = False
        
        # flush=True asegura que el texto aparece en terminal conforme llega
        print(chunk["response"], end="", flush=True)
    
    total_time = time.time() - start_time
    print(f"\n\nTiempo total: {total_time:.2f}s")


if __name__ == "__main__":
    run_ocr()
```

El prompt `"Text recognition:"` no es arbitrario — es el prompt de instrucción estándar con el que GLM-OCR fue entrenado. Usar un prompt diferente puede degradar la calidad de reconocimiento porque el modelo espera ese formato específico de instrucción.

### Paso 5: interpretar los resultados de docker stats

Mientras el script corre, ejecuta en otra terminal:

```bash
docker stats
```

Verás algo similar a:

```
CONTAINER ID   NAME            CPU %     MEM USAGE / LIMIT     MEM %
0ca7a28d6fe2   ollama-server   600.42%   4.359GiB / 7.607GiB   57.31%
```

Dos cosas llaman la atención aquí y vale la pena entenderlas en profundidad.

**CPU al 600%**: en Linux/Docker, el CPU% puede superar el 100% porque el porcentaje se calcula respecto a un único núcleo. 600% significa que el proceso está usando el equivalente a 6 núcleos completos a máxima capacidad. Esto corresponde exactamente al `num_thread 6` que configuramos — todos los hilos están trabajando al límite. Esto es correcto y esperado: es la fase de prefill visual, la parte compute-bound del pipeline.

**RAM al 57%**: el modelo de 0.9B en Q4 más el KV cache de 16.384 tokens de contexto ocupa unos 4.4 GB de los 7.6 GB disponibles en el sistema del ejemplo. El resto del sistema operativo ocupa lo demás. Si el MEM% se acercara al 100%, el sistema empezaría a hacer swapping a disco, lo que multiplicaría la latencia por 10x o más.

**El dato más revelador son los tiempos:**

```
TTFT: 174.96s
Total Processing Time: 183.39s
```

*Figura 8.6 — Pipeline GLM-OCR con Ollama: el 95% del tiempo total cae en el Vision Encoder (prefill visual, compute-bound); el decoder genera el Markdown en segundos una vez que el KV cache está construido.*

> **Descripción visual:** Diagrama de flujo horizontal. Bloques naranjas "Imagen documento" y "Base64 codificación", bloque verde "Resize 1024 px max", bloques azules "Ollama GLM-OCR", "Vision Encoder prefill visual", "Decoder tokens de texto", bloque naranja "KV Cache construido", bloque rojo "Markdown estructurado". Flechas punteadas etiquetadas "TTFT: 174s compute-bound" y "Decode: 9s memory-BW-bound" marcan los dos bottlenecks. Fondo blanco, estilo limpio.

El tiempo hasta el primer token es de **174.96 segundos — casi tres minutos**. El tiempo de generación posterior es de solo **8.43 segundos** (183.39 - 174.96). ¿Qué nos dice esto?

Que el 95.4% del tiempo total del pipeline está en la fase de prefill — específicamente, en el encoder visual procesando la imagen. El decoder LLM, que genera el texto token a token, es comparativamente instantáneo. Esta es la demostración práctica perfecta de por qué las fases de prefill y decode son tan distintas: el encoder visual realiza multiplicaciones de matrices masivas sobre las representaciones de la imagen (compute-bound al 100%), mientras que la generación de texto aprovecha el KV cache ya construido y produce tokens en cuestión de centésimas de segundo cada uno.

Esta observación tiene implicaciones prácticas directas:

1. **En producción, el cuello de botella de GLM-OCR es el encoder visual.** Más VRAM no te ayuda (no hay OOM). Más RAM no te ayuda. Lo que te ayuda es una GPU que pueda hacer el forward pass del encoder en paralelo masivo. Con una NVIDIA A10G (24 GB VRAM), el mismo pipeline corre en 2-5 segundos totales en lugar de 183.

2. **El TTFT de 175s no es un mal diseño — es la física del hardware.** Una CPU, por muy buena que sea, no puede competir con una GPU en operaciones matriciales masivas. La CPU del ejemplo procesa ~4 GFLOPs; una A100 hace 312 TFLOPs en BF16 — 78.000 veces más throughput de cómputo.

3. **La fase de decode es un argumento a favor del chunked prefill y la disaggregación.** Si este modelo tuviera que servir 100 usuarios concurrentes en CPU, el prefill de cada uno sería tan largo que el continuous batching no podría rescatarlo — los nuevos requests esperarían horas en cola. El salto a GPU resuelve el problema en el orden correcto.

### Paso 6: el SDK de GLM-OCR con PP-DocLayout-V3

Para documentos más complejos — el paper técnico de Qwen3, una presentación con columnas y tablas, documentación de código — la API básica de Ollama tiene limitaciones. El SDK oficial de GLM-OCR integra PP-DocLayout-V3, el modelo de detección de layout de PaddlePaddle, que primero analiza la estructura del documento y luego pasa cada región al encoder de visión de forma independiente.

¿Por qué este pipeline de dos etapas? Porque el encoder visual de GLM-OCR tiene una resolución fija — procesa la imagen como un todo. Para un documento A4 con tres columnas de texto, tabla con 20 filas, y dos figuras, procesar la imagen completa a 1024px hace que el detalle de texto en columnas sea demasiado pequeño para reconocerse correctamente. PP-DocLayout-V3 detecta primero las regiones del documento (bounding boxes de cada bloque), y luego el encoder procesa cada región de forma recortada y normalizada — el equivalente a leer cada párrafo en primer plano en lugar de intentar leer toda la página desde lejos.

```python
import requests
from PIL import Image
from glmocr import GlmOcr

LOCAL_FILENAME = "documento_complejo.jpg"


def run_sdk_ocr(image_path: str):
    # Resize a 1024px para optimizar la inferencia en CPU
    # (el sweet spot entre resolución suficiente y velocidad)
    with Image.open(image_path) as img:
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            img.save(image_path)
            print(f"Imagen redimensionada para inferencia en CPU.")
    
    print("Inicializando GLM-OCR SDK...")
    
    # GlmOcr como context manager gestiona la carga y descarga del modelo
    # config.yaml especifica las rutas al modelo, idioma, y parámetros del detector
    with GlmOcr(config_path="./config.yaml") as parser:
        print("Analizando estructura del documento...")
        result = parser.parse(image_path)
        
        # El resultado incluye el Markdown con toda la estructura preservada:
        # - Títulos como # ## ###
        # - Tablas como Markdown tables
        # - Fórmulas matemáticas como LaTeX
        # - Listas con su jerarquía
        print(result.markdown_result)


if __name__ == "__main__":
    run_sdk_ocr(LOCAL_FILENAME)
```

El `config.yaml` que acompaña al SDK configura:
- Ruta al modelo GLM-OCR (en formato safetensors, no GGUF — el SDK usa HuggingFace Transformers, no llama.cpp).
- Ruta al modelo PP-DocLayout-V3.
- Idiomas soportados para la detección.
- Umbrales de confianza para la detección de regiones.
- Modo de paralelización de regiones.

El output del SDK para la primera página del Qwen3 Technical Report incluye el abstract completo con su estructura, los títulos de las tablas de benchmarks, y el contenido de las tablas en formato Markdown perfectamente estructurado — incluyendo las celdas de cabecera alineadas a la izquierda (`:---`) y los valores numéricos de los benchmarks en cada celda. Esto es lo que hace a GLM-OCR útil para pipelines de procesamiento de documentos: el output no es texto plano sino Markdown estructurado que puede parsearse programáticamente.

### Paso 7: cuándo saltar a vLLM/SGLang

Ollama con GLM-OCR funciona perfectamente para uso personal, prototipado, o procesamiento de documentos de baja frecuencia (unos pocos documentos al día). Los límites se hacen evidentes cuando:

- **El throughput importa**: si necesitas procesar 100 documentos/hora, los 183 segundos por documento en CPU son prohibitivos. Con vLLM en una GPU A10G, ese tiempo cae a 3-5 segundos — 40-60x más rápido.
- **Hay múltiples usuarios concurrentes**: Ollama no implementa continuous batching. El segundo usuario espera a que el primero termine, incluyendo los 175 segundos de prefill. Con vLLM, múltiples requests de OCR pueden pipelinearse.
- **Necesitas endpoints compatibles con OpenAI a escala**: vLLM expone `/v1/chat/completions` con soporte de throughput alto. La configuración recomendada por la documentación oficial de GLM-OCR para producción es vLLM con `max_workers` y `connection_pool_size` ajustados para evitar errores 503 bajo carga.

```bash
# Servir GLM-OCR con vLLM (requiere GPU)
vllm serve THUDM/glm-ocr \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --tensor-parallel-size 1
```

Con esta configuración, vLLM aplica automáticamente todos los mecanismos del capítulo: continuous batching, PagedAttention, y chunked prefill si se configura. El tiempo de prefill visual cae a 2-4 segundos porque la GPU paraleliza las multiplicaciones de matrices que la CPU hacía en 175 segundos.

---

## Cierre del capítulo — El arco completo del fine-tuning al deployment

Hemos llegado al final del libro, y vale la pena hacer una pausa para ver el camino completo que hemos recorrido y cómo se conecta.

El **Capítulo 1** estableció los fundamentos: transformers, atención, pretraining. Aprendimos que los LLMs aprenden representaciones del lenguaje a través de la predicción del siguiente token sobre corpus masivos.

Los **Capítulos 2 y 3** construyeron las técnicas centrales del fine-tuning: SFT para alinear el modelo con instrucciones, y LoRA para hacerlo de forma eficiente modificando solo matrices de bajo rango. Aprendimos a adaptar modelos preentrenados a tareas específicas sin necesidad de actualizar todos los parámetros.

El **Capítulo 4** añadió QLoRA, que combina cuantización de 4 bits con LoRA para llevar el fine-tuning a hardware de consumo. La misma cuantización que hace posible ejecutar GLM-OCR en un portátil es la que permite finetunear modelos de 7B en una RTX 3090.

Los **Capítulos 5 y 6** abrieron la dimensión del alignment: RLHF, DPO, y GRPO para entrenar modelos que no solo generan texto coherente sino que alinean su comportamiento con preferencias humanas. Los mismos modelos de RLHF que has aprendido a entrenar son los que viven detrás de ChatGPT, Claude, y Gemini — y que, después de entrenados, deben servirse eficientemente.

El **Capítulo 7** expandió el dominio a modalidades no textuales: VLMs para visión y TTS para síntesis de voz. GLM-OCR, el modelo de este capítulo, es precisamente un VLM — y los principios de fine-tuning que aprendiste en el Capítulo 7 (cómo se entrena un vision encoder, cómo se conecta al decoder LLM, qué técnicas como MTP loss mejoran el rendimiento) explican por qué GLM-OCR es tan bueno con tan pocos parámetros.

Este **Capítulo 8** cierra el ciclo preguntando: ¿y una vez entrenado, cómo lo servimos? La respuesta no es obvia. El modelo más capaz del mundo, servido con un framework naive de static batching, puede ser más lento y caro que un modelo menor servido con continuous batching y PagedAttention. La inferencia eficiente es la diferencia entre un modelo que vive en un paper y un modelo que usan millones de usuarios.

La tensión central del capítulo — prefill vs decode queriendo cosas opuestas del hardware — se resuelve progresivamente en cinco generaciones de técnicas:

1. **Static batching**: agrupa requests, pero el batch es tan lento como el request más largo.
2. **Continuous batching**: libera slots al instante y los rellena en cada iteración — end-to-end.
3. **Chunked prefill**: divide los prefills largos para que no bloqueen el decode de otros requests.
4. **Prefill-decode disaggregation**: separa físicamente las dos fases en hardware optimizado para cada una.
5. **PagedAttention + kvcached**: gestiona la VRAM con la misma inteligencia con la que un SO gestiona la RAM.

Lo que ves en el lab con GLM-OCR — 175 segundos de prefill, 9 segundos de decode, 600% de CPU — es la manifestación directa de estos principios: la fase compute-bound dominando en CPU, y la fase memory-bandwidth-bound siendo casi gratuita una vez que el KV cache está construido.

El salto de Ollama en CPU a vLLM en GPU es el mismo salto conceptual que el libro entero ha estado describiendo: de prototipar algo que funciona a construir algo que escala. Las herramientas cambian. Los principios — gestión eficiente de recursos, eliminación de bottlenecks, separación de concerns computacionales — son los mismos en el entrenamiento que en la inferencia.

Si has llegado hasta aquí y entiendes por qué un ratio de importancia se recorta en PPO, por qué un adaptador LoRA de rango 8 funciona casi igual que un fine-tuning completo, y por qué el TTFT de un modelo VLM en CPU es tan alto, entonces has construido una base sólida sobre la que operar como ingeniero de LLMs en producción. El siguiente paso, inevitablemente, es ensuciarte las manos con tus propios modelos, tus propios datos, y tus propios casos de uso. Todo lo que necesitas para empezar está en este libro.

---

## Tags

#inferencia #serving #vllm #paged-attention #kv-cache #prefill #decode #chunked-prefill #disaggregation #ollama #llama-cpp #glm-ocr #multimodal #nivel/avanzado #tipo/lab #estado/completo

