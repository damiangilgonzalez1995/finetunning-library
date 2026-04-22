
The Neural Maze
The Neural Maze



The Finetuning Landscape - A Map of Modern LLM Training
Finetuning Sessions · Lesson 1 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Feb 11, 2026




Welcome to Lesson 1 of the Finetuning Sessions!

🙋 Here's a small plot twist: for this first lesson, we're not talking about finetuning!

Before jumping into techniques like SFT or LoRA, it's important to understand what happens before finetuning ever starts. That's exactly what this lesson is about.

We'll break down what pretraining really means, introduce the Transformer architecture that modern LLMs are built on, and explain the key differences between encoder-only, encoder–decoder, and decoder-only models.

Along the way, you'll build a clear mental map of the training pipeline of modern language models, from pretraining to alignment.

✅ This context will pay off later.

Many advanced techniques—e.g. LoRA—only truly make sense once you understand how Transformers work under the hood.

So, think of this lesson as setting the foundation. Once that's in place, everything that follows will click into place much faster.

Go Premium if you want the full experience of this course, with exclusive access to all the upcoming articles, labs, and office hours!

Transformer 101
Transformers One (2024) - IMDb


You know this is the first Transformer that came to mind!
At this point, you've probably heard the word Transformer so many times that it feels almost synonymous with "modern AI."

And in practice, that's not far from the truth: today, nearly every state-of-the-art language model is built on top of the Transformer architecture.

🙋 But here's a small surprise to kick things off: attention was not invented in the Transformer paper!

A LSTM neural network.


Before the Transformer era, LSTMs ruled the world of sequences (image from Understanding LSTM Networks)
Long before Transformers took over natural language processing, deep learning was already going through its first major boom. Architectures like convolutional neural networks (CNNs) dominated vision, while recurrent neural networks (RNNs)—especially LSTMs—were the workhorse of sequence modeling tasks such as machine translation, speech recognition, and text generation.

These models worked, but they came with clear limitations, especially when dealing with long sequences and long-range dependencies.

✅ This is where attention mechanisms first entered the picture.

The Attention Mechanism
Originally introduced as an improvement to encoder–decoder recurrent models for sequence-to-sequence tasks (like machine translation), attention was designed to solve a simple but fundamental problem: instead of compressing an entire input sequence into a single fixed-length vector (state), why not allow the model to look back at different parts of the input as needed?


The encoder-decoder architecture
At its simplest, attention is a weighted lookup mechanism.

You start with:

A set of keys: k1, k2, …, kn.

A corresponding set of values: v1, v2, …, vn.

A query that expresses what you're looking for.

The attention mechanism compared the query against all keys, assigns a weight to each one, and then produces a weighted sum of the values. This operation is often called attention pooling.




The Attention Mechanism
In other words:

🙋 Attention decides which pieces of information matter most, and combines them accordingly.

Early encoder–decoder models had to squeeze an entire input sentence into one vector before decoding. Attention removed this constraint by allowing the decoder to dynamically access all encoder states at every step. Longer sentences, richer context, better translations.

And this is where things get interesting … because at first, attention, was "just" an add-on! These mechanisms were trying to improve the encoder-decoder architecture in very specific applications.

But the implications, dear builders, ran much deeper …

Once you realise that this architecture can:

Query a set of representations

Select relevant information dynamically

Do so in a differentiable and parallelizable way

You no longer need recurrence at all!

🙋 This insight would end up reshaping neural network architectures across every domain.

Preparing the Ground for Transformers
So, as we already know by now, attention was not invented for Transformers, but one thing it's true. It radically expanded what attention mechanisms could do.

Several key evolutions turn basic attention pooling into the architecture that domains every domain nowadays.

Self-Attention



Self-attention
So far, attention has helped models look back at an input sequence when generating outputs, as in encoder–decoder models for machine translation.

✅ Self-attention takes this idea one step further.

Instead of having one sequence attend to another, each token in a sequence attends to all the other tokens in the same sequence! Simple and genius at the same time.

Imagine you feed a sequence of tokens into a model:

The cat sat on the mat
In self-attention:

Every token is turned into a query (Q), a key (K), and a value (V)

For each token, its query is compared with the keys of all tokens (including itself)

These comparisons determine how much attention the token pays to every other token

The token's new representation is built as a weighted sum of all values

So when the model updates the representation of "sat", it can:

Strongly attend to "cat" (who is sitting)

Attend to "mat" (where the action happens)

Largely ignore function words like "the"

Clear, right? Let's move on to another key technique: positional encodings.

Positional Encoding



Sine and cosine functions as positional encodings (image from Dive into Deep Learning)
Attention itself has no notion of order. To make sequences meaningful, positional information is injected into token representations, allowing the model to distinguish "first", "next", and "last" without relying on recurrence.

In the original Transformer, positional encodings are built using sine and cosine functions at different frequencies.

Each dimension of the positional encoding corresponds to a sinusoid:

Low-frequency sinusoids capture coarse position information

High-frequency sinusoids capture fine-grained differences

Together, they form a rich representation of position.

🙋 A quick reassurance before moving on: don't worry if the exact intuition behind sinusoidal positional encodings doesn't fully click right now. We're mentioning them because they were an important design choice in the original Transformer architecture, and it’s useful to know why they exist.

Multi-Head Attention



Scaled Dot-Product Attention and Multi-Head Attention (image from Attention Is All You Need)
So far, we've talked about attention as a general idea: queries (Q) interact with keys (K) to decide how values (V) should be combined.

Scaled dot-product attention is the specific, concrete way the Transformer implements this idea.

At a high level, it works like this:

Each token produces a query, key, and value

Queries are compared with keys using a dot product

The resulting scores are normalized with a softmax

These normalized scores are used to compute a weighted sum of the values

This is the mathematical formula, in case you are interested:

 
 
With a single attention head, the model is forced to average all relationships into one view.

Multi-head attention removes this constraint.

Different heads can specialize in different patterns, such as:

Short-range vs long-range dependencies

Syntactic relationships

Semantic similarity

This allows the model to jointly attend to information from different representation subspaces, all at once.

Bringing It All Together: The Transformer
If you combine all of these techniques, you arrive at the most important neural network architecture of our time.

This is the backbone behind models like BERT, ChatGPT, and Qwen. Different flavors, different objectives … but the same core idea.

At this point, it's hard not to recognize one of the most famous diagrams in modern deep learning, which I'll take the liberty of sharing here.

Ladies and gentlemen… meet the Transformer.




The Transformer Architecture (image from Attention Is All You Need)
So before we dive into LoRA or GRPO, remember this:

🙋 Understanding the structure we're working with is essential. That's exactly why we're taking the time to walk through these concepts!

This mental model will make every future lesson clearer, more intuitive, and far more effective!

The 3 Transformer architectures



The 3 Transformer Architectures
When people hear Large Language Models, many immediately think of systems like ChatGPT or Claude. That's understandable—but also a bit misleading.

⚠️ LLMs are not a single thing!

They come in three different Transformer architectures, each designed for different kinds of problems. Modern LLMs mostly rely on one of them—but the other two are just as important historically and practically.

Let's take a quick tour.

Encoder-Only Transformers
Encoder-only Transformers take an input sequence and turn it into rich representations, one per token.

All tokens attend to each other freely (bidirectional self-attention), which makes these models especially good at:

Understanding text

Classification

Retrieval

Semantic similarity

There's no text generation here—only encoding.

This family includes models like BERT, which popularized large-scale pretraining via masked language modeling. Encoder-only models shine when the goal is “understand this input” rather than “generate the next token.”

Encoder–Decoder Transformers
Encoder–decoder Transformers were the original Transformer design, created for sequence-to-sequence tasks like machine translation.

Here, the architecture is split into two parts:

The encoder reads and represents the input sequence

The decoder generates an output sequence, token by token

The decoder uses:

Cross-attention to look at the encoder outputs

Causal self-attention to avoid peeking at future tokens

This setup is ideal when:

Input and output are both sequences

Output length can vary freely

Models like T5 and BART belong to this family. They’re extremely flexible and powerful, especially for tasks like summarization, translation, and text-to-text learning.

Decoder-Only Transformers
Now we get to the architecture that powers modern LLMs.

Decoder-only Transformers remove the encoder entirely and rely on a single mechanism:

Causal self-attention over a growing sequence of tokens

Each token can only attend to tokens that came before it. The model is trained with a simple objective: predict the next token.

This design turns out to be shockingly powerful.

Almost every modern LLM follows this pattern. With enough data, parameters, and compute, decoder-only Transformers:

Learn language, reasoning, and structure

Perform tasks via prompting (in-context learning)

Scale remarkably well

🙋 This is the architecture we'll focus on for the rest of the course, because it's the foundation behind today's most capable LLMs

The Scaling Laws



Language modeling performance improves smoothly as we increase the model size, dataset size, and amount of compute used for training (image from Scaling Laws for Neural Language Models)
One last ingredient explains why Transformers became dominant: scaling.

Empirical studies have shown that Transformer performance improves smoothly as we increase:

Model size (parameters)

Training data (tokens)

Training compute

These improvements follow power-law scaling relationships, meaning that bigger models trained on more data tend to get predictably better—especially for language modeling.

Decoder-only architectures are particularly well-suited to this regime:

Simple objective

Massive unlabeled data

Efficient parallel training

This is why the biggest breakthroughs in recent years didn't come from radically new architectures—but from scaling Transformers to unprecedented sizes.

The LLM Training Pipeline



The LLM Training Pipeline
So far, we've talked about architectures. Now it's time to talk about how modern LLMs are actually trained.

When we refer to LLMs in this course, we'll be talking specifically about decoder-only language models—the family behind systems like ChatGPT, Claude, and Qwen. And while these models may look magical from the outside, their training follows a fairly well-defined pipeline.

🙋 That pipeline was clearly formalized in 2022 by OpenAI's InstructGPT.

Before InstructGPT, most language models were trained primarily as next-token predictors. They were good at completing text—but not necessarily good at following instructions, reasoning step by step, or aligning with human preferences.

InstructGPT marked a turning point. While it wasn't the first or the largest model of its time, it introduced and popularized a multi-stage training pipeline that has since become the standard for modern LLMs.

Diagram showing three-step methodology to train InstructGPT models.


SFT and RLHF steps (image from Aligning language models to follow instructions)
Concretely, it established a three-stage framework:

Pretraining on large-scale, raw text data

Supervised Fine-Tuning (SFT) on high-quality, task-oriented examples

Alignment via Reinforcement Learning from Human Feedback (RLHF)

This combination—pretraining + SFT + RLHF—proved far more effective than pretraining alone and laid the foundation for systems like ChatGPT and many successors.

As training methods evolved, people began describing this process in slightly different ways.

The three-stage view (pretraining → SFT → RLHF) is still widely used because of its historical and practical clarity.

A broader two-phase view is also common:

Pretraining, which builds general-purpose language capabilities

Post-training, which refines, adapts, and aligns those capabilities

Both perspectives describe roughly the same process. The difference is mostly conceptual.

If you want a great high-level walkthrough of this pipeline, we highly recommend the video below. It offers one of the best explanations available and complements this lesson beautifully!

👉 Watch the first 20 minutes, where Daniel Han covers the LLM Training Pipeline!


Pretraining 101
Now that we've mapped out the full LLM training pipeline, it's time to zoom in on the first—and most foundational—stage: pretraining.

Before models learn to follow instructions, answer questions, or behave nicely in a chat interface, they go through a much more fundamental phase. Pretraining is where a language model learns the basics of language itself—syntax, semantics, patterns, facts, and a surprising amount of what we often call world knowledge.

💸 It's also where the vast majority of a model's training compute and data are spent!

So what does pretraining actually look like?

What Happens During Pretraining



A good example of a pretrained (base) model. The model we deploy in Lab 0 falls into this category: it has been trained to complete text, not to answer questions or follow instructions. These are typically referred to as base models.
At its core, pretraining is simple.

A decoder-only Transformer is trained to predict the next token in a sequence, given all previous tokens. This is known as causal language modeling (CLM).

There are:

No instructions

No human feedback

No task-specific labels

Just raw text, one token at a time. Because pretraining uses the largest and most diverse dataset the model will ever see, this is the phase where it acquires:

Grammar and syntax

Style and structure

Facts and common sense

Code patterns

Multilingual capabilities (if trained on multilingual data)

🌎 This is why pretraining is often described as the phase where the model "learns the world."

Self-Supervised Learning



This is an example of a pretraining dataset that contains mathematical text. We will use this dataset in this week's lab.
Pretraining does not rely on labeled data.

Instead, it uses self-supervised learning. The supervision signal is already present in the data itself: the next token is the label.

This is what makes pretraining scalable. The internet provides an enormous supply of raw text, and as long as you can tokenize it, you can train on it.

That's why most pretraining corpora consist largely of web pages, books, articles or repositories!

The output of pretraining: A Base Model
The result of pretraining is what we usually call a base model (or foundation model).

At this stage, the model is extremely good at continuing text. It has absorbed a broad distribution of language patterns, facts about the world, and even coding styles.

What it can’t do—at least not reliably—is follow instructions, decide when to refuse a request, or optimize for helpfulness or safety. None of that has been taught yet. The model has learned what language looks like, not how it should behave.

In other words, it’s powerful—but not yet usable as a product. This is where a famous analogy comes in.

3 phases of ChatGPT development


The best visualization of the LLM training pipeline you'll ever see! (image from RLHF: Reinforcement Learning from Human Feedback)
Chip Huyen's Shoggoth meme captures this stage perfectly. During pretraining, the model is essentially a raw pattern-learning engine—absorbing everything it sees, without judgment or alignment. The structure is there. The power is there. But the behavior is… unpredictable.

This is where everything that comes after pretraining enters the picture.

Techniques like Supervised Fine-Tuning (SFT), LoRA, and QLoRA are not about teaching the model language from scratch. Instead, they shape and steer the capabilities already learned during pretraining.

They teach the model how to respond to instructions, how to structure answers, and how to behave in ways that are consistent with human expectations—without rewriting everything it already knows.

In other words, pretraining builds the raw intelligence. Fine-tuning and alignment are what turn that raw intelligence into something usable.

🙋 And that's exactly where we're headed in the next lessons.

Next Steps
Now that you understand the theory behind pretraining, a very reasonable question might be:

When would I ever need to use this?

It’s not like most of us are sitting on a million-dollar budget to pretrain a model from scratch… right? 😄

The good news is that pretraining doesn't always mean starting from zero. In practice, there are very real scenarios where pretraining—or a variant of it—makes a lot of sense. For example:

When you want to add new domain knowledge, like legal or medical text

When you want to support an underrepresented language

When your data distribution is very different from what the original model saw

In these cases, continued pretraining can be a powerful tool. Instead of rebuilding a model from scratch, you extend an existing base model by exposing it to new data, allowing it to absorb new patterns and knowledge while retaining what it already knows.




Continued Pretraining using Unsloth (image from Continued Pretraining with Unsloth)
This is exactly what we'll cover in this Friday's lab, walking you through how to implement it step by step using our tech stack.

See you there!

References
Olah, C. (2015, August 27). Understanding LSTM Networks. Colah’s Blog.

Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into Deep Learning.

Morgan, A. (2025, August 1). Pretraining: Breaking Down the Modern LLM Training Pipeline. Comet Blog.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling laws for neural language models. arXiv.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. arXiv.

Huyen, C. (2023, May 2). RLHF: Reinforcement learning from human feedback. Chip Huyen Blog.

NVIDIA. (2023). AI scaling laws: What they are and why they matter. NVIDIA Blog.

OpenAI. (2022, January 27). Aligning language models to follow instructions. OpenAI.

59 Likes
∙
5 Restacks
Discussion about this post
Write a comment...
Marcelo Acosta Cavalero 
Build With AWS
11 feb

Foundational articles are always hard because it’s not easy to know when to stop and move forward to the next subject. You struck a nice balance here and left enough crumbs for people to ask their LLM of choice about any gaps.

This clip lives rent-free in my head:

https://youtu.be/MO0r930Sn_8

Like (3)
Reply
Share
Ankana Mukherjee 
Ankana Mukherjee
17 feb

Great Article.

Like (1)
Reply
Share
9 more comments...

6 hands-on projects to grow as an AI Engineer
From zero to builder. One project at a time.
Aug 6, 2025 • Miguel Otero Pedrido

158

15

24


Building Agent projects without losing your mind
A clean, reusable structure that just works
Jun 18, 2025 • Miguel Otero Pedrido

114

14

16


One cookiecutter to build Agents in seconds
Building Agent APIs without losing your mind (and time!)
Sep 3, 2025 • Miguel Otero Pedrido

92

31

17



10:13
Let's build real agents, not just demos
A hands-on journey to building production-ready agents, not just prototypes.
May 14, 2025 • Miguel Otero Pedrido

277

20

41


20 books and 1 piece of advise for aspiring ML / AI Engineers
The books that helped me stop feeling lost (most of the time)
Jun 4, 2025 • Miguel Otero Pedrido

181

4

31


The AI Engineering Playbook
Your reference point for ML, AI, and Agent engineering
Sep 28, 2025 • Miguel Otero Pedrido

147


32


Systems Engineering in the Age of LLMs
A practical guide using modern AI assistants as a case study
Jan 21 • Antonio Zarauz Moreno and Miguel Otero Pedrido

68

7

11


Deploying a Multimodal Agent on AWS Lambda
The ultimate LangGraph Workshop: Part 2
Oct 15, 2025 • Miguel Otero Pedrido

45

4

7


Meet Ava: the Whatsapp Agent
Turning the Turing Test into a multimodal Whatsapp conversation
Feb 5, 2025 • Miguel Otero Pedrido

102

6

10


How to build production-ready Recommender Systems
An introduction to the four stage design for Recommender Systems
Mar 19, 2025 • Miguel Otero Pedrido

155

14

21


© 2026 Miguel Otero Pedrido · Privacy ∙ Terms ∙ Collection notice
Start your Substack
Get the app
Substack is the home for great culture
   
The Neural Maze
The Neural Maze



Modern Pretraining Strategies: A Hands-On Guide
Finetuning Sessions · Lab 1 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Feb 13, 2026
∙ Paid




Welcome to Lab 1 of the Finetuning Sessions!

After a comprehensive, introductory article about Transformers, attention mechanisms, and language model training pipeline, it's time to get our hands dirty and run experiments!

📕 If you haven't read Lesson 1's article, make sure to review it before going forward!

The Finetuning Landscape - A Map of Modern LLM Training
The Finetuning Landscape - A Map of Modern LLM Training
Miguel Otero Pedrido and Antonio Zarauz Moreno
·
11 feb
Read full story
We have prepared a walkthrough video for you so that all ideas are clearly explained and you make steps towards becoming a finetuning ninja 🥷

Here's a detailed breakdown of the main ideas covered.

Continued PreTraining (CPT)



The first stage of a language model training pipeline is usually general pretraining, but what happens when your model needs to speak "medical", "legal", or "highly specific codebase"?

👉 This is where Continued PreTraining (CPT) enters the chat.

While the initial pretraining phase (the "Foundational" stage) exposes a model to a massive crawl of the internet, that data is often a mile wide and an inch deep.

If you are building a model for a specialized industry, generic internet data won't cut it … You need a model that understands the latent relationships between niche technical terms—not just how to autocomplete a sentence.

🙋 Here's the nuance: CPT is fundamentally different from Supervised Finetuning (SFT)!

In SFT, we provide "prompt-response" pairs to teach the model how to act. In CPT, we go back to self-supervised learning. We feed the model raw, unlabeled text from your specific domain (like 50GB of internal engineering documents or medical journals) using the same "predict the next token" objective.

✅ Why go through this extra step?

Domain Vocabulary: It allows the model to learn specialized tokens and their contexts that were underrepresented in the original training set.

World Knowledge: It updates the model’s internal facts. If you're working in a rapidly evolving field like AI research, a model pretrained in 2023 won't know about 2025's breakthroughs unless you perform CPT.

Better Foundation for SFT: An SFT model is only as good as the "brain" underneath it. If the base model doesn't understand the concepts you're asking it to follow instructions on, the finetuning will likely result in "hallucination-heavy" outputs.

Curriculum Learning



While large language models often steal the spotlight, the real magic frequently happens at a smaller, more specialized scale.

In this lab, we are going to get hands-on with continued pre-training, taking a Small Language Model (SLM) and immersing it in the specialized domain of mathematics.

🙋 Rather than starting from scratch, we leverage a pre-trained foundation and refine its "brain" to handle the rigor and logic required for mathematical reasoning.

However, fine-tuning a model on a new domain isn't as simple as dumping a textbook into its memory. To get the best results—especially when dealing with the high-density logic of math—we need a strategic approach to how that information is ingested.

Plus, one of the main challenges regarding causal language modelling is saturation. If you feed a model a mountain of complex data right away, it might struggle to converge, or worse, it might "collapse" into learning low-quality patterns.

🎓 This is where we borrow a concept from human education: Curriculum Learning.

Imagine trying to teach a child calculus before they can add numbers. They might eventually memorize some formulas, but they won't understand the logic. Models are surprisingly similar. If we bombard a fresh model with highly dense, complex mathematical proofs or messy, noisy web data at the very start, the gradients can become unstable.

🙋 The "plot twist" here? The order and quality in which you show data to a model can drastically change the final performance of the weights.

In Curriculum Learning, we organize training data by signal-to-noise ratio or complexity. We start with the most "educational" data—highly curated, clean, and foundational—and gradually increase the difficulty. This is often implemented in two ways:

Data Quality Sorting: Start training on "Gold Standard" datasets (like Wikipedia or textbooks) and slowly introduce the "Long Tail" of the internet (like Reddit or raw web crawls).

Sequence Length Scaling: Start with shorter context windows (e.g., 512 tokens) so the model learns local syntax, then gradually increase to 4k, 8k, or 128k tokens to learn long-range dependencies.

✅ The result? By preventing the model from getting "overwhelmed" by noise early on, we ensure the weights settle into a more robust configuration. It leads to faster convergence, lower final loss, and a model that is significantly more resilient to "garbage in, garbage out" scenarios.

Distillation



A common practice in the last stages of language model training pipelines is knowledge distillation.

If you've ever wondered how a tiny 7B model can sometimes punch way above its weight class—sometimes even outperforming 70B models on specific tasks—this is often the secret sauce!

The idea is simple yet powerful: we take a massive, highly capable "teacher" model (like GPT-4 or Llama 3 405B) and use its "intelligence" to train a much smaller "student" model. But we aren't just asking the student to copy the teacher's homework.

How it works: In standard training, a model sees a "hard label" (e.g., the next word is "Paris"). In Distillation, the student looks at the Teacher's Soft Targets. The Teacher doesn't just say the answer is "Paris"; it provides a probability distribution. It says, "I am 90% sure it's Paris, 8% sure it's Lyon, and 2% sure it's London.”

This "Dark Knowledge"—the relative probabilities of the incorrect answers—contains crucial information about how the Teacher perceives the relationship between words.

By transfering internal semantics from an already trained model into a smaller one, we achieve:

Efficiency: You get a model that is 10x smaller and 10x faster to run in production, but retains 90% of the capability of the giant.

Reasoning Transfer: Through techniques like Chain-of-Thought Distillation, we can even teach a small model the step-by-step reasoning process of a large model by using the Teacher's rationales as the training data for the Student.

Maximizing GPUs capabilities
Training language models is costly and computationally intensive; therefore, having a good understanding of hardware requirements and usage is essential to master finetuning.

You don't want to be the person who hits "Run" only to see an Out of Memory (OOM) error five seconds later because you forgot to account for the optimizer states.

🙋 When we talk about maximizing GPUs, we are fighting a war on two fronts: Memory (VRAM) and Compute (FLOPS).

First, let's look at the Memory Wall. Many beginners think that if a model is 14GB (7B parameters in FP16), they can fit it on a 16GB GPU. Wrong!

During training, you don't just store the weights; you store the gradients, the optimizer states (which can be 3-4x the size of the weights!), and the activations for every layer.

To bypass this, we use several key optimization "tricks":

Mixed Precision (BF16): Most modern hardware (like H100s or A100s) supports Brain Float 16. It offers the range of a 32-bit float but the memory footprint of a 16-bit float, significantly speeding up training without the instability of standard FP16.

Gradient Accumulation: If your GPU is too small for a large batch size, you can calculate gradients over several smaller “micro-batches” and only update the weights after a certain number of steps.

Activation Checkpointing: Instead of storing all the intermediate "thoughts"(activations) of the model during the forward pass, we throw them away and re-calculate them during the backward pass. It's a trade-off: you save massive amounts of VRAM at the cost of about 25% more compute time.

🙋 One final tip: Always monitor your Compute Utilization (SM Utilization). If your GPU memory is full but your utilization is low, your bottleneck is likely your CPU or your data loader—not your GPU!

LLM Training Parallelism



Single GPUs are not enough for many finetuning tasks in real life.

Even if you have a top-of-the-line H100 with 80GB of VRAM, a 70B parameter model simply won’t fit once you add in the training overhead. To train the giants, we have to move from a single-player game to a multi-GPU orchestra.

Parallelism is the art of breaking a model or a dataset into pieces and spreading them across a cluster. There are three main "flavors" you need to know:

Data Parallelism (DP/DDP): The simplest form. You copy the entire model onto every GPU, but you give each GPU a different slice of the data. They all do their work and then "sync" their findings at the end of the step.

Tensor Parallelism (TP): This is for when a single layer of a model is too big for one GPU. We literally split the large weight matrices across multiple GPUs. This requires incredibly fast interconnects (like NVLink) because the GPUs have to talk to each other constantly during a single forward pass.

Pipeline Parallelism (PP): We split the model "vertically" by layers. GPU 0 handles layers 1-10, GPU 1 handles 11-20, and so on. It’s like an assembly line. The challenge here is "bubbles"—idle time where the later GPUs are waiting for the first ones to finish.

Sequence Parallelism (SP): This is essential when you have such long context (e.g., feeding an entire codebase to a language model) that have huge memory requirements.

✅ The Modern Standard: FSDP & DeepSpeed. Today, we often use Fully Sharded Data Parallelism (FSDP). It’s a hybrid approach that shards the model weights, gradients, and optimizer states across all available GPUs. It gives you the memory efficiency of model parallelism with the simplicity of data parallelism. Understanding which of these to use is the difference between a project that takes 2 days and one that takes 2 weeks.

Demo time!
Time for the video lab! 🙌

In this lab, we'll implement Continued Pretraining (CPT). We take a pretrained base model and immerse it in raw domain-specific text (mathematics) to reshape its internal representations.

Unlike instruction tuning, this is self-supervised domain adaptation — the model learns by predicting the next token across curated domain data.

And we don't stop at training!

Once the new model is pushed to the Hub, we'll deploy it behind an inference endpoint.

Now, download the code from the shared drive…

Get the code here!

… then press play and follow along! 👇


Next Steps
🎙️ This Sunday at 4:00 PM CET, we’re hosting the Lesson 1 Office Hours for the Finetuning Sessions.

We’ll recap the key ideas from Lesson 1, expand on the concepts we introduced, and answer your questions live!

See you on Sunday! 👋




38 Likes
∙
5 Restacks
Discussion about this post
Write a comment...
Santhanalakshmi 
14 feb

Great content. @Antonio, a small doubt, it's been mentioned that we are about to do only CPT but I could see SFTTraining() in the code as well. So is it something like we do CPT first and then doing SFT in the code ?

Like (2)
Reply
Share
2 replies
Xeno_invest 
2 mar

Great content!

Like
Reply
Share
4 more comments...

6 hands-on projects to grow as an AI Engineer
From zero to builder. One project at a time.
Aug 6, 2025 • Miguel Otero Pedrido

158

15

24


Building Agent projects without losing your mind
A clean, reusable structure that just works
Jun 18, 2025 • Miguel Otero Pedrido

114

14

16


One cookiecutter to build Agents in seconds
Building Agent APIs without losing your mind (and time!)
Sep 3, 2025 • Miguel Otero Pedrido

92

31

17



10:13
Let's build real agents, not just demos
A hands-on journey to building production-ready agents, not just prototypes.
May 14, 2025 • Miguel Otero Pedrido

277

20

41


20 books and 1 piece of advise for aspiring ML / AI Engineers
The books that helped me stop feeling lost (most of the time)
Jun 4, 2025 • Miguel Otero Pedrido

181

4

31


The AI Engineering Playbook
Your reference point for ML, AI, and Agent engineering
Sep 28, 2025 • Miguel Otero Pedrido

147


32


Systems Engineering in the Age of LLMs
A practical guide using modern AI assistants as a case study
Jan 21 • Antonio Zarauz Moreno and Miguel Otero Pedrido

68

7

11


Deploying a Multimodal Agent on AWS Lambda
The ultimate LangGraph Workshop: Part 2
Oct 15, 2025 • Miguel Otero Pedrido

45

4

7


Meet Ava: the Whatsapp Agent
Turning the Turing Test into a multimodal Whatsapp conversation
Feb 5, 2025 • Miguel Otero Pedrido

102

6

10


How to build production-ready Recommender Systems
An introduction to the four stage design for Recommender Systems
Mar 19, 2025 • Miguel Otero Pedrido

155

14

21


© 2026 Miguel Otero Pedrido · Privacy ∙ Terms ∙ Collection notice
Start your Substack
Get the app
Substack is the home for great culture
