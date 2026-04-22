
The Neural Maze
The Neural Maze



QLoRA Explained - How 4 Bit Quantization Unlocks Frontier Models
Finetuning Sessions · Lesson 4 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Mar 04, 2026




Welcome to Lesson 4 of the Finetuning Sessions!

In our last exploration, we looked at how LoRA revolutionized finetuning by moving us from updating billions of parameters to just a sliver of low-rank matrices.

It solved the compute problem and the storage problem. But for the practitioner sitting in front of a single GPU, a more formidable enemy remains …

The VRAM Wall!

Even when we aren't updating the base model weights, those weights still need to live somewhere. In standard 16-bit precision, a 70B parameter model is a 140GB "fixed cost" before you've even processed a single token of context.

This creates a zero-sum game: every gigabyte consumed by the model's "static" memory is a gigabyte stolen from your batch size and your context window.

Enter QLoRA (Quantized LoRA). If LoRA was about reducing what we train, QLoRA is about rethinking how we store what we aren't training.

It is the bridge that allows us to squeeze "frontier-class" intelligence into "consumer-class" hardware without sacrificing the mathematical integrity of the gradients.

Ready? Let's dive in! 👇

Go Premium if you want the full experience of this course, with exclusive access to all the upcoming articles, labs, and office hours!

The Fixed Cost of Training
black and silver sony cassette player


Image by Nana Dua (source: Unsplash)
In the economy of deep learning, VRAM is the ultimate finite resource.

When we initialize a training run, the GPU doesn't start with a blank slate, it begins with a massive, non-negotiable "down payment" known as Model States.

These are the static costs of doing business—the memory required to hold the model's weights, the gradients for backpropagation, and the internal states of the optimizer.

For a standard 7B parameter model in 16-bit precision (FP16 / BF16), the weights alone claim 14 GB. But loading the model is just the beginning. To actually learn, the system must also track the direction of the error (gradients) and the momentum of the update (optimizer states), turning a seemingly manageable model into a memory-hungry leviathan.

To understand the scale of this burden, we must look at the arithmetic of the AdamW optimizer, the industry workhorse.




Image from Decoupled Weight Decay Regularization
In a standard mixed-precision setup, AdamW maintains three distinct tensors for every single trainable parameter:

A master copy of the weight in FP32 to prevent rounding errors

A first-moment buffer (momentum)

And a second-moment buffer (variance)

This adds up to 12 bytes per parameter for the optimizer alone!

When you stack this on top of the 2 bytes for the weights and 2 bytes for the gradients, you are looking at a total fixed cost of roughly 16 bytes per parameter.

For a 70B model, this translates to a staggering 1.1 TB of VRAM before you've even fed the model its first token 😅

This fixed-cost allocation creates a rigid ceiling for what we call "Residual States"—the dynamic memory used for activations and the KV Cache.

Every gigabyte consumed by a redundant optimizer state or a high-precision weight is a gigabyte stolen from your training throughput. In practice, this forces a brutal trade-off …

To stay within the 80 GB limit of a flagship H100, practitioners are often forced to slash their batch sizes to the bare minimum. Low batch sizes don't just slow down training, they introduce "noise" into the gradient updates, often requiring more steps and more expensive compute time to reach convergence.

Beyond throughput, the "static" weight footprint directly dictates the maximum context length a model can handle.

In modern Transformer architectures, the KV Cache—which stores the keys and values of previous tokens to avoid recomputation—grows linearly with sequence length and batch size.

If your model weights and gradients occupy 90% of your VRAM, you are left with a tiny "buffer zone" for context. This is why many high-end models were historically capped at 2k or 4k tokens.

The VRAM was simply too cluttered with the "machinery" of training to leave room for the "memory" of the conversation!

The relationship can be modeled as a zero-sum game of VRAM occupancy:


By reducing VRAM_Fixed, we don't just make training "cheaper", we fundamentally expand the capabilities of the hardware. We shift the boundary of the possible, allowing a single GPU to simulate the memory capacity of a small cluster.

This is where the architectural brilliance of QLoRA shines.

While standard LoRA reduced the memory footprint of G and O by only updating a small subset of parameters, it left the massive W (the base model) untouched in 16-bit.

QLoRA attacks the final frontier of the fixed cost by quantizing that base model down to 4-bit.

By shrinking the "static" model weights from 2 bytes per parameter to just 0.5 bytes, QLoRA frees up nearly 75% of the weight-related VRAM. This "found" memory is then reinvested into what actually matters for performance: massive batch sizes for stable gradients and deep context windows that allow the model to reason over entire books rather than just paragraphs.

KV Cache and the Memory Bottleneck
If the model weights are the "static" library of an LLM, the KV Cache (Key-Value Cache) is its short-term working memory.

In the Transformer architecture, every time the model generates a new token, it needs to "look back" at every previous token in the sequence to calculate attention.

Recomputing these representations from scratch for every single step would be computationally suicidal (O(n^2) complexity).

To solve this, we store the intermediate Key and Value tensors in VRAM. This is a brilliant optimization for speed, but it introduces a secondary "VRAM Wall" that is often more treacherous than the model weights themselves.

What is the KV Cache? Secret of Fast LLM Inference | Towards AI


Image from The Secret Behind Fast LLM Inference: Unlocking the KV Cache
The KV Cache is a voracious consumer of memory.

Its size is a direct product of the sequence length, the batch size, the number of layers, and the hidden dimension of the model.

For a 70B model with a 128-dimension head and 80 layers, a single 4k context window can easily swallow 2.5 GB of VRAM.

This might seem manageable on an 80 GB H100, but in a production environment, you aren't serving one user … but dozens.

This is where concurrency enters the equation. If your model weights already occupy 40 GB, and each active user requires 2.5 GB for their context, you hit a "hard ceiling" of roughly 16 concurrent users before the GPU throws an Out-of-Memory (OOM) error.

This creates a high-stakes game of "VRAM Tetris". In the era of high-precision weights (FP16), the "static" model was so large that there was very little room left for the KV Cache. This forced a trade-off.

You could either have a massive model with a tiny context window, or a smaller model with a longer memory.

By using QLoRA to compress the base model down to 4-bit, we aren't just saving disk space, we are reclaiming the "concurrency real estate" on the GPU. Every 10 GB saved from the model weights is 10 GB that can be reallocated to the KV Cache, effectively doubling or tripling the number of users a single node can support!

However, the KV Cache problem isn't just about capacity; it's about memory bandwidth.

During inference, the GPU spends the vast majority of its time moving data from HBM (High Bandwidth Memory) to the streaming multiprocessors, rather than actually doing math.

This is known as being "memory-bound".

When the model is quantized to 4-bit, we are moving 4x less data per weight compared to FP16. This reduction in the "data movement tax" allows the GPU to feed the compute cores faster, which directly translates to higher tokens-per-second and lower latency for the end user.

Attention Variations — MQA vs GQA vs MHA vs MLA | by VerticalServe Blogs |  Medium


Image from Attention Variations — MQA vs GQA vs MHA vs MLA
The final piece of the puzzle is how quantization interacts with Grouped Query Attention (GQA) and FlashAttention.

Modern architectures use GQA to reduce the number of Key / Value heads, which already shrinks the KV Cache footprint.

When you combine GQA with a 4-bit quantized base model, the memory profile of the LLM transforms. We move from a world where "the model is the bottleneck" to a world where "the context is the bottleneck".

This shift is fundamental for building "Long Context" applications, such as RAG (Retrieval-Augmented Generation) systems that need to "read" entire technical manuals in a single prompt.

By lowering the "static" floor of the weights, we raise the "dynamic" ceiling of the context.

This allows us to push the boundaries of concurrency (how many people can talk to the model at once) and depth (how much the model can remember about the conversation).

In the next section, we will look at the "digital rulers" that make this compression possible: the transition from the fluid precision of FP32 to the jagged, efficient world of 4-bit integers.

From FP32 to the 4-Bit Horizon
In the early days of deep learning, we were generous with precision.

We treated VRAM as an infinite canvas and used FP32 (Single Precision), a 32-bit floating-point format that can represent numbers with staggering granularity.

Think of FP32 as a ruler with billions of sub-millimeter markings; it's perfect for scientific simulations where rounding errors can crash a rocket, but for a neural network, it's often overkill.

Most weights in a trained LLM don't need that level of resolution to convey their signal.

As we moved to FP16 and BF16 (Brain Float 16), we effectively "halved" the ruler. BF16, in particular, was a hardware masterstroke: it kept the same dynamic range (the exponent) as FP32 but sacrificed the precision (the mantissa), recognizing that a model cares more about the magnitude of a signal than its exact decimal place.




Image from Getting Immediate Speedups with NVIDIA A100 TF32
The transition to INT8 represented the first major shift from "floating" scales to "fixed" integers.

This was the era where Google's TPUs (Tensor Processing Units) began to dominate the efficiency conversation.

By forcing weights into an 8-bit integer format, TPUs could perform matrix multiplications with significantly less silicon area and power than floating-point units.

However, INT8 quantization is a rigid process; it requires "calibration" to ensure the clipping of values doesn't destroy the model's nuances.

While INT8 was a breakthrough for massive-scale inference in data centers, it often felt too "brittle" for the generative complexity of modern LLMs, leading to the "quantization tax" where model intelligence dropped sharply as memory usage fell.

TPU vs GPU: Comprehensive Technical Comparison


The leap to 4-bit is where the geometry changes entirely.

In a 4-bit world, we only have 16 possible values to represent a weight.

Imagine trying to paint a masterpiece with only 16 colors. If you distribute those colors evenly (linear quantization), you lose the subtle nuances of the "outliers"—those rare but vital weights that carry the most information.

This is the fundamental challenge of quantization …

How to map a continuous distribution of weights onto a tiny, discrete set of integers without collapsing the model's ability to reason.

To solve this, we turned to the statistics of the weights themselves.

Neural network weights typically follow a Normal (Gaussian) distribution, clustered tightly around zero.

Standard 4-bit integers waste their 16 slots by spacing them evenly across a range, often leaving empty "bins" where no weights exist. This is why NF4 (NormalFloat 4) became the backbone of QLoRA.

By mathematically aligning the 16 available "slots" with the actual statistical distribution of the weights, we can achieve 4-bit storage that performs almost identically to 16-bit.

It is an information-theory hack that proves we don't need more bits; we just need our bits to be in the right place!

The latest frontier in this "digital ruler" evolution is the Microscaling (MX) standard, specifically MXFP4.

Championed by the Open Compute Project, MXFP4 uses an E8M0 scaling format.

In this setup, a block of 32 elements shares a single 8-bit scale factor that is strictly a power-of-two (2^n). This is an "efficiency-first" design: by snapping scales to powers-of-two, hardware can perform scaling using simple bit-shifts rather than expensive multiplications.

It's a rugged, fast format designed for the brutal throughput requirements of massive AI factories, though it can sometimes struggle with the "outlier spikes" found in the most complex frontier models.

Introducing NVFP4 for Efficient and Accurate Low-Precision Inference |  NVIDIA Technical Blog


NVIDIA's answer to this within the Blackwell architecture is NVFP4, which pushes the "Microscaling" philosophy even further using an E4M3 scale.

Unlike the rigid power-of-two jumps of MXFP4, NVFP4's 8-bit scale includes 3 bits of mantissa, allowing for fractional scaling (e.g., 1.5x or 2.25x).

By reducing the block size from 32 down to 16 elements and providing this finer-grained scaling, NVFP4 can "hug" the actual data distribution much more tightly.

It is the most sophisticated digital ruler yet—a 4-bit format that behaves with the mathematical dignity of 8-bit silicon, enabling Blackwell to double its throughput without a noticeable drop in model "intelligence".

The Architecture of Quantization
The world of quantization is governed by a diverse taxonomy—GGUF, GPTQ, AWQ, and NVFP—each designed to navigate the trade-off between memory and intelligence.

At the highest level, we distinguish between "Weights-Only" methods and "Full Quantization"

Popular local formats like GGUF and GPTQ act as sophisticated compression for model weights; they allow a 70B model to reside in consumer-grade VRAM by storing weights in 4-bit, but they "dequantize" them back to 16-bit for the actual computation.

While this solves the storage crisis, it leaves the compute bottleneck intact.

To truly break the speed barrier, "Full Quantization" targets both weights and the volatile activations, allowing the GPU to perform the math itself in low precision. However, because activations change with every prompt, they are far more prone to "outlier" spikes that can degrade model reasoning.

To mitigate this degradation, different algorithms employ unique strategies for protecting model "salience".

GPTQ uses second-order optimization to minimize error layer-by-layer, adjusting unquantized weights to compensate for their compressed neighbors.

However, it can be "blind" to the most critical parameters.

AWQ (Activation-aware Weight Quantization) addresses this by identifying the 1% of "salient" weights—those that interact with high-magnitude activations—and scaling them up to protect them from quantization noise.

This surgical approach ensures that the model's "core logic" remains intact even as its overall footprint shrinks. By contrast, NVFP moves toward native hardware-level scaling, using micro-scaling factors to "hug" the weight distribution more tightly than traditional linear methods.

The breakthrough of QLoRA lies in its hybrid architecture.

It keeps the massive base model "frozen" in a statistically optimized 4-bit NormalFloat (NF4) format while utilizing high-precision 16-bit LoRA adapters to handle the learning.

This allows the model to maintain the stability of high-fidelity gradients while reaping the 75% VRAM savings of 4-bit storage. As we move into the Blackwell era with native NVFP4 support, the "dequantization" tax is being eliminated entirely.

The hardware now "thinks" in the same 4-bit language it uses for storage, finally aligning the math of the neural network with the silicon of the GPU.

This convergence represents the ultimate democratization of scale, transforming the VRAM wall into a gateway for local, frontier-class intelligence.

From Turing to Blackwell



The history of AI is, in many ways, the history of making precision smaller.

The journey began in 2018 with the Turing architecture (T4), which introduced the first-generation Tensor Cores. Before Turing, GPUs treated every number as a general-purpose calculation; Tensor Cores changed the game by building dedicated hardware "lanes" specifically for matrix multiplication.

Turing made FP16 the standard for mixed-precision training, effectively doubling throughput by sacrificing unnecessary bits.

However, at this stage, any move toward 4-bit was a software trick—a "simulated" quantization that saved memory but often cost more in compute overhead than it saved in storage.

With the arrival of Ampere (A100) in 2020, NVIDIA solved the stability problem of low precision.

Ampere introduced BF16 (Brain Float 16) and TF32, formats that kept the dynamic range of a 32-bit number while using only 16 bits of space.

Models that previously required a supercomputer could now fit on a single HGX node. Ampere also teased the first hardware support for sparsity, but it was still an 8-bit and 16-bit world. The silicon was getting smarter, but the 4-bit frontier remained a theoretical goal.

The Ada Lovelace (RTX 40-series/L40S) and Hopper (H100) architectures marked the true "Quantization Pivot".

These chips introduced the first-generation Transformer Engine, a sophisticated hardware-software layer that dynamically managed precision during training. For the first time, we had native FP8 support, which provided a 2x-4x boost in throughput over FP16.

Hopper became the engine of the LLM explosion precisely because it could handle the massive throughput of FP8 without losing the model's "reasoning" quality.

Yet, even on Hopper, 4-bit was still handled via INT8 emulation, a compromise that limited its peak efficiency.

The Blackwell architecture (B200 / GB200) represents the final destination: the arrival of the 2nd-Generation Transformer Engine and native NVFP4 support.

Blackwell doesn't just store 4-bit weights; it executes them natively in its 5th-generation Tensor Cores.

By using the micro-scaling techniques we discussed—specifically the E4M3 scaling factors—Blackwell can process 4-bit tensors with the accuracy of 8-bit ones. This hardware-level leap delivers up to 15 PetaFLOPS of 4-bit compute per GPU, a staggering 30x increase in inference performance compared to the Pascal era.

This evolution from Turing to Blackwell has fundamentally changed the "Fixed Cost" equation of Section 1!

We have moved from a world where hardware was a passive container for weights to a world where the silicon is an active partner in compression.

Ultimately, QLoRA is the software bridge that allows us to walk across this hardware landscape.

It allows us to take the massive, trillion-parameter dreams of the next decade and fit them into the 4-bit silicon realities of today. As Blackwell enters the data center, the "VRAM Wall" it's being rebuilt entirely out of 4-bit blocks, opening the maze to anyone with a single, modern GPU and a desire to build.

Inside QLoRA: NF4 and Paged Optimizers
While standard LoRA proved we could finetune massive models by only updating a sliver of their parameters, it left a glaring inefficiency: the massive "frozen" base model still had to sit in VRAM at 16-bit precision.

If you tried to simply cast that base model to a standard 4-bit integer (INT4), the model's reasoning would often collapse.

The breakthrough of QLoRA was the introduction of 4-Bit NormalFloat (NF4). NF4 is a data type custom-engineered for the specific statistical profile of neural network weights. Since weights naturally follow a zero-centered Gaussian distribution, NF4 uses information theory to space its 16 available "bins" more densely where the weights are most crowded and more sparsely at the edges.

This ensures that every bit is used with maximum efficiency, keeping the quantization error significantly lower than standard linear formats.

However, shrinking the weights is only half the battle.

When we finetune a model, we still have to deal with "gradient spikes"—sudden surges in memory usage during the backpropagation step that can trigger a dreaded Out-of-Memory (OOM) error.

QLoRA introduces Double Quantization to squeeze even more "juice" out of the VRAM. This process quantizes the quantization constants themselves, saving an additional 0.37 bits per parameter. While that might sound like a rounding error, on a 70B parameter model, it reclaims hundreds of megabytes of precious memory—room that can be reinvested into a larger batch size or a longer sequence length.




The final piece of the QLoRA puzzle is the Paged Optimizer.

Think of this as a "safety valve" for your GPU memory. Leveraging NVIDIA's Unified Memory feature, Paged Optimizers can automatically offload optimizer states to the CPU RAM when the GPU hits its limit, and then pull them back when they are needed for the next update.

This prevents the training run from crashing during a memory spike, allowing you to "over-provision" your GPU. It effectively turns your system's 128GB of DDR5 RAM into a temporary extension of your 24GB HBM, ensuring that the training process remains resilient even when you are pushing the absolute limits of your hardware.

By combining NF4, Double Quantization, and Paged Optimizers, QLoRA transforms a single consumer-grade GPU into a powerhouse capable of handling models that were previously the exclusive domain of multi-million dollar clusters.

It proves that in the era of massive AI, the most important innovations aren’t just about adding more parameters—they are about being smarter with the ones we already have.

We are no longer limited by the size of our silicon, but by the creativity of our constraints!

Next Steps



That's everything for today's article!

This Friday, we'll move from theory to practice in a hands-on QLoRA lab.

See you there! 👋

21 Likes
∙
3 Restacks
Discussion about this post
Write a comment...

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



Engineering QLoRA for memory-efficient LLM Finetuning
Finetuning Sessions · Lab 4 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Mar 06, 2026
∙ Paid




Welcome to Lab 4 of the Finetuning Sessions!

In today's lab, we're moving from theory to the terminal. We are stepping onto the 4-bit frontier with a hands-on QLoRA finetuning experiment.

If you haven't read last Wednesday's deep dive, make sure to review it before going forward! Understanding the geometry of the "VRAM Wall" is essential to mastering the code we are about to run.

QLoRA Explained - How 4 Bit Quantization Unlocks Frontier Models
QLoRA Explained - How 4 Bit Quantization Unlocks Frontier Models
Miguel Otero Pedrido and Antonio Zarauz Moreno
·
4 mar
Read full story
The shift from standard LoRA to Quantized LoRA (QLoRA) isn't just a minor optimization—it's a fundamental rethinking of the GPU memory map.

While LoRA taught us how to reduce the active parameter count, QLoRA attacks the "static" weight footprint that usually keeps frontier-class models out of reach for individual researchers.

Today, we are going deep into the implementation.

We won't just "run a script"; we will dissect how the Unsloth library leverages 4-bit NormalFloat (NF4) and Paged Optimizers to squeeze every drop of performance out of a single GPU.

We will walk through a live training job on Hugging Face infrastructure, analyzing how to reallocate the VRAM we "save" through quantization into massive context windows and stable gradients.

By the end of this lab, you won't just have a trained adapter—you'll have a first-principles understanding of how to orchestrate high-fidelity training on low-precision silicon.

From LoRA to QLoRA
⚠️ Sorry about the issue with the video. We had some technical problems during the recording. The voice and screen sharing work well, but there is some lag in the window where the person speaking appears. Thanks for your understanding!


In this first clip, we look at the script that drives our experiment.

Get the code here!

The beauty of the Unsloth library is that transitioning from a standard LoRA setup to a memory-efficient QLoRA setup is a single-parameter change, but its impact on your VRAM "geometry" is profound.

The 4-bit switch: Notice the load_in_4bit=True argument in the FastLanguageModel.from_pretrained call. By flipping this from False to True, you aren't just loading a smaller file; you are triggering the NF4 (NormalFloat 4-bit) quantization we discussed in the theory section. This immediately shrinks the "fixed cost" of your base model weights by 75%.

The PEFT Injection: In the get_peft_model block, we target the specific projection layers (q_proj, k_proj, etc.). In a QLoRA context, these 16-bit adapters are being mathematically stitched onto a 4-bit base. We keep the learning high-fidelity while the "memory" remains compressed.

The memory safety valve: Look at optim = "adamw_8bit". This ensures that even our optimizer states—the “silent killers” of VRAM—are compressed, allowing us to fit larger models into smaller hardware slots without losing the "momentum" of the Adam algorithm.

Hugging Face Jobs Hardware

When we run hf jobs uv run, we have to choose a "flavor".

This isn't just a pricing choice; it’s an architectural decision. Hugging Face provides a range of NVIDIA hardware that aligns perfectly with our generational analysis of Tensor Cores.

T4 (Small): The veteran. Ideal for testing logic or fine-tuning tiny models like Qwen3-0.6B. It supports FP16 but lacks the hardware-level "brains" for native 4-bit acceleration.

A10G (Medium): The workhorse of modern labs. It offers 24GB of VRAM and Ampere-level efficiency. This is where QLoRA truly shines, allowing you to fine-tune 7B or 13B models that would otherwise hit an OOM (Out of Memory) error on standard consumer cards.

A100 / H100 (Large/XL): The "Blackwell Preludes". Use these when you need the Transformer Engine to accelerate FP8 or when your context window (KV Cache) needs to span entire technical manuals.

The strategy: Always start small. Use an a10g-small to verify your loss curves over 50 steps before scaling up to an h100 for a production-grade run.

Why QLoRA?

Why go through the trouble of quantization? It isn't just about saving money; it's about expanding the "edge".

The "edge" reality: Most production AI doesn't live on an H100 cluster; it lives on a MacBook, a local workstation, or an edge device. QLoRA allows us to take a model trained in the cloud and "distill" its fine-tuned intelligence into a format that can run on consumer-grade silicon.

Context over capacity: By reducing the fixed weight cost to 4-bit, we "buy" space for the KV Cache. This makes QLoRA the go-to choice for RAG (Retrieval-Augmented Generation) applications where the model needs to "read" a 100-page PDF without forgetting the first sentence.

Rapid prototyping: QLoRA reduces the "iteration loop". You can test five different hyperparameters in the time it would take to run one full-precision fine-tuning session.

Comet ML Dashboard

A lab is only as good as its data. In this clip, we look at the Comet ML dashboard to interpret what the GPU is actually doing under the hood during our run.

System Metrics: Keep your eye on the GPU Utilization vs. GPU Memory charts. If your memory is a flat line at 95% but utilization is low, you are likely bottlenecked by your batch_size. QLoRA gives you the "slack" to push that batch size higher.

The Loss Curve: We look for the "LoRA Elbow"—that sharp initial drop in loss. Because we are using 4-bit, we watch for "spikes" that might indicate our learning rate is too aggressive for the quantized precision of the base model.

Hyperparameter Tracking: Comet captures our lora_r and lora_alpha. By comparing multiple runs, we can see exactly how much "capacity" we need to solve a specific instruction-following task.

Adapters and the Hub
The final step of the code is model.push_to_hub. This is a crucial "First Principles" moment that defines the modularity of modern AI.

The Tiny File: When you check your Hugging Face repo after this job, you won't see a 5GB model. You'll see a 100MB adapter. This is the "delta"—the concentrated intelligence we added during the lab.

Portability: These adapters are the ultimate "plug-and-play" components. You can take this tiny file and attach it to the base Qwen3 model on any device, anywhere. You've successfully separated the knowledge (the base) from the skill (the adapter).

The next step: In our next session, we'll look at how to merge these adapters back into the base model for deployment, or how to swap multiple adapters on the fly using a single base model instance.

Next Steps



🎙️ This Sunday at 4:00 PM CET, we're hosting the Lesson 4 Office Hours for the Finetuning Sessions.

We'll recap the key ideas from Lesson 4, expand on the concepts we introduced, and answer your questions live!

See you on Sunday! 👋

13 Likes
∙
3 Restacks
Discussion about this post
Write a comment...
Zeeshan Amber 
18 mar
Edited

Getting this error at the end of HF job - TypeError: Unsloth: You are pushing to Hugging Face, but Qwen3-0.6B-QLoRA-Finetuning is not a valid repo.

Is there any recommended format or length constraint for repo name?

Also, will I have to re-run the full job or is there a way to push the already trained model to repo after fixing the repo name?

Like
Reply
Share

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
