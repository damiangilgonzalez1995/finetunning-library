
The Neural Maze
The Neural Maze



Understanding LoRA from First Principles
Finetuning Sessions · Lesson 3 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Feb 25, 2026




Welcome to Lesson 3 of the Finetuning Sessions!

Low-Rank Adaptation (LoRA) has quietly transformed from a clever research trick into the default strategy for steering large-scale models in production.

What began as a parameter-efficient finetuning (PEFT) method is now an industry standard. Yet while libraries like Hugging Face's peft or Unsloth have democratized its use, they've also abstracted away the deeper mechanics.

🙋 The "how" is packaged into a few lines of code, but the "why" is left implicit. And that "why" is where real understanding lives!

To truly grasp LoRA, we have to step past the convenience of the API and return to first principles. Large models are not magic—they are compositions of weight matrices, and those matrices encode behavior.

If you understand how those weights shape representation and transformation, you begin to see why we can alter a model's behavior without retraining billions of parameters.

✅ In this article, we'll strip LoRA down to its foundations.

We'll start at the architectural level—what weight matrices actually do—and then descend into the linear algebra that makes low-rank updates not just efficient, but surprisingly expressive.

Ready? Let's go! 👇

Go Premium if you want the full experience of this course, with exclusive access to all the upcoming articles, labs, and office hours!

LoRA Foundations
Do you remember when, in Lesson 1, we dissected the three Transformer architectures?




If not, here's a quick mental refresh — think of it as clearing the cache before we continue … 🤣

Encoder-only models use bidirectional attention to produce contextualized representations of an input. They are designed for understanding tasks, where the goal is to extract meaning rather than generate new sequences.

Decoder-only models are autoregressive. Each token attends only to previous tokens, enabling next-token prediction and large-scale text generation. The LLMs we finetune in this course belong to this category!

Encoder–decoder models combine both components: an encoder first builds a representation of the input, and a decoder then generates a new sequence conditioned on that representation. They are commonly used for structured transformation tasks such as translation. This was also the original Transformer formulation introduced in Attention Is All You Need.

🙋 That's the structural landscape most modern models live in.

Now, even though almost every large-scale system today is some variant of the Transformer, understanding LoRA benefits from stepping slightly outside that family.

To see why low-rank adaptation works, we need to revisit a simpler architectural idea — one centered on compression and latent structure: the autoencoder.

Connecting Autoencoders and LoRA



Autoencoder architecture (source: What is an autoencoder?)
Although the architectures discussed above are Transformer-based, the concept of the autoencoder is particularly useful for understanding the logic behind LoRA.

An autoencoder is composed of two parts: an encoder that compresses the input into a lower-dimensional representation (the latent space), and a decoder that reconstructs the original input from that compressed representation.

This structure is commonly used for tasks such as feature extraction, denoising, and dimensionality reduction.

🤔 At this point, you might be wondering: how is this related to LoRA or large language models?

👉 The connection lies in dimensionality!

Just as an autoencoder demonstrates that high-dimensional input data can often be compressed into a much lower-dimensional latent space without losing its essential structure, LoRA is built on a similar assumption.

🙋 Instead of compressing data, LoRA assumes that the updates applied to a model's weights during finetuning can be represented in a lower-dimensional subspace!




Representing model updates in a lower-dimensional subspace! (Image from LoRA: Low-Rank Adaptation of Large Language Models)
Returning to Transformers
Regardless of the specific architecture — encoder-only, decoder-only, or encoder–decoder — a model's knowledge is encoded in large weight matrices.

In full finetuning (as in Lab 2), we adapt a pre-trained LLM to a new task by updating its weights. If we denote the original weight matrix as W, then finetuning learns a modification ΔW. The updated weights can therefore be written as:

Now, check the figure below.

This is exactly what is happening. We start with an input X, which is multiplied by the pre-trained weight matrix W.

During full finetuning, W is updated directly. In other words, the model learns a dense correction ΔW, and all parameters of W are allowed to change.




This approach is simple and effective … but creates two major issues.

➤ Issue 1 - VRAM Constraints
For very large models (e.g., 70B parameters), updating all weights requires storing:

Gradients

Optimizer states (e.g., Adam moments)

The updated parameters themselves

🙋 This makes full finetuning prohibitively expensive without access to large GPU clusters.

➤ Issue 2 - Catastrophic Forgetting
Full finetuning gives the model complete freedom in parameter space. But this freedom comes at a cost!

When adapting to a narrow task (e.g., medical coding), gradients can overwrite useful general-purpose knowledge. The model may improve on the new domain but degrade in reasoning, fluency, or broader capabilities.

Because full finetuning has effectively unlimited degrees of freedom, nothing constrains the update to remain "close" to the original solution.

The LoRA Hypothesis



LoRA, introduced by Microsoft researchers in 2021, proposes a structural solution.

The core intuition:

🙋 The weight update ΔW does not require full rank. The adaptation lies in a low-dimensional subspace.

This is sometimes referred to as the intrinsic rank hypothesis. So, instead of learning a full dense ΔW, LoRA factorizes it into two smaller matrices, A and B.

The updated weight matrix becomes:

Here is the key difference from full finetuning:

The original pre-trained matrix W is frozen.

Only A and B are trainable.

In the figure above, instead of modifying W directly, the input X flows through the frozen weights, and an additional low-rank correction BA is added to the transformation.

✅ The model's behavior changes, but the original parameters remain untouched.

Why LoRA is stable?
Two technical decisions make LoRA practical and stable.

➤ Decision 1 - Zero initialization of B
Matrix B is initialized to zero. This ensures:

at the start of training.

As a result, the model initially behaves exactly like the pre-trained base model. There is no sudden perturbation or instability at the beginning of finetuning.

The update grows gradually as training progresses.

➤ Decision 2 - The scaling factor α
In practice, the update is applied as:

 
 
The hyperparameter α controls the strength of the update.

Scaling by α/r:

Allows us to change the rank r

Without drastically retuning the learning rate

Improves numerical stability

It effectively decouples rank selection from update magnitude.

LoRA Hyperparameters



An example of how LoRA is implemented in practice using Unsloth.
Up to this point, we've focused on what LoRA does: it constrains weight updates to a low-rank subspace. But in practice, that constraint is controlled by a set of hyperparameters that determine how expressive, stable, and efficient the adaptation will be.

🙋 Even though LoRA dramatically reduces the number of trainable parameters, it does not eliminate the need for tuning.

In fact, because we are deliberately limiting the adaptation to a smaller space, choosing the right configuration becomes even more important. The hyperparameters define the size of that space, the strength of the update, and the way optimization unfolds over time.

In the following sections, we will examine the most important LoRA hyperparameters at a high level!

➤ LoRA Rank (r)
The rank r is the most important LoRA-specific hyperparameter.

It determines the dimensionality of the subspace in which the weight update lives. Since LoRA decomposes the update as ΔW=BA, the rank defines how expressive that update can be.

A small rank (e.g., 4 or 8) constrains adaptation heavily. This acts as a strong regularizer and is often sufficient for relatively simple tasks.

A larger rank (e.g., 32, 64, or higher) increases capacity, allowing the adapter to model more complex task shifts. However, increasing rank also increases memory usage and the risk of overfitting!

🙋 In practice, ranks between 8 and 32 work well for most instruction-tuning tasks. The key trade-off is capacity versus efficiency. Rank defines how many independent "directions" the model is allowed to move in weight space.

➤ LoRA Alpha (lora_alpha)
LoRA does not apply the raw product BA directly. Instead, the update is typically scaled:

 
 
The parameter α controls the strength of the update relative to the base model. If the rank defines the dimensionality of adaptation, alpha defines its magnitude.

🙋 A common and stable choice is α=r. Some practitioners use 2r to allow slightly more aggressive updates.

If alpha is too small, the adapter may struggle to influence the model. If too large, training can become unstable.

➤ Learning Rate
Even though we are only updating a small subset of parameters, the learning rate remains critical.

Too high a learning rate can cause divergence or noisy training. Too low may slow convergence or prevent the adapter from learning meaningful corrections.

Unlike rank, which controls capacity, the learning rate controls optimization dynamics.

➤ Target Modules
Up to now, we have described LoRA using a generic matrix W. However, in a Transformer, there is no single weight matrix.

🙋 There are many projection matrices, each serving a distinct role.

The target_modules parameter determines which linear layers receive LoRA adapters. Common targets include:

q_proj

k_proj

v_proj

o_proj

gate_proj

up_proj

down_proj

Targeting fewer modules reduces memory slightly, but may limit performance. Targeting all major linear layers usually produces results closer to full finetuning.

To understand why, we need to look inside a Transformer block!

The Attention Mechanism and the Transformer Model


Inside every self-attention layer, each token vector is projected into three different spaces (remember the queries (Q), keys (K) and values (V)?)

These projections serve distinct purposes:

Query (W_q) determines what each token is "looking for".

Key (W_k) determines how each token represents what it contains.

Value (W_v) determines what information is passed forward once attention is computed.

After attention weights are applied, the outputs from multiple heads are combined through another projection:

That final matrix, W_o​, controls how information from different heads is mixed back into the residual stream.

In addition to attention, each Transformer layer contains an MLP block with large projection matrices (gate_proj, up_proj, down_proj) that transform representations nonlinearly.

Each of these matrices can be enormous.

In a 7B model with hidden size 4096, a single projection matrix may contain over 16 million parameters.

Multiply that across dozens of layers, and it becomes clear why full finetuning is computationally expensive!

So, when we define as target module q_proj, for example, what we are actually doing is:

And the same applied for the rest of them!

🧪 In the lab, we'll experiment with different hyperparameter values and observe how they affect performance in practice.

Next Steps



That's everything for today's article!

We've covered the foundations of LoRA, its mathematical intuition, how it integrates into Transformer architectures, and the key hyperparameters that control its behavior.

You should now have both the conceptual understanding and the structural intuition behind low-rank adaptation!

This Friday, we'll move from theory to practice in a hands-on LoRA lab.

See you there! 👋

42 Likes
∙
7 Restacks
Discussion about this post
Write a comment...
manishlearnsai 
25 feb

There's a small typo in the article.

Just below this: https://theneuralmaze.substack.com/i/186056048/the-lora-hypothesis

'the-lora-hypothesis'

W' = BA should be ΔW = BA

Amazing article by the way. Liked the in depth explaination.

Like (3)
Reply
Share
1 reply by Miguel Otero Pedrido
Klement Gunndu
Klement’s Substack
28 feb

Building on your observation about "Welcome to Lesson 3 of the Finetuning Sessions!

Low-Rank Adaptation (LoRA) has quietly transformed from a clever resear" -- one pattern that complements this well is separating the orchestration layer from the execution layer. It makes the failure modes much more predictable.

Like (2)
Reply
Share
1 more comment...

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



Engineering LoRA for Real-World Finetuning
Finetuning Sessions · Lab 3 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Feb 27, 2026
∙ Paid




Welcome to Lab 3 of the Finetuning Sessions!

In today's lab, we’re going to run more finetuning experiment — specifically, LoRA finetuning.

📕 If you haven't read Lesson 3's article, make sure to review it before going forward!

Understanding LoRA from First Principles
Understanding LoRA from First Principles
Miguel Otero Pedrido and Antonio Zarauz Moreno
·
25 feb
Read full story
The shift from Full Finetuning (FFT) to Parameter-Efficient Finetuning (PEFT) has redefined how we approach model adaptation. At the center of this paradigm shift is Low-Rank Adaptation (LoRA). While often pitched simply as a "cheaper way to fine-tune", treating LoRA as a black box leaves performance on the table.

So, today, we are going deep.

We will dissect the exact mechanics of how LoRA slashes the "fixed costs" of memory during training, how to navigate its highly sensitive hyperparameter space, and how the concept of Multi-LoRA is revolutionizing both training and deployment.

The "Fixed Costs" of GPU Memory
Ventilador de caja en blanco y negro


Image by Thomas Foster (source: Unsplash)
Residual states consist of activations and temporary buffers, which scale dynamically with your sequence length and batch size.

But the true bottleneck for massive models lies in the model states—the "fixed costs" of training. Model states include the parameter weights themselves, the gradients, and the higher-order optimization quantities (like momentum and variance in the Adam optimizer).

If we perform Full Finetuning (FFT) on a 7-billion parameter model using the Adam optimizer in single precision (FP32), the math is punishing. You need 16 bytes per parameter:

4 bytes for the master weight.

4 bytes for the gradient.

8 bytes for the optimizer states (4 for momentum, 4 for variance).

For a 7B model, these fixed model states alone consume 112 GB of VRAM. This completely prices out consumer hardware!

LoRA fundamentally alters this equation by freezing the original weights and only calculating gradients and optimizer states for a tiny fraction of injected low-rank matrices.

If we assume LoRA adds just 1% of trainable parameters (e.g., ~70M parameters) and we store the frozen base model in bfloat16 (2 bytes per parameter), the fixed costs plummet:

Base weights (frozen): 2 bytes * 7B = 14 GB

LoRA trainable parameters (Adam, FP32): 16 bytes * 0.07B = 1.12 GB

The total memory required for model states drops from 112 GB to roughly 15.12 GB.

By neutralizing the gradient and optimizer footprint for 99% of the network, LoRA turns finetuning from a datacenter-scale problem into a workstation-scale task.

Navigating the Hyperparameter Maze
Asset 019c9e34-0ca9-70a7-883a-70f4f22286b6


The architectural simplicity of LoRA—where the weight update is parameterized as ΔW=BA—hides a complex optimization landscape.

Achieving parity with full finetuning requires strict adherence to hyperparameter best practices.

➤ Target Modules
Early LoRA implementations exclusively targeted the query (W_q) and value (W_v) projection matrices in the self-attention mechanism.

However, exhaustive empirical studies have shown that "attention-only" LoRA heavily underperforms. The best practice is now to apply LoRA to all major linear layers, specifically including the Multi-Layer Perceptron (MLP) layers (gate, up, and down projections).

The MLP layers house the vast majority of a model’s parameters, and targeting them is absolutely essential for driving domain specialization and complex reasoning.

➤ Rank (r) and the Alpha (α) Scaling Factor
The rank (r) determines the dimensionality of your bottleneck.

While tiny ranks (4 or 8) are sufficient for simple natural language tasks, complex domains like coding or mathematics require higher capacity, often necessitating ranks of 64, 128, or 256.

However, simply increasing the rank without adjusting the scaling factor (α) will cause the model to collapse into suboptimal, low-rank solutions.

The LoRA update is scaled by the term α / r. A mathematically sound and highly recommended heuristic is to maintain α = 2r (or at least α = r ). Keeping α fixed (e.g., α=8) while scaling up rank drastically reduces the effective rank of the updates, causing the model to underutilize its capacity and suffer from catastrophic forgetting.

➤ Learning Rate: The 10x Rule
Perhaps the most counterintuitive aspect of LoRA is its learning rate dynamics.

LoRA requires a learning rate that is generally an order of magnitude (10x) higher than what you would use for full finetuning.

For example, if the optimal learning rate for FFT is 2e-5, the optimal LoRA learning rate will likely sit around 2e-4. Because LoRA initializes matrix B to zero, updates at the beginning of training have an incredibly small impact on the network's outputs, necessitating this much more aggressive learning rate schedule to achieve convergence.

The Multi-LoRA Paradigm
Because LoRA drastically compresses the delta of a finetuned model into matrices that are mere megabytes in size, it unlocks an entirely new paradigm for both inference and training: Multi-LoRA.

➤ Multi-Tenant Serving
At deployment, you do not need to host independent 175-billion parameter models for different tasks.

You can load a single, frozen base model into VRAM and dynamically swap out or batch multiple tiny LoRA adapters!

Modern serving systems (like Punica or S-LoRA) can process different user requests, routing them through their respective LoRA adapters in a single batched forward pass, maximizing GPU arithmetic intensity.

➤ Multi-Job Training and Load Balancing
The Multi-LoRA concept is now pushing the boundaries of training efficiency.

Systems like mLoRA and LoRAFusion allow developers to concurrently finetune multiple independent LoRA adapters on the exact same frozen base model using the same set of GPUs.

Why is this revolutionary?

In real-world datasets, token sequence lengths vary wildly. In traditional distributed training, this causes severe load imbalances across GPUs, leading to idle time and massive pipeline bubbles.

By pooling samples across entirely different LoRA finetuning jobs, the training scheduler can solve a bin-packing problem, constructing perfectly balanced micro-batches.

This eliminates pipeline stalls, maximizes communication overlap, and dramatically increases the overall system throughput (tokens processed per second) compared to training jobs sequentially.

Demo time!
Time for the video lab! 🙌

Time to get our hands dirty! See how LoRA enables uncharted possibilities in finetuning landscape.

Download the training script from the shared drive:

Get the code here!

And watch the video!


Next Steps



🎙️ This Sunday at 4:00 PM CET, we’re hosting the Lesson 3 Office Hours for the Finetuning Sessions.

We'll recap the key ideas from Lesson 3, expand on the concepts we introduced, and answer your questions live!

See you on Sunday! 👋

15 Likes
∙
3 Restacks
Discussion about this post
Write a comment...
Rafael Gildin 
Rafael's Substack
2 mar

Thanks for the lesson. I was able to run the hf job, but for running it locally each part of the code how do you suggest pls? Probably running it in colab, but I've got an error when running there with t4. So the only way might be to use hf apps, with the same gpu you're using. What do you think about that?

Like
Reply
Share
1 reply
1 more comment...

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
