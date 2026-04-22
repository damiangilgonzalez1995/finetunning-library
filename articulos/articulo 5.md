
The Neural Maze
The Neural Maze



The RLHF Landscape - Aligning LLMs Beyond SFT
Finetuning Sessions · Lesson 5 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Mar 11, 2026




If you've ever fine-tuned an LLM, you've probably had this experience:

The model generates fluent, well-structured text … that is confidently, completely wrong. Or it gives a technically accurate answer in a tone so robotic that no user would trust it.

Sound familiar? 😄

Supervised Fine-tuning (SFT) gets you far. But at some point, you hit a ceiling. The model knows how to talk, but not what kind of talking humans actually value.

This is the alignment problem, and it's where most of us first run into reinforcement learning. Not because we went looking for RL. But because SFT alone wasn't closing the gap between "technically correct output" and "output a human would actually prefer".

Reinforcement Learning from Human Feedback (RLHF) is the family of techniques that addresses this. Instead of showing the model what to say through labeled examples, RLHF lets it learn which outputs humans prefer over others, a subtler but more powerful training signal.

It's the third and final stage of the modern LLM training pipeline. And understanding it has become essential for anyone building with these models.

In this article, we'll walk through the RLHF landscape from the ground up:

What RL is, how it maps onto language model training, and the key algorithms — PPO, DPO, and others — that you'll encounter in practice.

We'll keep the math light and focused on intuition. A follow-up article will cover the GRPO approach separately.

Ready? Let's go! 👇

Go Premium if you want the full experience of this course, with exclusive access to all the upcoming articles, labs, and office hours!

A Gentle Introduction to RL
black flat screen tv on brown wooden table


If you've spent most of your time in the supervised learning world (as most of us have), RL can feel alien at first.

Different training loop. Different terminology. And a level of instability that can be… unsettling. But here's the thing:

The core ideas are actually quite intuitive once you strip away the jargon!

The Core Idea
Let's train and play with Huggy 🐶 - Hugging Face Deep RL Course


Imagine you're training a dog.

You say "Sit!" — if the dog sits, it gets a treat. If it doesn't, you try again. Over many repetitions, the dog learns to associate the command with the action that earns it a reward.

Reinforcement learning works the same way … but with a computational agent instead of a dog!

In RL, there are five fundamental components. Let's go through each one:

Agent — the learner making decisions. For us, this will be the language model itself.

Environment — the world the agent acts in. It provides observations and feedback. In classic RL this is a game board or a simulated world. For LLMs, it's more abstract … but don't worry, we'll get there.

Action — what the agent can do at each step. Moving a chess piece, choosing a word, deciding how to respond.

Reward — a scalar signal telling the agent how well it did. Positive rewards reinforce good behavior. Negative rewards discourage bad behavior. This is the training signal that makes RL fundamentally different from supervised learning.

Policy — the agent's strategy for choosing actions given its current state. This is the function we want to optimize. Initially, it might be random. Through training, it learns to pick actions that maximize cumulative reward.

The RL Loop
A brief introduction to reinforcement learning


Training happens through a cycle of trial and error:

The agent observes its current state

It selects an action based on its current policy

The environment returns a reward and transitions to a new state

The agent updates its policy to favor actions that led to higher rewards

Repeat — many, many times

If this feels different from the train-on-a-dataset loop you're used to, that's because it is.

There's no fixed dataset here. The agent generates its own experience through exploration, and the quality of that experience depends on the current policy.

✅ This creates a feedback loop that can be powerful … but also tricky to stabilize.

Why Not Just Use SFT?



This is the question every ML engineer asks when they first encounter RLHF. And it's a good one!

The answer comes down to differentiability.

In supervised learning, we need a loss function whose gradient we can compute and backpropagate through the entire model.

But what if the feedback comes from a human who simply says "Response A is better than Response B"?

There's no smooth mathematical function connecting that judgment to the model's parameters. The evaluation process is a black box from the model's perspective.

🙋 RL bridges this gap.

It uses policy gradient methods that don't require differentiating through the reward-generating process. Instead, they use the reward signal to weight the gradients of the policy itself:

Actions that led to high rewards → reinforced.

Actions that led to low rewards → suppressed.

This lets us train from arbitrary feedback signals — including subjective human preferences.

RL in the Context of LLMs
Now that we have the RL basics down, let's connect the dots to what we actually care about: language models.

The Three-Stage Pipeline



If you've been following along in this course, you already know about pretraining and SFT. RLHF is the third piece of the puzzle.

Let's quickly recap the full pipeline:

Stage 1 - Pretraining: The model trains on massive text corpora with a self-supervised objective: predict the next token. This gives it language fluency, world knowledge, and rudimentary reasoning. But the result is essentially a very sophisticated autocomplete engine. No concept of being helpful. No sense of what to avoid.

Stage 2 - Supervised Fine-Tuning (SFT): The pre-trained model is fine-tuned on curated demonstrations of desirable assistant behavior. This teaches it to follow instructions, produce structured outputs, and mimic a helpful style. Effective — but limited to teaching by example.

Stage 3 - RLHF: This is where we move from “show the model what to say” to "teach the model what humans prefer". Instead of providing gold-standard outputs, we provide comparative judgments. The model learns to optimize for them.

✅ This stage is what separates a competent text generator from an aligned assistant.

Mapping RL Concepts to Text Generation
Here's where it gets interesting. The RL framework maps onto language modeling more naturally than you'd expect:

Policy → The language model (its parameters define token probabilities)

State → Current text sequence (prompt + tokens generated so far)

Action → Choosing the next token

Reward → Quality score for the complete generated output

Episode → One complete generation (prompt → final token)

🙋 See how the pieces line up? The model is the policy. Token generation is action selection. And the reward comes after the full response is produced.

The RLHF Process: Three Steps



Now let's look at how RLHF actually works in practice. There are three steps:

➤ Step 1 - Collect human preferences.
For a set of prompts, the model generates multiple candidate responses. Human annotators compare them pairwise: "Is Response A or Response B better?"

This produces a preference dataset.

➤ Step 2 - Train a reward model.
Using the preference data, we train a separate model to predict human preferences. This reward model learns to assign higher scores to outputs humans would rate as better.

It becomes a scalable proxy for human judgment — you can evaluate thousands of outputs without needing a human for each one.

➤ Step 3 - Optimize the policy with RL.
The language model generates responses. The reward model scores them. An RL algorithm updates the model to produce higher-scoring outputs.

⚠️ A critical detail here: there’s always a KL divergence penalty that prevents the model from drifting too far from the SFT baseline. Without it, the model quickly learns to exploit quirks in the reward model rather than genuinely improving. More on this later.

The Key Algorithms
Alright, this is the section that matters most in practice.

We'll go deep on PPO and DPO — the two algorithms you're most likely to actually use — and then survey several other approaches worth knowing about.

Let's start with the heavyweight.

PPO (Proximal Policy Optimization)
PPO is the default on-policy RLHF algorithm. It was used to train InstructGPT, and it's been a core component in many other frontier models.

If RLHF is the third stage of the pipeline, PPO has been the engine powering it for most of its history.

From REINFORCE to PPO
To understand PPO, it helps to start one step earlier with REINFORCE, the simplest policy gradient algorithm.

The idea: run the current policy, collect rewards, and compute the gradient of expected reward with respect to the model's parameters. Here's the policy gradient formula:


🙋 In plain English → For each action the model took, compute how changing the parameters would change the probability of that action, then weight that change by the total reward.

High reward → push toward those actions.

Low reward → push away.

Elegant. But there's a serious practical problem: high variance.

The total episode reward R(τ) is a noisy signal. It reflects everything that happened in the episode, not just whether a specific action was good. This makes training unstable, especially for long sequences.

PPO addresses this through several key refinements. Let's walk through them.

The Advantage Function
Instead of weighting by total return, PPO uses the advantage — how much better a specific action was compared to the average outcome from that state:


Where Q(s_t, a_t) is the expected return of taking action a_t in state s_t, and V(s_t) is the expected return from that state under the current policy, regardless of which action is taken.

✅ This dramatically reduces variance because we're measuring relative quality rather than absolute returns.

In practice, the advantage is estimated using Generalized Advantage Estimation (GAE), which blends short-term and long-term reward estimates to balance bias and variance.

The Clipped Surrogate Objective
This is PPO's signature contribution. Instead of allowing unbounded policy updates, PPO clips the probability ratio between the new and old policy:


Where:


The intuition: if a policy update would change an action's probability too much (beyond the ε clipping range), the gradient gets zeroed out.

This creates a "trust region" — the policy improves, but only within safe bounds. No single batch of experience can cause a catastrophic update.

The Four-Model Setup
Here's where PPO gets expensive. A complete PPO training run for RLHF requires four models in memory simultaneously:

Actor — the language model being optimized. This is the policy.

Critic — a value network that estimates expected future reward for each state. Needed to compute the advantage function.

Reward Model — trained on human preferences in step 2. Frozen during RL training. Scores the actor’s outputs.

Reference Model — a frozen copy of the SFT model. Used to compute the KL divergence penalty that keeps the actor from drifting too far.

💸 For a 70B parameter model, you're loading four large models simultaneously, plus generating text autoregressively at every training step. It works — and it works well — but the resource requirements are significant.

When to Reach for PPO
PPO shines when you need maximum alignment quality and have the compute to support it.

Its on-policy nature means training data always reflects the model's current behavior, creating a tight, self-correcting feedback loop.

If you're pushing a frontier model and alignment quality is paramount, PPO is the proven choice!

🎬 If you want a visual guide to PPO that really makes it click, I highly recommend going through this amazing video by the one and only Luis Serrano. One of the best explanations of PPO out there!


DPO (Direct Preference Optimization)
DPO arrived in 2023 and changed the practical conversation around RLHF.

🙋 Its central provocation: what if we could get RLHF-quality alignment without RL at all?

The Key Insight
Starting from the same RLHF objective that PPO optimizes — maximize expected reward while staying close to a reference policy — the DPO authors showed something surprising:

You can solve for the optimal policy in closed form.

This means the reward function can be reparameterized in terms of the policy itself. No separate reward model needed. No RL training loop at all.

The resulting loss function is surprisingly simple — a binary cross-entropy loss over preference pairs:


Where:

x is a prompt

y_w is the preferred (winning) response

y_l is the rejected (losing) response

π_ref is the frozen reference model

β controls how far the policy can deviate from the reference

🙋 In plain English → for each preference pair, increase the model's probability of generating the preferred response (relative to the reference) and decrease its probability of generating the rejected one.

That's it!

No reward model to train. No value network. No text generation during training. No RL at all.

Just a dataset of preference pairs and a loss function you optimize with standard gradient descent — the same way you'd fine-tune any other model.

Why DPO Took Off
DPO's appeal for practitioners is hard to overstate:

✅ Only two models in memory (policy + frozen reference)

✅ No text generation during training — runs at SFT-like speeds

✅ Minimal hyperparameter tuning (mostly just β)

If you're working with limited compute or need fast iteration, DPO is often the right starting point.

Limitations
But DPO isn't a free lunch.

⚠️ It's an off-policy method: it learns from a static preference dataset collected from a previous version of the model (or a different model entirely).

If the training model's distribution shifts significantly from the data-generating distribution, learning quality degrades. DPO also tends to overfit on preference data faster than PPO.

And several studies have found that PPO still outperforms DPO on harder alignment benchmarks when compute isn’t a constraint.

The fundamental trade-off is clear: DPO trades some alignment ceiling for dramatically lower cost and complexity.

🎬 And just like with PPO, if you want a visual walkthrough that makes DPO truly click, Luis Serrano has you covered again. Seriously, his ability to break down complex ideas is unmatched!


Other Approaches Worth Knowing
The RLHF space is active, and new methods appear regularly. Here’s a quick tour of the ones you’ll encounter:

REINFORCE / Vanilla Policy Gradients. The original policy gradient algorithm and the conceptual ancestor of PPO. Simple to understand, rarely used directly in production RLHF due to high variance. Worth knowing for building intuition.

Identity Preference Optimization (IPO). A DPO variant with a modified loss function that’s more robust to overfitting. Useful if you’re finding DPO overfits your preference data too quickly.

Kahneman-Tversky Optimization (KTO). An interesting departure. KTO only requires knowing whether each response is "good" or "bad" — no paired A-vs-B comparisons needed. This makes data collection significantly easier.

Rejection Sampling / Best-of-N. Generate N candidates, score them with a reward model, fine-tune on the best ones. Conceptually simple, surprisingly effective, and a practical complement to PPO.

RLAIF (RL from AI Feedback). Replace human annotators with an AI judge (often a larger LLM). Dramatically reduces cost and turnaround time, at the risk of inheriting the judge model’s biases. Increasingly common in practice.

PPO vs. DPO: A Practical Decision Framework



In practice, the choice often comes down to your constraints:

➤ Reach for PPO when:
You have significant GPU budget

Alignment quality is the top priority

Your task requires the model to explore beyond what static preference data captures

You're training a frontier model

➤ Reach for DPO when:
You need faster iteration cycles

You have a solid preference dataset available

Compute is limited

You're getting started with alignment

🙋 Many teams use a staged approach: DPO first for rapid alignment (fast and cheap), then PPO for further refinement if the quality bar demands it.

Practical Considerations
Before we wrap up, a few things that cut across all algorithms and are worth keeping in mind:

➤ Get the Data Right First
This is easy to underestimate.

Noisy, inconsistent, or poorly calibrated preference data will produce poorly aligned models — regardless of whether you use PPO, DPO, or anything else.

Careful annotation guidelines, inter-annotator agreement checks, and thoughtful prompt selection are at least as impactful as the choice of optimization algorithm.

✅ Get the data right first. Then worry about the algorithm.

➤ The KL Penalty Is More Important Than It Looks
Every RLHF method includes some mechanism to keep the aligned model close to its pre-RL baseline.

Without this constraint, models quickly learn to exploit quirks in the reward signal — producing degenerate outputs that score highly but are useless to actual humans.

✅ Tuning the strength of this constraint (the β in DPO, the KL coefficient in PPO) is one of the most consequential decisions in the entire pipeline.

➤ Reward Model Quality Is a Bottleneck
For methods that use an explicit reward model (PPO, rejection sampling), the quality of that model caps how well alignment can go.

Active research areas include reward model ensembles, process-based reward models (scoring intermediate reasoning steps, not just final answers), and iterative reward model refinement.

Next Steps
RLHF has moved from a niche research technique to a foundational part of how production LLMs are built.

For practitioners, the key insight is this: alignment is not a single algorithm. It's a pipeline — data collection, reward modeling, and policy optimization — where each step's quality compounds.

PPO remains the proven choice for maximum alignment quality, with its on-policy exploration creating a tight feedback loop between the model’s current behavior and the training signal.

DPO has made strong alignment accessible to a much wider set of practitioners by collapsing the RL complexity into a supervised-learning-style workflow.

And newer methods like IPO, KTO, and RLAIF continue to expand the toolkit.

🙋 In a follow-up article, we'll dive into GRPO (Group Relative Policy Optimization) — a newer approach that rethinks how we compare and score model outputs during training.

Also, remember that this Friday we'll move from theory to practice in our hands-on lab.




See you there builder! 👋

References
Hugging Face LLM Course — Chapter 12: Introduction to Reinforcement Learning and its Role in LLMs. A beginner-friendly overview of RL concepts and their mapping to language model training.

Cameron R. Wolfe — "Basics of Reinforcement Learning for LLMs." Deep (Learning) Focus newsletter. A thorough walkthrough of RL fundamentals, MDPs, Q-learning, and the formal framework underlying RLHF.

Yihua Zhang — "Navigating the RLHF Landscape: From Policy Gradients to PPO, GAE, and DPO for LLM Alignment." Hugging Face Community Blog. A detailed mathematical treatment covering policy gradients, REINFORCE, PPO with clipping, GAE derivation, and DPO.

Rafailov et al. — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023). The original DPO paper.

Schulman et al. — “Proximal Policy Optimization Algorithms” (2017). The original PPO paper.

Ouyang et al. — "Training language models to follow instructions with human feedback" (2022). The InstructGPT paper that popularized RLHF for LLMs.

17 Likes
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

179

4

30


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
