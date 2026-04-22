
The Neural Maze
The Neural Maze



Beyond Text: A Guide to Vision & TTS Finetuning
Finetuning Sessions · Lesson 7 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Mar 25, 2026




Let's kick off Lesson 7 with a simple question:

Why should we care about anything other than text? 🤔

Throughout this series, we've been finetuning models that do one thing: read text and write text.

And that makes sense, text-to-text is the bread and butter of LLM finetuning, and it's where the most mature tooling lives … But the world isn't text-only, dear builder.

I mean, think about it. A radiologist doesn't diagnose from a paragraph, they diagnose from an X-ray. Or podcasters, they don't type their episodes, they speak them! And increasingly, the models we're finetuning can handle all of these modalities.

What's remarkable is that the finetuning techniques you've already learned (LoRA, QLoRA, …) apply almost identically to these multimodal models. The architectures differ in how they encode or decode non-text signals, but the training loop is the same.

You attach adapters to the transformer backbone, prepare a dataset, and run SFTTrainer.

In this lesson, we'll focus on two multimodal directions that are both practically useful and surprisingly accessible:

Vision finetuning — teaching a model to understand images and generate text about them

Text-to-Speech (TTS) finetuning — teaching a model to generate human-like speech from text

We'll build the intuition for how each one works, explore real-world use cases, and then preview the two specific models we'll be finetuning hands-on in Friday's lab.

Ready? Let's go! 👇

Go Premium if you want the full experience of this course, with exclusive access to all the upcoming articles, labs, and office hours!

When Should You Go Multimodal?
Before diving into architectures, let's ground ourselves in the why.

Here are concrete scenarios where multimodal finetuning unlocks capabilities that text-only models simply can't provide.

And who knows … maybe one of these clicks with a problem you've been trying to solve, or inspires a product idea you hadn't considered.

💬 Any other use case worth mentioning? Let us know in the comments!

Vision Finetuning Use Cases
Let's look at four use cases where vision finetuning really shines.

🏥 Medical image analysis



From: Collaboration between clinicians and vision–language models in radiology report generation. Ryutaro Tanno et Al. Nature Medicine (2024)
A finetuned vision model can look at a chest X-ray and generate a structured radiology report.

This doesn't replace radiologists, but it can pre-fill reports, flag urgent findings, and reduce turnaround time. General-purpose VLMs often get medical terminology wrong or miss subtle findings … finetuning on domain-specific image-report pairs fixes this.

📄 Document understanding & OCR



Document understanding systems (bonus points if you recognize the guy in the image 😏)
Invoices, handwritten forms, receipts, architectural blueprints … these are all images that contain structured information.

A finetuned VLM can extract fields from a scanned invoice (vendor, amount, date, line items) and output structured JSON, or convert handwritten mathematical notation into LaTeX.

This is far more robust than traditional OCR pipelines because the model understands context, not just characters.

🏭 Industrial quality inspection
A camera on a production line captures images of manufactured parts. A finetuned model can flag defects (scratches, misalignments, color inconsistencies) and classify their severity.

General VLMs don't know what a "hairline crack on a ceramic tile" looks like; finetuning on a few hundred labeled examples teaches them.

🚜 Agricultural crop monitoring



Image from Few-Shot Image Classification of Crop Diseases Based on Vision–Language Models
Diseased crops cost billions in losses every year, and early detection is everything.

A vision model finetuned on labeled images of crop diseases can distinguish between dozens of conditions (fungal infections, nutrient deficiencies, pest damage) from a single photo.

This works at the individual farmer level (snap a photo in the field) or at scale (drone footage over entire plantations).

TTS Finetuning Use Cases
Same exercise, different modality. Here are three use cases where TTS finetuning really shines.

🎙️ Voice cloning for content creators

MrBeast: YouTuber topples T-Series for most subscribers - BBC News


TTS Voice Cloning … for Mr Beast? 🤣🤣🤣
A podcaster or YouTuber can finetune a TTS model on their own voice recordings.

The result is a synthetic voice that captures not just their timbre, but their pacing, emphasis patterns, and verbal quirks.

This enables generating full episodes from a script, creating audio versions of blog posts, or producing content in multiple languages (all in their own voice).

🏢 Brand-consistent customer service

Customer service management: background, advantages, functions


Source: https://otrs.com/blog/customer-service/customer-service-management/
Companies want their voice assistants, IVR systems, and phone bots to sound consistent and on-brand.

Finetuning a TTS model on a specific voice actor's recordings (with consent!!!) creates a scalable, consistent voice that can handle any text input without re-recording.

📚 Audiobook narration at scale

Publishers can finetune TTS models to produce expressive, narrator-quality audiobooks at a fraction of the cost and time of traditional recording.

Finetuning is key here, as base TTS models often sound flat over long passages. A model finetuned on a narrator's expressive readings learns their dramatic range.

Part I: Vision Finetuning
All modern vision-language models (VLMs) share the same high-level structure.

Regardless of whether you’re working with Qwen3-VL, Llama 3.2 Vision, Gemma 3, or any other VLM, the architecture follows three stages:




➤ Stage 1 (The Vision Encoder)
Turns an image into a sequence of feature vectors.

This is almost always a Vision Transformer (ViT), the same architecture used in CLIP, SigLIP, and other vision foundation models.

The image is split into small patches (typically 14×14 or 16×16 pixels), each patch is linearly projected into an embedding, and the full sequence is processed by transformer layers with self-attention.

The key output: a sequence of visual tokens (one per image patch) that encode spatial and semantic information about the image.

➤ Stage 2 (The Projection Layer)
Bridges the gap between vision and language.

The visual tokens from the encoder live in a different embedding space than the text tokens the LLM expects.

The projection layer (often a simple MLP or cross-attention module) transforms and compresses these visual tokens into the LLM's embedding dimension.

Some models also reduce the number of tokens here, for example, grouping 2×2 adjacent visual tokens into one, cutting the sequence length by 4×.

This matters because visual tokens are expensive: a 500×500 image can produce hundreds of tokens, and every token adds to the LLM's context length.

➤ Stage 3 (The LLM Decoder)
It's a standard autoregressive transformer, the same kind of model you've been finetuning throughout this course.

It receives a concatenated sequence: the projected visual tokens followed by the text tokens from the user's prompt. It then generates a text response, token by token.

Where LoRA Fits In
Here's the crucial insight for finetuning: you almost never need to touch the vision encoder.

The vision encoder was pretrained on millions (sometimes billions) of image-text pairs.

It already knows how to extract rich visual features from images. What it doesn't know is how to map those features to your specific domain's language — the terminology of radiology reports, the format of LaTeX equations, the structure of invoice JSON.

That's the LLM decoder's job. So when you finetune a vision model:

The vision encoder stays frozen (no gradients, no updates)

The projection layer may or may not be trainable (model-dependent)

The LLM decoder gets LoRA adapters on its attention layers — exactly the same as text finetuning

This means your training cost is essentially the same as finetuning a text-only LLM of the same size. The vision encoder adds inference cost (encoding the image), but not training cost.

The Math (for those who want it)
The forward pass through a VLM during finetuning looks like this:

Visual encoding (frozen):

Where N_v is the number of visual tokens (depends on image resolution) and d_v is the vision encoder's hidden dimension.

Projection (may be trainable):

Where N'_v <= N_v (compression may reduce token count) and d is the LLM's hidden dimension.

Concatenation and LLM forward pass (with LoRA):

Where h_t are the text token embeddings, and Δθ are the LoRA parameters. The loss is computed only on the text output tokens (not the visual tokens), exactly like standard SFT:

Key Practical Considerations
Image resolution matters. Higher resolution = more visual tokens = longer sequences = more VRAM. Most models support dynamic resolution (you don’t need to resize to a fixed 224×224), but you should aim for 300-1000px during training to balance quality and efficiency.

Keep dimensions consistent. If your training images have wildly different aspect ratios and sizes, the number of visual tokens per sample will vary a lot, making batching inefficient. Standardize where possible.

Mix general and domain data. If you finetune only on, say, radiology images, the model may "forget" how to handle general visual questions. A common strategy is to mix your domain-specific dataset with a general VQA dataset (e.g., 80% domain, 20% general).

Part II: TTS Finetuning



Traditional TTS was a complex, multi-stage pipeline: text normalization → phoneme conversion → prosody modeling → waveform synthesis. Each stage had its own model, its own failure modes, and its own engineering headaches.

Modern LLM-based TTS throws all of that away. The core idea is stunningly simple:

Treat audio as just another language.

Instead of predicting the next text token, the LLM predicts the next audio token.

The architecture is the same autoregressive transformer you already know. The only new ingredient is a neural audio codec — a "tokenizer for sound" that converts continuous audio waveforms into discrete tokens (and back).

The Neural Audio Codec: Tokenizing Sound
The codec is the key innovation that makes LLM-based TTS possible. Here's how it works:

Encoding (audio → tokens): The codec's encoder takes a raw audio waveform and compresses it into a sequence of discrete integer codes — just like a text tokenizer converts characters into token IDs. These codes are chosen from a learned codebook (think of it like a vocabulary, but for audio chunks).

Decoding (tokens → audio): The codec's decoder takes the discrete codes and reconstructs the audio waveform. The reconstruction isn't perfect (it's lossy compression, like MP3), but modern codecs achieve remarkably high quality at low bitrates.

Hierarchical structure: Most modern codecs (like SNAC, EnCodec, or DAC) use multiple quantization layers at different temporal resolutions. Think of it as encoding audio at three different "zoom levels" simultaneously:

Layer 1 (coarse, ~12 Hz) captures the big picture — rhythm, prosody, who’s speaking. It produces 1 token per time frame.

Layer 2 (mid, ~24 Hz) captures phonetic patterns and intonation contours. It runs twice as fast, so for every 1 token from Layer 1, Layer 2 produces 2 tokens.

Layer 3 (fine, ~48 Hz) captures acoustic texture — breathiness, crispness, subtle vocal qualities. It runs four times as fast as Layer 1, producing 4 tokens per frame.




How the LLM Generates Speech
With the codec in place, the training and inference pipeline is straightforward:

Training:

Take a dataset of (text, audio) pairs

Encode all audio through the codec to get audio token sequences

Extend the LLM's vocabulary to include the audio token IDs (so the model can predict them)

Train with standard next-token prediction: given the text tokens, predict the audio tokens

Inference:

Feed text tokens into the LLM

The LLM autoregressively generates audio token IDs

Pass the generated token IDs through the codec decoder

Out comes a waveform (that's the speech!)

The LLM doesn't "know" it's generating speech.

From its perspective, it's just predicting the next token in a sequence. The magic is that the audio codec provides a discrete token space that's structured enough for the LLM to learn meaningful patterns.

The Math (for those who want it)
Audio encoding (frozen codec):

Where c^l in {0, 1, ..., K-1}^{T_l} are the discrete codes at layer l, K is the codebook size, and T_l is the number of tokens at that layer's temporal resolution.

Interleaving into a flat sequence:

For a codec with 3 layers (like SNAC), codes are interleaved so that every 1 + 2 + 4 = 7 tokens represent one temporal frame:

Vocabulary extension:

The codes are offset to avoid collision with text tokens:

Where V_text is the original text vocabulary size.

Training objective (standard next-token prediction):

Where t are the text input tokens and s is the interleaved audio token sequence. Again, Δθ are the LoRA adapters.

Why Finetuning Beats Zero-Shot Cloning
Base TTS models can do "zero-shot voice cloning", where you provide a short audio sample of a voice, and the model generates new speech in that voice. But the results are often... okay. The model captures the basic timbre but misses pacing, emotional expression, and speaking quirks.

Finetuning on a specific voice's recordings teaches the model all of these subtleties.

Think of the difference like this: zero-shot cloning is like hearing someone speak for 10 seconds and then imitating them. Finetuning is like spending weeks studying their recordings until you can reproduce their delivery perfectly.

In practice, 30 minutes of clean, single-speaker audio is often enough for high-quality voice cloning through finetuning. The key is quality over quantity: consistent recording conditions, minimal background noise, and accurate transcriptions.

Key Practical Considerations
Token rate determines max duration. If your codec produces 83 tokens per second and your model’s max generation length is 2048 tokens, you can generate ~24 seconds of speech at most. Plan accordingly — you may need to chunk longer texts.

Repetition penalty is critical. Without it (or with too low a value), TTS models tend to get stuck in loops — repeating the same syllable or producing monotonic droning. A repetition_penalty >= 1.1 is typically required.

Smaller models are often better for TTS. Unlike text generation where bigger models = better quality, TTS prioritizes latency. A 3B parameter model that generates speech in real-time is more useful than a 70B model that takes 30 seconds per sentence. Models under 3B are the sweet spot!

Next Steps
In the upcoming lab, we'll go hands-on with two specific models. Here's what you need to know about each one — we won't repeat this theory on Friday, so this is your reference!

Qwen3-VL (8B) — For Vision Finetuning



Source: https://www.siliconflow.com/blog/qwen3-vl-8b-now-on-siliconflow-small-model-big-vision
Qwen3-VL is the latest and most capable vision-language model in Alibaba's Qwen series.

It comes in both dense (2B, 4B, 8B, 32B) and mixture-of-experts (30B-A3B, 235B-A22B) variants — we'll use the 8B dense model, which offers a great balance of capability and trainability (and can be finetuned for free on Colab with Unsloth).

What makes it special:

256K native context window: Qwen3-VL supports up to 256K tokens (extendable to 1M), which means you can feed it hundreds of pages of documents, or even hour-long videos, in a single prompt.

DeepStack integration: Unlike Qwen2.5-VL which only used the final ViT output, Qwen3-VL fuses features from multiple levels of the vision encoder. This captures both fine-grained details and high-level semantics — critical for tasks like small-text OCR or subtle visual differences.

Interleaved-MRoPE: An enhanced version of the multimodal rotary position embeddings from Qwen2.5-VL. It allocates full-frequency positional information across time, width, and height dimensions, which dramatically improves long-horizon video reasoning and spatial understanding.

Text-Timestamp alignment: For video inputs, the model no longer relies on relative position IDs to track time. Instead, it uses explicit textual timestamps, enabling precise event localization (e.g., “at 1:23, the player scores a goal”).

Expanded OCR: Supports 32 languages (up from 19 in Qwen2.5-VL), with improved robustness for low-light, blurry, or tilted text — and better handling of rare characters and domain-specific jargon.

Thinking mode: The Instruct and Thinking editions let you toggle chain-of-thought reasoning on or off, useful for complex visual math problems or multi-step document analysis.

In the lab, we'll finetune this model on a handwriting-to-LaTeX conversion task using QLoRA through Unsloth, so you can send it photos of handwritten equations and get clean LaTeX back.

Orpheus-TTS (3B) — For TTS Finetuning
How to deploy STT and TTS systems to production?


💡 If you want to see Orpheus in action inside a real-world project, check out the course I built with Jesús Copado, where we used it as the voice engine for a fully agentic call center: Building a Production-Ready Agent Call Center.

Orpheus-TTS, built by Canopy Labs, is a state-of-the-art open-source TTS system built on the Llama 3B backbone. It demonstrates that an LLM, when properly trained, can produce speech that rivals closed-source commercial services.

What makes it special:

Llama 3B backbone: This is literally a Llama model that has been trained to predict audio tokens. If you've finetuned Llama for text generation, you already understand the underlying architecture. The finetuning workflow is nearly identical.

SNAC codec (24kHz): Orpheus uses the SNAC (Multi-Scale Neural Audio Codec) operating at 24kHz sample rate. It has 3 quantization layers at 12, 24, and 48 Hz, interleaved into 7 tokens per frame (~83 tokens/second total). Each layer has a codebook of 4,096 entries.

Emotive tags: You can control speech expression with inline tags like <laugh>, <sigh>, <chuckle>, <gasp>, etc. These are part of the text prompt and guide the model’s generation style.

Multiple voices: The model ships with 8 preset voices (tara, leah, jess, leo, dan, mia, zac, zoe) that you select by prefixing the text prompt with the voice name, similar to how you’d use a system prompt.

Streaming capability: Orpheus can deliver ~200ms latency for real-time streaming applications, generating speech chunk by chunk as the audio tokens are produced.

In the lab, we'll finetune Orpheus on a custom voice dataset using QLoRA through Unsloth, so you can hear the difference between the base model's generic voices and your finetuned, personalized voice.

The key takeaway from this lesson is that multimodal finetuning isn't a fundamentally different skill from what you've already learned. The architectural innovations — vision encoders, audio codecs — handle the translation between modalities. Your job as a finetuner remains the same: attach LoRA adapters to the transformer backbone, prepare a good dataset, and run SFT.

What is different is the range of problems you can now solve. Text-only finetuning limits you to text-in, text-out tasks. With vision finetuning, you can build systems that understand the visual world. With TTS finetuning, you can build systems that speak with custom voices, emotions, and styles.

On Friday, we'll put this theory into practice.

See you in the lab! 🔬




Resources
Vision finetuning:

Unsloth Vision Finetuning docs

Qwen3-VL Technical Report

Qwen3-VL GitHub

Qwen3-VL HuggingFace model

Unsloth Qwen3-VL Colab notebook

Unsloth Qwen3-VL finetuning guide

TTS finetuning:

Unsloth TTS Finetuning docs

Orpheus-TTS GitHub

Orpheus-TTS HuggingFace model

SNAC codec paper

SNAC HuggingFace model

"LLM-based Audio Models" explainer

"Neural audio codecs" explainer

20 Likes
∙
4 Restacks
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



The Builder's Guide to Multimodal Finetuning (Vision + TTS)
Finetuning Sessions · Lab 7 / 8
Miguel Otero Pedrido and Antonio Zarauz Moreno
Mar 27, 2026




This is the applied lab for Lesson 7 of our Finetuning Sessions.

Haven't caught Wednesday's foundations article yet? Go give it a read! We broke down the intuition behind vision and TTS finetuning, walked through the main architectures, and explored why multimodal finetuning is such a big deal.

Beyond Text: A Guide to Vision & TTS Finetuning
Beyond Text: A Guide to Vision & TTS Finetuning
Miguel Otero Pedrido and Antonio Zarauz Moreno
·
25 mar
Read full story
By the end of this lab, you'll have finetuned two multimodal models with your own hands: a vision model that converts handwritten math into LaTeX, and a TTS model that generates speech in a custom voice.

Go Premium if you want the full experience of this course, with exclusive access to all the upcoming articles, labs, and office hours!

What You'll Need
Before we start, make sure you have:

A Google account (we'll use free Colab T4 GPUs for both finetuning runs)

A Hugging Face account with a write token (for saving your finetuned models)

Around 1-2 hours of total time (training runs included)

Both notebooks run entirely on free Colab instances (no paid GPUs required for this one).

Handwriting to LaTeX with Qwen3-VL
In this first part, we'll finetune Qwen3-VL (8B) to convert photos of handwritten mathematical formulas into clean LaTeX code.

This is a great example of a vision finetuning task: the base model can describe images in general terms, but it doesn't know how to produce precise LaTeX output from handwritten notation. Finetuning fixes that.

💻 Get the notebook here!


Step 1: Setup & Model Loading
We start by installing Unsloth and loading the Qwen3-VL 8B model in 4-bit quantization. This is the QLoRA setup you already know from previous lessons — the only difference is that we’re using FastVisionModel instead of FastLanguageModel.

from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

Notice anything? It's almost identical to loading a text model. Unsloth handles the vision encoder, the projection layer, and the LLM backbone (all under one call).




Step 2: Adding LoRA Adapters
Here's where it gets interesting. With vision models, Unsloth gives you granular control over what to finetune:

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,  # Finetune the ViT?
    finetune_language_layers   = True,  # Finetune the LLM?
    finetune_attention_modules = True,  # Finetune attention?
    finetune_mlp_modules       = True,  # Finetune MLP layers?

    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

As we discussed in the foundations article, you typically keep the vision encoder frozen. But Unsloth lets you experiment (try setting finetune_vision_layers = False) and see how it affects your results.

For our handwriting task, finetuning both vision and language layers gives the best outcome because the model needs to learn to read a very specific visual style (handwritten math notation).


Step 3: Dataset Preparation
We're using the LaTeX_OCR dataset — a collection of handwritten math formula images paired with their LaTeX representations.

from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")

Let’s peek at what we're working with:




The key step is converting each sample into the conversational format that vision models expect:

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

This is the standard format for all VLM finetuning: a user message with both text and image, and an assistant response with the expected output. If you've done SFT before, this should feel familiar — the only new element is the {"type": "image"} entry.

Step 4: Before Finetuning — Baseline Check
Before we train anything, let's see what the base model produces:

FastVisionModel.for_inference(model)

image = dataset[2]["image"]
instruction = "Write the LaTeX representation for this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image, input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


As you can see, the base model tries, but it doesn't produce clean, correct LaTeX. That's exactly what finetuning will fix.

Step 5: Training
Time to train! We use SFTTrainer with Unsloth's UnslothVisionDataCollator:

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
        max_steps = 30,  # Use num_train_epochs = 1 for a full run
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        # Required for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

trainer_stats = trainer.train()

A few things to note:

max_steps = 30 is just for the demo. For a real finetuning run, set num_train_epochs = 1 and remove max_steps.

The three extra SFTConfig fields at the bottom (remove_unused_columns, dataset_text_field, dataset_kwargs) are required for vision finetuning — don’t skip them.

On a free T4, this takes about 5 minutes for 30 steps.


Step 6: After Finetuning — The Difference
Now let's run inference again on the same image:


The model now produces precise LaTeX that, when rendered, matches the handwritten input. That's the power of vision finetuning with just 30 training steps.

Step 7: Saving Your Model
Save the LoRA adapters locally or push them to Hugging Face:

model.save_pretrained("qwen_lora")
tokenizer.save_pretrained("qwen_lora")

# Or push to Hugging Face:
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN")
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN")

You can also export to GGUF for local deployment with llama.cpp or Ollama:

# Save to q4_k_m GGUF
model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "q4_k_m")

Voice Cloning with Orpheus-TTS
Now let's switch modalities entirely. In this second part, we'll finetune Orpheus-TTS (3B) to generate speech in a specific voice. The base model ships with generic preset voices — finetuning will teach it a new one.

💻 Get the notebook here!


Step 1: Setup & Model Loading
This time we’re back to FastLanguageModel — remember, Orpheus is just a Llama 3B under the hood:

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

Note that we’re loading in full precision here (load_in_4bit = False). TTS models are more sensitive to quantization than text models — the audio quality can degrade noticeably with 4-bit. If you’re tight on VRAM, you can try True, but expect some quality loss.

Step 2: Adding LoRA Adapters
Standard LoRA setup, targeting all the usual attention and MLP projections:

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

Notice the r = 64 — higher than the r = 16 we used for vision. TTS finetuning benefits from a higher rank because the model needs to learn subtle acoustic patterns (voice timbre, pacing, intonation) that require more expressive capacity in the adapters.

Step 3: Dataset & Audio Tokenization
This is where TTS finetuning diverges the most from text finetuning. We're using the MrDragonFox/Elise dataset — a single-speaker voice dataset designed for TTS training.

from datasets import load_dataset
dataset = load_dataset("MrDragonFox/Elise", split = "train")




Now comes the critical step: encoding the audio into SNAC tokens. As we explained in the foundations article, the LLM doesn't work with raw audio — it works with discrete audio tokens produced by the SNAC codec.

from snac import SNAC
import torchaudio.transforms as T

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")

def tokenise_audio(waveform):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))

    return all_codes

Let’s unpack what's happening here, because this is the core of TTS data preparation:

Resample to 24kHz — SNAC expects 24kHz audio, so we resample from whatever the dataset provides.

Encode with SNAC — This produces three layers of codes at different temporal resolutions (12Hz, 24Hz, 48Hz).

Interleave into a flat sequence — The 7-token-per-frame pattern we discussed in the foundations article. Each frame contains 1 coarse + 2 mid + 4 fine tokens.

Offset by 128,266 — This shifts the audio token IDs into a range that doesn't collide with the LLM's text vocabulary. Each layer gets an additional offset of n × 4096 so the model can distinguish which layer each token belongs to.

We then apply this to the entire dataset:

dataset = dataset.map(add_codes, remove_columns=["audio"])
dataset = dataset.filter(lambda x: x["codes_list"] is not None)
dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)

Step 4: Formatting the Training Data
Each training sample is structured as a sequence of special tokens:

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

The structure is: [SOH] text tokens [EOH] [SOA] [SOS] audio tokens [EOS] [EOA]. The model learns to predict the audio tokens given the text — which is exactly the “speech as a language” paradigm we covered in the foundations.

There's also a deduplication step that removes consecutive frames with the same coarse token — this cleans up silent or repetitive sections:

def remove_duplicate_frames(example):
    vals = example["codes_list"]
    result = vals[:7]
    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(vals[i:i+7])
    example["codes_list"] = result
    return example

Step 5: Training
The training loop uses the standard Hugging Face Trainer (not SFTTrainer — since we’ve already formatted the input_ids manually):

from transformers import TrainingArguments, Trainer

trainer = Trainer(
    model = model,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,  # Use num_train_epochs = 1 for a full run
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

A few differences from the vision training:

per_device_train_batch_size = 1 — TTS sequences can be long (hundreds of audio tokens per sample), so we keep the batch size small.

max_steps = 60 — We use more steps than the vision example because audio patterns take longer to learn.

We use the standard Trainer instead of SFTTrainer since the data is already tokenized.


Step 6: Inference — Hearing the Results
This is the most satisfying part. Let's generate speech with our finetuned model:

FastLanguageModel.for_inference(model)
snac_model.to("cpu")  # Free up GPU for generation

prompts = [
    "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person.",
]

The inference pipeline is:

Tokenize the text prompt with special control tokens

Generate audio token IDs with the LLM

Decode those token IDs back into a waveform with SNAC

Play the audio!


Notice how the finetuned model picks up the <giggles> tag and produces a more natural, expressive delivery.

The base model's preset voices are decent, but the finetuned voice captures the specific characteristics of the Elise dataset — the pacing, the warmth, the small vocal quirks.

Step 7: Saving Your Model
Same as before — save the LoRA adapters or merge and push:

model.save_pretrained("orpheus_lora")
tokenizer.save_pretrained("orpheus_lora")

# Or push to Hugging Face:
# model.push_to_hub("your_name/orpheus_lora", token = "YOUR_HF_TOKEN")

For TTS, you'll likely want the merged 16-bit version for deployment, since inference quality matters more than model size:

# Merge to 16bit for best audio quality
model.save_pretrained_merged("orpheus_finetune_16bit", tokenizer, save_method = "merged_16bit")

Wrapping Up
Let's step back and appreciate what we just did:

We finetuned a vision model (Qwen3-VL, 8B parameters) to convert handwritten math into LaTeX — using the exact same LoRA + SFT workflow we've used all course

We finetuned a TTS model (Orpheus, 3B parameters) to generate speech in a custom voice — again, same LoRA workflow, just with audio tokens instead of text tokens

Both ran on free Colab GPUs

The total code difference between finetuning a text model, a vision model, and a TTS model? Maybe 20 lines

That's the message of this lesson: multimodal finetuning is not a new skill. It's the same skill, applied to new modalities. The encoders and decoders change, but the core — LoRA adapters on a transformer backbone, trained with SFT — stays the same.

Now go build something with it. And if you do, tell us about it in the comments (we'd love to see what you create! 🚀




Resources
Notebooks used in this lab:

Qwen3-VL (8B) Vision Finetuning

Orpheus-TTS (3B) Finetuning

Datasets:

LaTeX OCR

Elise (TTS)

Models:

Qwen3-VL 8B

Orpheus-TTS 3B

SNAC 24kHz

Documentation:

Unsloth Vision Finetuning

Unsloth TTS Finetuning

Unsloth Qwen3-VL Guide

24 Likes
∙
4 Restacks
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
