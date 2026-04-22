The Neural Maze
A Practical Guide to LLM Inference at Scale
Finetuning Sessions · Lesson 8 / 8
Miguel Otero Pedrido y Antonio Zarauz Moreno
abr 01, 2026


So far in this series, we've stayed firmly in model-land — architectures, training loops, finetuning tricks. All important stuff. But at some point, you need to actually serve the thing. And that's where it gets humbling.

🙋 Because here's what nobody tells you: a single LLM request involves two computationally opposite phases fighting over the same GPU.

One is massively parallel and compute-hungry. The other is sequential and starved for memory bandwidth. Making them coexist efficiently is basically the entire game of LLM inference — and it's led to a beautiful chain of engineering ideas, each one patching the flaw the previous one introduced.

That's what this post is about.

We'll start from the basics of how a single request flows through a model, and build all the way up to production-scale techniques like chunked prefill, prefill-decode disaggregation, PagedAttention, and elastic GPU sharing.

Let's get into it! 👇

Introduction to LLM Inference


Image from Understanding the Two Key Stages of LLM Inference: Prefill and Decode
At the heart of it, all large language models (LLMs) are simply sophisticated next-token predictors.

When a user submits a query to an AI service, the model first processes the entire prompt and then iteratively generates the response one token at a time. This generation process creates a highly variable input-to-output workload: user prompts can range from a handful of words to thousands of tokens, and the length of the generated response is unpredictable until the model decides to emit an end-of-sequence token.

🙋 Because of this dynamic and variable nature, serving LLMs at scale requires high-performance infrastructure that differs significantly from traditional deep learning inference paradigms.

To handle these variable workloads effectively, modern inference engines divide the generation process into two fundamentally distinct computational stages: the prefill phase and the decoding phase.

This dual-phase architecture dictates how hardware resources are allocated and forms the basis for all modern inference optimizations. Understanding the distinct computational characteristics of each phase is crucial for identifying bottlenecks, maximizing GPU utilization, and achieving high throughput in production environments.

The prefill phase is responsible for processing the user's initial prompt in its entirety to generate the very first token.

During this stage, the model computes the hidden representations for all input tokens concurrently in a single forward pass. Because it processes a large sequence of tokens simultaneously, this phase heavily utilizes the GPU's parallel processing capabilities through massive matrix multiplications, making it highly compute-bound.

The primary performance metric for this phase is the Time to First Token (TTFT), which is critical for maintaining responsiveness in real-time applications like chatbots.

As the prefill phase computes these token representations, it saves intermediate attention states—specifically the key and value vectors—in GPU memory to avoid redundant calculations in future steps. This stored data is known as the KV cache.

By preserving the keys and values of the initial prompt and every subsequently generated token, the model drastically reduces the computational cost of generating new tokens from quadratic to linear time. However, this optimization shifts the system’s bottleneck, as it requires maintaining and continuously accessing an ever-growing memory footprint for every active request.



Link: https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests
Following the prefill, the decoding phase takes over to sequentially generate the remainder of the response step-by-step.

In each decoding iteration, the model only processes the single newest token, relying on the saved KV cache to provide the necessary historical context. Because the compute cores only handle one token per sequence but must still load the massive model parameters and the entire KV cache from high-bandwidth memory (HBM) to the SRAM, the decoding phase is severely memory-bandwidth bound.

Consequently, the performance of this phase is measured by the Time Per Output Token (TPOT), and inference engines must rely on sophisticated batching strategies to prevent the GPU from being underutilized.

Decoding (I): The baseline of static batching
Modern GPUs are designed for highly parallel computational workloads, capable of executing trillions or even quadrillions of floating-point operations per second.

However, large language models (LLMs) often struggle to fully saturate these powerful chips because a significant portion of the GPU's memory bandwidth is bottlenecked by the constant need to load massive model parameters from memory.

To mitigate this inefficiency, inference serving systems employ batching, which allows the engine to load the model parameters once and use them to process multiple input sequences concurrently.

The simplest and most traditional approach to this technique is known as static batching.

In static batching, the inference server waits until a fixed number of user requests arrive, grouping them together to process as a single, unified batch.

Because traditional tensor operations require rectangular shapes, the system must add padding tokens to shorter prompts so that all sequences match the length of the longest prompt in the batch. Once the batch is assembled, the model executes its forward passes, generating new tokens for every sequence in lockstep. This process continues iteratively until the entire batch has completed its generation phase.



Link: https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests
While static batching is relatively straightforward to implement, it possesses a critical flaw when applied to LLMs: the entire batch is strictly tied to the longest-running request.

In real-world applications, output sequences vary drastically; one prompt might require a short, single-sentence response, while another demands a lengthy, step-by-step explanation.

Even if a shorter request emits its end-of-sequence token after just a few iterations, it must remain in the batch. The system cannot remove the finished request to free up memory, nor can it insert a new prompt, until the absolute longest sequence in that specific batch finishes generating its final token.

This rigid limitation directly leads to severe GPU underutilization and increased latency.

For every iteration where a completed sequence sits idle waiting for the longest sequence to finish, valuable compute resources are wasted.

Furthermore, new incoming requests are forced to wait in a queue until the currently active batch completely clears, which adds unnecessary delays for users. Ultimately, while static batching provides a baseline improvement over processing single requests sequentially, its inability to handle variable-length outputs makes it highly inefficient for production-scale LLM inference.

Decoding (II): Continuous batching


Link: https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests
To overcome the severe limitations of static batching, inference systems initially turned to dynamic batching, which collects incoming requests within a set time window before processing them.

However, while this balances throughput and latency better than static approaches, it still forces shorter requests to wait for the longest sequence in the batch to finish. The definitive solution to this inefficiency is continuous batching, also widely known as iteration-level scheduling.

This approach fundamentally rethinks request scheduling by ensuring that the GPU does not wait for all sequences in a batch to complete before starting new ones.

At its core, continuous batching operates by evaluating the batch composition dynamically at each decoding iteration.

As soon as a sequence within the batch finishes generating its response—typically indicated by emitting an end-of-sequence token—the system immediately removes it from the batch.

Instead of waiting for the rest of the batch to clear, a new request from the waiting queue is instantly inserted into the freed compute slot. This assembly-line mechanism keeps the compute resources constantly busy, completely eliminating the idle time that plagues traditional batching methods.

Implementing this continuous flow requires a departure from traditional tensor operations that rely on rectangular shapes and extensive padding.

To avoid the massive computational waste of padding when constantly swapping prompts, modern serving engines employ ragged batching, where prompts of uneven lengths are simply concatenated together into a single sequence.

The system then uses precise attention masks to seamlessly control token interactions, ensuring that tokens from one prompt never interact with tokens from another. This clever use of masking eliminates padding waste and allows the engine to mix sequences of drastically different lengths efficiently.

Managing this continuous influx of new requests requires the system to handle the prefill phase alongside ongoing decoding tasks. Because the initial prefill phase is highly compute-intensive and differs significantly from the memory-bound decoding phase, continuous batching frameworks must utilize specific scheduling policies to balance the two.

By intelligently mixing and scheduling prefill computations with ongoing decoding steps, continuous batching ensures that the GPU remains fully utilized at all times.

Ultimately, this iteration-level scheduling is the crucial optimization that allows production services to handle thousands of concurrent users with variable-length queries while drastically multiplying overall inference throughput.

Decoding (III): Chunked Prefill


Link: https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests
While continuous batching keeps the GPU highly occupied by constantly swapping in new requests, it introduces a severe new problem known as prefill-decode interference.

When a new user submits a long prompt, the system must execute a highly compute-intensive prefill phase. If this massive prefill job is inserted into the same batch as ongoing decoding tasks, the decoding requests are forced to wait for the long prefill computation to finish before they can generate their next token.

This massive compute stall significantly increases the Time Per Output Token (TPOT), or inter-token latency, creating a jarring and slow experience for users who are already in the middle of receiving their generated text.

To overcome these crippling compute stalls, modern inference engines implement a technique called chunked prefill.

Image

Instead of forcing the GPU to process a massive prompt in a single, monolithic forward pass, the system divides the long input sequence into smaller, strictly sized segments or "chunks".

By enforcing a strict token budget per batch, the engine prevents any single long prefill request from monopolizing the GPU's compute cycles and indefinitely delaying the progress of other requests.

Mechanically, chunked prefill relies on the iterative flexibility of the KV cache.

During the first forward pass, the engine processes the first chunk of the prompt and stores its KV states.

For the second chunk, it simply prepends these previously stored KV states to the new computation, updating the attention mask accordingly so no contextual information is lost.

This segmenting allows the system's scheduler to cleverly piggyback or mix smaller prefill chunks with the single-token decoding steps of other ongoing requests.

The scheduler can fill the batch with decoding tokens and then pack any remaining computational space with a prefill chunk, ensuring the GPU is perfectly utilized without ever starving the decoding sequences.

Despite its effectiveness at smoothing out inter-token latency, chunked prefill introduces its own unavoidable tradeoffs. Because the initial prompt is processed over multiple iterations and forced to share compute time with decoding tasks, the Time to First Token (TTFT) for that specific new request inherently increases.

Furthermore, chunked prefill causes significantly more memory access overhead for the prefill job. To compute each subsequent chunk, the KV cache of all the previously processed chunks must be repeatedly loaded from the GPU's High-Bandwidth Memory (HBM) into its SRAM, an overhead that scales quadratically with context length.

Yet, even with these costs, chunked prefill remains an essential mechanism for balancing high throughput with stable decoding latencies in dynamic LLM workloads.

Decoding (IV): Prefill-Decode Disaggregation
scaling

Link: https://haoailab.com/blogs/distserve-retro/
As inference systems scale to handle thousands of concurrent users, a fundamental architectural flaw in traditional continuous batching becomes apparent: the forced colocation of prefill and decoding phases on the same hardware.

Because prefill is heavily compute-bound and decoding is heavily memory-bandwidth bound, placing them on the same GPUs intrinsically couples their resource allocation and parallelism strategies.

In these colocated setups, systems are forced to batch these distinct computation types together to maximize overall throughput.

However, this leads to strong prefill-decoding interference, where latency-sensitive decoding steps are significantly delayed by lengthy prefill computations, forcing service providers to over-provision expensive compute resources to meet both Time to First Token (TTFT) and Time Per Output Token (TPOT) service-level objectives.

To break this compromise, modern high-performance inference architectures have introduced prefill-decode disaggregation, a paradigm shift that assigns the prefill and decoding computation to entirely separate GPUs.

In a disaggregated system, a dedicated "prefill instance" handles only the processing of the user's prompt to generate the very first token. Once this compute-intensive prefill phase is complete, the prefill instance transmits the generated intermediate states—specifically the KV cache—along with the first token to a separate "decoding instance".

This complete isolation fundamentally eliminates prefill-decoding interference, ensuring that bursty, long-context prompts never stall the continuous, step-by-step generation of ongoing responses.

Beyond eliminating interference, disaggregation allows each phase to scale independently using tailored resource allocation and model parallelism strategies optimized for their specific latency requirements.

For example, to meet stringent TTFT service-level objectives, prefill instances can utilize high degrees of intra-operator parallelism to accelerate the compute-heavy prompt processing. Conversely, because the decoding phase requires far less computation per token but struggles with GPU underutilization, the architecture can allocate multiple prefill instances to feed a single decoding instance. This funneling effect allows the decoding instance to accumulate a much larger batch size on dedicated hardware, maximizing throughput without sacrificing TPOT.

While disaggregation solves compute contention, it introduces a significant new challenge: the communication overhead of transferring massive KV caches across the network.

The intermediate states generated during the prefill phase can be exceptionally large; for instance, serving a single 512-token request on a 66-billion parameter model generates over a gigabyte of KV cache data.

When serving hundreds of requests per second, transferring this data from prefill GPUs to decoding GPUs demands immense network bandwidth—often requiring 90 Gbps or more just to render the transmission overhead invisible to the user. If not managed correctly, this data transfer can quickly replace compute stalls as the primary bottleneck in the inference pipeline.

afd

Link: https://haoailab.com/blogs/distserve-retro/
To mitigate these massive data transfer costs, disaggregated serving frameworks must employ sophisticated, topology-aware placement algorithms.

In clusters equipped with cutting-edge InfiniBand networks, prefill and decoding instances can be flexibly placed across different nodes without severe penalties.

However, in environments with limited cross-node bandwidth, the system must strategically colocate the corresponding prefill and decoding instances within the same physical node.

By doing so, the architecture can route the heavy KV cache transfers through ultra-fast intra-node connections, such as NVIDIA's NVLink, which boasts peak bandwidths of up to 600 GB/s.

This hardware-aware orchestration ensures that the system maintains high goodput and strict latency guarantees, making disaggregation a vital strategy for cost-effective, large-scale LLM serving.

Frameworks in Action
To truly understand how these theoretical optimizations translate into real-world performance, we must look at production-grade inference engines like vLLM.

vLLM successfully orchestrates continuous batching (iteration-level scheduling) while solving one of the most critical bottlenecks in LLM serving: memory fragmentation.

Traditional inference frameworks allocate a contiguous chunk of GPU memory ahead-of-time for a request's maximum possible context length. Because generation lengths are unpredictable, this rigid allocation leads to massive memory waste—often up to 80% memory fragmentation.

vLLM solves this through its core innovation, PagedAttention, which takes inspiration from operating system virtual memory and paging.

Introduction to vLLM and PagedAttention | Runpod Blog

Link: https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention
Instead of demanding contiguous memory, PagedAttention divides the KV cache into fixed-size blocks (or "pages") that can be stored non-contiguously in the GPU's memory.

The engine's Block Manager maintains a mapping between logical virtual blocks and physical memory blocks, allocating them on the fly only when they are actually needed during generation. This dynamic, just-in-time memory management practically eliminates waste, limiting memory fragmentation to under 4% (only occurring in the last partially filled block).

By drastically reducing memory waste, vLLM frees up vast amounts of VRAM. This recovered memory can be used to hold significantly more sequences at once, allowing the engine to drastically increase its batch size. When combined with a preemptive scheduler and continuous batching, this architecture yields massive performance gains, achieving up to 23x or 24x higher throughput compared to naive static batching systems like Hugging Face Transformers, while simultaneously reducing p50 latency.

Elastic Resource Allocation
As inference infrastructure scales, maximizing hardware utilization across variable, multi-tenant workloads becomes essential.

While hardware features like MIG partition the GPU physically, modern software projects like kvcached achieve elastic GPU sharing by bringing OS-style virtual memory abstraction directly to the LLM's KV cache.

Traditional serving engines must statically reserve physical GPU memory at startup, which is highly inefficient for dynamic workloads.

The kvcached daemon solves this by decoupling logical GPU virtual addressing from physical memory allocation.

When multiple LLMs share a GPU, kvcached reserves only the virtual address space initially; it then dynamically maps and allocates physical GPU memory strictly on-demand as cache blocks are actively used during inference.

Make GPU Sharing Flexible and Easy

Link: https://github.com/ovg-project/kvcached?tab=readme-ov-file
This elastic architecture provides several transformative benefits for production deployments:

Multi-LLM Serving: Multiple different models can concurrently share the same physical GPU memory pool elastically, replacing rigid memory partitioning and significantly reducing serving costs.

Serverless and Compound AI: Models can allocate memory only when actively serving requests and release it immediately when idle or finished. This enables true serverless LLM scaling with rapid cold-starts and allows complex multi-model pipelines (e.g., retrieval, reasoning, summarization) to share resources fluidly.

Workload Colocation: Because memory can be reclaimed instantly without modifying the underlying engine code, LLM inference can efficiently coexist on the same hardware alongside other memory-intensive GPU jobs like model training, fine-tuning, or vision workloads.

By implementing this fast-path/slow-path page allocation system, kvcached has been shown to deliver a 2x to 28x reduction in Time to First Token (TTFT) compared to static allocation systems when handling bursty, concurrent workloads.

Next Steps
If there's one thing to take away from all of this, it's that LLM inference is essentially a resource management problem disguised as a generation task.

Every optimization we covered — continuous batching, chunked prefill, disaggregation, PagedAttention — exists because prefill and decode want opposite things from the hardware, and the entire history of inference engineering has been about negotiating that tension more cleverly.

The good news? You don't need to build any of this from scratch. Frameworks like vLLM and kvcached package these ideas into production-ready systems.

But understanding why they work the way they do makes you a much better operator — you'll know which knobs to turn, what tradeoffs you're accepting, and when something is actually bottlenecked vs. just misconfigured.

On Friday, we'll get hands-on with this. See you in the lab! 🔬



48 Likes
∙
5 Restacks

The Neural Maze
Run the World's Best OCR on Your Own Laptop
Finetuning Sessions · Lab 8 / 8
Miguel Otero Pedrido y Antonio Zarauz Moreno
abr 03, 2026


In the previous article, "A Practical Guide to LLM Inference at Scale", we explored the theory behind serving large language models efficiently — from quantization strategies to deployment architectures.

But theory without practice is just PowerPoint.

In this hands-on guide, we're getting our hands dirty with a model that perfectly illustrates those principles in action: GLM-OCR, a 0.9B parameter vision-language model that ranks #1 on OmniDocBench V1.5, beating models 10x its size.

We'll take it from a local Docker container running on your laptop all the way to a production-ready pipeline, covering hardware optimization, custom model configuration, and the official SDK for complex document parsing. If you read the theory, this is where it clicks.

Why OCR?
The Optical Character Recognition (OCR) landscape is vast, but GLM-OCR stands out as a multimodal model specifically built for complex document understanding.

GLM-4.5V : Best Open-Sourced Vision model | by Mehul Gupta | Data Science  in Your Pocket | Medium

Source: https://medium.com/data-science-in-your-pocket/glm-4-5v-best-open-sourced-vision-model-51454f2ab21a
Developed by Z.ai and based on the GLM-V encoder-decoder architecture, it introduces advanced training techniques like Multi-Token Prediction (MTP) loss and full-task reinforcement learning to drastically improve recognition accuracy. Instead of relying on massive, unwieldy models, GLM-OCR proves that highly focused architectures can dominate specific tasks.

Despite its incredibly small size of just 0.9 billion parameters, GLM-OCR achieves a score of 94.62 on OmniDocBench V1.5, ranking #1 overall. This small footprint means it can run fully locally on standard consumer devices, like MacBooks or edge devices, without sacrificing capability.

It successfully rivals and often outperforms much larger, closed-source models across benchmarks for formula recognition, table extraction, and information extraction.

Introducing GLM-OCR: SOTA performance, optimized for complex document  understanding. With only 0.9B parameters, GLM-OCR delivers state-of-the-art  results across major document understanding benchmarks, including formula  recognition, table recognition ...

The secret to this "small but mighty" performance is its two-stage pipeline, which pairs the language decoder with the PP-DocLayout-V3 layout detection model. By first analyzing the document layout and then performing parallel recognition, GLM-OCR maintains robust performance on highly complex real-world scenarios, including code-heavy documents, intricate tables, and documents with rotating or staggered layouts.

Scaling Inference with vLLM
Running models locally with Ollama is perfect for testing, personal use, and CPU-only environments.

However, as your document parsing needs grow, you must consider the transition from a local machine to a robust cloud infrastructure. Cloud deployments allow you to serve the pipeline at scale, making use of time-slicing Kubernetes (k8s) configurations and worker-server deployments for maximum efficiency.

When moving to a production environment, the official GLM-OCR documentation strongly recommends transitioning to engines like vLLM or SGLang. These frameworks are specifically designed for high-concurrency services and provide significantly better performance and stability when you have access to one or multiple GPUs.



Link: https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking#comparison_2__tuned_ollama_versus_vllm
Using vLLM allows you to serve the model via an OpenAI-compatible /v1/chat/completions API endpoint. When configured properly—such as setting the max_workers and connection_pool_size appropriately in your SDK configuration to avoid 503 errors—vLLM ensures your pipeline can handle massive parallel OCR requests without crashing under load.

Introducing Ollama and llama.cpp
Before deploying, it is crucial to understand the ecosystem that makes local inference so accessible. At the core of this democratization is llama.cpp, a high-performance C++ engine designed to run LLMs on standard hardware with maximum efficiency.

While llama.cpp is incredibly powerful, it can require manual compilation and complex command-line arguments to operate.

This is where Ollama steps in as the "user interface" and manager. Ollama acts as a user-friendly wrapper around the llama.cpp backbone, allowing developers to download models, manage memory, and serve a clean API with simple commands. It handles the underlying complexity, bringing powerful language models to developers who may not be machine learning engineers.

To achieve this efficiency on local hardware, the engine relies heavily on quantization and the GGUF file format. Quantization shrinks the size of the model weights—such as using 2-bit (Q2) or 4-bit (Q4) representations instead of standard 16-bit floats—so the model can run on cheaper hardware without losing significant performance.

Download the code for this section here!

🔍 Step 0: Discover Your Hardware Specs
To get the best speed, you must identify your Physical Cores. LLMs perform best when assigned to physical cores rather than "logical" ones (Hyperthreading/SMT).

Windows (PowerShell)
Get-WmiObject -Class Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
macOS (Terminal)
sysctl -n hw.physicalcpu hw.logicalcpu
Linux (Terminal)
lscpu | grep -E '^CPU\(s\):|Core\(s\) per socket|Thread\(s\) per core'
The Rule of Thumb: For your thread configuration later, always aim for the Number of Cores, not the logical processors. If you have a hybrid CPU (like newer Intel chips), aim for the number of Performance Cores.

🐳 Step 1: Launch the Ollama Container
We use a Docker volume to ensure that once you download a multi-gigabyte model, it stays on your disk even if the container is deleted.

docker run -d \
  --name ollama-server \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama
📦 Step 2: Download the Model
Pull the GLM-OCR weights from the library. This usually requires about 2GB–4GB of space.

docker exec -it ollama-server ollama pull glm-ocr
⚙️ Step 3: The Modelfile
Standard settings often fail for OCR because images require more "memory space"(context) than simple text. We will create a custom version of the model with optimized parameters.

Enter the container:

docker exec -it ollama-server bash
Create a Modelfile:

cat <<EOF > GLM-Config
FROM glm-ocr
# Hardware & Context Settings
PARAMETER num_ctx 16384
PARAMETER num_thread 6

# Your Specific Generation Parameters
PARAMETER num_predict 8192
PARAMETER temperature 0
PARAMETER top_p 0.00001
PARAMETER top_k 1
PARAMETER repeat_penalty 1.1
EOF
Here it's important to point out that sampling parameters are typically specified at runtime (i.e., during model utilization), but given we're performing a very specific task, with a cutting-edge vision language model, sampling parameters are often hardcoded to obtain optimal results, and it’s not recommended to switch them.

Plus, maximum number of threads must follow the convention established in the previous section.

Deploy the updated model version:

ollama create glm-ocr-optimized -f GLM-Config
exit
🚀 Step 4: Using the API
With the server running, you can now send images to the model. Images must be sent as Base64 encoded strings.

import requests
import base64
import ollama
import sys
import time
from io import BytesIO
from PIL import Image

# 1. Configuration
IMAGE_URL = "https://marketplace.canva.com/EAE92Pl9bfg/6/0/1131w/canva-black-and-gray-minimal-freelancer-invoice-wPpAXSlmfF4.jpg" # Replace with your invoice link
MODEL_NAME = "glm-ocr-optimized"    # The model you created with specific parameters
MAX_DIMENSION = 1024          # Resize the longest edge to 1024px

def get_optimized_image_b64(url):
    """Downloads, resizes, and encodes the image."""
    print(f"📥 Downloading image...")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    # Calculate aspect ratio and resize
    original_width, original_height = img.size
    print(f"📐 Original Size: {original_width}x{original_height}")
    
    # Only resize if the image is actually larger than our limit
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
        print(f"🪄 Resized to: {img.width}x{img.height}")
    else:
        print("✅ Image is already small enough, skipping resize.")

    # Convert to Base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85) # JPEG is lighter than PNG for OCR
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def run_ocr():
    try:
        # Prepare the image
        image_b64 = get_optimized_image_b64(IMAGE_URL)

        print(f"🚀 Sending to Ollama (waiting for first token)...")
        start_time = time.time()
        first_token = True

        # 2. Invoke Ollama with streaming
        # The parameters (temp: 0, top_k: 1, etc.) are already in your Modelfile
        stream = ollama.generate(
            model=MODEL_NAME,
            prompt="Text recognition:",
            images=[image_b64],
            stream=True
        )

        for chunk in stream:
            if first_token:
                print(f"⏱️ Time to first token: {time.time() - start_time:.2f}s\n")
                first_token = False
            
            print(chunk['response'], end='', flush=True)

        print(f"\n\n✅ Total Processing Time: {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_ocr()

👀 Step 5: Inspecting results
When running via terminal the instruction docker stats while running the script, we will see something like:

CONTAINER ID   NAME            CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O   PIDS
0ca7a28d6fe2   ollama-server   600.42%   4.359GiB / 7.607GiB   57.31%    79.4kB / 5.25kB   0B / 0B     41
CONTAINER ID   NAME            CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O   PIDS
0ca7a28d6fe2   ollama-server   598.91%   4.359GiB / 7.607GiB   57.31%    79.4kB / 5.25kB   0B / 0B     41
CONTAINER ID   NAME            CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O   PIDS
0ca7a28d6fe2   ollama-server   598.91%   4.359GiB / 7.607GiB   57.31%    79.4kB / 5.25kB   0B / 0B     41
CONTAINER ID   NAME            CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O   PIDS
0ca7a28d6fe2   ollama-server   601.90%   4.359GiB / 7.607GiB   57.31%    79.4kB / 5.25kB   0B / 0B     41
Notice how the CPU usage is being maximized as the main task now that is generating the bottleneck is the vision encoder. It's not the decoding, which is not compute heavy, but the prefill stage. Indeed, by checking the logs:

📥 Downloading image...
📐 Original Size: 1131x1600
🪄 Resized to: 724x1024
🚀 Sending to Ollama (waiting for first token)...
⏱️ Time to first token: 174.96s

YOUR
LOGO
NO. 000001

INVOICE

Date: 02 June, 2030

Billed to:
Studio Shodwe
123 Anywhere St., Any City
hello@reallygreatsite.com

From:
Olivia Wilson
123 Anywhere St., Any City
hello@reallygreatsite.com

Item Quantity Price Amount
Logo 1 $500 $500
Banner (2x6m) 2 $45 $90
Poster (1x2m) 3 $55 $165

Total $755

Payment method: Cash
Note: Thank you for choosing us!

✅ Total Processing Time: 183.39s

It is noticeable that vision encoder and complete prefill took almost 3 minutes, and generation only 9 seconds. That is indeed where GPUs and specific hardware devices shine by applying vectorized operations, in the encoding; i.e., compute-heavy section of the pipeline.

So far results look pretty strong but this is just a toy example. By leveraging the layout detector, we will be able to adapt to a much broader range of situations, like the next one.

🧩 Step 6: GLM-OCR SDK
Together with the vision language model, Z.ai team provided a comprehensive client SDK that includes the safetensors version of the amazing PPDocLayoutV3 from PaddlePaddle, that in particular adapts to polygonal regions and edge cases.



Link: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
We will slightly adapt our code to an image taken form the first two pages of Qwen3 Technical Report:

import requests
from PIL import Image
from glmocr import GlmOcr

# --- Configuration ---
LOCAL_FILENAME = "7cf7af6c-0581-4fdc-a20f-7123aab8c0a2_3308x2339.jpg"

def run_sdk_ocr(image_path):
    # Optional: Resize to speed up CPU inference (1024px is the sweet spot)
    with Image.open(image_path) as img:
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            img.save(image_path)
            print(f"🪄 Resized {image_path} for faster CPU processing.")

    print(f"🚀 Initializing GLM-OCR SDK...")
    
    # Initialize the SDK in self-hosted mode
    with GlmOcr(config_path='./config.yaml') as parser:
        print("🔍 Analyzing document structure...")
        result = parser.parse(image_path)
        
        # Output the Markdown result
        print("\n" + "="*20 + " OCR RESULT " + "="*20)
        print(result.markdown_result)
        print("="*52)


if __name__ == "__main__":
    try:
        run_sdk_ocr(LOCAL_FILENAME)
        
    except Exception as e:
        print(f"❌ Error: {e}")

It's critical that you select the config.yaml file we provide alongside the article, as it will not be possible to run the example otherwise. It has been specifically tuned for it.

Let's check the results:

Starting Pipeline...
🪄 Resized 7cf7af6c-0581-4fdc-a20f-7123aab8c0a2_3308x2339.jpg for faster CPU processing.
🚀 Initializing GLM-OCR SDK...
Pipeline started!
GLM-OCR initialized in self-hosted mode
🔍 Analyzing document structure...
Stopping Pipeline...
Pipeline stopped!

==================== OCR RESULT ====================
Qwen3 Technical Report

Qwen Team

https://huggingface.co/qwen
https://modelscope.cn/organization/qwen
https://github.com/qwenLM/qwen3

Abstract

In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 253 billion. A key innovation in Qwen3 is the integration of thinking mode for complex, multi-step reasoning and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models, such as chat optimized models (e.g., GPT-3) and dedicated reasoning models (e.g., QX-32B)—and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their high competitive performance. Empirical evaluations demonstrate that Qwen3 achieves stated-theory results across benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive signal larger MoE models and proprietary models. Compared to its predecessor Qwen-2.5, Qwen3 expands multilingual support from 29 to 11 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0.

Table 7: Comparison among Qwen3-4B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.

| Architecture | Gemma-3-4B Base | Gemma-2.5-3B Base | Gemma-2.5-7B Base | Gemma-3-4B Base |
| :--- | :--- | :--- | :--- | :--- |
| # Total Params | 4B | 3B | 3B | 4B |
| # Activated Params | 4B | 3B | 3B | 4B |

General Tasks

| MMLU | 59.41 | 65.62 | 74.16 | 72.99 |
| :--- | :--- | :--- | :--- | :--- |
| MMLU-Redux | 56.91 | 63.68 | 71.08 | 72.79 |
| MMLU-Pro | 29.23 | 34.61 | 45.00 | 50.58 |
| SuperGQA | 17.98 | 20.31 | 26.33 | 28.43 |
| BBH | 17.87 | 16.24 | 16.30 | 17.29 |

GPQA | 24.24 | 26.26 | 36.36 | 36.87 |
| :--- | :--- | :--- | :--- | :--- |
| GSMK | 43.97 | 79.08 | 85.36 | 87.79 |
| MATH | 26.10 | 42.04 | 49.80 | 52.18 |

Coding Tasks

| EvalPlus | 43.23 | 46.28 | 62.18 | 63.53 |
| :--- | :--- | :--- | :--- | :--- |
| MultiPL-E | 28.06 | 39.65 | 50.72 | 53.13 |
| MMPT | 46.40 | 34.60 | 43.83 | 51.00 |
| CRUX-O | 34.00 | 36.50 | 48.50 | 55.00 |

Multilingual Tasks

| MGSM | 33.11 | 47.53 | 63.60 | 67.74 |
| :--- | :--- | :--- | :--- | :--- |
| MMLU-Pro | 59.62 | 65.55 | 73.31 | 71.42 |
| INCLUDE | 49.06 | 45.90 | 53.98 | 56.29 |

Table 8: Comparison among Qwen3-1.7B-Base, Qwen3-0.68-Base, and other strong open-source base-lines. The highest and second-best scores are shown in bold and underlined, respectively.

| Qwen2.5-0.58 Base | Qwen3-0.68 Base | Qwen2.5-1.58 Base | Qwen3-1.78 Base |
| :--- | :--- | :--- | :--- |
| Architecture | Gemma-3-4B Base | Gemma-2.5-3B Base | Gemma-2.5-7B Base | Gemma-3-4B Base |
| # Total Params | 0.58 | 0.68 | 1B | 1.78 |
| # Activated Params | 0.58 | 0.68 | 1B | 1.78 |

General Tasks

| MMLU | 52.11 | 26.26 | 60.90 | 62.63 |
| :--- | :--- | :--- | :--- | :--- |
| MMLU-Redux | 51.26 | 25.99 | 58.46 | 61.66 |
| MMLU-Pro | 24.74 | 9.72 | 28.53 | 36.76 |
| SuperGQA | 11.30 | 10.01 | 14.54 | 20.92 |
| BBH | 20.30 | 41.47 | 28.13 | 54.47 |

Coding Tasks

| EvalPlus | 32.77 | 24.75 | 24.24 | 28.28 |
| :--- | :--- | :--- | :--- | :--- |
| MultiPL-E | 24.75 | 24.75 | 62.54 | 72.44 |
| MMPT | 19.48 | 32.44 | 36.66 | 43.50 |
| CRUX-O | 12.10 | 27.00 | 3.80 | 36.40 |

Multilingual Tasks

| MGSM | 30.99 | 7.74 | 32.82 | 50.71 |
| :--- | :--- | :--- | :--- | :--- |
| MMLU-Pro | 31.53 | 60.16 | 60.27 | 63.27 |
| INCLUDE | 24.74 | 34.26 | 25.62 | 45.57 |
====================================================

You can now observe how the model has correctly gathered all the elements in the image and conveniently parsed them in the final markdown result.

Next Steps
This Sunday we're hosting our 8th and final Office Hours!

We'll cover two main topics:

Last week's session → Multimodal Finetuning

This week's session → LLM Deployment

In this lab we focused on local deployments, but during the office hours we'll go a step further — we'll walk you through the key parameters you can configure on Hugging Face Endpoints (with vLLM) to make your inference truly efficient.

See you there!




68 Likes
∙
6 Restacks

