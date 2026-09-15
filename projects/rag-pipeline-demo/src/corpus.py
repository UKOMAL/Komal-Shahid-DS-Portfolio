"""A support-doc corpus with exact-term traps and paraphrase traps.

Two things make this corpus useful rather than decorative:

1. Every document is TOPICALLY similar — GPUs, containers, deployment. That
   is the condition under which dense retrieval is weakest, because topic
   alone cannot separate the right answer from a plausible neighbour.
2. Vocabulary is shared deliberately across documents so that a latent
   semantic model has co-occurrence structure to learn from.
"""

DOCUMENTS = [
    {"id": "KB-1001", "text":
     "Error CUDA_ERR_4417 is raised when the container runtime CUDA version exceeds "
     "the host driver version. Downgrade the image to a CUDA line the host driver "
     "supports, or upgrade the driver on the host machine."},
    {"id": "KB-1002", "text":
     "Error CUDA_ERR_4471 indicates the NVIDIA container toolkit is missing on the "
     "host. Install the toolkit package and restart the docker daemon so the runtime "
     "can expose the device."},
    {"id": "KB-1003", "text":
     "If the GPU device is not visible inside a running container, confirm the "
     "container was started with the gpus all flag. Without that flag docker never "
     "passes the device through to the container."},
    {"id": "KB-1004", "text":
     "Part number A10G-24 is the GPU fitted to g5 instances. It carries 24 GB of "
     "device memory and suits transformer inference workloads that need more memory "
     "than a smaller card provides."},
    {"id": "KB-1005", "text":
     "Part number T4-16 is the GPU fitted to g4dn instances. It carries 16 GB of "
     "device memory and is the cheapest card for embedding generation and batch "
     "reranking workloads."},
    {"id": "KB-1006", "text":
     "GPU memory pressure during training is usually driven by batch size rather "
     "than model size. Reduce the batch, enable gradient checkpointing, or move the "
     "job to an instance with more device memory."},
    {"id": "KB-1007", "text":
     "A container should never download model weights on cold start. Bake the "
     "weights into the image or mount them from a volume, otherwise the first "
     "request a user makes pays for the whole download and feels extremely slow."},
    {"id": "KB-1008", "text":
     "Credentials for a containerised service running on an EC2 instance should be "
     "read from the instance profile through IMDS. Long lived access keys stored in "
     "environment variables are the finding every security review opens with."},
    {"id": "KB-1009", "text":
     "Model weights baked into a docker image increase image size but remove "
     "download latency at runtime. This is the usual trade for inference services "
     "where the first request must be fast."},
    {"id": "KB-1010", "text":
     "A health check that only confirms the process is alive will not catch a "
     "container that started without a usable GPU device. Have the check assert the "
     "accelerator is actually available so the failure surfaces at deploy time."},
    {"id": "KB-1011", "text":
     "Instance profile roles should be scoped to the specific bucket prefix and the "
     "specific actions a service needs. A wildcard policy granting every action on "
     "every resource will be flagged during any security review."},
    {"id": "KB-1012", "text":
     "Choosing an instance for inference starts with whether the model fits in "
     "device memory, then whether the workload is latency bound or throughput bound. "
     "Picking a training instance for an inference job wastes most of the spend."},
    {"id": "KB-1013", "text":
     "Cold start latency for a model service is dominated by weight loading. Keeping "
     "a warm pool of containers avoids making a user wait for initialisation on the "
     "first request of the day."},
    {"id": "KB-1014", "text":
     "The docker daemon must be restarted after installing the container toolkit, "
     "otherwise the runtime configuration is not picked up and the device stays "
     "invisible to new containers."},
    {"id": "KB-1015", "text":
     "Driver and runtime are separate layers. The host supplies the driver, the "
     "image supplies the runtime, and a mismatch between the two is the most common "
     "cause of accelerator problems in containerised workloads."},
    {"id": "KB-1016", "text":
     "Batch size is the first thing to reduce when a training job exhausts device "
     "memory. Gradient checkpointing trades compute for memory and often makes an "
     "otherwise impossible batch fit."},
]

# The exact token IS the answer. Topic cannot separate these.
EXACT_TERM_QUERIES = [
    ("what causes CUDA_ERR_4417", "KB-1001"),
    ("CUDA_ERR_4471 fix", "KB-1002"),
    ("specs for part number A10G-24", "KB-1004"),
    ("what is T4-16", "KB-1005"),
]

# Wording deliberately differs from the document. Lexical overlap is poor.
PARAPHRASE_QUERIES = [
    ("my graphics card isn't showing up inside docker", "KB-1003"),
    ("service is really slow the very first time someone calls it", "KB-1007"),
    ("how should an app on a server authenticate to the cloud", "KB-1008"),
    ("job ran out of memory while training", "KB-1006"),
]
