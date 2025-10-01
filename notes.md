# Notes for Code Review:

## Main-function Pipeline
   1) Build the saveing name for the model if *args.name* is empty
   2) Setup WandDb
   3) Build the latest checkpoint-path
   4) Start remote sync path if *args.remote_sync* is active
   5) Create CLIP model and transform
       * Parse schema and identifier -> "local-dir", "hf_hub", "ViT-B-16"
       * Load model config, pretrained condig and checkpoint path
       * Model config overwrites
       * Instantiate the model with model config
       * Load checkpoint if exits
       * Load image weights if exits
       * Load text weights if exits
       * Check if jit is activatetet
       * Attach the preprocess config to the model
   6) Log and print the moodels (ViT and LLM) and print all Parameters
   7) Transform model the DDP if *args.distributed* is true
   8) Select a optimizer -> wheater adamw or start *args.opt* with "timm/"
   9) Resume model from a checkpoint if *args.resume* is not empty
   10) Load the text tokenizer as HFTokenizer (Hugging Face tokenizer)
   11) Get the data from training
       * Check wheater the data is "train", "val" or "imagenet-val/v2"
       * Get dataset format (wds, csv, ...)
       * Figure out dataset size
       * WebDataset pipeline -> Load shards, split across nodes, read tars, shuffel sampels, decode images (pillow), apply processing, convert to (image, text)-tupels, batch them up
       * Wrap into WebLoader
   12) Select a scheduler for the optimizer
   13) Configurate logging with wandb
   14) Use compiled model if *args.torchcompile* is true
   15) Evaluation only mode if no "train" in data
   16) Create loss function
        * *SigLip-loss*
        * *CLIP-loss*
   17) Train one epoch 
       * Iterate over the dataloader -> get batches
       * Transmitt images and text to GPUs
       * Forward pass of the model with images and texts (one batch)
       * Calculate loss from the model_output
       * Sum up all losses (e.g. CLIP-loss and SigLIB-loss) and call the backward-pass
       * Make a optimizer step
       * Logging of the results
   18) Evaluate if "val" in data
   19) Saving checkpoints
   20) Remote sync 

## Arguments 
 1) *args.name*: For storing the model with a name
 2) Distribution:
    * *args.rank*: ID for each process across the entire job (relevant for SigLIB-loss)
    * *args.world_size*: Total number of processes (#GPUs complete)
    * *args.distribute*: Bool for training in parallel (multiple GPUs)
    * *args.horovod*: Alternative for PyTorch Distributed (DDP)
    * *args.dist_backend*: Communciation backend (common is NCCL (Nvidia Collective Communications Libary))
 3) *args.resume*: Path to latest checkpoint
 4) *args.destill*: Use knowledge distillation to train a (smaller) student model from a teacher model
 5) *args.opt*: Choose AdmW or a Timm-optimizer with "timm/.."
 6) *args.precision*: Use automatic mixed precision (amp) -> train with different precision where they are needed
 7) 

## Unknown Syntacs/Code 
 * *is_master()*-function: Check if rank is 0 -> then this process is mater 

## Notes:
 * DDP (DistributedDataParallel) handels splitting data across GPUs, synchronizing gradients and sync the processes
 * SLURM is a job scheduler -> allocate CPUs, GPUs, nodes, and lauches the training script
 * Timm is a model zoo for image models and optimizers
 * *Use_bnb_linear* is the bitsandbytes-library -> library for efficient low-bit training/inference 
 * *torch.jit* is a just-in-time compiler -> convert the model into a TorchScript that is a static graph representation
 * *autocast* is the automtic selected precition
 * *scaler* is a GradientScaler that scales up the precision of the loss before backpropagation
 * *logit scale* is the temperature of contrastive loss -> higer results in sharper softmax (focus on hardest positive/negatives)
 * *Attention Pooling* is a way to represent the whole token-embeeding in a vector (needed for contrastive learning)-> compute score for each token, apply softmax, comput weighted sum of all tokens
 * *forward_intermediates* outputs also the results of the intermediate results and allows for early stopping
 * *nn.Parameter* is a learnable tensor that will be updated during training and it appears in model.parameters()
 * *x.unsqueeze(0)* adds a leading dimention to the tensor x



{
  "mean": [
    0.632067511952372,
    0.2596067565908414,
    0.2685098403169425
  ],
  "std": [
    0.3184604777791743,
    0.41301071101841674,
    0.40244374335883953
  ]
}