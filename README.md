# Llama2-QLoRA-on-SageMaker

This experiment is based on Amazon SageMaker. Including, 

1/ Training a Bloke Llama2 (TheBloke/Llama-2-7B-fp16) QLoRA Adapter,

2/ merge the LLM and the adapter into one-model for faster inference later, 

3/ for the tuned model in S3, then use SageMaker LMI (Large Model Inference) container to host the adpated model.

    - in inference/, use Basic Python engine (HuggingFace Accelerate)
    
    - in inference-deepspeed/, use LMI optimized DJL-DeepSpeed engine



**Resources**:

**Notebook Instance** - CPU instance e.g. ml.c5.xlarge<br/>
**Training** - Single GPU Required, tested on ml.g5.xlarge ~ ml.g5.4xlarge<br/>
**Hosting** - Single GPU Required, tested on ml.g4dn.xlarge ~ ml.g5.4xlarge

<br/>

### Basics to Start Experiment on SageMaker
https://github.com/haozhx23/SageMaker-Utils/blob/main/0-How-to-start.md

<br/>

### Refs
https://huggingface.co/TheBloke/Llama-2-7b-chat-fp16
https://github.com/artidoro/qlora<br/>
https://arxiv.org/abs/2305.14314<br/>
https://www.philschmid.de/sagemaker-llama2-qlora<br/>
https://www.linkedin.com/pulse/enhancing-language-models-qlora-efficient-fine-tuning-vraj-routu