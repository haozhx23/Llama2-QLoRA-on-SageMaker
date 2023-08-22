# Llama2-QLoRA-on-SageMaker

This experiment is based on Amazon SageMaker. Training a Bloke Llama2 adapter with QLoRA, merged the LLM and adapter to one-model for faster inference later, save model to S3, and then use SageMaker LMI (Large Model Inference) container to host the adpated model.

Resources:

**Notebook Instance** - ml.c5.xlarge<br/>
**Training** - ml.g5.xlarge ~ ml.g5.4xlarge<br/>
**Hosting** - ml.g4dn.xlarge ~ ml.g5.4xlarge

<br/>

### Basics to Start Experiment on SageMaker
https://github.com/haozhx23/SageMaker-Utils/blob/main/0-How-to-start.md

<br/>

### Refs
https://github.com/artidoro/qlora<br/>
https://arxiv.org/abs/2305.14314<br/>
https://www.philschmid.de/sagemaker-llama2-qlora<br/>
https://www.linkedin.com/pulse/enhancing-language-models-qlora-efficient-fine-tuning-vraj-routu