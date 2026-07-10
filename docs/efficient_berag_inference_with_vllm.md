# Efficient BERAG Inference with vLLM

We have developed a version of vLLM that supports BERAG inference. 
    At a high level, given a request with K documents, we create K normal vLLM request, each corresponding to one branch in the BERAG ensemble, and treat them as a request group (BERAG group). The vLLM scheduler then perform scheduling on a group basis, so that the overall BERAG request can advance after all sub-requests  belongs to a BERAG group can advance. This implementation allows us to take advantage of many of the inference optmization techniques already implemented in vLLM, for example, PagedAttention and Iteration-based Scheduling, to name a few. 

We now use our implementation (referred to as the "vLLM-BERAG" engine) to accelerate and benchmark BERAG inference. 
    For acceleration, we will use vLLM-BERAG as the inference to process prepared VQA examples on E-VQA. 
        This is similar to `src/vllm_vqa_inference.py`, which uses vLLM for standard, concatenative RAG. 
        We will write a single file that handles standard RAG and BERAG with our vLLM-BERAG engine.
    For benchmarking BERAG, we will simulate the open-serving scenario with batch size = 1. 
        This would allow us to directly compare the Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT) metrics between BERAG and RAG. 

In the following sections, we first describe background information about the vLLM-BERAG inference engine, the VQA inference pipeline, and the BEFT models that will be used for inference. We then describe detailed implementation plans. 

# Background

## vLLM-BERAG engine

The vLLM-BERAG engine is under `src/infer/vllm-berag`. The general design is described in `src/infer/vllm-berag/berag-doc/design_doc.md`. 

From a user point of view, vLLM-BERAG provides a function `generate_berag()` that allows user to specify prompts, documents, query images, etc. 
    This is a different function than the standard `generate()` function, which is used for standard RAG. 
    The signature of `generate_berag()` can be found at `src/infer/vllm-berag/vllm/v1/engine/llm_engine.py`. Images are passed in as follows:

```python
{
    "prompt": shared_prefix,
    "multi_modal_data": {"image": query_image},
    "multi_modal_uuids": {"image": [query_image_uuid]},
}
```

Note that although in principle BERAG can also handle multi-modal content in its document, at the moment this is not yet developed. On E-VQA, all documents remain textual.


## VQA Inference Pipeline

The VQA inference pipeline can be found in `src/vllm_vqa_inference.py`. It is a simple, three-step procedure:
* First, the script reads in the test dataset and process it into prompts that can be accepted by the vLLM engine.
* Then, the script initializes an vLLM engine instance, and pass the prepared requests to the engine for processing. 
* Finally, the script evaluates the generated response with external evaluator modules. 

For E-VQA, BErt Match (BEM) is used. This requires old, TensorFlow library. We may separate inference and evaluation, as we likely need a separate environment for running the BEM evaluation. However, we can use one script. 

## BEFT models

The BEFT models are trained with our version of LlamaFactory, which implements a BEFT trainer. BEFT gives a LoRA weight and a `.pt` weight file for the prior head. The vLLM-BERAG engine should be provided with the correct model and the correct prior head weight. 
   

# Implementation

## Using vLLM-BERAGA

We will create a virtual environment under `src/infer/vllm-berag`. This environment will be used for inference.

We will need to make sure that we pass in the prior head weights for vLLM-BERAG, so that meaningful prior is obtained. 

We will merge the LoRA weight into the model to give a standalone BEFT model. Note that sometimes vLLM may not support LoRA request for newer models. 

We may need to adjust the default prior head setting to align with what training actually does. 
    We have set the hidden state dimension to 1024 and used a 2-layer MLP
    We use the -4 input token position (counting the generation header) in training. This also needs to be respected. 
    The exact prior head architecture can be found here `src/train/LlamaFactory-0.9.5-beft/src/llamafactory/train/beft/trainer.py`. It is important that we have the same architecture in vLLM-BERAG. 

## VQA Inference Pipeline

We can borrow the overall structure from `src/vllm_vqa_inference.py`. Just make sure that at different modes (`rag` or `beft`), we use different functions to add requests. 

We will use the prompt defined in `src/vllm_vqa_inference.py`. We can do a cross-check to make sure that the prompt are similarly defined as in the data curation scripts under `scripts/curate/evqa`, to ensure train-test alignment. 
