# ActiveLabel — LLMs and vision-language models (future plan)

Status: **planned, not started.** This document records how ActiveLabel can grow to use and train
large language models (LLMs) and vision-language models (VLMs), what the current design already
gives for that, and which decisions keep the door open in the meantime. Nothing here is built yet.

## Two ways in, in this order

### 1. LLM/VLM as a tool inside the image pipeline (first)

These add value to the existing projects without changing their data:

| Feature | What it does | Where it fits |
| --- | --- | --- |
| Zero-shot pre-annotation | Boxes or masks from a text prompt ("people, cars") with open-vocabulary detectors and VLMs, before any model is trained | Pre-annotation of new tasks, next to the current model |
| Label check | Asks a VLM whether a labelled object is what the label says; flags doubtful ones for review | Curation (step 3) |
| Natural-language search | Finds images by description ("night, rain, truck") with image-text embeddings | Images (step 1), FiftyOne views |
| Captions and visual Q&A | Text per image, for documentation or new datasets | New project type |
| Assistant | Explains metrics, compares models, suggests the next step | Training and Models screens |

Each one is a new kind of worker (same protocol as the training workers) and, where needed, a new
field in the labels; the core stays as it is.

### 2. LLM/VLM as the trained model (later)

Fine-tuning (LoRA/QLoRA or full) on text or image+text data. This needs:

- new data types: text, conversations, image+text pairs;
- an annotation tool for text (CVAT does not annotate text; the tool is still to be chosen);
- new metrics: perplexity, ROUGE/BLEU, exact match, evaluation by a judge model;
- in practice a GPU, and a check of its memory before a model can be chosen.

## What the current design already gives

- Projects → tasks → jobs, and sync with an annotation tool in both directions.
- Frozen dataset versions with a fingerprint: every run knows exactly which data it saw.
- A separate training environment, packages installed on demand, GPU detection, one heavy process
  at a time, temporary folders removed at the end.
- Workers as separate processes that talk in JSON lines: independent of the framework
  (transformers, peft, vLLM, llama.cpp…).
- The model catalogue drives the screen: a new model is data, not new interface code.
- Runs, live metrics, grid search, K-fold, versioned models and exports.

## Decisions to keep the door open (apply as the code is touched)

1. **Generic data item.** A dataset version lists *items* with a type (`image`, `frame`, `text`,
   `image_text`), not only images; image-only code checks the type.
2. **Open label schema.** Besides `bbox`, `polygons` and `keypoints`, labels can carry `text` and
   free `attributes`.
3. **Task kinds as tables.** Task kinds, metrics and workers keep coming from tables
   (`training/catalog.py`, `WORKER_FILES`); no scattered `if detection` checks.
4. **One environment per family.** Vision models and LLMs in separate training environments, so
   their package versions never conflict.
5. **Model format on the card.** The model card says the format (`pt`, `onnx`, `safetensors`,
   `gguf`) and whether it is an adapter (LoRA) or a merged model.
6. **Cache policy.** Downloads are temporary today. Base LLMs are many gigabytes: an optional,
   visible, persistent model cache (with its size and a clean-up button) will be needed.
7. **Licence in the catalogue.** Each model lists its licence (Llama, Gemma, Qwen and others have
   their own terms), shown before anything is downloaded.

## Security and personal data

- Text is much more likely than images to contain personal data: projects with text get the same
  care as any personal data (minimum necessary, no sharing outside the project).
- Models run locally by default. An external AI service may receive project data only if the project owner
  explicitly approves it; no other service is used with that data.
- Tokens for model hubs are kept in the system keyring, like the CVAT credentials, never in files.

## Suggested delivery (small pull requests, when the time comes)

1. Generic data item and open label schema (no new feature, only the model).
2. Persistent model cache and environments per family.
3. Zero-shot pre-annotation with an open-vocabulary detector.
4. Label check with a VLM in Curation.
5. Natural-language search.
6. Text projects (with a text annotation tool to be chosen) and LoRA fine-tuning.
