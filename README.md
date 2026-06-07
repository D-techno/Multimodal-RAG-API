<<<<<<< HEAD
# Multimodal RAG API

A production-style Retrieval-Augmented Generation (RAG) system built using:

- CLIP embeddings
- FAISS vector search
- TinyLlama (1.1B) LLM
- FastAPI backend
- Dockerized deployment

---

## Architecture

User Query  
→ CLIP Embedding  
→ FAISS Retrieval  
→ TinyLlama Generation  
→ JSON Response  

---

## Features

- Multimodal embedding pipeline
- Vector similarity search
- Context-aware generation
- Hallucination mitigation
- Latency tracking
- FastAPI REST API
- Swagger documentation
- Docker support

---

## Running Locally

```bash
uvicorn app.main:app --reload
=======
---
title: RAG API
emoji: ⚡
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> dd9b33f6efac0488cd2d856184fffb45310e7383
