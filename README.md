# 🌿 CoffeeGuard - End-to-End MLOps Platform for Coffee Leaf Disease Detection

An end-to-end, production-style machine learning system: data → training with
experiment tracking → containerized serving API → CI/CD → cloud deployment →
live monitoring and drift detection. Built entirely on free infrastructure.


## Why this project

Coffee leaf diseases (rust, miner, phoma, cercospora) cause major yield loss
for smallholder farmers, including in Ethiopia, one of the world's largest
coffee producers. CoffeeGuard classifies a photo of a coffee leaf into a
disease category in real time via a REST API, and — unlike most ML portfolio
projects — ships the *entire* lifecycle a production ML system needs, not
just a Jupyter notebook.


## Architecture
```
┌─────────────┐   ┌──────────────┐   ┌────────────────┐   ┌───────────────┐
│  Data (DVC) │──▶│ Training     │──▶│ Model Registry  │──▶│ FastAPI       │
│  ImageFolder│   │ (PyTorch +   │   │ (MLflow, local) │   │ Inference API │
└─────────────┘   │  MLflow)     │   └────────────────┘   └───────┬───────┘
                   └──────────────┘                                │
┌─────────────────────────────────────────────────────────────────┘
│
▼
┌──────────────┐   ┌───────────────┐   ┌─────────────────┐   ┌────────────┐
│ Docker image │──▶│ GitHub Actions│──▶│ GHCR + Render     │──▶│ Prediction │
│              │   │ CI/CD         │   │ (deployment)     │   │ logging    │
└──────────────┘   └───────────────┘   └─────────────────┘   └─────┬──────┘
                                                                     │
                                                                     ▼
                                                    ┌────────────────────────────┐
                                                    │ Evidently drift reports +  │
                                                    │ Streamlit monitoring dash  │
                                                    └────────────────────────────┘
```
## Tech stack
| Layer | Tool | Notes |
|---|---|---|
| Modeling | PyTorch 2.3, torchvision (EfficientNet-B0 transfer learning) | Runs on free Kaggle/Colab GPU |
| Data versioning | DVC | Git-based, free |
| Experiment tracking | MLflow | Self-hosted, free |
| Serving | FastAPI + Uvicorn | Async REST API |
| Containerization | Docker | Multi-stage-ready `Dockerfile` |
| CI/CD | GitHub Actions | Test → build → push → deploy, on every push to `main` |
| Container registry | GitHub Container Registry (GHCR) | Free, unlimited public |
| Deployment | Render (free web service, Docker build) | Free, public URL  |
| Monitoring | Evidently AI (drift) + SQLite prediction log + Streamlit dashboard | Free |
| Testing | pytest | Unit + API contract tests |