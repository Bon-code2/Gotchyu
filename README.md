# Gotchyu

Gotchyu is a locally-hosted web application that analyzes text to detect AI-generated content. Built with FastAPI and Python, it utilizes a GPT-2 perplexity-based scoring model to calculate the probability of synthetic generation and highlights suspicious text via a web interface.

## Features

* **Perplexity Analysis:** Uses Negative Log-Likelihood (NLL) variance to evaluate text predictability and assign an AI-probability score.
* **FastAPI Backend:** Asynchronous backend processing serving Jinja2 HTML templates.
* **Rate Limiting:** Integrated SlowAPI to prevent abuse and manage request volume.
* **Automated Data Harvesting:** Silently saves high-quality human text (scoring under 10% AI probability with a minimum of 150 words) for future model retraining.
* **Storage Guardrails:** Data harvesting is strictly capped at 2,000 local files to prevent storage bloat, with no IP tracking or metadata storage.

## Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** PyTorch, Hugging Face Transformers (GPT-2)
* **Frontend:** HTML, CSS, Jinja2 Templates
* **Middleware:** SlowAPI

## Local Setup

### Prerequisites
* Python 3.8+
* Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/gotchyu.git](https://github.com/yourusername/gotchyu.git)
   cd gotchyu
