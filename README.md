Here is a humble, informative README you can use for your project.

---

# LLM-Assisted Grounded Theory Analysis (GTA) Pipeline

Welcome to this experimental repository! This modest project attempts to automate and replicate the early stages of Grounded Theory Analysis (GTA) on qualitative interview transcripts using Large Language Models (LLMs).

While an LLM can never truly replace the nuanced, deeply contextual understanding of a human qualitative researcher, this tool is designed to serve as a supportive starting point to help explore, organize, and synthesize large volumes of qualitative data.

## 🛠️ How It Works

The pipeline follows a classic three-stage Grounded Theory approach:

1. **Phase 0: Data Extraction & Chunking** Reads raw interview PDF files from a target directory and safely splits them into manageable, character-length chunks to fit within the LLM's context window.
2. **Phase 1: Open Coding** The LLM scans the raw text chunks to extract base concepts, generating a structured JSON array of open codes alongside their original text passages.
3. **Phase 2: Axial Coding** The open codes are synthesized and grouped into relational categories. This stage establishes connections between the initial codes and outputs the relationships as structured JSON.
4. **Phase 3: Selective Coding** The pipeline takes the grouped axial relations and synthesizes them into a cohesive final narrative or core theory, saved as a readable Markdown file.

## 🧩 Key Features

* **Flexible LLM Backend:** The system is built to be model-agnostic. You can run it entirely locally using Hugging Face pipelines, or hook it up to proprietary models via OpenRouter
* **Traceable Outputs:** Intermediate outputs for both Open and Axial coding are saved locally as `.json` files, ensuring you can review the LLM's logic and trace the generated theory back to the raw data.

## 📚 Datasets

This project was built and tested using rich, publicly available qualitative datasets from the Qualitative Data Repository (QDR):

* *Teaching with Shared Data for Learning Qualitative Data Analysis: A Multi-Sited Case Study* (Furlong et al., 2025)
* *Barriers and Facilitators to Implementation of Mindfulness in Motion for Firefighters* (Steinberg et al., 2024)
* *Relationship Quality: A Multi-Country Investigation* (Silan & Ciruelas, 2026)

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed, along with the following primary dependencies:

```bash
pip install PyPDF2 transformers torch openai python-dotenv
```

### Configuration

1. **API Keys:** If you plan to use proprietary models via OpenRouter, create a `.env` file in the root directory and add your API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

2. **Setup your Data:** Place your interview PDFs into the `data/` directory (e.g., `data/RelationshipQuality`).
3. **Select your Model:** In `main.py`, set `MODEL_TO_USE` to either `"local"` or `"proprietary"`.

### Running the Pipeline

Simply execute the main script:

```bash
python main.py
```

Output files (`output_open_codes.json`, `output_axial_codes.json`, and `output_final_theory.md`) will be generated inside a dedicated output folder within your data directory.

## 🌱 A Gentle Disclaimer

Qualitative analysis is a deeply human endeavor. This code is shared humbly as an exploration of how AI can assist in the heavy lifting of data synthesis. Please critically review all generated codes and theories, and use this tool to augment—rather than replace—your own analytical lens! Feel free to fork, experiment, and improve upon the prompts and pipeline.
