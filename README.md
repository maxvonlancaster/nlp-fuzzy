# Science Repository

This is a repository for scientific ideas and research on the topics of fuzzy logic, NLP, ML and other interesting areas of computer sciences.

## Notebooks

1. NLP Direction

    - [😄 FL application in NLP sentiment analysis](/src/semantic_fuzzy.ipynb) by [@mehalyna](https://www.github.com/mehalyna)
    - [⚡ FL application in NLP sentiment analysis + energy](/src/semantic_two_dim.ipynb) by [@mehalyna](https://www.github.com/mehalyna)
    - [Ⓜ️ Application of ML and Fuzzy Logic to mood detection in Texts](/src/semantic_fuzzy_ml.ipynb)
    - [🏛️ Application of ML and Fuzzy Logic to political analysis of Texts](/src/political-nlp.ipynb) by [@12-deadblue](https://www.github.com/12-deadblue)
    - [😜 Sarcasm detection](/src/sarcasm_detection.ipynb)
    - [🤖 RAG system](/src/rag.ipynb) by [@wanesssssty](https://www.github.com/wanesssssty)
    - [✂️ Text Summarization](/src/summarization.ipynb) - Exploratory notebook
    - [📊 Text Summarization - Comprehensive Evaluation](/src/summarization_evaluation.ipynb) - **Publication-ready evaluation framework**

2. Math Econ

    - [📈 Timeseries Analysis with FL](/src/time_series.ipynb) by [@cryme666](https://www.github.com/cryme666)
    - [💵 Fuzzy Utility](/src/fuzzy-utility.ipynb)
    - [🏃 Customer Churn Prevention](/src/churn-retail.ipynb) by [@mehalyna](https://www.github.com/mehalyna)

3. Discrete Mathematics

    - [♟️ Generating Functions](/src/generating-functions.ipynb) by [@12-deadblue](https://www.github.com/12-deadblue)

4. Interesting Projects

    - [❤️ Heart Diseases Analysis](/src/medical-prediction.ipynb) by [@UserAgent0007](https://github.com/UserAgent0007)
    - [🩺 Medical Research General](/src/medical-prediction.ipynb) by [@UserAgent0007](https://github.com/UserAgent0007)

---

## Featured Research: Text Summarization

### Publication-Ready Evaluation Framework

A comprehensive research framework for **extractive text summarization using unsupervised machine learning** has been developed, transforming initial exploratory work into a publication-ready empirical study.

**Key Features:**
- ✅ **Large-scale evaluation:** 1000+ documents (MTSamples medical corpus)
- ✅ **Strong baseline comparisons:** Lead-N, Random, TextRank, LexRank, LSA
- ✅ **Statistical rigor:** Bootstrap confidence intervals, paired t-tests
- ✅ **Comprehensive analysis:** Ablation studies, error analysis, case studies
- ✅ **Publication-ready:** Suitable for COLING, EMNLP, NAACL conferences

**Methods Evaluated:**
- **K-Means Clustering:** Semantic grouping with centroid-based sentence selection
- **Kohonen Self-Organizing Maps (SOM):** Topological mapping preserving neighborhood relationships
- **TextRank:** Graph-based ranking with PageRank
- **LexRank:** Graph-based with transformer embeddings
- **LSA:** Latent Semantic Analysis with SVD

**Evaluation Metrics:**
- ROUGE-1, ROUGE-2, ROUGE-L (F-measure)
- Statistical significance testing (paired t-test, α=0.05)
- 95% confidence intervals via bootstrap resampling
- Per-document error analysis and failure pattern detection

**Framework Components:**
- [`src/summarization_evaluation.ipynb`](/src/summarization_evaluation.ipynb) - Main evaluation notebook (11 sections)
- [`src/helpers/`](/src/helpers/) - Modular evaluation infrastructure
  - `dataset_loader.py` - Multi-corpus support (MTSamples, CNN/DailyMail, Wiki Movies)
  - `baselines.py` - Standard comparison methods
  - `evaluation.py` - ROUGE metrics with statistical analysis
  - `analysis.py` - Error analysis and variance explanation
  - `ablation.py` - Component contribution testing
  - `visualization.py` - Publication-ready figures
  - `case_studies.py` - Qualitative analysis generation
- [`thesis/`](/thesis/) - Complete documentation
  - `2025-summarization.md` - Main research paper
  - `methodology_detailed.md` - Full experimental protocol
  - `algorithms.md` - Pseudocode and complexity analysis
  - `results_summary.md` - Comprehensive results template

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full evaluation (60-90 minutes for 1000 documents)
jupyter notebook src/summarization_evaluation.ipynb

# Results are exported to:
# - results/tables/ (CSV and LaTeX)
# - results/figures/ (PNG at 300 DPI)
# - results/evaluation_results.json (complete data)
```

**Research Status:** Framework complete and tested. Full 1000-document evaluation can be executed to generate final publication results.

**Citation:**
```
Melnyk, H. (2025). Unsupervised Machine Learning Methods for Text Summarization:
A Comprehensive Evaluation of Clustering and Topological Approaches.
[Details to be added upon publication]
```

---

## Our Team


| Avatar | Name | GitHub Profile | Areas of Interest| Role |
|--------|------|---------------|----|----|
| <img src="https://www.github.com/12-deadblue.png" width="50" height="50"> | **Andrii Lazoryk** | [@12-deadblue](https://www.github.com/12-deadblue) | Generating functions, NLP | Hedgehog |
| <img src="https://www.github.com/cryme666.png" width="50" height="50">  | **Valentyn Vikovan** | [@cryme666](https://www.github.com/cryme666) | NLP, Time Series Forecasting | Hedgehog |
| <img src="https://www.github.com/wanesssssty.png" width="50" height="50"> | **Anastasia Rarenko** | [@wanesssssty](https://www.github.com/wanesssssty) | RAG Systems | Hedgehog |
| <img src="https://www.github.com/Olexandr26.png" width="50" height="50"> | **Oleksandr Mykchailyk** | [@Olexandr26](https://www.github.com/Olexandr26) | NLP, FL | Hedgehog |
| TBA | **Liza Kvasnytska** | TBA | NLP | Hedgehog |
| TBA | **Dmytro** | TBA | Text Summarization | Hedgehog |
| TBA | **Kyryllo Kravtsov** | TBA | TBA | Hedgehog |
| TBA | **Nicu** | TBA | TBA | Hedgehog |
| <img src="https://www.github.com/maxvonlancaster.png" width="50" height="50">  | **Vasyl Melnyk** | [@maxvonlancaster](https://www.github.com/maxvonlancaster) | | Owl |



## Ideas (TODO):

- Sentiment (emotion in text, energicity of text, ...)
- disinformation  
- sarcasm (hard)
- medicine?
- movies?
- text professionality 
- manipulation in text 
- shortness/brevity of text 
- financial markers 
- marketing: is product represented on market?
- cv parsing (with genetic alg and fl) 

- market analysis (i.e. dollar / crypto value based on news)
- air attack prognosis (based on tg chanels / radio)
- fuzzy timeseries prognosis (may be + genetic algrthms)
- RAG system (Nastya)

- text summarizing (may be local news, etc.) (Dmitro)
- Also: web parsing, + web app for text summarization 

---

