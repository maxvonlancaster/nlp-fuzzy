## 1️⃣ Representation & Geometry of Language (often weak)

### 🔹 Tokenization theory (beyond BPE)

* Unigram LM tokenization (SentencePiece)
* Tokenization as a compression problem
* Tokenization mismatch effects (train vs inference)
* Subword regularization

**Why weak:** Most people “use tokenizer” without understanding consequences.

---

### 🔹 Embedding space geometry

* Anisotropy of embedding spaces
* Hubness problem
* Whitening / isotropy fixes
* Cosine vs dot-product semantics

---

### 🔹 Contextual representation collapse

* Layer-wise semantic specialization
* Why middle layers encode syntax
* Why last layers are task-specific

---

## 2️⃣ Training Dynamics of Transformers (very under-known)

### 🔹 Attention is not explanation

* Counterfactual attention
* Attention ≠ feature importance

---

### 🔹 Scaling laws

* Chinchilla scaling
* Compute-optimal training
* Parameter vs data scaling trade-offs

---

### 🔹 Pretraining instabilities

* Loss spikes
* Gradient explosion in attention
* Softmax saturation

---

## 3️⃣ Probabilistic & Information-Theoretic NLP (major gap)

### 🔹 Language modeling as density estimation

* Cross-entropy vs perplexity meaning
* Calibration vs likelihood
* Exposure bias

---

### 🔹 Mutual information in NLP

* MI between representations and labels
* Information bottleneck for language models

---

### 🔹 Entropy & surprisal

* Surprisal theory (psycholinguistics)
* Why surprisal correlates with reading time

---

## 4️⃣ Advanced Sequence Modeling Concepts

### 🔹 Long-context failures

* Attention quadratic bottleneck
* Recency bias
* Positional encoding pathologies

---

### 🔹 Alternatives to attention

* State space models (S4, Mamba)
* Linear attention
* RWKV

---

## 5️⃣ Optimization & Fine-tuning Pitfalls (very common weakness)

### 🔹 Catastrophic forgetting

* Adapter vs LoRA vs full fine-tuning
* Layer freezing strategies

---

### 🔹 Prompt vs parameter learning

* Prompt tuning
* Prefix tuning
* Soft prompts vs hard prompts

---

### 🔹 Loss surfaces in NLP

* Sharp vs flat minima
* Why overfitting looks different than in CV

---

## 6️⃣ Evaluation & Metrics (HUGE interview gap)

### 🔹 Metric mismatch

* BLEU ≠ quality
* ROUGE limitations
* F1 instability in NER

---

### 🔹 Distribution shift

* In-domain vs out-of-domain generalization
* Dataset leakage

---

### 🔹 Human vs automatic evaluation

* Inter-annotator agreement
* Krippendorff’s alpha

---

## 7️⃣ Linguistic Structure (often ignored)

### 🔹 Syntax induction

* Probing classifiers
* Structural probes

---

### 🔹 Semantics vs pragmatics

* Implicature
* Coreference resolution subtleties

---

### 🔹 Discourse modeling

* Coherence modeling
* Rhetorical Structure Theory (RST)

---

## 8️⃣ Robustness, Safety & Bias (interview favorite)

### 🔹 Spurious correlations

* Shortcut learning
* Dataset artifacts

---

### 🔹 Adversarial NLP

* Token-level attacks
* Gradient-based text attacks

---

### 🔹 Fairness metrics in NLP

* Group vs individual fairness
* Bias amplification

---

## 9️⃣ Interpretability beyond SHAP (advanced)

### 🔹 Probing vs causal analysis

* Linear probes limitations
* Causal mediation analysis

---

### 🔹 Concept activation vectors

* TCAV for text
* Neuron-level interpretability

---

## 🔟 Retrieval-Augmented & Memory Models (modern gap)

### 🔹 Dense vs sparse retrieval

* Dual encoders vs cross encoders
* ANN search errors

---

### 🔹 Hallucination mechanisms

* Parametric vs non-parametric memory
* Faithfulness vs fluency trade-offs

---

## 1️⃣1️⃣ Theoretical NLP (rare but impressive)

### 🔹 Formal language theory

* Transformers and regular languages
* Expressivity limits

---

### 🔹 Generalization theory

* In-context learning as Bayesian inference
* Memorization vs abstraction

---

## 1️⃣2️⃣ One killer interview question (test yourself)

> **Why does perplexity decrease but generation quality get worse?**

If you hesitate — this is a gap.

---

## How to use this list

If you want, I can:

* **Assess you** by asking 10 diagnostic questions
* Build a **custom study plan**
* Go deep into **1–2 topics with math**
* Map these topics to **FAANG-style interviews**

Just tell me how deep you want to go.





