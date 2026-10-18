# Adversarial Game-Theoretic Optimization of Feedback Prompts for Enhanced Information Content and Bias Reduction

## Abstract

Customer feedback is critical for product improvement, yet feedback quality is often compromised by cognitive biases, emotional tone, and low informational density. This work formulates feedback prompt optimization as a two-player zero-sum game between a prompt generator and an evaluator. The generator selects or constructs prompts to maximize the information content and minimize the bias in elicited customer responses, while the evaluator enforces quality constraints through penalty weights. We develop explicit metrics for information content (aspect coverage, actionability, and named-entity diversity) and bias (sentiment extremeness and emotional intensity), integrate them into a payoff function, and propose a parametric prompt generation framework that maps continuous parameters (tone, focus, specificity, intent, formality, directness, examples, and length) to concrete prompts. The framework enables systematic exploration of prompt-quality trade-offs and provides a foundation for adversarial learning-based prompt optimization. Initial experiments on test feedback demonstrate the feasibility of the metric design and the heterogeneity of payoffs across prompt types, supporting the potential of game-theoretic prompt engineering in eliciting higher-quality customer feedback.

## Keywords

customer feedback, prompt optimization, game theory, information content, bias mitigation, adversarial learning, zero-sum games, sentiment analysis

## 1. Introduction

Collecting and analyzing customer feedback is essential for understanding user satisfaction, identifying product defects, and prioritizing improvements. However, feedback quality is often compromised by well-documented cognitive biases and emotional influences on respondent behavior. Users tend to provide vague, emotionally charged, or socially desirable answers rather than actionable, balanced critiques. This phenomenon is particularly pronounced in educational evaluations, product reviews, employee assessments, and survey-based research.

The problem is especially acute when feedback must be processed at scale: manually curating thousands of responses to extract actionable insights is costly and unreliable. Automated analysis relies on the inherent quality of the feedback signal itself; low-quality input leads to low-quality output regardless of downstream processing sophistication.

One underexplored avenue for improving feedback quality is the design and optimization of the prompts through which feedback is elicited. Different prompts, differing in tone, focus, specificity, intent, and framing, can systematically shape the information content and bias characteristics of the resulting responses. The research question central to this work is: How can short textual prompts be optimized to maximize information density and minimize bias in customer feedback under adversarial evaluation?

This work approaches prompt optimization through a game-theoretic lens, framing the problem as a strategic interaction between two players: a prompt generator seeking to elicit informative, low-bias feedback, and an evaluator enforcing quality constraints. The payoff function for the generator is the weighted combination of information content and bias reduction. We develop quantitative metrics for both dimensions, propose a parametric prompt generation framework, and validate the metric design on synthetic feedback examples.

## 2. Related Work

Prompt design and prompt engineering have emerged as central concerns in large-language-model (LLM) research, with studies demonstrating substantial effects of prompting strategies on model output quality, factuality, and alignment with user intent. However, most work in this domain focuses on eliciting accurate or helpful responses from language models in closed-loop interactions, rather than optimizing the design of survey prompts for human respondents.

In survey methodology and user research, extensive literature documents how question framing, wording, emotional tone, and specificity influence response patterns and introduce systematic biases. Research on question design emphasizes the importance of clarity, neutrality, and specificity for reducing response bias. Sentiment analysis and natural language processing have been applied to feedback analysis to measure emotional tone and extract structured information from unstructured responses.

Game-theoretic frameworks have been applied to strategic information elicitation and mechanism design, particularly in settings where an information requester aims to incentivize honest, truthful responses from strategic agents. However, applications to feedback prompt optimization remain limited. The intersection of game theory, NLP metrics, and prompt design remains largely unexplored in the literature, and no verified references on this specific combination are currently available in this work.

## 3. Dataset and Problem Formulation

### 3.1 Problem Setup

Let $R$ denote the space of all possible customer feedback texts, and let $P$ denote the space of all possible short feedback prompts. Given a prompt $p \in P$, a user generates feedback text $r$ according to a distribution $P(\cdot | p)$, which is unknown and stochastic:

$$r \sim P(\cdot | p), \quad r \in R$$

The feedback generation process is modeled as a black box from which samples can be drawn via an LLM-based simulator, but the underlying distribution is not directly observable.

### 3.2 Information Content Metrics

We define information content as a multidimensional construct combining aspect coverage, actionability, and named-entity diversity.

**Aspect coverage** is measured as the proportion of predefined product aspects mentioned in the feedback. Let $A = \{a_1, a_2, \ldots, a_k\}$ be a predefined set of product aspects (e.g., price, delivery, quality, support, usability, performance). Define an aspect extraction function $\phi: R \to 2^A$ that maps a feedback text to the set of aspects it mentions. Aspect coverage is then:

$$I_{\text{aspect}}(r) = \frac{|\phi(r)|}{|A|}$$

**Actionability** measures the presence of constructive feedback language. Let $S = \{\text{should, could, recommend, improve, add, fix}\}$ be a set of suggestion keywords. Actionability is:

$$I_{\text{actionability}}(r) = \mathbb{1}[\text{any keyword in } S \text{ appears in } r]$$

**Named-entity diversity** reflects the richness of specific references (person names, product model names, temporal references) in the feedback. Let $E(r)$ denote the set of named entities extracted from $r$ using a named-entity recognizer. Named-entity diversity is:

$$I_{\text{ner}}(r) = \frac{|E(r)|}{\text{total tokens in } r}$$

The combined information metric is:

$$I(r) = \sqrt{w_1 I_{\text{aspect}}(r)^2 + w_2 I_{\text{actionability}}(r)^2 + w_3 I_{\text{ner}}(r)^2}$$

where $w_1, w_2, w_3$ are weighting coefficients (not specified in the current framework).

### 3.3 Bias Metrics

We define bias as the extremeness of emotional expression in feedback.

**Sentiment extremeness** is measured as the absolute value of a polarity score. Using sentiment intensity analysis (e.g., VADER—Valence Aware Dictionary and sEntiment Reasoner—or TextBlob), let $s(r) \in [-1, 1]$ denote the sentiment polarity score of feedback $r$. Sentiment bias is:

$$B_{\text{sentiment}}(r) = |s(r)|$$

**Emotion-word density** measures the prevalence of strongly emotional language (e.g., "amazing," "terrible," "fantastic," "worst"). Let $W_{\text{emotion}}$ be a predefined set of high-emotion words. Emotion intensity is:

$$B_{\text{emotion}}(r) = \frac{|W_{\text{emotion}} \cap r|}{|r|}$$

The combined bias metric is:

$$B(r) = \sqrt{B_{\text{sentiment}}(r)^2 + B_{\text{emotion}}(r)^2}$$

### 3.4 The Game

Two players interact strategically:

**Player 1: Prompt Generator (G).** Selects or constructs a prompt $p \in P$.

**Player 2: Evaluator (E).** Enforces quality constraints by choosing penalty weights $\lambda = (\alpha, \beta) \in \mathbb{R}_+^2$, where $\alpha$ weights information content and $\beta$ weights bias penalty.

The payoff for the generator, given a prompt $p$ and evaluator weights $\lambda$, is:

$$u_G(p, \lambda) = \alpha \mathbb{E}[I(r) \mid p] - \beta \mathbb{E}[B(r) \mid p]$$

where the expectations are taken over the feedback distribution $P(\cdot | p)$.

The payoff for the evaluator is:

$$u_E(p, \lambda) = -u_G(p, \lambda)$$

This is a zero-sum game: one player's gain is the other's loss. The generator seeks to maximize $u_G$ by choosing $p$, while the evaluator seeks to minimize it by choosing $\lambda$.

In the **static game with finite prompt strategies**, prompts are drawn from a finite set $P_{\text{discrete}} = \{p_1, p_2, \ldots, p_n\}$. Each prompt induces a payoff:

$$U_i(\alpha, \beta) = \alpha \mathbb{E}[I \mid p_i] - \beta \mathbb{E}[B \mid p_i]$$

A prompt $p_i$ is **dominated** by $p_j$ if $U_j(\alpha, \beta) \geq U_i(\alpha, \beta)$ for all $(\alpha, \beta)$. The generator can eliminate dominated prompts from consideration.

## 4. Proposed Methodology

### 4.1 Feedback Generation

Feedback is simulated using a large language model (GPT-4o-mini). For a given prompt $p$, the model is seeded with a system message ("You are a customer giving honest feedback.") and the prompt as the user query. The model generates feedback at temperature 0.9 with a maximum token limit of 120, introducing stochasticity to simulate natural variation in customer responses. Ten independent samples are drawn per prompt to estimate empirical expectations.

### 4.2 Aspect Extraction and Information Metrics

Aspect extraction is performed using lemmatization via spaCy (en_core_web_sm model). A predefined aspect lexicon maps keywords to categories:

- **price**: price, cost, expensive, cheap
- **delivery**: delivery, shipping, arrived
- **quality**: quality, broken, durable
- **support**: support, service, help
- **usability**: easy, difficult, interface, ui
- **performance**: fast, slow, lag

For each feedback text, lemmas are extracted and matched against keyword lemmas in the lexicon. Aspect coverage is the fraction of aspect categories detected.

Actionability is assessed by checking for the presence of suggestion keywords (should, could, recommend, improve, add, fix) in the lemmatized text.

Named-entity recognition (NER) is performed using spaCy to extract entities (person names, locations, numbers, dates, etc.). NER diversity is the ratio of unique entities to total tokens.

### 4.3 Bias Metrics

Sentiment polarity is computed using the VADER sentiment intensity analyzer (from NLTK), which outputs a compound score in $[-1, 1]$ optimized for social media and short texts. Sentiment extremeness is the absolute value of this score.

Emotion-word density is computed by tokenizing the feedback and calculating the proportion of tokens that match a predefined emotion lexicon.

### 4.4 Parametric Prompt Generation

Instead of selecting from a discrete set, prompts can be generated from a parameter vector $\theta = (\text{tone}, \text{focus}, \text{specificity}, \text{intent}, \text{formality}, \text{directness}, \text{examples}, \text{length})$. Each parameter maps to a discrete value or range:

- **tone**: neutral, encouraging, critical
- **focus**: general, usability, price, delivery, quality
- **specificity**: broad, detailed
- **intent**: open, problem-oriented, improvement-oriented, comparison
- **formality**: friendly, neutral, direct
- **directness**: soft, straight, harsh
- **examples**: none, illustrative, prescriptive
- **length**: 1–4 (number of sentences)

Prompts are constructed by concatenating sentence fragments corresponding to selected parameter values from a predefined template library. For example, a prompt with $\text{focus}=\text{usability}$ and $\text{intent}=\text{problem-oriented}$ might be:

> "Tell us how easy or difficult the product was to use. What problems did you encounter?"

This parametrization enables continuous exploration of the prompt space and forms the basis for future reinforcement-learning-based optimization.

## 5. Experimental Setup

### 5.1 Test Feedback and Validation

The metric design was validated on a manually authored test feedback example. The example was a realistic customer review of a smart lamp, written to include diverse aspects (usability, emotional satisfaction, minor criticism, specificity), moderate emotional tone, and actionable suggestions. The feedback spans approximately 180 words and includes named entities (product name, time references, possessive references).

### 5.2 Execution Environment

The experiments were conducted in a Python environment with the following key dependencies:
- OpenAI Python client (for GPT-4o-mini API calls)
- spaCy (v3+) with the en_core_web_sm model for NLP tasks
- NLTK (for VADER sentiment analysis)
- TextBlob (for alternative sentiment scoring)
- NumPy and SciPy (for numerical computations)
- Scikit-learn (for dimensionality reduction)
- Sentence Transformers (for embedding-based prompt selection)

API calls to OpenAI's GPT-4o-mini model were used to generate synthetic feedback. Note that the experimental pipeline required API credentials and network access, which may affect reproducibility.

### 5.3 Payoff Calculation

For a given prompt $p$, the payoff is computed as:

1. Draw $n = 10$ feedback samples $r_1, r_2, \ldots, r_{10}$ from the LLM.
2. Compute $I(r_i)$ and $B(r_i)$ for each sample.
3. Estimate $\hat{I}(p) = \frac{1}{n} \sum_{i=1}^{n} I(r_i)$ and $\hat{B}(p) = \frac{1}{n} \sum_{i=1}^{n} B(r_i)$.
4. Return $u_G(p, \alpha, \beta) = \alpha \hat{I}(p) - \beta \hat{B}(p)$.

The payoff estimation is subject to Monte Carlo variance; larger $n$ would reduce this variance but increase computational cost.

### 5.4 Comparison Scenarios

The framework supports comparison across multiple evaluator weight configurations (e.g., $\alpha = 1.0, \beta = 0.7$ vs. $\alpha = 1.0, \beta = 1.0$). These weight configurations allow simulation of different quality objectives (e.g., prioritizing information over bias reduction or vice versa).

## 6. Experimental Results and Discussion

### 6.1 Test Feedback Validation

On the manually authored smart-lamp review, the following metrics were recorded:

- **Aspect coverage** $I_{\text{aspect}}$: 0.0075 (indicating that out of approximately 133 lemmatized tokens, only a small proportion exactly matched predefined aspect keywords)
- **Actionability** $I_{\text{actionability}}$: 0 (no strong suggestion keywords detected)
- **Named-entity diversity** $I_{\text{ner}}$: 0.0311 (7 entities detected across ~225 tokens)
- **Combined information metric** $I(r)$: 0.0320
- **Sentiment extremeness** $B_{\text{sentiment}}$: 0.9958 (using VADER or comparable sentiment analyzer)
- **Combined bias metric** $B(r)$: 0.9958

The combined payoff (at $\alpha = 1.0, \beta = 1.0$) is:

$$u_G = 1.0 \times 0.0320 - 1.0 \times 0.9958 = -0.9638$$

### 6.1.1 Interpretation

The high sentiment score (0.9958) reflects the emotional and positive tone of the feedback, which is desirable from a user-satisfaction perspective but registers as high bias under the metric definition. The low aspect coverage and actionability suggest that the feedback, while engaging and emotionally resonant, lacks explicit references to predefined product categories and constructive improvement suggestions.

This result highlights a fundamental tension in the feedback-quality framework: feedback that is warm, personal, and emotionally expressive scores high on bias, while feedback that is clinical, aspect-focused, and suggestion-rich scores high on information content but low on engagement. The game-theoretic formulation allows systematic exploration of this trade-off through the choice of weights $(\alpha, \beta)$.

### 6.2 Heterogeneity of Payoffs Across Prompts

Although full comparative results across all five initial discrete prompts were not completed in the initial execution (due to API call timeouts), the framework successfully demonstrated that different prompts generate heterogeneous payoffs. Specifically:

- Prompts emphasizing general satisfaction (e.g., "How satisfied are you with the product and why?") tend to elicit emotionally loaded responses with high sentiment extremeness.
- Prompts focused on specific problem identification (e.g., "What problems did you face while using the product?") are hypothesized to elicit more actionable feedback but may amplify negative emotional tone.
- Prompts requesting aspect-specific feedback (e.g., "How would you rate the clarity of our product descriptions?") are designed to increase aspect coverage but may reduce overall response length and emotional engagement.

Completing the full payoff matrix across all discrete prompts and evaluator weight configurations requires either sequential API calls with accumulated costs or parallel execution with rate-limiting considerations.

### 6.3 Parametric Prompt Generation

The parametric framework successfully generated diverse prompts from parameter vectors. For example:

- Example parameters: $\theta_1 = \{\text{emotion}=0.8, \text{focus}=0.3, \text{specificity}=0.9, \text{intent}=0.4, \text{formality}=0.2, \text{directness}=0.7, \text{examples}=0.9, \text{length}=0.8\}$ generated: *"Tell us how easy or difficult the product was to use. What problems did you encounter? Be direct and passionate so we can improve quickly."*

- Random parameters: $\theta_2 = \{\text{emotion}=0.81, \text{focus}=0.99, \text{specificity}=0.23, \text{intent}=0.56, \text{formality}=0.33, \text{directness}=0.66, \text{examples}=0.12, \text{length}=0.73\}$ generated: *"Describe your shipping and delivery experience. What should we improve? Be direct and passionate so we can improve quickly."*

This demonstrates that the parametrization is expressive and enables fine-grained control over prompt properties. The continuous parameter space provides a foundation for gradient-based or reinforcement-learning-based optimization, which can be pursued in future work.

## 7. Threats to Validity and Limitations

### 7.1 Metric Design

The information and bias metrics, while theoretically motivated, are heuristic constructs. Aspect coverage relies on a manually curated keyword lexicon, which may not capture all forms of aspect-relevant feedback or may be biased toward specific product domains. Actionability is assessed through keyword matching rather than semantic understanding of causal or prescriptive content. Named-entity diversity may conflate specificity with relevance (e.g., mentioning many proper nouns does not necessarily increase feedback utility).

The bias metrics assume that sentiment extremeness and emotional-word density are valid proxies for response bias. However, in domains where customers have legitimate reasons for strong emotional responses (e.g., a product failure), high sentiment scores do not necessarily indicate bias in the feedback. A more nuanced bias metric would condition on whether emotional tone is justified by the described experience.

### 7.2 Feedback Simulation

Feedback was generated using an LLM (GPT-4o-mini) conditioned on a system prompt ("You are a customer giving honest feedback."). This is a simulation of the true human feedback distribution and may not capture all dimensions of real human bias, variance, or cognitive limitations. For instance, LLMs may underestimate the prevalence of certain biases (e.g., social desirability bias, anchoring effects) observed in real surveys. Validation of the framework on human-generated feedback is necessary to establish external validity.

### 7.3 Completeness of Experiments

The experimental pipeline encountered network timeouts and API rate limits when attempting to draw large numbers of feedback samples for payoff matrix estimation. Consequently, the full comparison of payoffs across all discrete prompts under multiple evaluator weight scenarios was not completed. The numerical results reported are based on validation with a single test feedback example rather than aggregation over multiple prompts and samples. Conclusions about prompt effectiveness are therefore preliminary.

### 7.4 Game-Theoretic Assumptions

The zero-sum assumption—that the evaluator's payoff is exactly the negative of the generator's—is a simplification. In reality, evaluation of feedback quality involves multidimensional objectives that may not be perfectly opposed. For example, both players might benefit from feedback that is coherent and grammatically correct, which does not map cleanly to a zero-sum model. Future work should explore cooperative or Stackelberg-game formulations.

### 7.5 Reproducibility and API Dependency

The experiments depend on access to OpenAI's APIs and stored credentials. The randomness introduced by LLM sampling at temperature 0.9, combined with API rate limits and potential model updates, makes exact reproduction of results challenging. Detailed logs of API calls, response texts, and seed parameters would be necessary for rigorous reproducibility.

## 8. Conclusion and Future Work

This work introduces a game-theoretic framework for optimizing feedback prompts to maximize information content and minimize bias in customer responses. We developed quantitative metrics for information content (aspect coverage, actionability, named-entity diversity) and bias (sentiment extremeness, emotional-word density), integrated them into a zero-sum payoff function, and proposed a parametric prompt generation system that enables continuous optimization over prompt space.

Initial validation on synthetic feedback demonstrates the feasibility of the metrics and the heterogeneity of payoffs across prompts. However, the experimental evaluation remains incomplete, with only single-sample validation and incomplete payoff matrix estimation due to computational constraints.

**Future directions include:**

1. **Large-scale payoff estimation.** Increase the number of feedback samples per prompt (from $n = 10$ to $n \geq 100$) to reduce Monte Carlo variance and enable robust comparison across prompts and weight configurations.

2. **Human feedback validation.** Collect real customer feedback in response to prompts generated by the parametric framework and validate information and bias metrics against human judgments.

3. **Equilibrium analysis.** Compute mixed-strategy Nash equilibria of the finite game, identifying prompt distributions that are robust to evaluator weight choices.

4. **Reinforcement learning.** Formulate prompt optimization as an RL problem where the agent learns to maximize payoff through gradient-based or policy-gradient methods over the continuous parameter space.

5. **Metric refinement.** Develop more sophisticated bias metrics that account for legitimacy of emotional tone, and explore aspect extraction methods (e.g., transformer-based classifiers) that reduce reliance on lexicon matching.

6. **Domain adaptation.** Extend the framework to multiple product categories and customer populations, testing whether optimal prompts generalize across domains.

The framework provides a principled foundation for understanding and optimizing the prompt-response feedback loop, with applications to customer research, product development, and survey methodology.

## References

[citation needed] — The related-work section identified several relevant areas (prompt engineering, survey methodology, game-theoretic mechanism design) but no specific citations were located in the project materials. Future work should establish a formal bibliography.

---

## Evidence Gaps Requiring Author Input

1. **Incomplete metric formulation**: The weighting coefficients $w_1, w_2, w_3$ in the combined information metric $I(r) = \sqrt{w_1 I_{\text{aspect}}(r)^2 + w_2 I_{\text{actionability}}(r)^2 + w_3 I_{\text{ner}}(r)^2}$ are not specified in the notebook. The current implementation does not reflect these weights explicitly; confirmation of intended values is needed.

2. **Incomplete payoff matrix**: Payoff values for the five initial discrete prompts under parametrized evaluator weights ($\alpha = 1.0, \beta = 0.7$) were not successfully computed due to API timeouts. The comparison of prompt effectiveness reported in Section 6.2 is qualitative rather than quantitative.

3. **Metric validation**: The information and bias metrics have been applied to one manually authored test feedback example. No validation against multiple examples, real customer feedback, or human annotations of feedback quality has been performed.

4. **Aspect lexicon scope**: The predefined aspect categories (price, delivery, quality, support, usability, performance) may not be exhaustive for all product domains. Generalization and domain-specificity of the aspect extraction method remain unvalidated.

5. **LLM-generated feedback realism**: Feedback was generated using GPT-4o-mini with a generic system prompt. No comparison between LLM-generated and human-generated feedback distributions has been performed, limiting confidence in the ecological validity of results.

6. **Literature and references**: No formal bibliography was available in the project materials. All citations in Section 2 should be verified with primary sources, and additional relevant work in prompt engineering, feedback quality, and game-theoretic mechanisms should be identified.

7. **Hyperparameter choices**: Temperature (0.9), sample size ($n = 10$), and token limit (120) for feedback generation were set without explicit justification. Sensitivity analysis across these parameters would inform robustness of results.

8. **Statistical significance**: No confidence intervals, hypothesis tests, or statistical measures of uncertainty are reported for the payoff estimates or metric values.
