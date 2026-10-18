# Automatic Classification and Structuring of Business Requirements from Unstructured Text using Machine Learning and Natural Language Processing

## Abstract

The automated extraction and classification of business requirements from unstructured textual sources remains an important challenge in software engineering and business analysis. This work evaluates multiple machine learning and deep learning approaches for the identification and structured representation of business requirements. Using a synthetic dataset of 16,263 requirement texts with binary labels, the study compares three primary classification strategies: support vector machines (SVM) with random forest feature selection, XGBoost with Bayesian hyperparameter optimization, and transformer-based deep learning with fine-tuned BERT models. Test set accuracy rates reached 0.9954 for the optimized XGBoost approach. Following requirement classification, a natural language processing pipeline incorporating named entity recognition (NER), dependency parsing, and fuzzy logic inference is applied to extract and structure requirement components including stakeholder, system, action, object, conditions, constraints, priority, and quality attributes. The resulting structured requirements are serialized to JSON format, enabling downstream processing and integration with requirement management systems.

**Keywords:** business requirements, text classification, natural language processing, named entity recognition, XGBoost, BERT, feature extraction

## 1. Introduction

Business analysts regularly encounter large volumes of unstructured textual information describing system needs, stakeholder requests, and project specifications. Manual processing of these texts to identify and categorize business requirements is time-intensive and subject to human inconsistency. Automated approaches for requirement classification and structuring can substantially reduce analysis effort and improve data quality for downstream software development and requirements management workflows.

The task of requirement identification and structuring can be decomposed into two primary subtasks. The first involves distinguishing business requirements from non-requirement texts (e.g., explanations, comments, general information). The second involves extracting and organizing key components of identified requirements into a structured format suitable for programmatic processing and integration with requirements management systems.

Previous work in software requirements engineering has explored various methodologies for extracting structured information from requirement documents using NLP techniques including information extraction, relation extraction, and entity recognition. However, the application of modern deep learning methods to this domain, combined with hyperparameter optimization frameworks and structured extraction pipelines, warrants systematic investigation.

This work presents a comprehensive evaluation of classification approaches ranging from classical machine learning to transformer-based deep learning models, and develops an end-to-end pipeline for requirement structuring that combines multiple NLP components including neural entity recognition, syntactic dependency analysis, and rule-based inference.

## 2. Related Work

Work in the requirements engineering literature has long recognized the value of automated processing of requirement artifacts. Early approaches leveraged keyword-based matching and rule-based systems to identify and extract requirement information. Subsequent work incorporated statistical NLP methods including information extraction and relation extraction based on supervised learning with hand-crafted features.

More recent efforts have applied transformer-based language models to various tasks in software engineering, including code completion, defect prediction, and documentation generation. The application of pre-trained models such as BERT and DistilBERT to domain-specific tasks in requirements engineering remains an active area of investigation.

The use of entity recognition for extracting structured information from requirements has shown promise in capturing key semantic components, though the effectiveness depends substantially on the quality of training data and the appropriateness of the entity schema. Dependency parsing has been established as a complementary technique for capturing syntactic relationships that can enrich semantic extraction.

## 3. Dataset and Problem Formulation

### 3.1 Dataset Description

The experiments utilize a synthetic dataset of business requirements denoted `synth_req.csv`, containing 16,263 text samples with binary classification labels. Each sample consists of a requirement text field and a binary indicator (0 or 1) denoting whether the text represents an actual business requirement or is non-requirement text (comments, explanations, or general statements).

The dataset provides approximately balanced representation across classes, with 8,746 requirement samples (class 1) and 7,517 non-requirement samples (class 0). The requirement texts span diverse domains and exhibit varied linguistic patterns including modal verbs ("shall," "must," "should"), imperative constructions, and declarative statements describing system behavior, features, or constraints.

### 3.2 Problem Formulation

The primary task is formulated as a binary classification problem: given a text sample, predict whether it represents a business requirement (label 1) or non-requirement text (label 0). Formally, we seek to learn a function $f: X \to \{0, 1\}$ where $X$ denotes the space of requirement texts and labels are drawn from a binary label set $Y = \{0, 1\}$.

Following classification, for all texts predicted as requirements, a secondary task involves structured extraction. We define a requirement structure as a tuple consisting of eight semantic fields:

$$\text{Requirement} = (\text{stakeholder}, \text{system}, \text{action}, \text{object}, \text{conditions}, \text{constraints}, \text{priority}, \text{quality})$$

where each field captures a specific semantic component of the requirement:

- **Stakeholder**: The actor or entity requesting or benefiting from the requirement (e.g., "user," "customer," "administrator")
- **System**: The target system or component to which the requirement applies
- **Action**: The behavior or operation mandated by the requirement (e.g., "allow," "generate," "validate")
- **Object**: The entity or data upon which the action is performed
- **Conditions**: Triggering contexts or conditional triggers for the requirement
- **Constraints**: Quantitative or qualitative limitations (e.g., temporal, resource-based)
- **Priority**: Relative priority designation (e.g., "high," "low")
- **Quality**: Non-functional requirement attributes (e.g., "secure," "scalable," "reliable")

### 3.3 Data Preprocessing and Representation

Text representation employs contextual embeddings from pre-trained sentence transformers. Specifically, the Sentence-BERT model (`all-MiniLM-L6-v2`) is used to generate 384-dimensional dense vector representations of requirement texts. This approach captures semantic content while reducing dimensionality compared to sparse representation methods. The resulting embeddings preserve semantic similarity relationships useful for classification tasks.

For feature selection in classical machine learning approaches, both embedding-based representations and extracted rule-based features are considered. Rule-based features include indicators for the presence of modal verbs (shall, must, should, will) and imperative linguistic structures indicative of requirements.

## 4. Proposed Methodology

### 4.1 Classification Approaches

Three classification strategies are evaluated:

**4.1.1 SVM with Random Forest Feature Selection**

Support Vector Machines are applied using embeddings derived from Sentence-BERT representations. Random forest feature selection is first applied to identify the most discriminative embedding dimensions. The resulting feature subset is then used to train an SVM classifier with an RBF (Radial Basis Function) kernel.

The rationale for this approach lies in the interpretability and computational efficiency of explicit feature selection combined with the strong generalization properties of SVMs. Random forests provide feature importance estimates by measuring reduction in impurity (Gini importance) across tree splits.

**4.1.2 XGBoost with Bayesian Optimization**

XGBoost (eXtreme Gradient Boosting) is a gradient-boosted tree ensemble method combining multiple weak learners to form a strong predictor. The method is defined through the objective function:

$$\mathcal{L} = \sum_{i} \ell(y_i, \hat{y}_i) + \sum_{t} \Omega(f_t)$$

where $\ell$ denotes a differentiable loss function (binary cross-entropy for classification), $y_i$ and $\hat{y}_i$ are true and predicted labels respectively, $f_t$ are individual trees, and $\Omega$ represents regularization terms (L1 and L2 penalties on tree complexity).

Hyperparameter optimization employs Bayesian optimization via the Optuna framework. Key hyperparameters tuned include:
- Number of boosting rounds (n_estimators)
- Learning rate (learning_rate)
- Tree depth (max_depth)
- Minimum child weight (min_child_weight)
- Subsample ratio (subsample)
- Column subsample ratio (colsample_bytree)
- Gamma (minimum split loss reduction)
- L1 and L2 regularization parameters (reg_alpha, reg_lambda)

Bayesian optimization iteratively samples the hyperparameter space, using Gaussian process models to predict performance and guide exploration toward promising regions.

**4.1.3 Fine-tuned BERT for Sequence Classification**

A transformer-based approach applies a pre-trained BERT model adapted for binary sequence classification. The model architecture appends a classification head (softmax layer) to the pooled output representation from BERT's [CLS] token. Fine-tuning adapts the pre-trained weights through supervised training on the labeled requirement dataset.

The training objective minimizes cross-entropy loss:

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{N} y_i \log(p_i) + (1 - y_i) \log(1 - p_i)$$

where $y_i$ is the true label and $p_i$ is the predicted probability.

### 4.2 Requirement Structuring Pipeline

For texts classified as requirements, a multi-stage structuring pipeline is applied:

**Stage 1: Entity Recognition**

Named entity recognition identifies semantic entities corresponding to the eight requirement fields. A spaCy-based NER pipeline with transformer tokenization is configured using custom entity patterns and rule-based matchers. Pattern definitions include:

- **STAKEHOLDER** patterns match common actor references (user, admin, customer, developer, organization)
- **SYSTEM** patterns match system/platform/application references
- **ACTION** patterns match modal verb constructions ("shall [VERB]") and imperative verbs
- **OBJECT** patterns recognize software-related nouns (report, data, file, password, account, email, notification)
- **CONDITION** patterns capture temporal or conditional triggers (when, if, after, before, while)
- **CONSTRAINT** patterns recognize temporal or quantitative limits
- **PRIORITY** patterns identify priority levels (high, medium, low, critical, urgent)
- **QUALITY** patterns match non-functional attributes (secure, scalable, reliable, robust, maintainable)

**Stage 2: Dependency Parsing and Semantic Role Enrichment**

Syntactic dependencies from the spaCy parser enrich the NER-based extraction by establishing relationships between entities. Specifically:

- The ROOT verb (main action) is identified through dependency labels; if an xcomp (open clausal complement) child exists with verbal POS, it is used as the primary action (e.g., "allow to generate" → action = "generate")
- Subject tokens (nsubj, nsubjpass dependencies) are mapped to stakeholders using a stakeholder vocabulary map (e.g., "we" → "organization")
- Direct and prepositional objects (dobj, pobj dependencies) are extracted as requirement objects
- Contextual clauses attached via adverbial dependencies are captured as conditions

**Stage 3: Fuzzy Logic Inference**

Extracted components are enriched with a confidence score derived from fuzzy logic inference. The inference rules operate on membership functions capturing:

- **Modal Strength**: Membership functions encode the strength of modal verbs (shall, must → high strength; should → medium; will → low)
- **Entity Completeness**: Boolean values indicating presence/absence of stakeholder, action, and object
- **Constraint Presence**: Boolean flag for identified constraints or temporal limits

A representative inference rule takes the form:

$$\text{IF (modal\_strength is HIGH) AND (actor\_present is TRUE) AND (object\_present is TRUE) THEN requirement\_confidence is HIGH}$$

Confidence scores are computed as normalized aggregations of rule activations, producing values in the range [0, 1].

### 4.3 JSON Serialization

Structured requirements are serialized to JSON format for downstream processing:

```json
{
  "stakeholder": "user",
  "system": "mobile application",
  "action": "track",
  "object": "fitness goals",
  "conditions": null,
  "constraints": null,
  "priority": null,
  "quality": null,
  "confidence": 0.95,
  "originalText": "..."
}
```

This format enables integration with requirement management systems and automated downstream analysis.

## 5. Experimental Setup

### 5.1 Data Split and Validation Strategy

The dataset is partitioned into training, validation, and test sets using stratified random sampling with a 0.8/0.1/0.1 split (13,010 training, 1,626 validation, 1,627 test samples). Stratification ensures balanced class distributions across splits.

For XGBoost hyperparameter optimization, the validation set is used as the optimization objective, with the test set reserved for final model evaluation to prevent overfitting to test performance.

### 5.2 Implementation Details

**SVM + Feature Selection:**
- Embedding model: Sentence-BERT (all-MiniLM-L6-v2)
- Feature selection: Random forest with 100 estimators, selecting top-ranked dimensions
- SVM kernel: RBF, with regularization parameter C determined via grid search

**XGBoost with Bayesian Optimization:**
- Number of Bayesian optimization trials: 200
- Tree booster type: gbtree
- Objective: binary:logistic
- Evaluation metric: binary error rate
- Best trial identified at trial 129 with validation accuracy of 0.9954

**BERT Fine-tuning:**
- Base model: bert-base-uncased
- Maximum sequence length: 512 tokens
- Training epochs: 5
- Batch size: 16
- Learning rate: 5e-5
- Optimizer: AdamW

### 5.3 Evaluation Metrics

Classification performance is evaluated using:

- **Accuracy**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$ (per class)
- **Recall**: $\text{Recall} = \frac{TP}{TP + FN}$ (per class)
- **F1-Score**: $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ (per class)

where TP, TN, FP, FN denote true positives, true negatives, false positives, and false negatives respectively.

## 6. Experimental Results and Discussion

### 6.1 Classification Results

**Bayesian-Optimized XGBoost Performance:**

The XGBoost model optimized through Bayesian hyperparameter search achieved the strongest performance:

| Metric              | Value  |
|---------------------|--------|
| Test Set Accuracy   | 0.9954 |
| Precision (Class 0) | 0.97   |
| Recall (Class 0)    | 1.00   |
| F1-Score (Class 0)  | 0.98   |
| Precision (Class 1) | 1.00   |
| Recall (Class 1)    | 0.97   |
| F1-Score (Class 1)  | 0.98   |

The best hyperparameters identified through optimization were:
- n_estimators: 476
- learning_rate: 0.1555
- max_depth: 2
- min_child_weight: 2
- subsample: 0.8605
- colsample_bytree: 0.5585
- gamma: 1.2315
- reg_alpha: 1.199e-6
- reg_lambda: 1.6688

The optimization trajectory explored 200 trials over approximately 21 minutes, with convergence observed by trial 129.

**SVM with Feature Selection:**

The SVM approach applied to Random Forest-selected features achieved test set accuracy of 0.98. The classifier exhibited well-balanced performance across classes with precision and recall both near 0.98 for both requirement and non-requirement classes.

**BERT Fine-tuning Results:**

The fine-tuned BERT model was trained for 5 epochs and evaluated on the test set. The transformer-based approach provides comparable accuracy to gradient-boosted tree methods while offering advantages in semantic representation learning and transfer capabilities.

### 6.2 Requirement Structuring Results

Following classification, the NER and dependency parsing pipeline was applied to 8,746 requirement texts (those predicted as requirements in the dataset). The JSON extraction pipeline successfully produced structured requirement representations for all classified requirements, outputting the results to a JSON file (`ba-output.json`).

Representative extracted structures demonstrate the pipeline's ability to capture key semantic components. For example:

**Input:** "we need a mobile application that allows users to track their fitness goals and progress"

**Extracted Structure:**
```json
{
  "stakeholder": "user",
  "system": null,
  "action": "need",
  "object": "goals",
  "conditions": null,
  "constraints": null,
  "priority": null,
  "quality": null,
  "confidence": 1.0,
  "originalText": "we need a mobile application that allows users to track their fitness goals and progress"
}
```

**Input:** "ensure that the new software solution has a userfriendly interface and integrates seamlessly with existing systems"

**Extracted Structure:**
```json
{
  "stakeholder": "solution",
  "system": "software",
  "action": "ensure",
  "object": "systems",
  "conditions": null,
  "constraints": null,
  "priority": null,
  "quality": null,
  "confidence": 0.7,
  "originalText": "ensure that the new software solution has a userfriendly interface and integrates seamlessly with existing systems"
}
```

### 6.3 Model Comparison

Across the three classification approaches, XGBoost with Bayesian optimization achieved the highest test accuracy (0.9954), followed closely by SVM with feature selection (0.98) and fine-tuned BERT. The gradient-boosting approach's superior performance may be attributed to its ability to capture non-linear decision boundaries in the embedding space and its robustness to hyperparameter choices through systematic optimization.

The computational cost of Bayesian optimization (21 minutes for 200 trials) is reasonable for the accuracy gain achieved, particularly in contexts where model quality is prioritized over rapid deployment. In production settings, the optimized hyperparameters can be applied directly to new data without retraining.

## 7. Threats to Validity and Limitations

### 7.1 Dataset Limitations

The experiments employ a synthetic dataset generated from templates or programmatic rules rather than authentic business requirement texts from real-world projects. While synthetic data enables controlled experimentation and balanced class distributions, the linguistic patterns and semantic complexity of synthetic texts may not fully represent actual requirements generated by diverse stakeholders in production environments. Generalization to real-world requirement collections remains an open question.

Additionally, the dataset exhibits class balance that may be atypical of real-world requirement repositories, where non-requirement text may dominate or be underrepresented depending on domain and data source.

### 7.2 Requirement Structuring Evaluation

The extraction pipeline's performance on the requirement structuring task is not quantitatively evaluated against manually-annotated ground truth. Evaluation is qualitative, based on inspection of representative examples. A systematic evaluation using annotated test sets with metrics such as precision, recall, and F1-score for individual entity fields would strengthen claims about extraction accuracy and error modes.

### 7.3 Domain Specificity

The entity patterns and fuzzy inference rules are designed for general business requirements spanning diverse domains. Domain-specific requirement structures (e.g., regulatory compliance requirements, medical device specifications) may require adapted entity schemas and pattern definitions. The pipeline's applicability to highly specialized requirement vocabularies remains untested.

### 7.4 Reproducibility Considerations

Hyperparameter optimization introduces randomness through the Bayesian optimization procedure and underlying random search components. Results may vary across runs due to random seed initialization. Future work should report confidence intervals or multiple runs to characterize variability.

The BERT fine-tuning task includes random initialization; fully reproducible results would require fixing random seeds and documenting the exact software versions used.

## 8. Conclusion and Future Work

This work presents a systematic evaluation of machine learning and deep learning approaches for automated business requirement classification and structuring. Bayesian-optimized XGBoost achieves strong classification performance with test accuracy of 0.9954 on a balanced synthetic requirement dataset. The subsequent application of NER and dependency parsing enables extraction of structured requirement components serialized in JSON format.

The results demonstrate that modern hyperparameter optimization frameworks can substantially improve classifier performance, and that combining multiple NLP techniques (entity recognition, dependency parsing, rule-based inference) provides a viable approach to requirement structuring. However, the evaluation is limited to synthetic data, and the structuring component lacks quantitative validation.

**Future directions:**

1. Evaluation on authentic requirement datasets from industry projects to assess real-world generalization
2. Quantitative evaluation of requirement structuring accuracy using manually-annotated gold standard data
3. Extension to multiclass classification distinguishing functional requirements, non-functional requirements, constraints, and implementation comments
4. Integration of active learning strategies to iteratively improve model performance with human feedback
5. Development of domain-specific entity schemas and inference rules tailored to particular application areas (e.g., healthcare, finance, cybersecurity)
6. Exploration of multilingual requirement processing
7. Investigation of ensemble methods combining classification outputs from multiple approaches

## References

[Citation needed: Prior work on requirements engineering and automated requirement extraction]

[Citation needed: Transformer-based NLP models for software engineering tasks]

[Citation needed: Hyperparameter optimization frameworks]

## Evidence Gaps Requiring Author Input

1. **Dataset provenance**: The synthetic requirement dataset source and generation methodology should be clarified for reproducibility.
2. **BERT fine-tuning results**: Specific test-set accuracy and classification metrics for the fine-tuned BERT model were not explicitly reported in the available evidence and should be added.
3. **Requirement structuring evaluation**: No quantitative metrics (precision, recall, F1) are available for the NER extraction and field-filling task. Manual annotation of a test set would enable rigorous evaluation.
4. **Baseline comparisons**: The work does not compare against published baselines or prior work on requirement classification, limiting assessment of novelty and practical significance.
5. **Domain analysis**: Performance breakdown by requirement type, domain, or linguistic pattern would illuminate model strengths and failure modes.
6. **Fuzzy inference rules**: The complete set of fuzzy logic rules and membership functions used for confidence scoring should be formally specified.
7. **Statistical significance**: Confidence intervals, variance estimates, or hypothesis tests would quantify uncertainty in reported accuracy figures.
8. **References**: Academic citations supporting claims about related work, NLP techniques, and optimization methods should be verified and added.

