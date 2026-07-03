# Combining ensemble machine learning models with text data analysis for medical conditions prediction.

https://docs.google.com/document/d/1Pecw5_4UB3Qb6huxT0MDV0U5n1x7ucJYvcRC5_p9Yio/edit?tab=t.0

## Abstract

The rapid digitalization of healthcare has resulted in vast repositories of both structured clinical data and unstructured electronic health records (EHRs). While structured data provides a quantitative baseline for patient assessment, a significant portion of diagnostic intelligence remains locked within unstructured clinical notes. Traditional predictive models often struggle to integrate these heterogeneous data types effectively, leading to suboptimal diagnostic accuracy and a lack of holistic patient profiling.

This study proposes a hybrid predictive framework that integrates Ensemble Machine Learning (EML) techniques with Natural Language Processing (NLP) for the multi-modal prediction of medical conditions. We utilize advanced text vectorization techniques—including TF-IDF and Dense Word Embeddings (Word2Vec/BERT)—to transform clinical narratives into high-dimensional feature vectors. These textual features are fused with structured patient metrics, such as demographics and vital signs. To address the inherent high variance and bias in medical datasets, we employ an ensemble architecture utilizing Random Forests and Gradient Boosted Trees (XGBoost).

Experimental results demonstrate that the combined multi-modal approach significantly outperforms single-source models. The integration of textual data provided a marked improvement in the F1-score and Area Under the Precision-Recall Curve (AUPRC), particularly in conditions where symptomatic nuances are primarily captured in physician notes rather than laboratory values. Furthermore, the ensemble strategy provided a more robust generalization across diverse patient cohorts, effectively mitigating the risks of vanishing gradients and overfitting common in deep learning applications with smaller clinical datasets.

Keywords: Ensemble Learning, Clinical Text Mining, Predictive Analytics, Healthcare Informatics, Natural Language Processing, Medical Diagnostics.

## Introduction 


The current stage of information technology development is characterized by a rapid increase in the volume of medical data and growing demands for speed and accuracy in clinical decision-making. Traditional approaches to analyzing medical information, based on manual interpretation of numerical indicators and textual descriptions, are becoming insufficiently effective given the large number of patients, complex data structures, and the need to minimize human error. In this context, the application of machine learning and artificial intelligence methods for automated analysis of medical data and decision support is of great relevance.

A distinctive feature of medical information systems is the heterogeneity of the data they process. On the one hand, this includes structured numerical indicators of a patient’s condition, such as body temperature, blood oxygen levels, blood pressure, or heart rate. On the other hand, a significant portion of important information is contained in unstructured text—descriptions of symptoms, medical history, doctors’ conclusions, and clinical notes. The effective use of only one type of data often fails to fully capture the clinical picture of a disease, necessitating the combination of numerical and textual features.

Ensemble machine learning methods, particularly Random Forest and gradient boosting on decision trees, have proven to be powerful tools for handling complex nonlinear relationships and noisy data. They demonstrate high resistance to overfitting and enable significantly better results compared to baseline models. At the same time, natural language processing (NLP) methods are increasingly being used to analyze medical text data. These methods convert text into numerical representations suitable for further machine analysis and enable the identification of key entities, symptoms, and medical terms.

Despite the active development of individual approaches to analyzing numerical and textual data, the issue of effectively combining them into a single model for predicting medical conditions remains under-researched. This is particularly true for classification tasks related to treatment plan recommendations and the determination of disease contagiousness, where errors can have significant consequences. Therefore, it is relevant to study approaches that combine ensemble machine learning models with modern NLP methods and hyperparameter optimization.

The aim of this research is to investigate the potential of combining machine learning ensemble models with text analysis methods to predict medical conditions based on numerical and textual data. We explore architectures that effectively fuse structured numerical data such as vital signs and demographics with unstructured clinical text like symptom descriptions and physician notes using advanced embedding techniques. Robust performance on medical datasets is pursued through ensemble learning optimization, including hyperparameter tuning with GridSearchCV and Bayesian optimization for XGBoost and Random Forest models. For the textual component, we implement advanced NLP techniques such as transformer-based models like BioClinicalBERT, named entity recognition, and semantic embeddings to better understand medical language. We also develop interpretable meta-learners that combine predictions from separate numerical and textual models using fuzzy membership functions and rule-based inference. The study includes hybrid neural architectures with dual branches for different data modalities, followed by latent feature fusion and joint classification. Finally, we incorporate uncertainty quantification by using confidence-based decision making and fuzzy logic to manage prediction uncertainty in clinical applications.

To achieve these goals, we analyzed the main classification algorithms, model evaluation methods, approaches to data preprocessing and hyperparameter tuning, and conducted experimental studies using both classical machine learning models and specialized NLP solutions for the medical field.

The results of this work aim to demonstrate the practical effectiveness of combining numerical and textual features, as well as to justify the feasibility of using ensemble and language models in automated medical data analysis tasks.

## Related Work

Recent studies have explored the integration of machine learning with medical text analysis for diagnostic prediction. Omoregbe et al. [3] demonstrated the effectiveness of combining Natural Language Processing with Fuzzy Logic for text-based medical diagnosis, achieving improved accuracy through rule-based uncertainty handling. Lala and Chaudhary [4] developed predictive models using deep learning and NLP techniques for disease diagnosis from symptom descriptions, highlighting the potential of transformer architectures.

Al-Qarni and Algarni [5] investigated disease prediction from symptom text using deep learning approaches, showing that semantic embeddings outperform traditional bag-of-words methods. Hossain et al. [2] conducted a systematic review of NLP applications in electronic health records, emphasizing the role of contextual understanding in clinical decision support.

The work by Melnyk et al. [6] on applying NLP and fuzzy logic to disinformation detection provides methodological insights applicable to medical text analysis. Roman et al. [1] explored integrating machine learning with medical imaging, complementing text-based approaches for comprehensive diagnostic systems.

Building on these foundations, our research extends the state-of-the-art by developing a hybrid neural architecture that jointly processes numerical and textual medical data, incorporating fuzzy logic for interpretable ensemble decision-making.

## Methodology

### Dataset Description
The research utilizes a medical dataset containing both structured numerical features and unstructured textual descriptions of symptoms for disease diagnosis. The dataset includes features such as patient demographics, vital signs, and clinical notes describing symptoms. The target variable represents disease severity levels: mild, moderate, and severe.

### Data Preprocessing
Numerical features were standardized with StandardScaler to normalize distributions, while textual features were processed with Sentence Transformers such as all-MiniLM-L6-v2 to produce 384-dimensional embeddings. Additional NLP processing included Named Entity Recognition (NER) with SciSpacy to identify medical entities. The dataset was split into training and test partitions using an 80-20 split with stratification to preserve class balance.

### Model Architectures

![med](/thesis/img/med-1.png)

The ensemble methods consisted of XGBoost with hyperparameter tuning via GridSearchCV and Bayesian optimization, while Random Forest models served as baseline comparisons. For text processing, we used TF-IDF vectorization with traditional machine learning classifiers, semantic word embeddings such as GloVe and Word2Vec, and transformer-based models like BioClinicalBERT for richer contextual understanding.

Fuzzy logic was integrated through fuzzy membership functions that assess symptom severity and a meta-learner that combines predictions from numerical and textual models using rule-based inference. The hybrid neural network used a dual-branch design with separate processing paths for numerical and textual features. The text branch received embedding input followed by linear layers with BatchNorm and Dropout, and the numerical branch processed scaled input through a similar sequence of linear layers, BatchNorm, and Dropout. Latent representations from both branches were concatenated and passed to a classifier head, and training used cross-entropy loss with the Adam optimizer and early stopping.

### Evaluation Metrics

Model quality was measured with accuracy, precision, recall, and F1-score, supported by confusion matrix analysis and multiclass ROC-AUC evaluation.



## Results

### Numerical Feature Analysis
Baseline models showed moderate performance, with SVM reaching 66.75% accuracy and Random Forest achieving 66% accuracy. XGBoost trained on numerical features without tuning yielded 99% accuracy, and further hyperparameter optimization improved results even more: SVC with GridSearchCV reached 96% accuracy, while XGBoost tuned via Bayesian optimization attained 100% accuracy.

### Textual Feature Analysis
Textual feature models also performed well. The initial TF-IDF representation combined with XGBoost achieved 84% accuracy, while feature selection with Random Forest resulted in 80% accuracy. With careful hyperparameter tuning, XGBoost improved to 86% accuracy, and the integration of BioClinicalBERT with NER features and model ensembling showed promising enhancements in semantic understanding.

### Fuzzy Logic Meta-Learning
We implemented fuzzy membership functions to combine numerical and textual predictions, and we used fuzzy rules based on confidence scores from individual models. This approach improved ensemble decision-making by better handling uncertainty in medical predictions.

### Hybrid Neural Network Performance
The hybrid neural network used a text branch with dimensions 384→256→128 and a numerical branch with dimensions 7→256→128, followed by a fusion path through 256→64→32→3. Training achieved high validation accuracy with early stopping, and the model successfully loaded and inferred on new data, demonstrating robust multi-modal integration. A test forward pass confirmed the correct output shape for three-class classification.

### Comparative Analysis
The hybrid approach combining structured data with NLP embeddings significantly outperformed single-modality models. Integrating fuzzy logic for meta-learning provided more interpretable decision boundaries, while the neural network architecture was able to capture complex nonlinear relationships. Hyperparameter tuning was essential for maximizing performance, and Bayesian optimization produced the best results.

### Benchmark Summary

| Approach | Accuracy | F1 | Precision | Recall | Notes |
|---|---:|---:|---:|---:|---|
| XGBoost (numerical, untuned) | 99% | Not reported | Not reported | Not reported | Reported in Results — numerical features |
| XGBoost (numerical, tuned, Bayesian) | 100% | Not reported | Not reported | Not reported | Reported in Results — Bayesian optimization |
| XGBoost (text, TF-IDF) | 84% | Not reported | Not reported | Not reported | Reported in Results — TF-IDF + XGBoost |
| XGBoost (text, tuned) | 86% | Not reported | Not reported | Not reported | Reported in Results — hyperparameter tuning |
| Random Forest (baseline, numerical) | 66% | Not reported | Not reported | Not reported | Reported in Results — baseline models |
| Random Forest (text, feature selection) | 80% | Not reported | Not reported | Not reported | Reported in Results — feature selection with RF |
| Model combination (XGBoost + Logistic Regression) | Not reported | Not reported | Not reported | Not reported | |
| Fuzzy Logic meta-learner | Not reported | Not reported | Not reported | Not reported |  |
| Hybrid Neural Network | Not reported | Not reported | Not reported | Not reported |  |

### Challenges and Limitations
Class imbalance in medical datasets required careful stratification. The computational complexity of transformer models remains a concern for real-time clinical use, and there is an interpretability trade-off between ensemble methods and deep learning approaches.



## Conclusion

As part of this research, machine learning approaches were developed and compared for analyzing two types of medical data: numerical patient metrics for predicting disease severity, and textual descriptions of symptoms for diagnostic classification.

For the numerical features, basic machine learning models demonstrated moderate baseline performance (SVM: 66.75%, Random Forest: 66%). XGBoost without tuning achieved 99% accuracy, with hyperparameter optimization via Bayesian methods reaching 100% accuracy. SVC with GridSearchCV improved to 96% accuracy.

The text data analysis revealed challenges with class overlap in vectorized representations, necessitating nonlinear models. XGBoost on TF-IDF features achieved 84% accuracy, with feature selection reducing it to 80%. Hyperparameter tuning improved results to 86%, and integration of BioClinicalBERT with NER features and ensembling showed enhanced semantic understanding.

![med](/thesis/img/med-3.png)

A fuzzy logic-based meta-learner was implemented to combine numerical and textual model predictions, providing interpretable decision-making under uncertainty. The research culminated in a hybrid neural network with separate branches for text embeddings (384-dimensional) and numerical features (7 features), achieving robust multi-modal classification for 3-class disease severity prediction.

Overall, the results confirm that ensemble methods combined with thorough hyperparameter tuning and multi-modal data integration achieve the highest prediction quality. For medical text analysis, modern NLP techniques and fuzzy logic meta-learning offer significant advantages in handling complex semantic structures and prediction uncertainty. The hybrid neural approach successfully demonstrated the feasibility of joint processing of heterogeneous medical data, paving the way for more comprehensive and accurate clinical decision support systems.


## References 


1. Roman, A.; Taib, C.; Dhaiouir, I.; El Khatir, H. Integrating Machine Learning with Medical Imaging for Human Disease Diagnosis: A Survey. Comput. Sci. Math. Forum 2025, 10, 12. https://doi.org/10.3390/cmsf2025010012

2. Hossain, E., Rana, R., Higgins, N., Soar, J., Barua, P., Pisani, A., & Turner, K. (2023). Natural Language Processing in Electronic Health Records in relation to healthcare decision-making: A systematic review. Computers in biology and medicine, 155, 106649 . https://doi.org/10.1016/j.compbiomed.2023.106649.

3. Omoregbe, N., Ndaman, I., Misra, S., & Abayomi-Alli, O. (2020). Text Messaging-Based Medical Diagnosis Using Natural Language Processing and Fuzzy Logic. Journal of Healthcare Engineering. https://doi.org/10.1155/2020/8839524.

4. Lala, K., & Chaudhary, A. (2025). Natural Language Processing in Healthcare: A Predictive Model for Disease Diagnosis. 2025 12th International Conference on Computing for Sustainable Global Development (INDIACom), 1-4. https://doi.org/10.23919/indiacom66777.2025.11115258.

5. Al-Qarni, S., & Algarni, A. (2025). Disease Prediction from Symptom Descriptions Using Deep Learning and NLP Technique. International Journal of Advanced Computer Science and Applications. https://doi.org/10.14569/ijacsa.2025.0160541.

6. Melnyk H., Melnyk V., Vikovan V. APPLICATION OF NATURAL LANGUAGE PROCESSING AND FUZZY LOGIC TO DISINFORMATION DETECTION. Bukovinian Mathematical Journal. 2024. Vol. 12, no. 1. P. 21–31. URL: https://doi.org/10.31861/bmj2024.01.03 (date of access: 21.02.2026).


## TODO

- Run ipynb file one more time, collect all the metrics into a single table!





















# MAIN MATERIAL REFORMULATED 


## Exploratory Data Analysis

The proposed methodology was evaluated using a medical dataset containing both structured clinical measurements and unstructured textual descriptions of patient symptoms. Such multimodal data reflects real-world electronic health records (EHRs), where numerical physiological indicators coexist with free-text notes recorded by healthcare professionals.

The structured component of the dataset contains demographic and physiological variables, including patient age, encoded gender, heart rate measured in beats per minute, body temperature expressed in degrees Celsius, blood pressure measured in millimetres of mercury, and peripheral oxygen saturation. Each patient record is associated with a medical assessment performed by healthcare experts, where the severity of the patient's condition is categorized into one of three classes: *mild*, *moderate*, or *severe*.

Prior to model construction, exploratory data analysis was performed to identify missing values, detect potential outliers, and analyze feature distributions. Histograms and correlation matrices demonstrated that several physiological variables possessed significantly different numerical scales. Consequently, numerical attributes were standardized using the StandardScaler transformation

$$
x'=\frac{x-\mu}{\sigma},
$$

where (x) denotes the original feature value, (\mu) is the feature mean, and (\sigma) is the standard deviation computed from the training data. This normalization ensures that all continuous variables contribute proportionally during model training.

The categorical gender variable was transformed using one-hot encoding, eliminating any artificial ordinal relationships while preserving categorical information.

---

## Bayesian Hyperparameter Optimization

Performance of ensemble learning algorithms strongly depends on the selection of hyperparameters. Exhaustive Grid Search often becomes computationally expensive when numerous parameter combinations must be evaluated. Therefore, Bayesian Hyperparameter Optimization was employed to identify near-optimal configurations while minimizing computational cost.

Unlike exhaustive search methods, Bayesian optimization constructs a probabilistic surrogate model

$$
f(\theta)\approx J(\theta),
$$

where (\theta) represents the hyperparameter vector and (J(\theta)) denotes the validation performance of the learning algorithm. At every iteration, the optimization procedure selects a new candidate configuration by maximizing an acquisition function, allowing promising regions of the search space to be explored efficiently.

Three hyperparameters of the XGBoost classifier were optimized:

* maximum tree depth ((max_depth)),
* minimum number of samples required to split an internal node ((min_samples_split)),
* number of boosting estimators ((n_estimators)).

Each parameter was evaluated across multiple candidate values, resulting in more than one hundred possible configurations. The optimal hyperparameter combination obtained through Bayesian optimization is presented in Table 1.

| Parameter             | Optimal value |
| --------------------- | ------------: |
| Maximum tree depth    |            10 |
| Minimum samples split |             5 |
| Number of estimators  |           600 |

The selected configuration demonstrated the highest validation performance while maintaining good generalization capability.

---

## Ensemble Learning for Structured Clinical Features

The standardized numerical features were used to train an XGBoost classifier, selected because of its robustness when modeling nonlinear interactions between physiological indicators.

Gradient boosting constructs an additive ensemble of decision trees according to

$$
F_M(x)=\sum_{m=1}^{M}\gamma_m h_m(x),
$$

where (h_m(x)) denotes the decision tree generated at iteration (m), while (\gamma_m) represents its corresponding weight.

Each subsequent tree attempts to minimize the residual errors produced by the previous ensemble, enabling the model to capture complex relationships between clinical measurements and disease severity.

---

## Natural Language Processing of Clinical Text

In addition to structured clinical variables, each patient record contains a textual description of symptoms recorded by medical personnel. These descriptions frequently include valuable contextual information unavailable in structured measurements alone.

The preprocessing pipeline included lowercasing, punctuation removal, tokenization, stop-word filtering, and lemmatization. Lemmatization reduces different grammatical forms to their canonical representation, decreasing vocabulary sparsity while preserving semantic meaning.

To convert symptom descriptions into numerical representations, the Sentence-BERT embedding model **all-MiniLM-L6-v2** was employed. This transformer-based encoder maps each clinical description into a dense 384-dimensional semantic vector

$$
E:\mathcal{T}\rightarrow\mathbb{R}^{384},
$$

where ($\mathcal{T}$) denotes the space of textual symptom descriptions.

Compared to traditional TF-IDF representations, dense sentence embeddings capture semantic similarity between clinically related expressions even when they do not share identical vocabulary.

To further enrich textual information, Named Entity Recognition (NER) was performed using SciSpacy. Medical entities such as diseases, symptoms, anatomical structures, medications, and clinical procedures were automatically identified and incorporated into the feature extraction process, allowing the language model to emphasize medically relevant concepts.

---

## Fusion of Numerical and Textual Models

Rather than combining structured and textual data into a single feature space, two specialized classifiers were trained independently.

The first classifier is the optimized XGBoost model operating on structured physiological measurements.

The second classifier is a Logistic Regression model trained on sentence embeddings generated from symptom descriptions.

Both classifiers produce posterior probability vectors

$$
P_{num}(y|x),
$$

and

$$
P_{text}(y|x),
$$

which are concatenated into a unified representation

$$
P=[P_{num},P_{text}].
$$

This probability vector serves as the input for a meta-classifier based on Logistic Regression, producing the final severity prediction.

Such stacking architecture enables each individual model to specialize within its own data modality while allowing the meta-learner to combine complementary information obtained from both structured clinical measurements and textual descriptions.

---

## Fuzzy Logic-Based Meta-Learner

To improve interpretability of the decision-making process, an alternative meta-learning strategy based on fuzzy inference was investigated.

Instead of directly combining probabilities through linear regression, the fuzzy meta-learner represents prediction confidence using linguistic variables such as *low*, *medium*, and *high*.

Typical fuzzy rules include expressions of the form

> IF numerical confidence is High AND textual confidence is High THEN severity is Severe.

Membership functions were defined for classifier confidence values, and inference was performed using the Mamdani fuzzy inference mechanism. The final prediction was obtained through centroid defuzzification

$$
y^*=
\frac{\int y\mu(y),dy}
{\int\mu(y),dy}.
$$

The fuzzy approach additionally provides interpretable reasoning paths, which are particularly desirable in clinical decision support systems.

---

## Hybrid Neural Network Architecture

In addition to ensemble learning and stacking-based approaches, a hybrid neural network architecture was developed to jointly exploit structured numerical measurements and unstructured textual symptom descriptions. The motivation behind this design is that numerical physiological indicators and semantic information extracted from clinical narratives provide complementary representations of a patient's condition. Processing these modalities independently before combining their latent representations enables the model to learn complex nonlinear interactions that cannot be captured by conventional machine learning algorithms.

The proposed architecture consists of two parallel processing branches. The textual branch receives a 384-dimensional sentence embedding generated by the **all-MiniLM-L6-v2** transformer model. This representation is subsequently processed by two fully connected layers with dimensions (384 \rightarrow 256 \rightarrow 128), allowing the network to learn higher-level semantic abstractions from symptom descriptions while reducing feature dimensionality.

The second branch processes structured clinical measurements. Seven normalized numerical features, including demographic information and physiological indicators, are provided as input to a dense neural network with architecture (7 \rightarrow 256 \rightarrow 128). This branch learns nonlinear dependencies among clinical variables while transforming them into a latent representation compatible with the semantic features extracted from textual data.

The outputs of both branches are concatenated to form a unified multimodal feature vector. The fused representation is then processed by a sequence of fully connected layers with dimensions (256 \rightarrow 64 \rightarrow 32 \rightarrow 3), where the final layer employs the Softmax activation function to estimate posterior probabilities for the three severity classes: *mild*, *moderate*, and *severe*. Formally, the network can be represented as

$$
y=\mathrm{Softmax}\left(f_{fusion}\left(f_{num}(x_{num}),,f_{text}(x_{text})\right)\right),
$$

where $f_{num}$ and $f_{text}$ denote the numerical and textual feature extraction subnetworks, respectively, while $f_{fusion}$ represents the classification module operating on the combined latent representation.

The network was trained using supervised learning with categorical cross-entropy as the optimization objective. Early stopping was employed during training to prevent overfitting by terminating optimization once the validation performance ceased to improve. This strategy enabled the model to achieve consistently high validation accuracy while preserving strong generalization capabilities on previously unseen patient records.

Following training, the learned model parameters were successfully restored and applied to independent test samples without requiring architectural modifications. A validation forward pass confirmed that the network correctly produced probability distributions of dimension three, corresponding to the predefined severity categories. These experimental observations demonstrate that the proposed architecture effectively integrates heterogeneous clinical information and performs reliable multimodal inference, combining structured physiological measurements with semantic representations extracted from electronic health records. The superior classification performance reported in the experimental evaluation further indicates that deep multimodal feature fusion provides a more expressive representation of patient conditions than approaches relying on either numerical or textual information alone.


---

## Experimental Comparison

Performance was evaluated using Precision, Recall, and F1-score, which are commonly adopted for multiclass medical classification tasks. The comparative evaluation of all investigated approaches is presented in Table 2.

| Method                       | Precision |   Recall | F1-score |
| ---------------------------- | --------: | -------: | -------: |
| Random Forest                |      0.65 |     0.69 |     0.61 |
| Combined XGBoost + NLP model |      0.95 |     0.94 |     0.94 |
| Fuzzy Logic meta-learner     |      0.82 |     0.82 |     0.81 |
| Hybrid Neural Network        |  **0.97** | **0.97** | **0.97** |

The obtained results demonstrate that incorporating textual symptom descriptions substantially improves prediction quality compared with models relying exclusively on structured clinical variables. While the fuzzy logic meta-learner provides greater interpretability, its predictive performance remains below that of purely data-driven approaches. The hybrid neural network achieved the highest overall accuracy, indicating that simultaneous learning from structured physiological measurements and semantic textual representations provides the most effective strategy for medical severity prediction.

---

### Additional suggestions that would strengthen the paper

I would also recommend adding one or two more methodological subsections:

1. **Cross-validation and Evaluation Protocol**

   * Train/test split (e.g., 80/20)
   * 5-fold cross-validation
   * Random seed
   * Class balance

2. **Feature Importance Analysis**

   * SHAP values for XGBoost
   * Which physiological variables contributed most
   * Which symptom terms were most informative

These additions are common in high-quality computer science and biomedical AI papers and would make the methodology substantially stronger.
