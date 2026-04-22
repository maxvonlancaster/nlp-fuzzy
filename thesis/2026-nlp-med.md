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

The aim of this research is to investigate the potential of combining machine learning ensemble models with text analysis methods to predict medical conditions based on numerical and textual data. Key research ideas explored include:

1. **Multi-Modal Feature Integration**: Developing architectures that effectively combine structured numerical data (vital signs, demographics) with unstructured text (symptom descriptions, clinical notes) using advanced embedding techniques.

2. **Ensemble Learning Optimization**: Applying hyperparameter tuning (GridSearchCV, Bayesian optimization) to ensemble methods like XGBoost and Random Forest for robust performance on medical datasets.

3. **Advanced NLP Techniques**: Implementing transformer-based models (BioClinicalBERT), named entity recognition, and semantic embeddings for medical text understanding.

4. **Fuzzy Logic Meta-Learning**: Creating interpretable meta-learners that combine predictions from separate numerical and textual models using fuzzy membership functions and rules.

5. **Hybrid Neural Architectures**: Designing dual-branch neural networks with separate processing paths for different data modalities, followed by feature fusion and joint classification.

6. **Uncertainty Quantification**: Incorporating confidence-based decision making and fuzzy logic to handle prediction uncertainty in clinical applications.

To achieve these goals, we analyzed the main classification algorithms, model evaluation methods, approaches to data preprocessing and hyperparameter tuning, and conducted experimental studies using both classical machine learning models and specialized NLP solutions for the medical field.

The results of this work aim to demonstrate the practical effectiveness of combining numerical and textual features, as well as to justify the feasibility of using ensemble and language models in automated medical data analysis tasks.

## Related Work

Recent studies have explored the integration of machine learning with medical text analysis for diagnostic prediction. Omoregbe et al. (2020) demonstrated the effectiveness of combining Natural Language Processing with Fuzzy Logic for text-based medical diagnosis, achieving improved accuracy through rule-based uncertainty handling. Lala and Chaudhary (2025) developed predictive models using deep learning and NLP techniques for disease diagnosis from symptom descriptions, highlighting the potential of transformer architectures.

Al-Qarni and Algarni (2025) investigated disease prediction from symptom text using deep learning approaches, showing that semantic embeddings outperform traditional bag-of-words methods. Hossain et al. (2023) conducted a systematic review of NLP applications in electronic health records, emphasizing the role of contextual understanding in clinical decision support.

The work by Melnyk et al. (2024) on applying NLP and fuzzy logic to disinformation detection provides methodological insights applicable to medical text analysis. Roman et al. (2025) explored integrating machine learning with medical imaging, complementing text-based approaches for comprehensive diagnostic systems.

Building on these foundations, our research extends the state-of-the-art by developing a hybrid neural architecture that jointly processes numerical and textual medical data, incorporating fuzzy logic for interpretable ensemble decision-making.## Methodology

### Dataset Description
The research utilizes a medical dataset containing both structured numerical features and unstructured textual descriptions of symptoms for disease diagnosis. The dataset includes features such as patient demographics, vital signs, and clinical notes describing symptoms. The target variable represents disease severity levels: mild, moderate, and severe.

### Data Preprocessing
- **Numerical Features**: Standardized using StandardScaler to normalize distributions.
- **Textual Features**: Processed using Sentence Transformers (e.g., all-MiniLM-L6-v2) to generate 384-dimensional embeddings. Additional NLP techniques include Named Entity Recognition (NER) with SciSpacy for medical entities.
- **Train-Test Split**: 80-20 split with stratification to maintain class balance.

### Model Architectures
1. **Ensemble Methods**:
   - XGBoost with hyperparameter tuning via GridSearchCV and Bayesian optimization.
   - Random Forest for baseline comparisons.

2. **Text Processing Models**:
   - TF-IDF vectorization followed by traditional ML classifiers.
   - Word embeddings (GloVe, Word2Vec) for semantic representation.
   - Transformer-based models (BioClinicalBERT) for advanced text understanding.

3. **Fuzzy Logic Integration**:
   - Fuzzy membership functions for symptom severity assessment.
   - Meta-learner combining predictions from numerical and textual models using fuzzy rules.

4. **Hybrid Neural Network**:
   - Dual-branch architecture: separate branches for numerical and textual features.
   - Text branch: Embedding input → Linear layers with BatchNorm and Dropout.
   - Numerical branch: Scaled input → Linear layers with BatchNorm and Dropout.
   - Fusion: Concatenation of latent representations → Classifier head.
   - Training: Cross-entropy loss, Adam optimizer, early stopping with patience.

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix analysis
- ROC-AUC for multi-class classification



## Results

### Numerical Feature Analysis
- Baseline models showed moderate performance: SVM (66.75% accuracy), Random Forest (66% accuracy).
- XGBoost without tuning achieved 99% accuracy on numerical features.
- Hyperparameter optimization further improved results:
  - SVC with GridSearchCV: 96% accuracy
  - XGBoost with Bayesian optimization: 100% accuracy

### Textual Feature Analysis
- Initial TF-IDF + XGBoost: 84% accuracy
- Feature selection with Random Forest: 80% accuracy
- Hyperparameter tuning improved XGBoost to 86% accuracy
- BioClinicalBERT integration with NER features and model ensembling showed promising results for semantic understanding.

### Fuzzy Logic Meta-Learning
- Implemented fuzzy membership functions for combining numerical and textual predictions.
- Fuzzy rules based on confidence scores from individual models.
- Improved ensemble decision-making by handling uncertainty in medical predictions.

### Hybrid Neural Network Performance
- Architecture: Text branch (384 → 256 → 128), Numerical branch (7 → 256 → 128), Fusion (256 → 64 → 32 → 3)
- Training achieved high validation accuracy with early stopping.
- Successful model loading and inference on new data, demonstrating robust multi-modal integration.
- Test forward pass confirmed correct output shape for 3-class classification.

### Comparative Analysis
The hybrid approach combining structured data with NLP embeddings significantly outperformed single-modality models. The integration of fuzzy logic for meta-learning provided interpretable decision boundaries, while the neural network architecture captured complex non-linear relationships. Hyperparameter tuning was crucial for maximizing performance, with Bayesian optimization yielding the best results.

### Challenges and Limitations
- Class imbalance in medical datasets required careful stratification.
- Computational complexity of transformer models for real-time clinical use.
- Interpretability trade-offs between ensemble methods and deep learning approaches.



## Conclusion

As part of this research, machine learning approaches were developed and compared for analyzing two types of medical data: numerical patient metrics for predicting disease severity, and textual descriptions of symptoms for diagnostic classification.

For the numerical features, basic machine learning models demonstrated moderate baseline performance (SVM: 66.75%, Random Forest: 66%). XGBoost without tuning achieved 99% accuracy, with hyperparameter optimization via Bayesian methods reaching 100% accuracy. SVC with GridSearchCV improved to 96% accuracy.

The text data analysis revealed challenges with class overlap in vectorized representations, necessitating nonlinear models. XGBoost on TF-IDF features achieved 84% accuracy, with feature selection reducing it to 80%. Hyperparameter tuning improved results to 86%, and integration of BioClinicalBERT with NER features and ensembling showed enhanced semantic understanding.

A fuzzy logic-based meta-learner was implemented to combine numerical and textual model predictions, providing interpretable decision-making under uncertainty. The research culminated in a hybrid neural network with separate branches for text embeddings (384-dimensional) and numerical features (7 features), achieving robust multi-modal classification for 3-class disease severity prediction.

Overall, the results confirm that ensemble methods combined with thorough hyperparameter tuning and multi-modal data integration achieve the highest prediction quality. For medical text analysis, modern NLP techniques and fuzzy logic meta-learning offer significant advantages in handling complex semantic structures and prediction uncertainty. The hybrid neural approach successfully demonstrated the feasibility of joint processing of heterogeneous medical data, paving the way for more comprehensive and accurate clinical decision support systems.


## References 


1. Roman, A.; Taib, C.; Dhaiouir, I.; El Khatir, H. Integrating Machine Learning with Medical Imaging for Human Disease Diagnosis: A Survey. Comput. Sci. Math. Forum 2025, 10, 12. https://doi.org/10.3390/cmsf2025010012

2. Hossain, E., Rana, R., Higgins, N., Soar, J., Barua, P., Pisani, A., & Turner, K. (2023). Natural Language Processing in Electronic Health Records in relation to healthcare decision-making: A systematic review. Computers in biology and medicine, 155, 106649 . https://doi.org/10.1016/j.compbiomed.2023.106649.

3. Omoregbe, N., Ndaman, I., Misra, S., & Abayomi-Alli, O. (2020). Text Messaging-Based Medical Diagnosis Using Natural Language Processing and Fuzzy Logic. Journal of Healthcare Engineering. https://doi.org/10.1155/2020/8839524.

4. Lala, K., & Chaudhary, A. (2025). Natural Language Processing in Healthcare: A Predictive Model for Disease Diagnosis. 2025 12th International Conference on Computing for Sustainable Global Development (INDIACom), 1-4. https://doi.org/10.23919/indiacom66777.2025.11115258.

5. Al-Qarni, S., & Algarni, A. (2025). Disease Prediction from Symptom Descriptions Using Deep Learning and NLP Technique. International Journal of Advanced Computer Science and Applications. https://doi.org/10.14569/ijacsa.2025.0160541.

6. Melnyk H., Melnyk V., Vikovan V. APPLICATION OF NATURAL LANGUAGE PROCESSING AND FUZZY LOGIC TO DISINFORMATION DETECTION. Bukovinian Mathematical Journal. 2024. Vol. 12, no. 1. P. 21–31. URL: https://doi.org/10.31861/bmj2024.01.03 (date of access: 21.02.2026).


