# Combining ensemble machine learning models with text data analysis for medical conditions prediction

https://docs.google.com/document/d/1Pecw5_4UB3Qb6huxT0MDV0U5n1x7ucJYvcRC5_p9Yio/edit?tab=t.0

## Abstract





## Introduction 



## Related Work

The current stage of information technology development is characterized by a rapid increase in the volume of medical data and growing demands for speed and accuracy in clinical decision-making. Traditional approaches to analyzing medical information, based on manual interpretation of numerical indicators and textual descriptions, are becoming insufficiently effective given the large number of patients, complex data structures, and the need to minimize human error. In this context, the application of machine learning and artificial intelligence methods for automated analysis of medical data and decision support is of great relevance [1, 2, 3, 4].

A distinctive feature of medical information systems is the heterogeneity of the data they process. On the one hand, this includes structured numerical indicators of a patient’s condition, such as body temperature, blood oxygen levels, blood pressure, or heart rate. On the other hand, a significant portion of important information is contained in unstructured text—descriptions of symptoms, medical history, doctors’ conclusions, and clinical notes. The effective use of only one type of data often fails to fully capture the clinical picture of a disease, necessitating the combination of numerical and textual features [5].

Ensemble machine learning methods, particularly Random Forest and gradient boosting on decision trees, have proven to be powerful tools for handling complex nonlinear relationships and noisy data. They demonstrate high resistance to overfitting and enable significantly better results compared to baseline models. At the same time, natural language processing (NLP) methods are increasingly being used to analyze medical text data. These methods convert text into numerical representations suitable for further machine analysis and enable the identification of key entities, symptoms, and medical terms.

Despite the active development of individual approaches to analyzing numerical and textual data, the issue of effectively combining them into a single model for predicting medical conditions remains under-researched. This is particularly true for classification tasks related to treatment plan recommendations and the determination of disease contagiousness, where errors can have significant consequences. Therefore, it is relevant to study approaches that combine ensemble machine learning models with modern NLP methods and hyperparameter optimization [6].

## Methodology



## Results



## Conclusion

As part of this research, machine learning approaches were developed and compared for analyzing two types of medical data: numerical patient metrics for predicting or recommending a treatment plan, and textual descriptions of symptoms for determining the contagiousness of a disease.

For the task involving numerical features, basic machine learning models, specifically SVM and Random Forest, demonstrated moderate accuracy (66.75% and 66%, respectively), indicating the limited ability of these approaches to capture complex nonlinear relationships in the data without additional tuning. At the same time, the use of XGBoost in combination with data preprocessing allowed for significantly higher results—up to 99% accuracy. Further systematic hyperparameter tuning helped improve model quality: for SVC using GridSearchCV, accuracy rose to 96%, and for XGBoost, after Bayesian optimization, the maximum accuracy of 100% was achieved.

The text data analysis section demonstrates that, in the vector representation of symptoms, there is significant class overlap, which complicates their linear separation and suggests the use of nonlinear models or specialized linguistic representations. On vectorized text features, the XGBoost model achieved 84% accuracy; applying feature selection using Random Forest yielded a result of 80%; and further tuning of XGBoost hyperparameters allowed the accuracy to be increased to 86%. In addition, an approach utilizing BioClinicalBERT, additional NER features, and model ensembling was implemented, enabling the integration of information from textual and numerical sources.

Overall, the results confirm that, within the scope of this work, the highest prediction quality is achieved by boosting models combined with thorough data preprocessing and systematic hyperparameter tuning. For medical text analysis tasks, the most promising approaches are modern NLP methods and model ensembling, which allow for more effective consideration of the complex semantic structure of symptoms.


## References 


1. Roman, A.; Taib, C.; Dhaiouir, I.; El Khatir, H. Integrating Machine Learning with Medical Imaging for Human Disease Diagnosis: A Survey. Comput. Sci. Math. Forum 2025, 10, 12. https://doi.org/10.3390/cmsf2025010012

2. Hossain, E., Rana, R., Higgins, N., Soar, J., Barua, P., Pisani, A., & Turner, K. (2023). Natural Language Processing in Electronic Health Records in relation to healthcare decision-making: A systematic review. Computers in biology and medicine, 155, 106649 . https://doi.org/10.1016/j.compbiomed.2023.106649.

3. Omoregbe, N., Ndaman, I., Misra, S., & Abayomi-Alli, O. (2020). Text Messaging-Based Medical Diagnosis Using Natural Language Processing and Fuzzy Logic. Journal of Healthcare Engineering. https://doi.org/10.1155/2020/8839524.

4. Lala, K., & Chaudhary, A. (2025). Natural Language Processing in Healthcare: A Predictive Model for Disease Diagnosis. 2025 12th International Conference on Computing for Sustainable Global Development (INDIACom), 1-4. https://doi.org/10.23919/indiacom66777.2025.11115258.

5. Al-Qarni, S., & Algarni, A. (2025). Disease Prediction from Symptom Descriptions Using Deep Learning and NLP Technique. International Journal of Advanced Computer Science and Applications. https://doi.org/10.14569/ijacsa.2025.0160541.

6. Melnyk H., Melnyk V., Vikovan V. APPLICATION OF NATURAL LANGUAGE PROCESSING AND FUZZY LOGIC TO DISINFORMATION DETECTION. Bukovinian Mathematical Journal. 2024. Vol. 12, no. 1. P. 21–31. URL: https://doi.org/10.31861/bmj2024.01.03 (date of access: 21.02.2026).


