# Text Summarization Case Studies

Generated: 2026-10-22 18:27
Number of cases: 3

---

## Case Study 1: HIGH Performance

**Document ID:** 296
**Word Count:** 256
**Sentence Count:** 28
**Average ROUGE-1:** 0.6216

### Original Text (excerpt)

PREOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,POSTOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,PROCEDURE PERFORMED:,  Needle-localized excisional biopsy, left breast.,ANESTHESIA:,  Local with sedation.,SPECIMEN:  ,Left breast with specimen mammogram.,COMPLICATIONS:,  None.,HISTORY: , The patient is a 71-year-old black female who had a routine mammogram, which demonstrated suspicious microcalcifications in the left breast.  She had no palpable mass on physical exam.  She does have significant family history with two daughters having breast cancer.  The patient also has a history of colon cancer.  A surgical biopsy was recommended and she was scheduled electively.,PROCEDURE:,  After proper informed consent was obtained, she was placed in the operative suite.  This occurred after undergoing preoperative needle localization.  She was placed in the operating room in the supine position.  She was given sedation by the Anesthesia Department.  The left breast was prepped and draped in the usual sterile fashion.  The skin was infiltrated with local and a curvilinear incision was made in the left lower outer quadrant.  The breast tissue was grasped with Allis clamps and a core of tissue was removed around the localization wire.  There were some fibrocystic changes noted.  The specimen was then completely removed and was sent to Radiology for mammogram.  The calcifications were seen in specimen per Dr. X.  Meticulous hemostasis was achieved with electrocautery.  The area was irrigated and suctioned.,The aspirant was clear.  The skin was then reapproximated using #4-0 undyed Vicryl in a running subcuticular fashion.  Steri-Strips and sterile dressing on the patient's bra were applied.  The patient tolerated the procedure well and was transferred to recovery room in stable condition.

### Reference Summary

 Needle-localized excisional biopsy, left breast.  The patient is a 71-year-old black female who had a routine mammogram, which demonstrated suspicious microcalcifications in the left breast.  She had no palpable mass on physical exam.  She does have significant family history with two daughters having breast cancer.

### Method Summaries and Scores

#### lead (ROUGE-1: 0.7805)

PREOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,POSTOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,PROCEDURE PERFORMED:,  Needle-localized excisional biopsy, left breast.,ANESTHESIA:,  Local with sedation.,SPECIMEN:  ,Left breast with specimen mammogram.,COMPLICATIONS:,  None.,HISTORY: , The patient is a 71-year-old black female who had a routine mammogram, which demonstrated suspicious microcalcifications in the left breast. She had no palpable mass on physical exam. She does have significant family history with two daughters having breast cancer.

#### random (ROUGE-1: 0.6612)

PREOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,POSTOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,PROCEDURE PERFORMED:,  Needle-localized excisional biopsy, left breast.,ANESTHESIA:,  Local with sedation.,SPECIMEN:  ,Left breast with specimen mammogram.,COMPLICATIONS:,  None.,HISTORY: , The patient is a 71-year-old black female who had a routine mammogram, which demonstrated suspicious microcalcifications in the left breast. She does have significant family history with two daughters having breast cancer. There were some fibrocystic changes noted.

#### lexrank (ROUGE-1: 0.6341)

PREOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,POSTOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,PROCEDURE PERFORMED:,  Needle-localized excisional biopsy, left breast.,ANESTHESIA:,  Local with sedation.,SPECIMEN:  ,Left breast with specimen mammogram.,COMPLICATIONS:,  None.,HISTORY: , The patient is a 71-year-old black female who had a routine mammogram, which demonstrated suspicious microcalcifications in the left breast. She had no palpable mass on physical exam. The left breast was prepped and draped in the usual sterile fashion.

#### textrank (ROUGE-1: 0.5161)

PREOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,POSTOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,PROCEDURE PERFORMED:,  Needle-localized excisional biopsy, left breast.,ANESTHESIA:,  Local with sedation.,SPECIMEN:  ,Left breast with specimen mammogram.,COMPLICATIONS:,  None.,HISTORY: , The patient is a 71-year-old black female who had a routine mammogram, which demonstrated suspicious microcalcifications in the left breast. The patient also has a history of colon cancer. The left breast was prepped and draped in the usual sterile fashion.

#### lsa (ROUGE-1: 0.5161)

PREOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,POSTOPERATIVE DIAGNOSIS:,  Suspicious microcalcifications, left breast.,PROCEDURE PERFORMED:,  Needle-localized excisional biopsy, left breast.,ANESTHESIA:,  Local with sedation.,SPECIMEN:  ,Left breast with specimen mammogram.,COMPLICATIONS:,  None.,HISTORY: , The patient is a 71-year-old black female who had a routine mammogram, which demonstrated suspicious microcalcifications in the left breast. The patient also has a history of colon cancer. The left breast was prepped and draped in the usual sterile fashion.

### Analysis

**Overall Performance:** High (avg ROUGE-1: 0.622)

**Best Method:** lead (ROUGE-1: 0.780)
**Worst Method:** lsa (ROUGE-1: 0.516)
**Performance Spread:** 0.264

**Key Observations:**
- All methods performed well, indicating the document is well-suited for extractive summarization
- The reference summary likely contains sentences very similar to the source
- Strong agreement: all methods selected the same first sentence

---

## Case Study 2: MEDIUM Performance

**Document ID:** 379
**Word Count:** 230
**Sentence Count:** 16
**Average ROUGE-1:** 0.1996

### Original Text (excerpt)

ABNORMAL CHOLESTEROL RESULT LETTER,Recently you had a cholesterol test done.  The cholesterol levels were abnormal.  These are usually associated with increased risk for stroke and heart attack.  I am writing this letter to you to let you know that your levels are high enough that I think intervention is the next best step.  I would like you to make an appointment, if you are interested in treatment for this.  There are several treatment options available at this time.  Diet is one of the options, although there is limited reduction in total cholesterol and LDL cholesterol with dieting.  Most of the time under strict diet patients can achieve a 15% reduction in cholesterol.  If your cholesterol levels are moderate to severely elevated, usually diet is not the first line of therapy.  If you are diabetic or have hypertension these two also increase your risk with the combination of hypercholesterolemia.  Most of the time cholesterol that is elevated is from your genetic background and is independent of diet.  New research shows that treatment of high cholesterol can decrease your risk of developing Alzheimer’s disease.,Again, I am recommending at least we discuss medical treatment for hypercholesterolemia.  Please consider calling for a followup appointment concerning your cholesterol levels.  If you should have any further questions, please do not hesitate to contact me.  I am enclosing a copy of your lab results for your review.

### Reference Summary

 Letter to the patient to let him know about his abnormal cholesterol test results.

### Method Summaries and Scores

#### lexrank (ROUGE-1: 0.2917)

ABNORMAL CHOLESTEROL RESULT LETTER,Recently you had a cholesterol test done. The cholesterol levels were abnormal. If your cholesterol levels are moderate to severely elevated, usually diet is not the first line of therapy.

#### lead (ROUGE-1: 0.2857)

ABNORMAL CHOLESTEROL RESULT LETTER,Recently you had a cholesterol test done. The cholesterol levels were abnormal. These are usually associated with increased risk for stroke and heart attack.

#### lsa (ROUGE-1: 0.1739)

The cholesterol levels were abnormal. I would like you to make an appointment, if you are interested in treatment for this. Please consider calling for a followup appointment concerning your cholesterol levels.

#### textrank (ROUGE-1: 0.1667)

The cholesterol levels were abnormal. If your cholesterol levels are moderate to severely elevated, usually diet is not the first line of therapy. Please consider calling for a followup appointment concerning your cholesterol levels.

#### random (ROUGE-1: 0.0800)

If you are diabetic or have hypertension these two also increase your risk with the combination of hypercholesterolemia. New research shows that treatment of high cholesterol can decrease your risk of developing Alzheimer’s disease.,Again, I am recommending at least we discuss medical treatment for hypercholesterolemia. If you should have any further questions, please do not hesitate to contact me.

### Analysis

**Overall Performance:** Medium (avg ROUGE-1: 0.200)

**Best Method:** lexrank (ROUGE-1: 0.292)
**Worst Method:** random (ROUGE-1: 0.080)
**Performance Spread:** 0.212

**Key Observations:**
- Methods showed moderate performance with some variation
- Different approaches captured different aspects of the content
- Partial agreement: some methods converged on similar content

---

## Case Study 3: LOW Performance

**Document ID:** 264
**Word Count:** 204
**Sentence Count:** 26
**Average ROUGE-1:** 0.0000

### Original Text (excerpt)

CHIEF COMPLAINT:, Followup on diabetes mellitus, hypercholesterolemia, and sinusitis. ,SUBJECTIVE:,  A 70-year-old female who was diagnosed with diabetes mellitus last fall.  She has been checking her Accu-Cheks generally once a day, and they range from 82 to a high of 132.  She feels well.  She walks 1-2 miles most days.  If the weather is bad, they go to a local mall.  Otherwise they walk outside.  She does complain of some sinus congestion and drainage for the last several days.  She is up-to-date on her mammogram that she had at the Baylis Building.  She sees Dr. Cheng for her gynecological care.  She is a nonsmoker.  Denies abdominal pain, nausea, vomiting, diarrhea, constipation, blood in her urine, blood in her stools.  She has nocturia x 1.  Denies swelling in her ankles.  She checks her feet regularly., ,PAST MEDICAL HISTORY:,  Refer to chart.,MEDICATIONS:, Refer to chart.,ALLERGIES:, Refer to chart.,PHYSICAL EXAMINATION: ,Vitals: Wt:  185 B/P: 142/70.  When she checks it at the mall and other places, it is usually about 120/56. T: 96.5 P:  84 and regular.,General: A 70-year-old female who does not appear to be in acute distress.  ,HEENT: She has frontal and maxillary sinus tenderness on the right to palpation.  The right TM is slightly dull.

### Reference Summary

 Sample progress note - Gen Med.

### Method Summaries and Scores

#### lead (ROUGE-1: 0.0000)

CHIEF COMPLAINT:, Followup on diabetes mellitus, hypercholesterolemia, and sinusitis. ,SUBJECTIVE:,  A 70-year-old female who was diagnosed with diabetes mellitus last fall. She has been checking her Accu-Cheks generally once a day, and they range from 82 to a high of 132.

#### random (ROUGE-1: 0.0000)

She is up-to-date on her mammogram that she had at the Baylis Building. When she checks it at the mall and other places, it is usually about 120/56. T: 96.5 P:  84 and regular.,General: A 70-year-old female who does not appear to be in acute distress.

#### textrank (ROUGE-1: 0.0000)

,SUBJECTIVE:,  A 70-year-old female who was diagnosed with diabetes mellitus last fall. She does complain of some sinus congestion and drainage for the last several days. T: 96.5 P:  84 and regular.,General: A 70-year-old female who does not appear to be in acute distress.

#### lsa (ROUGE-1: 0.0000)

,SUBJECTIVE:,  A 70-year-old female who was diagnosed with diabetes mellitus last fall. She does complain of some sinus congestion and drainage for the last several days. Denies swelling in her ankles.

#### lexrank (ROUGE-1: 0.0000)

CHIEF COMPLAINT:, Followup on diabetes mellitus, hypercholesterolemia, and sinusitis. ,SUBJECTIVE:,  A 70-year-old female who was diagnosed with diabetes mellitus last fall. T: 96.5 P:  84 and regular.,General: A 70-year-old female who does not appear to be in acute distress.

### Analysis

**Overall Performance:** Low (avg ROUGE-1: 0.000)

**Best Method:** lead (ROUGE-1: 0.000)
**Worst Method:** lexrank (ROUGE-1: 0.000)
**Performance Spread:** 0.000

**Key Observations:**
- All methods struggled with this document
- Possible causes: abstractive reference summary, complex content structure, or domain-specific language
- Partial agreement: some methods converged on similar content

---
