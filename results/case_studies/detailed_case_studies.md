# Text Summarization Case Studies

Generated: 2026-04-11 20:53
Number of cases: 3

---

## Case Study 1: HIGH Performance

**Document ID:** 53
**Word Count:** 174
**Sentence Count:** 28
**Average ROUGE-1:** 0.6046

### Original Text (excerpt)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition.  The right groin was prepped and draped in routine fashion.  An arterial sheath was inserted into the right femoral artery.,Left and right coronary arteries were studied with a 6FL4 and 6FR4 Judkins catheters respectively.  Cine coronary angiograms were done in multiple views.,Left heart catheterization was done using the 6-French pigtail catheter.  Appropriate pressures were obtained before and after the left ventriculogram, which was done in the RAO view.,At the end of the procedure, the femoral catheter was removed and Angio-Seal was applied without any complications.,FINDINGS:,1.  LV is normal in size and shape with good contractility, EF of 60%.,2.  LMCA normal.,3.  LAD has 20% to 30% stenosis at the origin.,4.  LCX is normal.,5.  RCA is dominant and normal.,RECOMMENDATIONS: , Medical management, diet, and exercise.  Aspirin 81 mg p.o. daily, p.r.n. nitroglycerin for chest pain.  Follow up in the clinic.

### Reference Summary

 Chest pain and non-Q-wave MI with elevation of troponin I only.  Left heart catheterization, left ventriculography, and left and right coronary arteriography.

### Method Summaries and Scores

#### textrank (ROUGE-1: 0.6957)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition. LMCA normal.,3. LCX is normal.,5.

#### random (ROUGE-1: 0.6234)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition. RCA is dominant and normal.,RECOMMENDATIONS: , Medical management, diet, and exercise. daily, p.r.n.

#### lead (ROUGE-1: 0.4948)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition. The right groin was prepped and draped in routine fashion. An arterial sheath was inserted into the right femoral artery.,Left and right coronary arteries were studied with a 6FL4 and 6FR4 Judkins catheters respectively.

### Analysis

**Overall Performance:** High (avg ROUGE-1: 0.605)

**Best Method:** textrank (ROUGE-1: 0.696)
**Worst Method:** lead (ROUGE-1: 0.495)
**Performance Spread:** 0.201

**Key Observations:**
- All methods performed well, indicating the document is well-suited for extractive summarization
- The reference summary likely contains sentences very similar to the source
- Strong agreement: all methods selected the same first sentence

---

## Case Study 2: MEDIUM Performance

**Document ID:** 51
**Word Count:** 335
**Sentence Count:** 34
**Average ROUGE-1:** 0.1923

### Original Text (excerpt)

CC:, Progressive lower extremity weakness.,HX: ,This 52y/o RHF had a h/o right frontal glioblastoma multiforme (GBM) diagnosed by brain biopsy/partial resection, on 1/15/1991. She had been healthy until 1/6/91, when she experienced a generalized tonic-clonic type seizure during the night. She subsequently underwent an MRI brain scan and was found to have a right frontal enhancing lesion in the mesial aspect of the right frontal lobe at approximately the level of the coronal suture. There was minimal associated edema and no mass effect. Following extirpation of the tumor mass, she underwent radioactive Iodine implantation and 6020cGy radiation therapy in 35 fractions. In 11/91 she received BCNU and Procarbazine chemotherapy protocols. This was followed by four courses of 5FU/Carboplatin (3/92, 6/92, 9/92 ,10/92) chemotherapy.,On 10/12/92 she presented for her 4th course of 5FU/Carboplatin and complained of non-radiating dull low back pain, and proximal lower extremity weakness, but was still able to ambulate. She denied any bowel/bladder difficulty.,PMH: ,s/p oral surgery for wisdom tooth extraction.,FHX/SHX: ,1-2 ppd cigarettes. rare ETOH use. Father died of renal CA.,MEDS: ,Decadron 12mg/day.,EXAM: ,Vitals unremarkable.,MS: Unremarkable.,Motor: 5/5 BUE, LE: 4+/5- prox, 5/5 distal to hips. Normal tone and muscle bulk.,Sensory: No deficits appreciated.,Coord: Unremarkable.,Station: No mention in record of being tested.,Gait: Mild difficulty climbing stairs.,Reflexes: 1+/1+ throughout and symmetric. Plantar responses were down-going bilaterally.,INITIAL IMPRESSION:, Steroid myopathy. Though there was enough of a suspicion of "drop" metastasis that an MRI of the L-spine was obtained.,COURSE:, The MRI L-spine revealed fine linear enhancement along the dorsal aspect of the conus medullaris, suggestive of subarachnoid seeding of tumor. No focal mass or cord compression was visualized. CSF examination revealed: 19RBC, 22WBC, 17 Lymphocytes, and 5 histiocytes, Glucose 56, Protein 150. Cytology (negative). The patient was discharged home on 10/17/92, but experienced worsening back pain and lower extremity weakness and became predominantly wheelchair bound within 4 months. She was last seen on 3/3/93 and showed signs of worsening weakness (left hemiplegia: R > L) as her tumor grew and spread. She then entered a hospice.

### Reference Summary

 MRI L-spine - History of progressive lower extremity weakness, right frontal glioblastoma with lumbar subarachnoid seeding.

### Method Summaries and Scores

#### random (ROUGE-1: 0.2154)

CC:, Progressive lower extremity weakness.,HX: ,This 52y/o RHF had a h/o right frontal glioblastoma multiforme (GBM) diagnosed by brain biopsy/partial resection, on 1/15/1991. In 11/91 she received BCNU and Procarbazine chemotherapy protocols. Plantar responses were down-going bilaterally.,INITIAL IMPRESSION:, Steroid myopathy.

#### lead (ROUGE-1: 0.1818)

CC:, Progressive lower extremity weakness.,HX: ,This 52y/o RHF had a h/o right frontal glioblastoma multiforme (GBM) diagnosed by brain biopsy/partial resection, on 1/15/1991. She had been healthy until 1/6/91, when she experienced a generalized tonic-clonic type seizure during the night. She subsequently underwent an MRI brain scan and was found to have a right frontal enhancing lesion in the mesial aspect of the right frontal lobe at approximately the level of the coronal suture.

#### textrank (ROUGE-1: 0.1798)

CC:, Progressive lower extremity weakness.,HX: ,This 52y/o RHF had a h/o right frontal glioblastoma multiforme (GBM) diagnosed by brain biopsy/partial resection, on 1/15/1991. Following extirpation of the tumor mass, she underwent radioactive Iodine implantation and 6020cGy radiation therapy in 35 fractions. The patient was discharged home on 10/17/92, but experienced worsening back pain and lower extremity weakness and became predominantly wheelchair bound within 4 months.

### Analysis

**Overall Performance:** Medium (avg ROUGE-1: 0.192)

**Best Method:** random (ROUGE-1: 0.215)
**Worst Method:** textrank (ROUGE-1: 0.180)
**Performance Spread:** 0.036

**Key Observations:**
- Methods showed moderate performance with some variation
- Different approaches captured different aspects of the content
- Strong agreement: all methods selected the same first sentence

---

## Case Study 3: LOW Performance

**Document ID:** 3
**Word Count:** 509
**Sentence Count:** 35
**Average ROUGE-1:** 0.0000

### Original Text (excerpt)

OPERATIVE NOTE:, The patient was taken to the operating room and placed in the supine position on the operating room table. The patient was prepped and draped in usual sterile fashion. An incision was made in the groin crease overlying the internal ring. This incision was about 1.5 cm in length. The incision was carried down through the Scarpa's layer to the level of the external oblique. This was opened along the direction of its fibers and carried down along the external spermatic fascia. The cremasteric fascia was then incised and the internal spermatic fascia was grasped and pulled free. A hernia sac was identified and the testicle was located. Next the internal spermatic fascia was incised and the hernia sac was dissected free inside the internal ring. This was performed by incising the transversalis fascia circumferentially. The hernia sac was ligated with a 3-0 silk suture high and divided and was noted to retract into the abdominal cavity. Care was taken not to injure the testicular vessels. Next the abnormal attachments of the testicle were dissected free distally with care not to injure any long loop vas and these were divided beneath the testicle for a fair distance. The lateral attachments tethering the cord vessels were freed from the sidewalls in the retroperitoneum high. This gave excellent length and very adequate length to bring the testicle down into the anterior superior hemiscrotum. The testicle was viable. This was wrapped in a moist sponge.,Next a hemostat was passed down through the inguinal canal down into the scrotum. A small 1 cm incision was made in the anterior superior scrotal wall. Dissection was carried down through the dartos layer. A subdartos pouch was formed with blunt dissection. The hemostat was then pushed against the tissues and this tissue was divided. The hemostat was then passed through the incision. A Crile hemostat was passed back up into the inguinal canal. The distal attachments of the sac were grasped and pulled down without twisting these structures through the incision. The neck was then closed with a 4-0 Vicryl suture that was not too tight, but tight enough to prevent retraction of the testicle. The testicle was then tucked down in its proper orientation into the subdartos pouch and the subcuticular tissue was closed with a running 4-0 chromic and the skin was closed with a running 6-0 subcuticular chromic suture. Benzoin and a Steri-Strip were placed. Next the transversus abdominis arch was reapproximated to the iliopubic tract over the top of the cord vessels to tighten up the ring slightly. This was done with 2 to 3 interrupted 3-0 silk sutures. The external oblique was then closed with interrupted 3-0 silk suture. The Scarpa's layer was closed with a running 4-0 chromic and the skin was then closed with a running 4-0 Vicryl intracuticular stitch. Benzoin and Steri-Strip were applied. The testicle was in good position in the dependent portion of the hemiscrotum and the patient had a caudal block, was awakened, and...

### Reference Summary

 Orchiopexy & inguinal herniorrhaphy.

### Method Summaries and Scores

#### lead (ROUGE-1: 0.0000)

OPERATIVE NOTE:,  The patient was taken to the operating room and placed in the supine position on the operating room table. The patient was prepped and draped in usual sterile fashion. An incision was made in the groin crease overlying the internal ring.

#### random (ROUGE-1: 0.0000)

The testicle was viable. Benzoin and a Steri-Strip were placed. Benzoin and Steri-Strip were applied.

#### textrank (ROUGE-1: 0.0000)

Next the internal spermatic fascia was incised and the hernia sac was dissected free inside the internal ring. The hemostat was then passed through the incision. The external oblique was then closed with interrupted 3-0 silk suture.

### Analysis

**Overall Performance:** Low (avg ROUGE-1: 0.000)

**Best Method:** lead (ROUGE-1: 0.000)
**Worst Method:** textrank (ROUGE-1: 0.000)
**Performance Spread:** 0.000

**Key Observations:**
- All methods struggled with this document
- Possible causes: abstractive reference summary, complex content structure, or domain-specific language
- No agreement: each method selected different content

---
