# Text Summarization Case Studies

Generated: 2026-10-22 18:10
Number of cases: 3

---

## Case Study 1: HIGH Performance

**Document ID:** 832
**Word Count:** 139
**Sentence Count:** 16
**Average ROUGE-1:** 0.6526

### Original Text (excerpt)

REVIEW OF SYSTEMS,There was no weight loss, fevers, chills, sweats.  There is no blurring of the vision, itching, throat or neck pain, or neck fullness.  There is no vertigo or hoarseness or painful swallowing.  There is no chest pain, shortness of breath, paroxysmal nocturnal dyspnea, or chest pain with exertion.  There is no shortness of breath and no cough or hemoptysis.  No melena, nausea, vomiting, dysphagia, abdominal pain, diarrhea, constipation or blood in the stools.  No dysuria, hematuria or excessive urination.  No muscle weakness or tenderness.  No new numbness or tingling.  No arthralgias or arthritis.  There are no rashes.  No excessive fatigability, loss of motor skills or sensation.  No changes in hair texture, change in skin color, excessive or decreased appetite.  No swollen lymph nodes or night sweats.  No headaches.  The rest of the review of systems is negative.

### Reference Summary

 There was no weight loss, fevers, chills, sweats.  There is no blurring of the vision, itching, throat or neck pain, or neck fullness.  There is no vertigo or hoarseness or painful swallowing. 

### Method Summaries and Scores

#### lead (ROUGE-1: 0.9552)

REVIEW OF SYSTEMS,There was no weight loss, fevers, chills, sweats. There is no blurring of the vision, itching, throat or neck pain, or neck fullness. There is no vertigo or hoarseness or painful swallowing.

#### random (ROUGE-1: 0.7222)

REVIEW OF SYSTEMS,There was no weight loss, fevers, chills, sweats. There is no blurring of the vision, itching, throat or neck pain, or neck fullness. No melena, nausea, vomiting, dysphagia, abdominal pain, diarrhea, constipation or blood in the stools.

#### lsa (ROUGE-1: 0.5429)

REVIEW OF SYSTEMS,There was no weight loss, fevers, chills, sweats. There is no chest pain, shortness of breath, paroxysmal nocturnal dyspnea, or chest pain with exertion. There is no shortness of breath and no cough or hemoptysis.

#### lexrank (ROUGE-1: 0.5429)

REVIEW OF SYSTEMS,There was no weight loss, fevers, chills, sweats. There is no chest pain, shortness of breath, paroxysmal nocturnal dyspnea, or chest pain with exertion. There is no shortness of breath and no cough or hemoptysis.

#### textrank (ROUGE-1: 0.5000)

REVIEW OF SYSTEMS,There was no weight loss, fevers, chills, sweats. There is no chest pain, shortness of breath, paroxysmal nocturnal dyspnea, or chest pain with exertion. The rest of the review of systems is negative.

### Analysis

**Overall Performance:** High (avg ROUGE-1: 0.653)

**Best Method:** lead (ROUGE-1: 0.955)
**Worst Method:** textrank (ROUGE-1: 0.500)
**Performance Spread:** 0.455

**Key Observations:**
- All methods performed well, indicating the document is well-suited for extractive summarization
- The reference summary likely contains sentences very similar to the source
- Strong agreement: all methods selected the same first sentence

---

## Case Study 2: MEDIUM Performance

**Document ID:** 684
**Word Count:** 413
**Sentence Count:** 27
**Average ROUGE-1:** 0.2074

### Original Text (excerpt)

PREOPERATIVE DIAGNOSIS:,  Dental caries.,POSTOPERATIVE DIAGNOSIS: , Dental caries.,PROCEDURE: , Dental restorations and extractions.,CLINICAL HISTORY: , This 23-year-old male is a client of the ABC Center because of his disability, the nature of which is unclear to me at this time; however, he reportedly has several issues that qualify him as disabled.  He has had multiple severe carious lesions that warrant multiple extractions at this time.  It is also unclear to me as to how his prior or existing restorations were accomplished.  In any case, he has been cleared for the procedure today.  He has his history and physical in the chart.,PROCEDURE: , The patient was brought to the operating room at 11 o'clock and placed in the supine position.  Dr. X administered the general anesthetic, after which a throat pack was placed.  Available full mouth x-rays were reviewed.  These x-rays were taken at another location.  Teeth 2, 4, 10, 12, 13, 15, 18, 20, 27, and 31 were all in varying degrees of severe decay from complete destruction of the crowns to pulp exposures with periapical radiolucencies.  All of these aforementioned teeth were extracted using combinations of forceps and elevators.  Hemostasis in all of these sites was accomplished with direct pressure using gauze packs.  ,Tooth 5 had caries in the distal surface extending to the occlusal as well as another carious lesion in the buccal.  These carious lesions and his tooth were excavated, and the tooth was restored with amalgam involving these surfaces.  ,Tooth 6 had caries on the facial surface, which was excavated, and the tooth was restored with composite.  ,Tooth 7 had caries involving the distal surface.  ,Tooth 8 likewise had caries involving the distal surface, and both of these distal lesions extended into incisal area.  These carious lesions were excavated, and both of these teeth were restored with composite.  ,Tooth 9 had caries in a mesial surface and a buccal surface, which was excavated, and this tooth was restored with composite.  ,Tooth 28 caries in the mesial surface extending to the occlusal, which was excavated, and the tooth was restored with amalgam, and tooth 30 had carries in the buccal surface, which was excavated, and the tooth was restored with amalgam.  ,A prophylaxis was done, primarily using a rotating rubber cup and some minor scaling, and the mouth was irrigated and suctioned thoroughly.  The throat pack was removed, and the patient was awakened and brought to the recovery room in good condition at 1330 hours.  There was negligible blood loss.

### Reference Summary

 Dental restorations and extractions.  Dental caries.  He has had multiple severe carious lesions that warrant multiple extractions at this time.

### Method Summaries and Scores

#### lead (ROUGE-1: 0.3883)

PREOPERATIVE DIAGNOSIS:,  Dental caries.,POSTOPERATIVE DIAGNOSIS: , Dental caries.,PROCEDURE: , Dental restorations and extractions.,CLINICAL HISTORY: , This 23-year-old male is a client of the ABC Center because of his disability, the nature of which is unclear to me at this time; however, he reportedly has several issues that qualify him as disabled. He has had multiple severe carious lesions that warrant multiple extractions at this time. It is also unclear to me as to how his prior or existing restorations were accomplished.

#### random (ROUGE-1: 0.2276)

PREOPERATIVE DIAGNOSIS:,  Dental caries.,POSTOPERATIVE DIAGNOSIS: , Dental caries.,PROCEDURE: , Dental restorations and extractions.,CLINICAL HISTORY: , This 23-year-old male is a client of the ABC Center because of his disability, the nature of which is unclear to me at this time; however, he reportedly has several issues that qualify him as disabled. Teeth 2, 4, 10, 12, 13, 15, 18, 20, 27, and 31 were all in varying degrees of severe decay from complete destruction of the crowns to pulp exposures with periapical radiolucencies. ,Tooth 6 had caries on the facial surface, which was excavated, and the tooth was restored with composite.

#### lsa (ROUGE-1: 0.1739)

,Tooth 7 had caries involving the distal surface. ,Tooth 8 likewise had caries involving the distal surface, and both of these distal lesions extended into incisal area. ,Tooth 9 had caries in a mesial surface and a buccal surface, which was excavated, and this tooth was restored with composite.

#### lexrank (ROUGE-1: 0.1471)

,Tooth 6 had caries on the facial surface, which was excavated, and the tooth was restored with composite. ,Tooth 7 had caries involving the distal surface. ,Tooth 9 had caries in a mesial surface and a buccal surface, which was excavated, and this tooth was restored with composite.

#### textrank (ROUGE-1: 0.1000)

,Tooth 6 had caries on the facial surface, which was excavated, and the tooth was restored with composite. ,Tooth 9 had caries in a mesial surface and a buccal surface, which was excavated, and this tooth was restored with composite. ,Tooth 28 caries in the mesial surface extending to the occlusal, which was excavated, and the tooth was restored with amalgam, and tooth 30 had carries in the buccal surface, which was excavated, and the tooth was restored with amalgam.

### Analysis

**Overall Performance:** Medium (avg ROUGE-1: 0.207)

**Best Method:** lead (ROUGE-1: 0.388)
**Worst Method:** textrank (ROUGE-1: 0.100)
**Performance Spread:** 0.288

**Key Observations:**
- Methods showed moderate performance with some variation
- Different approaches captured different aspects of the content
- Partial agreement: some methods converged on similar content

---

## Case Study 3: LOW Performance

**Document ID:** 559
**Word Count:** 476
**Sentence Count:** 53
**Average ROUGE-1:** 0.0000

### Original Text (excerpt)

CC: ,Headache.,HX:, This 51 y/o RHM was moving furniture several days prior to presentation when he struck his head (vertex) against a door panel. He then stepped back and struck his back on a trailer hitch. There was no associated LOC but he felt "dazed." He complained a HA since the accident. The following day he began experiencing episodic vertigo lasting several minutes with associated nausea and vomiting. He has been lying in bed most of the time since the accident. He also complained of transient left lower extremity weakness. The night before admission he went to his bedroom and his girlfriend heard a loud noise. She found him on the floor unable to speak or move his left side well. He was taken to a local ER. In the ER experienced a spell in which he stared to the right for approximately one minute. During this time he was unable to speak and did not seem to comprehend verbal questions. This resolved. ER staff noted decreased left sided movement and a left Babinski sign.,He was given valium 5 mg, and DPH 1.0g. A HCT was performed and he was transferred to UIHC.,PMH:, DM, Coronary Artery Disease, Left femoral neuropathy of unknown etiology. Multiple head trauma in past (?falls/fights).,MEDS:, unknown oral med for DM.,SHX:, 10+pack-year h/o Tobacco use; quit 2 years ago. 6-pack beer/week. No h/o illicit drug use.,FHX:, unknown.,EXAM: ,70BPM, BP144/83, 16RPM, 36.0C,MS: Alert and oriented to person, place, time. Fluent speech.,CN: left lower facial weakness with right gaze preference. Pupils 3/3 decreasing to 2/2 on exposure to light. Optic disks flat.,MOTOR: decreased spontaneous movement of left-sided extremities. 5/4 strength in both upper and lower extremities. Normal muscle tone and bulk.,SENSORY: withdrew equally to noxious stimulation in all four extremities. GAIT/STATION/COORDINATION: not tested.,The general physical exam was unremarkable.,During the exam the patient experienced a spell during which his head turned and eyes deviated to the leftward, and his right hand twitched. The entire spell lasted one minute.,During the episode he was verbally unresponsive. He appeared groggy and lethargic after the event.,HCT without contrast: 11/18/92: right frontal skull fracture with associated minimal epidural hematoma and small subdural hematoma, as well as some adjacent subarachnoid blood and brain contusion.,LABS:, CBC, GS, PT/PTT were all WNL.,COURSE:, The patient was diagnosed with a right frontal SAH/contusion and post traumatic seizures. DPH was continued and he was given a Librium taper for possible alcoholic withdrawal. A neurosurgical consult was obtained. He did not receive surgical intervention and was discharged 12/1/92. Neuropsychological testing on 11/25/92 revealed: poor orientation to time or place and poor attention. Anterograde verbal and visual memory was severely impaired. Speech became mildly dysarthric when fatigued. Defective word finding. Difficulty copying 2 of 3 three dimensional figures. Recent head injury as well as a history of ETOH abuse and multiple prior head injuries probably contribute to his deficits.

### Reference Summary

 HCT: SAH, Contusion, Skull fracture

### Method Summaries and Scores

#### lead (ROUGE-1: 0.0000)

CC: ,Headache.,HX:, This 51 y/o RHM was moving furniture several days prior to presentation when he struck his head (vertex) against a door panel. He then stepped back and struck his back on a trailer hitch. There was no associated LOC but he felt "dazed."

#### random (ROUGE-1: 0.0000)

ER staff noted decreased left sided movement and a left Babinski sign.,He was given valium 5 mg, and DPH 1.0g. The entire spell lasted one minute.,During the episode he was verbally unresponsive. DPH was continued and he was given a Librium taper for possible alcoholic withdrawal.

#### textrank (ROUGE-1: 0.0000)

He also complained of transient left lower extremity weakness. During this time he was unable to speak and did not seem to comprehend verbal questions. ER staff noted decreased left sided movement and a left Babinski sign.,He was given valium 5 mg, and DPH 1.0g.

#### lsa (ROUGE-1: 0.0000)

She found him on the floor unable to speak or move his left side well. During this time he was unable to speak and did not seem to comprehend verbal questions. ER staff noted decreased left sided movement and a left Babinski sign.,He was given valium 5 mg, and DPH 1.0g.

#### lexrank (ROUGE-1: 0.0000)

He complained a HA since the accident. He has been lying in bed most of the time since the accident. He also complained of transient left lower extremity weakness.

### Analysis

**Overall Performance:** Low (avg ROUGE-1: 0.000)

**Best Method:** lead (ROUGE-1: 0.000)
**Worst Method:** lexrank (ROUGE-1: 0.000)
**Performance Spread:** 0.000

**Key Observations:**
- All methods struggled with this document
- Possible causes: abstractive reference summary, complex content structure, or domain-specific language
- No agreement: each method selected different content

---
