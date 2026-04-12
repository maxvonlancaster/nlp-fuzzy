# Text Summarization Case Studies

Generated: 2026-04-12 08:43
Number of cases: 3

---

## Case Study 1: HIGH Performance

**Document ID:** 53
**Word Count:** 174
**Sentence Count:** 28
**Average ROUGE-1:** 0.6114

### Original Text (excerpt)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition.  The right groin was prepped and draped in routine fashion.  An arterial sheath was inserted into the right femoral artery.,Left and right coronary arteries were studied with a 6FL4 and 6FR4 Judkins catheters respectively.  Cine coronary angiograms were done in multiple views.,Left heart catheterization was done using the 6-French pigtail catheter.  Appropriate pressures were obtained before and after the left ventriculogram, which was done in the RAO view.,At the end of the procedure, the femoral catheter was removed and Angio-Seal was applied without any complications.,FINDINGS:,1.  LV is normal in size and shape with good contractility, EF of 60%.,2.  LMCA normal.,3.  LAD has 20% to 30% stenosis at the origin.,4.  LCX is normal.,5.  RCA is dominant and normal.,RECOMMENDATIONS: , Medical management, diet, and exercise.  Aspirin 81 mg p.o. daily, p.r.n. nitroglycerin for chest pain.  Follow up in the clinic.

### Reference Summary

 Chest pain and non-Q-wave MI with elevation of troponin I only.  Left heart catheterization, left ventriculography, and left and right coronary arteriography.

### Method Summaries and Scores

#### textrank (ROUGE-1: 0.6957)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition. LMCA normal.,3. LCX is normal.,5.

#### lsa (ROUGE-1: 0.6316)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition. The right groin was prepped and draped in routine fashion. nitroglycerin for chest pain.

#### random (ROUGE-1: 0.6234)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition. RCA is dominant and normal.,RECOMMENDATIONS: , Medical management, diet, and exercise. daily, p.r.n.

#### lead (ROUGE-1: 0.4948)

PROCEDURES: , Left heart catheterization, left ventriculography, and left and right coronary arteriography.,INDICATIONS: , Chest pain and non-Q-wave MI with elevation of troponin I only.,TECHNIQUE:  ,The patient was brought to the procedure room in satisfactory condition. The right groin was prepped and draped in routine fashion. An arterial sheath was inserted into the right femoral artery.,Left and right coronary arteries were studied with a 6FL4 and 6FR4 Judkins catheters respectively.

### Analysis

**Overall Performance:** High (avg ROUGE-1: 0.611)

**Best Method:** textrank (ROUGE-1: 0.696)
**Worst Method:** lead (ROUGE-1: 0.495)
**Performance Spread:** 0.201

**Key Observations:**
- All methods performed well, indicating the document is well-suited for extractive summarization
- The reference summary likely contains sentences very similar to the source
- Strong agreement: all methods selected the same first sentence

---

## Case Study 2: MEDIUM Performance

**Document ID:** 34
**Word Count:** 359
**Sentence Count:** 68
**Average ROUGE-1:** 0.1958

### Original Text (excerpt)

SUBJECTIVE:,  Patient presents with Mom and Dad for her 5-year 3-month well-child check.  Family has not concerns stating patient has been doing well overall since last visit.  Taking in a well-balanced diet consisting of milk and dairy products, fruits, vegetables, proteins and grains with minimal junk food and snack food.  No behavioral concerns.  Gets along well with peers as well as adults.  Is excited to start kindergarten this upcoming school year.  Does attend daycare.  Normal voiding and stooling pattern.  No concerns with hearing or vision.  Sees the dentist regularly.  Growth and development:  Denver II normal passing all developmental milestones per age in areas of fine motor, gross motor, personal and social interaction and speech and language development.  See Denver II form in the chart.,ALLERGIES:,  None.,MEDICATIONS: , None.,FAMILY SOCIAL HISTORY:,  Unchanged since last checkup.  Lives at home with mother, father and sibling.  No smoking in the home.,REVIEW OF SYSTEMS:,  As per HPI; otherwise negative.,OBJECTIVE:,Vital Signs:  Weight 43 pounds.  Height 42-1/4 inches.  Temperature 97.7.  Blood pressure 90/64.,General:  Well-developed, well-nourished, cooperative, alert and interactive 5-year -3month-old white female in no acute distress.,HEENT:  Atraumatic, normocephalic.  Pupils equal, round and reactive.  Sclerae clear.  Red reflex present bilaterally.  Extraocular muscles intact.  TMs clear bilaterally.  Oropharynx:  Mucous membranes moist and pink.  Good dentition.,Neck:  Supple, no lymphadenopathy.,Chest:  Clear to auscultation bilaterally.  No wheeze or crackles.  Good air exchange.,Cardiovascular:  Regular rate and rhythm.  No murmur.  Good pulses bilaterally.,Abdomen:  Soft, nontender.  Nondistended.  Positive bowel sounds.  No masses or organomegaly.,GU:  Tanner I female genitalia.  Femoral pulses equal bilaterally.  No rash.,Extremities:  Full range of motion.  No cyanosis, clubbing or edema.,Back:  Straight.  No scoliosis.,Integument:  Warm, dry and pink without lesions.,Neurological:  Alert.  Good muscle tone and strength.  Cranial nerves II-XII grossly intact.  DTRs 2+/4+ bilaterally.,ASSESSMENT/PLAN:,1. Well 5-year 3-month-old white female.,2. Anticipatory guidance for growth and diet development and safety issues as well as immunizations.  Will receive MMR, DTaP and IPV today.  Discussed risks and benefits as well as possible side effects and symptomatic treatment.  Gave 5-year well-child check handout to mom.  Completed school pre-participation physical.  Copy in the chart.  Completed vision and hearing screening.  Reviewed results with family.,3. Follow up in one year for next well-child check or as needed for acute care.

### Reference Summary

 A 3-month well-child check. 

### Method Summaries and Scores

#### lsa (ROUGE-1: 0.3333)

No concerns with hearing or vision. Well 5-year 3-month-old white female.,2. Gave 5-year well-child check handout to mom.

#### textrank (ROUGE-1: 0.2500)

No concerns with hearing or vision. TMs clear bilaterally. Well 5-year 3-month-old white female.,2.

#### lead (ROUGE-1: 0.2000)

SUBJECTIVE:,  Patient presents with Mom and Dad for her 5-year 3-month well-child check. Family has not concerns stating patient has been doing well overall since last visit. Taking in a well-balanced diet consisting of milk and dairy products, fruits, vegetables, proteins and grains with minimal junk food and snack food.

#### random (ROUGE-1: 0.0000)

No murmur. Good pulses bilaterally.,Abdomen:  Soft, nontender. Will receive MMR, DTaP and IPV today.

### Analysis

**Overall Performance:** Medium (avg ROUGE-1: 0.196)

**Best Method:** lsa (ROUGE-1: 0.333)
**Worst Method:** random (ROUGE-1: 0.000)
**Performance Spread:** 0.333

**Key Observations:**
- Methods showed moderate performance with some variation
- Different approaches captured different aspects of the content
- Partial agreement: some methods converged on similar content

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

#### lsa (ROUGE-1: 0.0000)

Benzoin and a Steri-Strip were placed. The external oblique was then closed with interrupted 3-0 silk suture. Benzoin and Steri-Strip were applied.

### Analysis

**Overall Performance:** Low (avg ROUGE-1: 0.000)

**Best Method:** lead (ROUGE-1: 0.000)
**Worst Method:** lsa (ROUGE-1: 0.000)
**Performance Spread:** 0.000

**Key Observations:**
- All methods struggled with this document
- Possible causes: abstractive reference summary, complex content structure, or domain-specific language
- No agreement: each method selected different content

---
