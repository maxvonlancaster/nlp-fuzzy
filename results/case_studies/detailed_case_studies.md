# Text Summarization Case Studies

Generated: 2026-04-12 08:56
Number of cases: 3

---

## Case Study 1: HIGH Performance

**Document ID:** 921
**Word Count:** 335
**Sentence Count:** 27
**Average ROUGE-1:** 0.6254

### Original Text (excerpt)

TITLE OF OPERATION: , Ligation (clip interruption) of patent ductus arteriosus.,INDICATION FOR SURGERY: , This premature baby with operative weight of 600 grams and evidence of persistent pulmonary over circulation and failure to thrive has been diagnosed with a large patent ductus arteriosus originating in the left-sided aortic arch.  She has now been put forward for operative intervention.,PREOP DIAGNOSIS:  ,1.  Patent ductus arteriosus.,2.  Severe prematurity.,3.  Operative weight less than 4 kg (600 grams).,COMPLICATIONS: , None.,FINDINGS: , Large patent ductus arteriosus with evidence of pulmonary over circulation.  After completion of the procedure, left recurrent laryngeal nerve visualized and preserved.  Substantial rise in diastolic blood pressure.,DETAILS OF THE PROCEDURE: , After obtaining information consent, the patient was positioned in the neonatal intensive care unit, cribbed in the right lateral decubitus, and general endotracheal anesthesia was induced.  The left chest was then prepped and draped in the usual sterile fashion and a posterolateral thoracotomy incision was performed.  Dissection was carried through the deeper planes until the second intercostal space was entered freely with no damage to the underlying lung parenchyma.  The lung was quite edematous and was retracted anteriorly exposing the area of the isthmus.  The pleura overlying the ductus arteriosus was inside and the duct dissected in a nearly circumferential fashion.  It was then test occluded and then interrupted with a medium titanium clip.  There was preserved pulsatile flow in the descending aorta.  The left recurrent laryngeal nerve was identified and preserved.  With excellent hemostasis, the intercostal space was closed with 4-0 Vicryl sutures and the muscular planes were reapproximated with 5-0 Caprosyn running suture in two layers.  The skin was closed with a running 6-0 Caprosyn suture.  A sterile dressing was placed.  Sponge and needle counts were correct times 2 at the end of the procedure.  The patient was returned to the supine position in which palpable bilateral femoral pulses were noted.,I was the surgical attending present in the neonatal intensive care unit and in-charge of the surgical procedure throughout the entire length of the case.

### Reference Summary

 Ligation (clip interruption) of patent ductus arteriosus.  This premature baby with operative weight of 600 grams and evidence of persistent pulmonary over circulation and failure to thrive has been diagnosed with a large patent ductus arteriosus originating in the left-sided aortic arch. 

### Method Summaries and Scores

#### lead (ROUGE-1: 0.7963)

TITLE OF OPERATION: , Ligation (clip interruption) of patent ductus arteriosus.,INDICATION FOR SURGERY: , This premature baby with operative weight of 600 grams and evidence of persistent pulmonary over circulation and failure to thrive has been diagnosed with a large patent ductus arteriosus originating in the left-sided aortic arch. She has now been put forward for operative intervention.,PREOP DIAGNOSIS:  ,1. Patent ductus arteriosus.,2.

#### textrank (ROUGE-1: 0.7963)

TITLE OF OPERATION: , Ligation (clip interruption) of patent ductus arteriosus.,INDICATION FOR SURGERY: , This premature baby with operative weight of 600 grams and evidence of persistent pulmonary over circulation and failure to thrive has been diagnosed with a large patent ductus arteriosus originating in the left-sided aortic arch. Patent ductus arteriosus.,2. After completion of the procedure, left recurrent laryngeal nerve visualized and preserved.

#### random (ROUGE-1: 0.7350)

TITLE OF OPERATION: , Ligation (clip interruption) of patent ductus arteriosus.,INDICATION FOR SURGERY: , This premature baby with operative weight of 600 grams and evidence of persistent pulmonary over circulation and failure to thrive has been diagnosed with a large patent ductus arteriosus originating in the left-sided aortic arch. After completion of the procedure, left recurrent laryngeal nerve visualized and preserved. It was then test occluded and then interrupted with a medium titanium clip.

#### lsa (ROUGE-1: 0.1739)

After completion of the procedure, left recurrent laryngeal nerve visualized and preserved. The left recurrent laryngeal nerve was identified and preserved. With excellent hemostasis, the intercostal space was closed with 4-0 Vicryl sutures and the muscular planes were reapproximated with 5-0 Caprosyn running suture in two layers.

### Analysis

**Overall Performance:** High (avg ROUGE-1: 0.625)

**Best Method:** lead (ROUGE-1: 0.796)
**Worst Method:** lsa (ROUGE-1: 0.174)
**Performance Spread:** 0.622

**Key Observations:**
- All methods performed well, indicating the document is well-suited for extractive summarization
- The reference summary likely contains sentences very similar to the source
- Partial agreement: some methods converged on similar content

---

## Case Study 2: MEDIUM Performance

**Document ID:** 604
**Word Count:** 759
**Sentence Count:** 28
**Average ROUGE-1:** 0.2107

### Original Text (excerpt)

PREOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,POSTOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,PROCEDURE: , Lumbar re-exploration for removal of fractured internal fixation plate from T11 through L2 followed by a repositioning of the L2 pedicle screws and evaluation of the fusion from T11 through L2 followed by a bilateral hemilaminectomy and diskectomy for decompression at L4-5 and L5-S1 with posterior lumbar interbody fusion using morselized autograft bone and the synthetic spacers from the Capstone system at L4-5 and L5-S1 followed by placement of the pedicle screw fixation devices at L3, L4, L5, and S1 and insertion of a 20 cm fixation plate that range from the T11 through S1 levels and then subsequent onlay fusion using morselized autograft bone and bone morphogenetic soaked sponge at L1-2 and then at L3-L4, L4-L5, and L5-S1 bilaterally.,DESCRIPTION OF PROCEDURE: ,This is a 68-year-old lady who presents with a history of osteomyelitis associated with the percutaneous vertebroplasty that was actually treated several months ago with removal of the infected vertebral augmentation and placement of a posterior pedicle screw plate fixation device from T11 through L2. She subsequently actually done reasonably well until about a month ago when she developed progressive severe intractable pain. Imaging study showed that the distal hardware at the plate itself had fractured consistent with incomplete fusion across her osteomyelitis area. There was no evidence of infection on the imaging or with her laboratory studies. In addition, she developed a pretty profound stenosis at L4-L5 and L5-S1 that appeared to be recurrent as well. She now presents for revision of her hardware, extension of fusion, and decompression.,The patient was brought to the operating room, placed under satisfactory general endotracheal anesthesia. She was placed on the operative table in the prone position. Back was prepared with Betadine, iodine, and alcohol. We elliptically excised her old incision and extended this caudally so that we had access from the existing hardware fixation all the way down to her sacrum. The locking nuts were removed from the screw post and both plates refractured or significantly weakened and had a crease in it. After these were removed, it was obvious that the bottom screws were somewhat loosened in the pedicle zone so we actually tightened one up and that fit good snugly into the nail when we redirected so that it actually reamed up into the upper aspect of the vertebral body in much more secure purchase. We then dressed the L4-L5 and L5-S1 levels which were profoundly stenotic. This was a combination of scar and overgrown bone. She had previously undergone bilateral hemilaminectomies at L4-5 so we removed scar bone and actually cleaned and significantly...

### Reference Summary

 Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.

### Method Summaries and Scores

#### lsa (ROUGE-1: 0.2250)

PREOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,POSTOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,PROCEDURE: , Lumbar re-exploration for removal of fractured internal fixation plate from T11 through L2 followed by a repositioning of the L2 pedicle screws and evaluation of the fusion from T11 through L2 followed by a bilateral hemilaminectomy and diskectomy for decompression at L4-5 and L5-S1 with posterior lumbar interbody fusion using morselized autograft bone and the synthetic spacers from the Capstone system at L4-5 and L5-S1 followed by placement of the pedicle screw fixation devices at L3, L4, L5, and S1 and insertion of a 20 cm fixation plate that range from the T11 through S1 levels and then subsequent onlay fusion using morselized autograft bone and bone morphogenetic soaked sponge at L1-2 and then at L3-L4, L4-L5, and L5-S1 bilaterally.,DESCRIPTION OF PROCEDURE:  ,This is a 68-year-old lady who presents with a history of osteomyelitis associated with the percutaneous vertebroplasty that was actually treated several months ago with removal of the infected vertebral augmentation and placement of a posterior pedicle screw plate fixation device from T11 through L2. We then dressed the L4-L5 and L5-S1 levels which were profoundly stenotic. We used 10 x 32 mm spacers at both L4-L5 and L5-S1.

#### lead (ROUGE-1: 0.2175)

PREOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,POSTOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,PROCEDURE: , Lumbar re-exploration for removal of fractured internal fixation plate from T11 through L2 followed by a repositioning of the L2 pedicle screws and evaluation of the fusion from T11 through L2 followed by a bilateral hemilaminectomy and diskectomy for decompression at L4-5 and L5-S1 with posterior lumbar interbody fusion using morselized autograft bone and the synthetic spacers from the Capstone system at L4-5 and L5-S1 followed by placement of the pedicle screw fixation devices at L3, L4, L5, and S1 and insertion of a 20 cm fixation plate that range from the T11 through S1 levels and then subsequent onlay fusion using morselized autograft bone and bone morphogenetic soaked sponge at L1-2 and then at L3-L4, L4-L5, and L5-S1 bilaterally.,DESCRIPTION OF PROCEDURE:  ,This is a 68-year-old lady who presents with a history of osteomyelitis associated with the percutaneous vertebroplasty that was actually treated several months ago with removal of the infected vertebral augmentation and placement of a posterior pedicle screw plate fixation device from T11 through L2. She subsequently actually done reasonably well until about a month ago when she developed progressive severe intractable pain. Imaging study showed that the distal hardware at the plate itself had fractured consistent with incomplete fusion across her osteomyelitis area.

#### random (ROUGE-1: 0.2057)

PREOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,POSTOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,PROCEDURE: , Lumbar re-exploration for removal of fractured internal fixation plate from T11 through L2 followed by a repositioning of the L2 pedicle screws and evaluation of the fusion from T11 through L2 followed by a bilateral hemilaminectomy and diskectomy for decompression at L4-5 and L5-S1 with posterior lumbar interbody fusion using morselized autograft bone and the synthetic spacers from the Capstone system at L4-5 and L5-S1 followed by placement of the pedicle screw fixation devices at L3, L4, L5, and S1 and insertion of a 20 cm fixation plate that range from the T11 through S1 levels and then subsequent onlay fusion using morselized autograft bone and bone morphogenetic soaked sponge at L1-2 and then at L3-L4, L4-L5, and L5-S1 bilaterally.,DESCRIPTION OF PROCEDURE:  ,This is a 68-year-old lady who presents with a history of osteomyelitis associated with the percutaneous vertebroplasty that was actually treated several months ago with removal of the infected vertebral augmentation and placement of a posterior pedicle screw plate fixation device from T11 through L2. We elliptically excised her old incision and extended this caudally so that we had access from the existing hardware fixation all the way down to her sacrum. This corrected the deformity and helped to preserve the correction of the stenosis and then after we cannulated the pedicles of L4, L5 and S1 tightened the pedicle screws in L3.

#### textrank (ROUGE-1: 0.1946)

PREOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,POSTOPERATIVE DIAGNOSIS: , Recurrent degenerative spondylolisthesis and stenosis at L4-5 and L5-S1 with L3 compression fracture adjacent to an instrumented fusion from T11 through L2 with hardware malfunction distal at the L2 end of the hardware fixation.,PROCEDURE: , Lumbar re-exploration for removal of fractured internal fixation plate from T11 through L2 followed by a repositioning of the L2 pedicle screws and evaluation of the fusion from T11 through L2 followed by a bilateral hemilaminectomy and diskectomy for decompression at L4-5 and L5-S1 with posterior lumbar interbody fusion using morselized autograft bone and the synthetic spacers from the Capstone system at L4-5 and L5-S1 followed by placement of the pedicle screw fixation devices at L3, L4, L5, and S1 and insertion of a 20 cm fixation plate that range from the T11 through S1 levels and then subsequent onlay fusion using morselized autograft bone and bone morphogenetic soaked sponge at L1-2 and then at L3-L4, L4-L5, and L5-S1 bilaterally.,DESCRIPTION OF PROCEDURE:  ,This is a 68-year-old lady who presents with a history of osteomyelitis associated with the percutaneous vertebroplasty that was actually treated several months ago with removal of the infected vertebral augmentation and placement of a posterior pedicle screw plate fixation device from T11 through L2. We then dressed the L4-L5 and L5-S1 levels which were profoundly stenotic. Once we placed the plate onto the screws and locked them in position, we then packed the remaining BMP sponge and morselized autograft bone through the plate around the incomplete fracture healing at the L1 level and then dorsolaterally at L4-L5 and L5-S1 and L3-L4, again the goal being to create a dorsal fusion and enhance the interbody fusion as well.

### Analysis

**Overall Performance:** Medium (avg ROUGE-1: 0.211)

**Best Method:** lsa (ROUGE-1: 0.225)
**Worst Method:** textrank (ROUGE-1: 0.195)
**Performance Spread:** 0.030

**Key Observations:**
- Methods showed moderate performance with some variation
- Different approaches captured different aspects of the content
- Strong agreement: all methods selected the same first sentence

---

## Case Study 3: LOW Performance

**Document ID:** 388
**Word Count:** 716
**Sentence Count:** 69
**Average ROUGE-1:** 0.0000

### Original Text (excerpt)

HISTORY OF PRESENT ILLNESS: , I have seen ABC today. He is a very pleasant gentleman who is 42 years old, 344 pounds. He is 5'9". He has a BMI of 51. He has been overweight for ten years since the age of 33, at his highest he was 358 pounds, at his lowest 260. He is pursuing surgical attempts of weight loss to feel good, get healthy, and begin to exercise again. He wants to be able to exercise and play volleyball. Physically, he is sluggish. He gets tired quickly. He does not go out often. When he loses weight he always regains it and he gains back more than he lost. His biggest weight loss is 25 pounds and it was three months before he gained it back. He did six months of not drinking alcohol and not taking in many calories. He has been on multiple commercial weight loss programs including Slim Fast for one month one year ago and Atkin's Diet for one month two years ago.,PAST MEDICAL HISTORY: , He has difficulty climbing stairs, difficulty with airline seats, tying shoes, used to public seating, difficulty walking, high cholesterol, and high blood pressure. He has asthma and difficulty walking two blocks or going eight to ten steps. He has sleep apnea and snoring. He is a diabetic, on medication. He has joint pain, knee pain, back pain, foot and ankle pain, leg and foot swelling. He has hemorrhoids.,PAST SURGICAL HISTORY: , Includes orthopedic or knee surgery.,SOCIAL HISTORY: , He is currently single. He drinks alcohol ten to twelve drinks a week, but does not drink five days a week and then will binge drink. He smokes one and a half pack a day for 15 years, but he has recently stopped smoking for the past two weeks.,FAMILY HISTORY: , Obesity, heart disease, and diabetes. Family history is negative for hypertension and stroke.,CURRENT MEDICATIONS:, Include Diovan, Crestor, and Tricor.,MISCELLANEOUS/EATING HISTORY: ,He says a couple of friends of his have had heart attacks and have had died. He used to drink everyday, but stopped two years ago. He now only drinks on weekends. He is on his second week of Chantix, which is a medication to come off smoking completely. Eating, he eats bad food. He is single. He eats things like bacon, eggs, and cheese, cheeseburgers, fast food, eats four times a day, seven in the morning, at noon, 9 p.m., and 2 a.m. He currently weighs 344 pounds and 5'9". His ideal body weight is 160 pounds. He is 184 pounds overweight. If he lost 70% of his excess body weight that would be 129 pounds and that would get him down to 215.,REVIEW OF SYSTEMS: , Negative for head, neck, heart, lungs, GI, GU, orthopedic, or skin. He also is positive for gout. He denies chest pain, heart attack, coronary artery disease, congestive heart failure, arrhythmia, atrial fibrillation, pacemaker, pulmonary embolism, or CVA. He denies venous insufficiency or thrombophlebitis. Denies shortness of breath, COPD, or...

### Reference Summary

 Consult for laparoscopic gastric bypass.

### Method Summaries and Scores

#### lead (ROUGE-1: 0.0000)

HISTORY OF PRESENT ILLNESS: , I have seen ABC today. He is a very pleasant gentleman who is 42 years old, 344 pounds. He is 5'9".

#### random (ROUGE-1: 0.0000)

He drinks alcohol ten to twelve drinks a week, but does not drink five days a week and then will binge drink. Heart is regular rhythm and rate. He will need to see a nutritionist and mental health worker.

#### textrank (ROUGE-1: 0.0000)

He drinks alcohol ten to twelve drinks a week, but does not drink five days a week and then will binge drink. He will need to go to Dr. XYZ as he previously had a sleep study. We will need another sleep study.

#### lsa (ROUGE-1: 0.0000)

He drinks alcohol ten to twelve drinks a week, but does not drink five days a week and then will binge drink. He will need to go to Dr. XYZ as he previously had a sleep study. We will need another sleep study.

### Analysis

**Overall Performance:** Low (avg ROUGE-1: 0.000)

**Best Method:** lead (ROUGE-1: 0.000)
**Worst Method:** lsa (ROUGE-1: 0.000)
**Performance Spread:** 0.000

**Key Observations:**
- All methods struggled with this document
- Possible causes: abstractive reference summary, complex content structure, or domain-specific language
- Partial agreement: some methods converged on similar content

---
