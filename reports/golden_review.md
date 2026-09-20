# Golden set — curation pass

84 pairs across 10 documents. Each already passed automated checks: its evidence quote was located in the chunk it cites, the answer clears a length floor, and the evidence is not text repeated across more than two documents.

**12 are pre-flagged** below. If you are short on time, read only those — the other 72 look clean to me.

**What automation cannot judge, and you can:**

1. **Is it worth asking?** A question answered by administrative boilerplate (where to mail comments, a phone number, a disclaimer) verifies perfectly and measures nothing. `ADMIN?` is a guess, not a verdict — some "where" questions are real procedure questions about a protocol.
2. **Is the answer actually correct?** I can prove the quote exists. I cannot prove the answer follows from it. That is the part only you can do.
3. **Is it trivially extractive?** If the answer restates the evidence, it tests copying rather than comprehension. `TRIVIAL-COPY?`.
4. **Would someone ask this without knowing the answer already?** `SELF-REF?`, `TERSE-Q?`, `ALLCAPS?` (a quoted disclaimer rather than substance).

**How to reply:** give me the numbers to drop — "drop 3, 7, 12-14". Anything you do not mention, I keep. If a pair is fixable rather than junk, say so and I will redraft just that one.

---

## ctgov-NCT00567567-Prot_SAP_000

**1.** _definition_  ⚠️ **ALLCAPS?**
- **Q:** How is the research protocol defined in terms of its purpose for medical practice?
- **A:** The protocol is a research plan developed by the Children's Oncology Group to investigate study questions and should not be used to direct the practice of medicine or provide individualized medical care.
- **Evidence** (`normalized`): 'THIS PROTOCOL IS THE RESEARCH PLAN DEVELOPED BY THE CHILDREN ’S ONCOLOGY GROUP TO INVESTIGATE A PARTICULAR STUDY QUESTION OR SET OF STUDY QUESTIONS AND SHOULD NOT BE USED TO DIRECT THE PRACTICE OF MEDICINE BY ANY PERSON OR TO PROVIDE INDIVI'

**2.** _condition_
- **Q:** Under what clinical conditions will a patient be removed from protocol therapy during induction?
- **A:** Patients are taken off protocol therapy if they develop progressive disease (PD) at the end of induction, are unable to obtain adequate PBSC for transplant, or have a GFR < 60 ml/min/1.73 m2.
- **Evidence** (`normalized`): 'Patients who develop progressive disease (PD) at the end of induction, those w ho are unable to obtain adequate PBSC for transplant or who have GFR < 60 ml/min/1.73 m 2 will be taken off protocol therapy.'

**3.** _requirement_
- **Q:** What standard must central venous lines meet prior to starting induction chemotherapy in ANBL0532?
- **A:** All patients must have a double lumen central venous line placed before beginning induction chemotherapy.
- **Evidence** (`normalized`): 'All patients will have a double lumen central venous line placed prior to beginning induction chemotherapy.'

**4.** _procedure_
- **Q:** How should mesna and cyclophosphamide dosing be modified during Cycles 4 and 6 if a patient develops hematuria following Cycle 4?
- **A:** If microscopic or gross hematuria occurs after Induction Cycle 4 cyclophosphamide, mesna is administered as a 24-hour continuous infusion with Induction Cycle 6 cyclophosphamide as MESNA 560 mg/m² (or 18.7 mg/kg if < 12 kg) with each cyclophosphamide infusion, followed by MESNA 1800 mg/m² (or 60 mg/kg if < 12 kg) over 18 hours post-infusion.
- **Evidence** (`normalized`): 'give mesna as a 24 hour continuous infusion with Induction Cycle 6 cyclopho sphamide as follows: MESNA 560 mg/m²(or 18.7 mg/kg if < 12 kg) with each cyclophosphamide infusion, then MESNA 1800 mg/m² (or 60 mg/kg if < 12 kg) in required fluid'

**5.** _procedure_  ⚠️ **ADMIN?**
- **Q:** Where should immunocytochemistry bone marrow aspirate samples be shipped, and via which delivery schedule?
- **A:** Samples must be shipped overnight via Federal Express (Priority Overnight) to Dr. Robert C. Seeger at the Neuroblastoma Biology Reference Laboratory, Children's Hospital of Los Angeles, for Tuesday through Friday delivery.
- **Evidence** (`normalized`): 'Ship via Federal Express Prior Overnight using the COG Federal Express account number (2504-648 1-9). Specimens may be shipped to CHLA on Monday – Thursday for Tuesday – Friday delivery.'

**6.** _definition_
- **Q:** How is the Planning Target Volume 2 (PTV2) defined in relation to the Clinical Target Volume 2 (CTV2)?
- **A:** PTV2 is defined as the CTV2 with a geometric margin of 0.5 to 1.0 cm.
- **Evidence** (`exact`): 'The PTV2 is defined as the CTV2 with a geometric margin of 0.5-1.0 cm.'

**7.** _requirement_
- **Q:** What pre-apheresis blood product support requirement is specified for platelet counts?
- **A:** Patients should have a platelet count of > 50,000 on the day of apheresis and may receive a platelet transfusion to reach this requirement.
- **Evidence** (`normalized`): 'Patients should have platelet count of > 50,000 on the day of apheresis. May receive a platelet transfusion to reach 50,000 requirement.'

**8.** _scope_
- **Q:** What information is explicitly not protected by the Certificate of Confidentiality?
- **A:** The Certificate of Confidentiality does not protect against mandatory reporting of suspected child abuse, reportable communicable diseases, or possible threats of harm to self or others.
- **Evidence** (`normalized`): 'The Certific ate of Confidentiality will not protect against the requir ed reporting by hospital staff of information on suspected child abuse, reportable communi cable diseases, and/or possible threat of harm to self or others.'

**9.** _scope_  ⚠️ **SELF-REF?**
- **Q:** Which neuroblastoma patients belong in study ANBL0532 according to the study assignment table?
- **A:** ANBL0532 includes Stage 2a/2b with MYCN amplification; Stage 3 with MYCN amplification or age > 547 days with unfavorable histology; Stage 4 with MYCN amplification, age 365–<547 days with DI=1 or unfavorable histology, or age > 547 days; and Stage 4s < 365 days with MYCN amplification.
- **Evidence** (`exact`): 'ANBL0532 2a/2b any amp any any \nany degree of \nresection'

## ctgov-NCT02942264-Prot_SAP_000

**10.** _condition_
- **Q:** Under what condition can patients with prior bevacizumab use be included in the study?
- **A:** Patients who received bevacizumab for symptom management, such as cerebral edema or pseudo progression, can be included in the study.
- **Evidence** (`normalized`): 'Patients who received bevacizumab for symptom management, including but not limited to cerebral edema, pseudo progression can be included in the study.'

**11.** _requirement_
- **Q:** What surgical sample tissue specimens are required before a patient can enter the clinical trial?
- **A:** A representative paraffin tissue block measuring at least 5 mm x 5 mm or unstained slides from previous surgery must be obtained to confirm histological diagnosis and molecular markers.
- **Evidence** (`normalized`): 'A representative paraffin tissue block (at least 5 mm x 5 mm) or unstained slides will be obtained from previous surgery to confirm the histological diagnosis and molecular markers before entry into the clinical trial.'

**12.** _procedure_
- **Q:** How frequently must subjects be contacted during the survival follow-up phase after completing study therapy?
- **A:** Subjects in the survival follow-up phase should be contacted approximately every 6 months to evaluate survival status until death, withdrawal of consent, or study termination.
- **Evidence** (`normalized`): 'the subject moves into the survival follow up phase and should be contacted approximately within every 6 months to assess for survival status until death, withdrawal of consent, or the end of the study, whichever occurs first.'

**13.** _quantity_
- **Q:** How long prior to the first dose of temozolomide is the initial dose of Zotiraciclib administered in Cycle 1?
- **A:** In Cycle 1, one dose of Zotiraciclib (TG02) is administered 3 days prior to the first dose of temozolomide (TMZ).
- **Evidence** (`normalized`): 'one dose of Zotiraciclib (TG02) will be administered 3 days prior to first dose of TMZ in the first cycle.'

**14.** _scope_
- **Q:** Which sensitive CYP1A2 and CYP2D6 substrates with narrow therapeutic indices are noted as requiring cautious use?
- **A:** The sensitive CYP1A2 substrates with narrow therapeutic indices are theophylline and tizanidine, and the sensitive CYP2D6 substrates are thioridazine and tamoxifen.
- **Evidence** (`normalized`): 'Sensitive CYP1A2 substrates with narrow therapeutic indices include theophylline and tizanidine, and sensitive CYP2D6 substrates with narrow therapeutic indices include thioridazine and tamoxifen.'

**15.** _procedure_
- **Q:** What is the reporting timeframe requirement if a patient becomes pregnant during the study?
- **A:** Study treatment must be discontinued immediately, and the pregnancy must be reported to the Sponsor no later than 24 hours after the Investigator becomes aware of it.
- **Evidence** (`normalized`): 'If a patient becomes pregnant during the course of the study, the study treatment should be discontinued immediately, and the pregnancy reported to the Sponsor no later than 24 hours of when the Investigator becomes aware of it.'

**16.** _quantity_
- **Q:** What score shift is defined as the minimum clinically meaningful change for symptom severity and interference measures?
- **A:** A difference of at least 2 points is classified as the minimum clinically meaningful change in symptom severity and interference measures.
- **Evidence** (`normalized`): 'Differences of at least 2 points will be classified as the minimum clinically meaningful change in the symptom severity and symptom interference measures.'

**17.** _quantity_
- **Q:** How many days must a subject be removed from prior cytotoxic therapy before registration?
- **A:** Subjects must be removed from prior cytotoxic therapy for at least 4 weeks (28 days) at the time of registration.
- **Evidence** (`exact`): '≥ 4 weeks (28 days) from prior cytotoxic therapy,'

**18.** _condition_
- **Q:** What rule applies in the BOIN design if the number of DLTs triggers an 'Eliminate' action at a dose level?
- **A:** When a dose is eliminated, future patients are prevented from receiving that or higher doses, the dose is automatically de-escalated to the next lower level, and if the lowest dose is eliminated, the trial is stopped for safety with no MTD selected.
- **Evidence** (`normalized`): '(a) “Eliminate” means that we eliminate the current and higher doses from the trial to prevent treating any future patients at these doses because they are overly toxic. (b) When we eliminate a dose, we automatically de-escalate the dose to'

## ctgov-NCT04660552-Prot_003

**19.** _scope_
- **Q:** What is the targeted age range for children participating in the transcranial photobiomodulation feasibility study?
- **A:** The study targets male and female participants between 2 years and 6 years of age (inclusive).
- **Evidence** (`exact`): '1. Male or female participants between 2 years and 6 years of age (inclusive), of all races.'

**20.** _requirement_  ⚠️ **TRIVIAL-COPY?**
- **Q:** Which specific exclusion criteria apply to prior light-activated drug therapies before joining the study?
- **A:** Any use of light-activated drugs (photodynamic therapy) within 14 days prior to study enrollment is an exclusion criterion.
- **Evidence** (`exact`): '7. Any use of light-activated drugs (photodynamic therapy) within 14 days prior to study enrollment'

**21.** _procedure_
- **Q:** How will random assignment to treatment and control groups be carried out?
- **A:** A random sequence of 30 participant numbers will be created using a random number generator program (like random.org), and then the assignment to treatment and control conditions will be made by simply alternating down that randomized sequence.
- **Evidence** (`exact`): 'Once the program generates the random sequence of 30 participants (in the example above, the sequence goes 23, 25, 12, 15, etc), we will simply alternate the assignment to treatment and control conditions.'

**22.** _quantity_
- **Q:** How many total participants does the transcranial photobiomodulation study plan to enroll?
- **A:** The investigators plan to enroll up to 30 participants in total (with 15 participants planned for each group).
- **Evidence** (`exact`): 'The investigators propose to enroll up to 30 participants of both genders'

**23.** _condition_
- **Q:** Under what circumstance must a child participant be withdrawn from the study regarding session attendance?
- **A:** A child must be withdrawn from the study if a parent cannot reschedule a cancelled session within a 3 day window.
- **Evidence** (`exact`): 'cancelled session within a 3 day window, then the child will have to be withdrawn from the study.'

**24.** _definition_
- **Q:** What primary mechanism explains how photobiomodulation affects brain mitochondria at the cellular level?
- **A:** Photobiomodulation works by having the mitochondria in brain cells absorb light and produce more ATP (energy molecules).
- **Evidence** (`exact`): 'the mitochondria in brain cells absorbs the light and produces more ATP (energy molecules).'

**25.** _quantity_
- **Q:** How long will raw research data and identifying parental information be stored following the study?
- **A:** Raw research data will be kept for 6 years, whereas identifying parental information will be retained for 1 year.
- **Evidence** (`exact`): 'The raw research data will be stored for 6 years. The identifying parental information will be kept for a year, for potential follow up interviews.'

**26.** _procedure_
- **Q:** What daily observations are parents required to document regarding their child's behavior?
- **A:** Parents will keep a daily diary recording words spoken, comprehension of instructions, eye contact, sleep patterns, number of tantrums, anxiety, social interaction, and eating.
- **Evidence** (`exact`): 'Parents will be asked to keep a daily diary of the child’s behavior, specifically – words spoken, comprehension of instructions, eye contact, his sleep pattern, number of tantrums, anxiety, social interaction and eating.'

**27.** _condition_
- **Q:** What condition triggers an immediate shutdown of the entire study?
- **A:** The entire study will be immediately discontinued if two participants suddenly manifest signs of developmental regress that were unanticipated and not seen in prior photobiomodulation studies.
- **Evidence** (`exact`): 'If two participants suddenly manifests signs of regress (not anticipated, not seen before in prior photobiomodulation studies), we will immediately discontinue the study.'

**28.** _requirement_
- **Q:** Are financial compensation or transportation costs provided to participants and their families?
- **A:** No financial compensation will be provided to participants, and parents are responsible for paying their own transportation costs to the research facilities.
- **Evidence** (`exact`): 'No financial compensation will be provided.  Parents of participants will be responsible for the cost of transportation to the research facilities.'

## fda-113923

**29.** _definition_
- **Q:** How is a supplier defined under 21 CFR part 507?
- **A:** A supplier is the establishment that manufactures/processes the animal food, raises the animal, or grows the food provided to a receiving facility without further manufacturing/processing by another establishment, except for minor activities such as adding labeling.
- **Evidence** (`normalized`): 'Part 507 defines a “supplier” as the establishment that manufactures/processes the animal food, raises the animal, or grows the food that is provided to a receiving facility without further manufacturing/processing by another establishment,'

**30.** _definition_
- **Q:** What is the financial threshold for a business to be classified as a very small business under animal food regulations?
- **A:** A very small business is defined as a business (including subsidiaries and affiliates) averaging less than $2,500,000 per year in sales of animal food plus market value held without sale, adjusted for inflation, during the 3-year period preceding the applicable calendar year.
- **Evidence** (`normalized`): 'Very small business means a business (including any subsidiaries and affiliates) averaging less than $2,500,000, adjusted for inflation, per year, during the 3-year period preceding the applicable calendar year in sales of animal food plus '

**31.** _requirement_
- **Q:** What documentation must be maintained when an onsite audit is conducted as part of supplier verification?
- **A:** Documentation must include the name of the supplier audited, the audit procedures, the dates conducted, audit conclusions, corrective actions taken for significant deficiencies, and evidence that the audit was performed by a qualified auditor.
- **Evidence** (`exact`): '507.175(c)(7) Documentation of the conduct of an \nonsite audit, including (i) The name of \nthe supplier subject to the onsite audit; \n(ii) Documentation of audit procedures; \n(iii) The dates the audit was conducted; \n(iv) The conclusions of'

**32.** _procedure_
- **Q:** How can computerized systems be utilized in receiving procedures to verify suppliers?
- **A:** Receiving personnel cross-reference purchase orders, supplier names, materials, and quantities with previously entered system data to verify approved status and order accuracy, utilizing safeguard mechanisms to block unapproved suppliers and generating lists of approved suppliers on demand.
- **Evidence** (`normalized`): 'When raw materials and other ingredients are delivered to a facility, the receiving personnel cross reference the purchase order number, supplier name, material received, and the quantity of material received with the information previously'

**33.** _condition_
- **Q:** When can an annual onsite audit be bypassed if a SAHCODHA hazard requires a preventive control?
- **A:** An annual onsite audit is not required if there is a written determination in the food safety plan, prepared by or under the oversight of a PCQI, stating that other verification activities or less frequent audits provide adequate assurance that the hazard is controlled.
- **Evidence** (`normalized`): 'The exception to the requirement to conduct an annual onsite audit when the hazard requiring a preventive control is a SAHCODHA hazard is when there is a written determination that other verification activities and/or less frequent onsite a'

**34.** _condition_
- **Q:** Under what condition is an onsite audit conducted by an accredited certification body's audit agent exempt from 21 CFR part 1 subpart M requirements?
- **A:** The audit is not subject to subpart M requirements if it is conducted solely to meet the supply-chain program requirements of part 507.
- **Evidence** (`normalized`): 'Section 21 CFR 507.135(d) specifies that if an onsite audit is solely conducted to meet the requirements of part 507 by an audit agent of a certification body that is accredited in accordance with regulations in part 1, subpart M, the audit'

**35.** _quantity_
- **Q:** What amount of animal food ingredient is permitted for cattle feeding studies under the research exemption?
- **A:** Up to 200 pounds of an animal food ingredient may be needed when conducting a feeding study in cattle under the research or evaluation exemption.
- **Evidence** (`normalized`): '200 pounds of an animal food ingredient may be needed if conducting a feeding study in cattle.'

**36.** _scope_  ⚠️ **TRIVIAL-COPY?**
- **Q:** Which entities fall outside the definitions of receiving facilities and suppliers under Subpart E?
- **A:** Entities such as brokers, food distributors, and cold storage facilities are neither receiving facilities nor suppliers under Subpart E because they are not manufacturers/processors.
- **Evidence** (`normalized`): 'Under subpart E, entities such as brokers, food distributors, and cold storage facilities are neither receiving facilities that are required to establish a supply-chain program nor suppliers, because such entities are not manufacturers/proc'

## fda-133009

**37.** _procedure_  ⚠️ **ADMIN?**
- **Q:** Where can written comments regarding the FDA's compliance policy guidance for limited modifications to marketed tobacco products be mailed?
- **A:** Written comments may be submitted to the Dockets Management Staff (HFA-305), Food and Drug Administration, 5630 Fishers Lane, Room 1061, Rockville, MD 20852.
- **Evidence** (`normalized`): 'Alternatively, submit written comments to the Dockets Management Staff (HFA-305), Food and Drug Administration, 5630 Fishers Lane, Room 1061, Rockville, MD 20852.'

**38.** _scope_
- **Q:** Which two types of limited modifications to pre-existing new tobacco products are covered by the FDA's compliance policy?
- **A:** The policy covers (1) modifications to battery-operated tobacco products solely to comply with UL 8139, and (2) modifications to liquid nicotine products solely to comply with the Child Nicotine Poisoning Prevention Act of 2015 (CNPPA) flow restrictor requirements for liquid nicotine containers.
- **Evidence** (`normalized`): 'FDA sets out its compliance policy for premarket review requirements for two types of limited modifications to new tobacco products that were on the market as of August 8, 2016: (1) modifications to battery-operated tobacco products solely '

**39.** _quantity_
- **Q:** How many e-cigarettes had UL certified as meeting the UL 8139 standard as of August 27, 2019?
- **A:** As of August 27, 2019, UL had certified 14 e-cigarettes as complying with the UL 8139 standard.
- **Evidence** (`normalized`): 'As of August 27, 2019, UL has certified 14 e-cigarettes as complying with this standard.'

**40.** _requirement_
- **Q:** How does the FDA recommend a manufacturer submit modifications made to comply with UL 8139 if the product is modified after submitting a marketing application?
- **A:** The FDA recommends that manufacturers submit an amendment to the original application that describes the modifications implemented for UL 8139 compliance.
- **Evidence** (`normalized`): 'FDA recommends that manufacturers submit an amendment to the original application that describes the modifications implemented for UL 8139 compliance.'

**41.** _condition_
- **Q:** What product changes are explicitly cited as non-necessary for UL 8139 compliance?
- **A:** Changes that are not necessary to comply with UL 8139 include changes to the coil design or other heating element, changes in the wicking material or amount, changes in the power supplied to the product, and changes to the method of aerosolization of the e-liquid.
- **Evidence** (`normalized`): 'Changes to the product that are not necessary to comply with UL 8139, include, but are not limited to: changes to the coil design or other heating element (e.g., number of coils, material, resistance, length, or diameter); changes in the wi'

**42.** _definition_
- **Q:** How is special packaging defined under 16 CFR 1700.1?
- **A:** Special packaging is defined as packaging that is significantly difficult for children under age five to open or obtain a toxic or harmful amount of the substance contained therein within a reasonable time and not difficult for normal adults to use properly.
- **Evidence** (`normalized`): 'Special packaging is defined as packaging that is “significantly difficult for children under age five to open or obtain a toxic or harmful amount of the substance contained therein within a reasonable time and not difficult for normal adul'

**43.** _definition_  ⚠️ **TRIVIAL-COPY?**
- **Q:** What are flow restrictors and how do they function as liquid container packaging protection?
- **A:** Flow restrictors are typically adapters added to the neck of a bottle to limit the release of a liquid, serving as a passive secondary line of protection intended to limit the amount of a liquid substance a child can gain access to.
- **Evidence** (`normalized`): 'Flow restrictors, on the other hand, are a passive secondary line of protection. Flow restrictors are typically adapters added to the neck of a bottle to limit the release of a liquid and are intended to limit the amount of a liquid substan'

**44.** _requirement_
- **Q:** What maximum volume of liquid flow is allowed under the special packaging restricted flow requirement in 16 CFR 1700.15(d)?
- **A:** The regulation requires special packaging from which the flow of liquid is restricted so that not more than 2 milliliters of the contents can be obtained when the inverted, opened container is taken or squeezed once or when otherwise activated once.
- **Evidence** (`normalized`): '16 CFR 1700.15(d) requires “special packaging from which the flow of liquid is so restricted that not more than 2 milliliters of the contents can be obtained when the inverted, opened container is taken or squeezed once or when the containe'

**45.** _quantity_  ⚠️ **TRIVIAL-COPY?**
- **Q:** How many calls were made to U.S. poison centers between 2001 and 2016 regarding e-cigarette and e-liquid exposures in children under five?
- **A:** Between 2001 and 2016, there were 7,707 calls to U.S. poison centers regarding exposure to e-cigarettes and e-liquids for children younger than five years old.
- **Evidence** (`normalized`): 'Between 2001 and 2016, there were 7,707 calls to U.S. poison centers regarding exposure to e-cigarettes and e-liquids for children younger than five years old'

## fda-171592

**46.** _definition_
- **Q:** How is a prescription drug product defined under section 581(13) of the FD&C Act?
- **A:** A product is defined under section 581(13) of the FD&C Act as a prescription drug for human use in a finished dosage form for administration to a patient without substantial further manufacturing.
- **Evidence** (`normalized`): 'Product is defined in section 581(13) of the FD&C Act as a prescription drug for human use in a finished dosage form for administration to a patient without substantial further manufacturing.'

**47.** _scope_
- **Q:** Which entity types are listed as trading partners subject to enhanced drug distribution security under section 582(g)(1)?
- **A:** The trading partners listed as subject to enhanced drug distribution security requirements are manufacturers, wholesale distributors, dispensers, and repackagers.
- **Evidence** (`normalized`): 'This guidance is for trading partners 2 — namely manufacturers, wholesale distributors, dispensers, and repackagers3 — who are subject to requirements for enhanced drug distribution security under section 582(g)(1)'

**48.** _condition_
- **Q:** Under what condition may manufacturers provide transaction tracing documentation in a paper format to a subsequent purchaser before November 27, 2023?
- **A:** A manufacturer may provide transaction information, history, and statements in paper format if the purchaser is either a State licensed health care practitioner authorized to prescribe medication, or a licensed individual who dispenses product in the usual course of professional practice under the supervision or direction of a licensed prescribing health care practitioner.
- **Evidence** (`normalized`): 'except that the manufacturer may provide transaction history, transaction information, and transaction statements in a paper format if the subsequent purchaser is either a: (1) State licensed health care practitioner authorized to prescribe'

**49.** _procedure_
- **Q:** What process must trading partners follow to handle saleable returns under section 582(g)(1)(F) starting November 27, 2023?
- **A:** Trading partners must have systems and processes in place to accept saleable returns by associating the saleable return product with its corresponding transaction information and transaction statement.
- **Evidence** (`normalized`): 'Have systems and processes in place to accept saleable returns under appropriate conditions, i.e., being able to associate the saleable return product with the transaction information and transaction statement associated with that product.'

**50.** _requirement_
- **Q:** What information must be included in the transaction information under section 582(g)(1)(B) of the FD&C Act as of November 27, 2023?
- **A:** Section 582(g)(1)(B) requires transaction information to include package level product identifiers (consisting of the NDC, serial number, lot number, and expiration date) for each package included in the transaction.
- **Evidence** (`normalized`): 'Section 582(g)(1)(B) of the FD&C Act requires that, as of November 27, 2023, the transaction information required to be exchanged under section 582 include the product identifier (i.e., the standardized numerical identifier consisting of th'

**51.** _quantity_
- **Q:** Until what specific date does FDA intend to delay enforcing the package-level product identifier requirement under section 582(g)(1)(B)?
- **A:** FDA does not intend to take enforcement action regarding the package-level product identifier requirement under section 582(g)(1)(B) until November 27, 2024.
- **Evidence** (`normalized`): 'FDA does not intend to take action to enforce this requirement until November 27, 2024.'

**52.** _condition_
- **Q:** How does FDA's enforcement policy apply to products introduced into commerce prior to November 27, 2024?
- **A:** FDA does not intend to enforce the package-level product identifier requirement for transaction information on product introduced into commerce by a manufacturer or repackager before November 27, 2024, throughout its subsequent transactions through expiry.
- **Evidence** (`normalized`): 'In addition, FDA does not intend to take action to enforce the requirement under section 582(g)(1)(B) of the FD&C Act with respect to product that is introduced in a transaction into commerce by the product’s manufacturer or repackager befo'

**53.** _procedure_
- **Q:** How are trading partners recommended to respond to requests for information when assisting investigations during the enforcement transition period?
- **A:** Under the compliance policy, FDA recommends that trading partners respond to such requests with their relevant transaction information if they directly transacted the products subject to the request.
- **Evidence** (`normalized`): 'Under this compliance policy, we recommend that trading partners respond to such requests with its relevant transaction information— if it directly transacted the product(s) that are subject to the request.'

**54.** _requirement_
- **Q:** What system capabilities are required regarding information retrieval going back to the manufacturer during an investigation?
- **A:** Trading partners must have systems and processes to facilitate gathering the information required to produce transaction information for a product going back to the manufacturer in the event of a recall, suspect or illegitimate product investigation, or an authorized trading partner request.
- **Evidence** (`normalized`): 'Have systems and processes in place to facilitate the gathering of information needed to produce the transaction information for a product going back to'

## fda-69900

**55.** _scope_
- **Q:** Which specific domestic livestock species were the focus of FDA's Risk Assessment on SCNT?
- **A:** The Risk Assessment specifically focused on cattle, swine, sheep, and goats.
- **Evidence** (`exact`): 'focuses on those domestic livestock that \nhave been cloned, i.e., cattle, swine, sheep, and goats.'

**56.** _procedure_
- **Q:** What voluntary request did CVM make to companies in July 2001 regarding livestock cloning?
- **A:** CVM requested that companies voluntarily refrain from introducing meat or milk from animal clones or their progeny into the human or animal food supply while the risk assessment process was pending.
- **Evidence** (`exact`): 'CVM also requested that companies voluntarily refrain \nfrom introducing meat or milk from animal clones or their progeny into the human or \nanimal food supply pending completion of the risk assessment process.'

**57.** _condition_  ⚠️ **ADMIN?**
- **Q:** Where can the evaluated data from the Risk Assessment be accessed?
- **A:** All of the evaluated data in the Risk Assessment are available either in peer-reviewed publications or in the Risk Assessment itself.
- **Evidence** (`exact`): 'All of the data evaluated in the Risk Assessment are either available in peer-reviewed \npublications, or in the Risk Assessment itself.'

**58.** _requirement_
- **Q:** What measures does the FDA recommend for using clones to produce animal feed?
- **A:** FDA does not recommend any additional measures for using clones of any age or species for producing animal feed, as no unique risks were identified.
- **Evidence** (`exact`): 'FDA \ntherefore does not have recommendations for any additional measures related to the use \nof clones of any age or species for the production of feed for animals that are based on \nthe fact that the animals are derived from cloning.'

**59.** _condition_
- **Q:** What did FDA conclude regarding the food safety of cattle, swine, and goat clones?
- **A:** FDA concluded that there is sufficient information to determine that food from cattle, swine, and goat clones is as safe to eat as food from their conventionally-bred counterparts.
- **Evidence** (`exact`): 'concluded that there is sufficient information to \ndetermine that food from cattle, swine, and goat clones is as safe to eat as that from their \nmore conventionally-bred counterparts.'

**60.** _requirement_
- **Q:** What is the FDA's current recommendation regarding introducing edible products from sheep clones into human food?
- **A:** Because insufficient information was available on sheep clones, the FDA continues to recommend that edible products from clones of animals other than cattle, swine, or goat (such as sheep) not be introduced into the human food supply.
- **Evidence** (`exact`): 'recommend that edible products from clones from animals other than cattle, swine, or goat \n(e.g., sheep) not be introduced into the human food supply.'

**61.** _definition_
- **Q:** How are clone progeny defined in the context of SCNT technology?
- **A:** Clone progeny are defined as the sexually-reproduced offspring of clones.
- **Evidence** (`exact`): 'clone progeny, the sexually-reproduced offspring of clones, rather than from the clones \nthemselves.'

## fda-71787

**62.** _procedure_  ⚠️ **ADMIN?**
- **Q:** Where should written public comments on CPG Sec. 315.100 be submitted?
- **A:** Written comments regarding the document may be submitted to the Division of Dockets Management (HFA-305), Food and Drug Administration, 5630 Fishers Lane, rm. 1061, Rockville, MD 20852.
- **Evidence** (`normalized`): 'submit written comments regarding this document to the Division of Dockets Management (HFA-305), Food and Drug Administration, 5630 Fishers Lane, rm. 1061, Rockville, MD 20852.'

**63.** _definition_
- **Q:** What does the word "should" represent when used in FDA guidance documents?
- **A:** In FDA guidance documents, the use of the word "should" means that something is suggested or recommended, but not required.
- **Evidence** (`normalized`): 'The use of the word “should” in Agency guidance means that something is suggested or recommended, but not required.'

**64.** _scope_
- **Q:** Do FDA guidance documents create legally enforceable responsibilities?
- **A:** No, FDA's guidance documents do not establish legally enforceable responsibilities; instead, they describe the Agency's current thinking on a topic and should be viewed only as recommendations.
- **Evidence** (`normalized`): 'FDA’s guidance documents, including this guidance, do not establish legally enforceable responsibilities. Instead, guidance describes the Agency’s current thinking on a topic and should be viewed only as recommendations'

**65.** _definition_
- **Q:** Which government entity enforces the Federal Denture Act of 1942?
- **A:** The Federal Denture Act of 1942 is enforced by the Department of Justice.
- **Evidence** (`normalized`): 'The Federal Denture Act *of 1942* (Section 1821 of Title 18 *of the United States Code (18 USC 1821)) is *enforced by the Department of Justice.*'

**66.** _condition_
- **Q:** Under 18 USC 1821, what state law prohibitions trigger restrictions on transporting dentures into that jurisdiction?
- **A:** The restrictions apply where state laws prohibit the taking of impressions/casts by unlicensed persons, the construction or supply of dentures without authorization from a licensed dentist, or the construction/supply of dentures from casts made by unlicensed persons.
- **Evidence** (`normalized`): '1. the taking of impressions or casts of the human mouth or teeth by a person not licensed under such laws to practice dentistry; 2. the construction or supply of dentures by a person other than, or without the authorization or prescription'

**67.** _quantity_
- **Q:** What is the maximum prison sentence specified for violations under 18 USC 1821?
- **A:** The statute specifies a prison sentence of not more than one year, a fine under that title, or both.
- **Evidence** (`exact`): 'Shall be fined *under this title* or imprisoned not more than one year, or both."'

**68.** _procedure_
- **Q:** To which entity should reports of Federal Denture Act violations be referred first?
- **A:** Reports of violations of the Federal Denture Act should first be referred to the local office of the Office of Criminal Investigations (OCI) for review.
- **Evidence** (`normalized`): 'Reports of violations of the Federal Denture Act should *first* be referred to the *local office of the Office of Criminal Investigations (OCI)* for review.'

**69.** _procedure_  ⚠️ **TRIVIAL-COPY?**
- **Q:** What actions may the Office of Criminal Investigations take after consulting with CDRH/OC regarding Federal Denture Act violations?
- **A:** The OCI may refer the matter to the appropriate United States Attorney for consideration for criminal prosecution or to the Center for Devices and Radiological Health's Office of Compliance (CDRH/OC).
- **Evidence** (`normalized`): "After consulting with the Center for Devices and Radiological Health's Office of Compliance (CDRH/OC), the OCI may refer the matter to the appropriate United States Attorney for consideration for criminal prosecution or to CDRH/OC."

**70.** _scope_
- **Q:** What options are available to CDRH/OC if a Federal Denture Act matter is referred to it by OCI?
- **A:** CDRH/OC may take appropriate non-criminal action for any violations of the Federal Food, Drug, and Cosmetic Act or refer the report to state or local regulatory authorities.
- **Evidence** (`normalized`): 'The latter office may take appropriate non-criminal action for any violations of the Federal Food, Drug, and Cosmetic Act or refer the report to state or local regulatory authorities.*'

## fda-75426

**71.** _scope_
- **Q:** What primary purpose do FDA advisory committees serve?
- **A:** FDA advisory committees provide independent expert advice to the agency regarding complex scientific, technical, and policy issues, including product development and evaluation.
- **Evidence** (`normalized`): "FDA's advisory committees provide independent expert advice to the agency on a range of complex scientific, technical, and policy issues, including questions related to the development and evaluation of products regulated by FDA."

**72.** _condition_
- **Q:** Under what circumstances are votes typically omitted during FDA advisory committee meetings?
- **A:** Votes are typically omitted during meetings held to discuss clinical trial design development or guidance document development.
- **Evidence** (`normalized`): 'votes are typically not taken at meetings to discuss the development of a clinical trial design or the development of a guidance document.'

**73.** _condition_
- **Q:** Why are secret ballots usually considered inappropriate for advisory committee voting?
- **A:** Secret ballots are generally not appropriate because each advisory committee member's expert opinion should be clearly understood and attributed to that specific expert.
- **Evidence** (`normalized`): 'The use of secret ballots, long a hallmark of the American electoral experience, generally is not appropriate in the advisory committee context because the expert opinion of each member should be clearly understood and identified with that '

**74.** _requirement_
- **Q:** What design standards should be met when drafting a question for an advisory committee vote?
- **A:** Questions presented for a vote must contain minimal qualifiers, must not be leading, and should avoid using double or triple negatives.
- **Evidence** (`normalized`): 'The question presented for a vote should have minimal qualifiers, not be leading, and should avoid the use of double or triple negatives.'

**75.** _procedure_
- **Q:** How should voting be conducted during advisory committee meetings according to FDA recommendations?
- **A:** Voting should occur simultaneously through methods such as a show of hands, showing 'yes' or 'no' cards, or casting written ballots, with specific methods announced at the meeting start.
- **Evidence** (`exact`): 'Voting should be done simultaneously.'

**76.** _requirement_
- **Q:** What public disclosure requirements follow the conclusion of an advisory committee vote?
- **A:** Promptly after the vote is taken, the names of the committee members and their corresponding votes must be read aloud and entered into the public record.
- **Evidence** (`normalized`): 'whatever method of voting is employed, the names of the committee members and their respective votes should be read aloud and otherwise made part of the public record shortly after the vote is taken.'

**77.** _condition_
- **Q:** When can an advisory committee question be discussed or clarified relative to the vote?
- **A:** Discussion and clarification should happen before voting starts. Once voting has begun, no discussion or clarification of the question is allowed while members cast their votes.
- **Evidence** (`normalized`): 'The question put to the vote should not be the subject of further discussion or clarification while the voting is underway (i.e., whereas a discussion and clarification of the question is encouraged before the vote, there should be no discu'

**78.** _procedure_
- **Q:** What protocol must an Advisory Committee Chair follow if they wish to introduce a vote on a new question not originally posed by FDA?
- **A:** The Chair must verify with the DFO or senior FDA officials that the new question fits the meeting purpose, aligns with meeting notices, and does not interfere with completed conflict-of-interest screenings.
- **Evidence** (`normalized`): 'If the Chair wants to put another question to a vote on his/her own initiative, the Chair should first check with the DFO or other senior FDA officials present to be sure that the question is appropriate for the meeting, that it is consiste'

## fda-84475

**79.** _requirement_
- **Q:** How frequently are bottled water manufacturers required to test their finished products for DEHP?
- **A:** Bottled water manufacturers must monitor finished bottled water products for DEHP as often as necessary, but at least once per year under current good manufacturing practice (CGMP) regulations.
- **Evidence** (`normalized`): 'bottled water manufacturers are required to monitor their finished bottled water products for DEHP as often as necessary, but at least once each year under the current good manufacturing practice (CGMP) regulations for bottled water.'

**80.** _scope_
- **Q:** Under what circumstance are bottled water manufacturers exempt from annual source water testing for DEHP?
- **A:** Bottlers are required to test source water for DEHP at least once a year unless they fulfill the criteria for source water testing exemptions specified under the CGMP regulations.
- **Evidence** (`normalized`): 'Bottled water manufacturers also are required to monitor for DEHP at least once each year in their source water, unless the bottlers meet the criteria for source water testing exemptions under the CGMP regulations.'

**81.** _quantity_
- **Q:** What proposed allowable level for DEHP was deferred by the FDA in its 1996 final rule?
- **A:** The FDA deferred final action on a proposed allowable level of 0.006 milligrams/liter (mg/L) for DEHP in response to a public comment.
- **Evidence** (`normalized`): 'FDA deferred final action on the proposed allowable level of 0.006 milligrams/liter (mg/L) for the chemical DEHP in response to a comment.'

**82.** _quantity_
- **Q:** What is the maximum allowable level set by the FDA for DEHP in bottled water under 21 CFR 165.110(b)(4)(iii)(C)?
- **A:** The allowable level established by the FDA for DEHP in bottled water is 0.006 milligram per liter (mg/l).
- **Evidence** (`normalized`): 'The allowable level established by FDA for DEHP in bottled water is 0.006 milligram per liter (mg/l) (21 CFR 165.110(b)(4)(iii)(C)).'

**83.** _procedure_
- **Q:** What analytical test methods are listed by FDA for checking compliance with the DEHP standard in bottled water?
- **A:** Compliance is determined using EPA Method 506, Rev. 1.1 ("Determination of phthalate and adipate esters in drinking water...") and EPA Method 525.2, Rev. 2.0 ("Determination of organic compounds in drinking water...").
- **Evidence** (`normalized`): 'Method 506, Rev. 1.1—“Determination of phtha late and adipate esters in drinking water by liquid/liquid extraction or liquid/solid extraction and gas chromatography with photoionization detection,” U.S. EPA, 1995, EPA/600/R–95/131 (21 CFR 1'

**84.** _procedure_  ⚠️ **ADMIN?**
- **Q:** Where can the specific publication containing EPA Method 506, Rev. 1.1 and EPA Method 525.2, Rev. 2.0 be accessed online?
- **A:** Both methods are contained in "Methods for the Determination of Organic Compounds in Drinking Water, Supplement III" (EPA/600/R-95/131, August 1995), which can be accessed online at http://www.epa.gov/nscep/index.html.
- **Evidence** (`normalized`): 'can be found in “Methods for the Determination of Organic Compounds in Drinking Water, Supplement III,” EPA National Exposure Research Laboratory, EPA/600/R–95/131, August 1995, which can be accessed online at http://www.epa.gov/nscep/index'
