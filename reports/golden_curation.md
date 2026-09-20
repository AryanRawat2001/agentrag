# Golden set — curation worksheet

24 of 159 pairs carry at least one flag (15.1%). Flags are **triage, not verdicts**: each one names something a bag-of-words check cannot see and a human can, and a flagged pair is as likely to be a legitimate paraphrase as an error.

| severity | pairs |
|---|---:|
| high | 20 |
| medium | 4 |
| low | 0 |

**Judged: 24 of 24 flagged pairs.** 0 still unread.

1 further verdict is kept for pairs the current detectors no longer flag. That is not backlog: a human read the pair when an earlier detector flagged it, and the fixed detector now agrees with the verdict. Such verdicts are retained so a rejection cannot be silently undone by a detector change — see `revalidate`, which honours them.

| flag | count | what it means |
|---|---:|---|
| `actor_not_in_source` | 3 | an obligation is attached to a party the source never names |
| `modal_added_to_unmodalised_source` | 2 | answer reads an obligation into a source that states none — an imperative, a bare specification, or an enumerated statutory duty |
| `modal_strengthened` | 15 | answer asserts more obligation than the source ('should' -> 'must') |
| `modal_strengthened_in_context` | 15 | against the single sentence that addresses it, the answer asserts more obligation than the source |
| `polarity_inverted` | 2 | the answer states an obligation in the polarity the source never uses |

## `ctgov-NCT00567567-Prot_SAP_000::gold::002` — high

**Q.** What standard must central venous lines meet prior to starting induction chemotherapy in ANBL0532?

**A.** All patients must have a double lumen central venous line placed before beginning induction chemotherapy.

**Closest source sentence.** 4.1.1 Central Line All patients will have a double lumen central venous line placed prior to beginning induction chemotherapy.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "will" (descriptive)
  - answer: `All patients must have a double lumen central venous line placed before...`
  - source: `...SC for transplant or who have GFR < 60 ml/min/1.73 m 2 will be taken off protocol therapy. Refer to protocol sect...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "will" (descriptive)
  - answer: `All patients must have a double lumen central venous line placed before...`
  - source: `4.1.1 Central Line All patients will have a double lumen central venous line placed prior to beginning induction chemotherapy.`

**Verdict: keep** — source: "All patients WILL have a double lumen central venous line placed prior to beginning induction chemotherapy." In a protocol, will states what the protocol mandates, so must is a fair reading and the content is exact.

## `ctgov-NCT00567567-Prot_SAP_000::gold::005` — high

**Q.** Where should immunocytochemistry bone marrow aspirate samples be shipped, and via which delivery schedule?

**A.** Samples must be shipped overnight via Federal Express (Priority Overnight) to Dr. Robert C. Seeger at the Neuroblastoma Biology Reference Laboratory, Children's Hospital of Los Angeles, for Tuesday through Friday delivery.

**Closest source sentence.** Neuroblastoma Biology Reference Laboratory Smith Research Tower, Room #509 Children's Hospital of Los Angeles 4546 Sunset Blvd.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "will" (descriptive)
  - answer: `Samples must be shipped overnight via Federal Express (Priority Ove...`
  - source: `...rate samples coll ected in sodium heparin tubes sample will be shipped overnight at room temperature for immunocy...`

**Verdict: keep** — source is imperative: "Ship via Federal Express Prior Overnight". An imperative obliges; the flagged will and may belong to other clauses. Address and delivery window are correct.

## `ctgov-NCT01511614-Prot_SAP_000::gold::004` — high

**Q.** How is the Rapid Visual Information Processing Task (RVIP) performed by participants?

**A.** Participants monitor a continuous stream of single digits on a screen and must press a button when they identify three consecutive odd or three consecutive even numbers.

**Closest source sentence.** The task has participants monitor a continuous stream of single digits presented on a screen in front of them and respond with a button press when they notice three odd or three even numbers presented consecutively (e.g., 5,9,7 or 4,2,8).

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "will" (descriptive)
  - answer: `...r a continuous stream of single digits on a screen and must press a button when they identify three consecutive od...`
  - source: `...lly inserted, and before the PM scan session, a nurse will take a blood sample from participants that will later...`

**Verdict: keep** — source describes task mechanics ("respond with a button press when they notice three odd or three even numbers"). The answer must press is framing, and the task content -- three consecutive odd or even digits -- is exact.

## `ctgov-NCT02942264-Prot_SAP_000::gold::012` — high

**Q.** What surgical sample tissue specimens are required before a patient can enter the clinical trial?

**A.** A representative paraffin tissue block measuring at least 5 mm x 5 mm or unstained slides from previous surgery must be obtained to confirm histological diagnosis and molecular markers.

**Closest source sentence.** A representative paraffin tissue block (at least 5 mm x 5 mm) or unstained slides will be obtained from previous surgery to confirm the histological diagnosis and molecular markers before entry into the clinical trial.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "will" (descriptive)
  - answer: `...5 mm x 5 mm or unstained slides from previous surgery must be obtained to confirm histological diagnosis and mole...`
  - source: `2.2.3 Unstained paraffin tissue from surgical samples will be obtained. A representative paraffin tissue block (...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "will" (descriptive)
  - answer: `...5 mm x 5 mm or unstained slides from previous surgery must be obtained to confirm histological diagnosis and mole...`
  - source: `A representative paraffin tissue block (at least 5 mm x 5 mm) or unstained slides will be obtained from previous surgery to confirm the histological diagnosis and molecular markers before entry into the clinical trial.`

**Verdict: keep** — protocol will be obtained -> must be obtained; tissue-block dimensions (5 mm x 5 mm) and stated purpose are exact.

## `ctgov-NCT02942264-Prot_SAP_000::gold::016` — high

**Q.** What is the reporting timeframe requirement if a patient becomes pregnant during the study?

**A.** Study treatment must be discontinued immediately, and the pregnancy must be reported to the Sponsor no later than 24 hours after the Investigator becomes aware of it.

**Closest source sentence.** 8.5 REPORTING PREGNANCY 8.5.1 Maternal exposure If a patient becomes pregnant during the course of the study, the study treatment should be discontinued immediately, and the pregnancy reported to the Sponsor no later than 24 hours of when the Investigator becomes aware of it.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "should" (recommended)
  - answer: `Study treatment must be discontinued immediately, and the pregnancy must be...`
  - source: `...nt during the course of the study, the study treatment should be discontinued immediately, and the pregnancy report...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "should" (recommended)
  - answer: `Study treatment must be discontinued immediately, and the pregnancy must be...`
  - source: `8.5 REPORTING PREGNANCY 8.5.1 Maternal exposure If a patient becomes pregnant during the course of the study, the study treatment should be discontinued immediately, and the pregnancy reported to the Sponsor no later tha`

**Verdict: reject** — source: "the study treatment SHOULD be discontinued immediately, and the pregnancy reported to the Sponsor no later than 24 hours". Answer: "MUST be discontinued" and "MUST be reported". A recommendation restated as a requirement, on a pregnancy safety instruction. The 24-hour figure is right; the obligation level is not.

## `ctgov-NCT04125745-Prot_SAP_000::gold::021` — high

**Q.** What are the storage requirements and temperature limits for CXA-10 capsules?

**A.** CXA-10 capsules must be refrigerated between 2 and 8°C (35 to 46°F). They can be kept at room temperature (20 to 25°C) for up to 48 hours, but should never be stored in direct sunlight or at temperatures exceeding 25°C.

**Closest source sentence.** CXA-10 capsules should be stored refrigerated between 2 and 8°C (35 to 46°F).

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "will" (descriptive)
  - answer: `CXA-10 capsules must be refrigerated between 2 and 8°C (35 to 46°F). They c...`
  - source: `...bility records of study drug received, dispensed, used will be maintained throughout the course of the study. Th...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "should" (recommended)
  - answer: `CXA-10 capsules must be refrigerated between 2 and 8°C (35 to 46°F). They c...`
  - source: `CXA-10 capsules should be stored refrigerated between 2 and 8°C (35 to 46°F).`
- **polarity_inverted** (high) — answer uses "should" negated; the cited source only ever states it unnegated
  - answer: `...room temperature (20 to 25°C) for up to 48 hours, but should never be stored in direct sunlight or at temperatures...`
  - source: `...essible only to authorized personnel. CXA-10 capsules should be stored refrigerated between 2 and 8°C (35 to 46°F)...`

**Verdict: reject** — source: "CXA-10 capsules SHOULD be stored refrigerated between 2 and 8C". Answer: "MUST be refrigerated". A storage recommendation restated as a requirement.

## `ctgov-NCT04125745-Prot_SAP_000::gold::023` — high

**Q.** Under what circumstances will the clinical trial be halted for safety review after the first two subjects are enrolled?

**A.** The study will be discontinued and halted for data review if the first two enrolled subjects experience any unexpected fatal or life-threatening events that are attributable to the study drug.

**Closest source sentence.** For safety reasons, we may discontinue the study if the first two subjects enrolled experience any unexpected fatal or life-threatening events that can be attributed to the study drug;

- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "will" (descriptive) where the source says "can" (permissive)
  - answer: `The study will be discontinued and halted for data review if the firs...`
  - source: `For safety reasons, we may discontinue the study if the first two subjects enrolled experience any unexpected fatal or life-threatening events that can be attributed to the study drug;`

**Verdict: reject** — source: "For safety reasons, we MAY discontinue the study if the first two subjects enrolled experience any unexpected fatal or life-threatening events". Answer: "The study WILL be discontinued and halted". A discretionary safety action restated as a commitment. Caught only by the sentence-level check: the chunk-level comparison mislabelled it as weakened, because the chunk contains required in another clause.

## `ctgov-NCT04125745-Prot_SAP_000::gold::025` — high

**Q.** What is the reporting timeline for notifying the FDA about a fatal or life-threatening adverse event?

**A.** The FDA must be notified via telephone or facsimile within 7 calendar days of the study receiving information about a fatal or life-threatening human adverse event.

**Closest source sentence.** The FDA will be notified by telephone or facsimile transmission of a human adverse event that is fatal or life-threatening no later than 7 calendar days after receiving the respective human adverse event information, followed by the subsequent submission of a written IND Safety Report.

- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "will" (descriptive)
  - answer: `The FDA must be notified via telephone or facsimile within 7 calend...`
  - source: `The FDA will be notified by telephone or facsimile transmission of a human adverse event that is fatal or life-threatening no later than 7 calendar days after receiving the respective human adverse event information, fol`

**Verdict: keep** — the protocol commits: "The FDA WILL be notified by telephone or facsimile ... no later than 7 calendar days". The answer says must be notified within 7 calendar days -- same commitment, same deadline.

## `ctgov-NCT04125745-Prot_SAP_000::gold::027` — high

**Q.** What dietary instructions regarding fat content must subjects follow when taking CXA-10?

**A.** Subjects should take CXA-10 once daily in the morning with food. The meal or snack can contain up to approximately 30% fat, but high-fat meals must be avoided.

**Closest source sentence.** Subjects will be instructed to take their CXA -10 dose once daily in the morning with food.

- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "will" (descriptive)
  - answer: `...ontain up to approximately 30% fat, but high-fat meals must be avoided.`
  - source: `Subjects will be instructed to take their CXA -10 dose once daily in the morning with food.`
- **polarity_inverted** (high) — answer uses "should" unnegated; the cited source only ever states it negated
  - answer: `Subjects should take CXA-10 once daily in the morning with food. The m...`
  - source: `...Dosing will then commence at home. However, subjects should not take their morning dose of study medication at ho...`

**Verdict: keep** — the disputed claim is verbatim in the source: "A high fat meal MUST be avoided." The sentence-level check selected the wrong sentence because the first clause of the answer dominated the overlap score, and the polarity flag compared should across two unrelated clauses.

## `ctgov-NCT04660552-Prot_003::gold::024` — high

**Q.** Under what circumstance must a child participant be withdrawn from the study regarding session attendance?

**A.** A child must be withdrawn from the study if a parent cannot reschedule a cancelled session within a 3 day window.

**Closest source sentence.** cancelled session within a 3 day window, then the child will have to be withdrawn from the study.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "will" (descriptive)
  - answer: `A child must be withdrawn from the study if a parent cannot resched...`
  - source: `...ancelled session within a 3 day window, then the child will have to be withdrawn from the study. • If the child b...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "will" (descriptive)
  - answer: `A child must be withdrawn from the study if a parent cannot resched...`
  - source: `cancelled session within a 3 day window, then the child will have to be withdrawn from the study.`

**Verdict: keep** — source: "then the child WILL HAVE TO BE withdrawn from the study". Will have to be is an obligation, so must be withdrawn is correct; the lexicon ranks the bare will and cannot see the periphrastic form.

## `fda-100907::gold::025` — high

**Q.** How many days from the publication of the notice in the Federal Register are allowed for submitting comments on the draft guidance?

**A.** Comments on the draft guidance should be submitted within 180 days of the publication of the notice announcing its availability in the Federal Register.

**Closest source sentence.** Although you can comment on any guidance at any time (see 21 CFR 10.115(g)(5)), to ensure that FDA considers your comment on this draft guidance before we begin work on the final version of the guidance, submit either electronic or written comments on the draft guidance within 180days of publication in the Federal Register of the notice announcing the availability of the draft guidance.

- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "should" (recommended) where the source says "can" (permissive)
  - answer: `Comments on the draft guidance should be submitted within 180 days of the publication of the...`
  - source: `Although you can comment on any guidance at any time (see 21 CFR 10.115(g)(5)), to ensure that FDA considers your comment on this draft guidance before we begin work on the final version of the guidance, submit either el`

**Verdict: keep** — the 180-day comment window and should be submitted both match the guidance; the selected sentence was the you can comment at any time aside rather than the deadline clause.

## `fda-136987::gold::044` — high

**Q.** How often are firms typically asked to provide Recall Status Reports to the FDA?

**A.** Recall Status Reports are usually requested on a monthly basis, though they may be required more frequently if indicated.

**Closest source sentence.** You will be asked to provide Recall Status Reports to your DRC after initiating a recall (usually on Contains Nonbinding Recommendations 16 a monthly basis but more frequently when indicated).

- **modal_strengthened** (high) — answer asserts "required" (mandatory); strongest in source is "will" (descriptive)
  - answer: `...ually requested on a monthly basis, though they may be required more frequently if indicated.`
  - source: `...ling firm (or sub recalling firm if such is the case) will then be requested by FDA to take appropriate actions,...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "required" (mandatory) where the source says "will" (descriptive)
  - answer: `...ually requested on a monthly basis, though they may be required more frequently if indicated.`
  - source: `You will be asked to provide Recall Status Reports to your DRC after initiating a recall (usually on Contains Nonbinding Recommendations 16 a monthly basis but more frequently when indicated).`

**Verdict: keep** — the answer hedges correctly -- MAY BE REQUIRED more frequently if indicated, against a source reading usually on a monthly basis but more frequently when indicated. The flag fires on required inside a permissive phrase, which a max-strength rule cannot see past.

## `fda-171592::gold::053` — high

**Q.** What process must trading partners follow to handle saleable returns under section 582(g)(1)(F) starting November 27, 2023?

**A.** Trading partners must have systems and processes in place to accept saleable returns by associating the saleable return product with its corresponding transaction information and transaction statement.

**Closest source sentence.** 15 and (5) Have systems and processes in place to accept saleable returns under appropriate conditions, i.e., being able to associate the saleable return product with the transaction information and transaction statement associated with that product.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "can" (permissive)
  - answer: `Trading partners must have systems and processes in place to accept saleable...`
  - source: `...3, wholesale distributors, dispensers, and repackagers can use either paper-based or electronic-based methods 17...`

**Verdict: keep** — source enumerates statutory duties under section 582(g)(1)(F): "(5) Have systems and processes in place to accept saleable returns...". The guidance is non-binding but the enumerated items restate the statute, which obliges.

## `fda-171592::gold::058` — high

**Q.** What system capabilities are required regarding information retrieval going back to the manufacturer during an investigation?

**A.** Trading partners must have systems and processes to facilitate gathering the information required to produce transaction information for a product going back to the manufacturer in the event of a recall, suspect or illegitimate product investigation, or an authorized trading partner request.

**Closest source sentence.** Contains Nonbinding Recommendations 3 (3) Have systems and processes in place to promptly respond with the transaction information and transaction statement for a product upon a request by the Secretary,13 or other appropriate Federal or State official, in the event of a recall or for investigations of suspect or illegitimate product;14 (4) Have systems and processes in place to facilitate the gat

- **modal_strengthened** (high) — answer asserts "required" (mandatory); strongest in source is "recommendations" (recommended)
  - answer: `...and processes to facilitate gathering the information required to produce transaction information for a product going...`
  - source: `...ion 582(g)(1)(C) of the FD&C Act. Contains Nonbinding Recommendations 3 (3) Have systems and processes in place to prom...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "required" (mandatory) where the source says "recommendations" (recommended)
  - answer: `...and processes to facilitate gathering the information required to produce transaction information for a product going...`
  - source: `Contains Nonbinding Recommendations 3 (3) Have systems and processes in place to promptly respond with the transaction information and transaction statement for a product upon a request by the Secretary,13 or other appro`

**Verdict: keep** — same document and pattern as ::053 -- an enumerated statutory duty, restated.

## `fda-75426::gold::085` — high

**Q.** What design standards should be met when drafting a question for an advisory committee vote?

**A.** Questions presented for a vote must contain minimal qualifiers, must not be leading, and should avoid using double or triple negatives.

**Closest source sentence.** • The question presented for a vote should have minimal qualifiers, not be leading, and should avoid the use of double or triple negatives.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "encouraged" (recommended)
  - answer: `Questions presented for a vote must contain minimal qualifiers, must not be leading, and s...`
  - source: `...• The Chair and DFO of an advisory committee are encouraged to generate a robust discussion about the matter at i...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "should" (recommended)
  - answer: `Questions presented for a vote must contain minimal qualifiers, must not be leading, and s...`
  - source: `• The question presented for a vote should have minimal qualifiers, not be leading, and should avoid the use of double or triple negatives.`

**Verdict: reject** — source: "The question presented for a vote SHOULD have minimal qualifiers, not be leading, and should avoid the use of double or triple negatives." Answer: "MUST contain minimal qualifiers, MUST not be leading". Two strengthenings in one answer, in advisory-committee guidance where should is explicitly non-binding.

## `fda-75426::gold::087` — high

**Q.** What public disclosure requirements follow the conclusion of an advisory committee vote?

**A.** Promptly after the vote is taken, the names of the committee members and their corresponding votes must be read aloud and entered into the public record.

**Closest source sentence.** Further, whatever method of voting is employed, the names of the committee members and their respective votes should be read aloud and otherwise made part of the public record shortly after the vote is taken.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "recommendations" (recommended)
  - answer: `...of the committee members and their corresponding votes must be read aloud and entered into the public record.`
  - source: `...l of Economics, 107, 797-817. 5 Contains Nonbinding Recommendations 6 include a simultaneous show of hands, a simul...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "should" (recommended)
  - answer: `...of the committee members and their corresponding votes must be read aloud and entered into the public record.`
  - source: `Further, whatever method of voting is employed, the names of the committee members and their respective votes should be read aloud and otherwise made part of the public record shortly after the vote is taken.`

**Verdict: reject** — source: "the names of the committee members and their respective votes SHOULD be read aloud". Answer: "MUST be read aloud and entered into the public record."

## `fda-75426::gold::089` — high

**Q.** What protocol must an Advisory Committee Chair follow if they wish to introduce a vote on a new question not originally posed by FDA?

**A.** The Chair must verify with the DFO or senior FDA officials that the new question fits the meeting purpose, aligns with meeting notices, and does not interfere with completed conflict-of-interest screenings.

**Closest source sentence.** If the Chair wants to put another question to a vote on his/her own initiative, the Chair should first check with the DFO or other senior FDA officials present to be sure that the question is appropriate for the meeting, that it is consistent with the topic identified in the meeting notices, and that it will not affect the conflict-of-interest screening that had been completed prior to the meeting

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "encouraged" (recommended)
  - answer: `The Chair must verify with the DFO or senior FDA officials that the n...`
  - source: `ion is encouraged before the vote, there should be no discussion of the...`
- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "should" (recommended)
  - answer: `The Chair must verify with the DFO or senior FDA officials that the n...`
  - source: `If the Chair wants to put another question to a vote on his/her own initiative, the Chair should first check with the DFO or other senior FDA officials present to be sure that the question is appropriate for the meeting,`

**Verdict: reject** — source: "the Chair SHOULD first check with the DFO or other senior FDA officials". Answer: "The Chair MUST verify with the DFO or senior FDA officials."

## `fda-78268::gold::094` — high

**Q.** How should files be named when submitting an amendment, update, or supplement in electronic format?

**A.** Each filename for an amendment, update, or supplement must begin with the date of submission in the YYYY-MM-DD format.

**Closest source sentence.** (For example, an amendment or update to a FAP would include a separate file directed to each topic addressed in the amendment or update, whereas an amendment or supplement to a GRAS notice would include a single file regardless of the number of topics addressed in the amendment or supplement.) • Begin each filename with the date (YYYY-MM-DD) the amendment, update or supplement is submitted.

- **modal_strengthened** (high) — answer asserts "must" (mandatory); strongest in source is "recommendations" (recommended)
  - answer: `Each filename for an amendment, update, or supplement must begin with the date of submission in the YYYY-MM-DD fo...`
  - source: `...nd Supplements in Electronic Format 21. What special recommendations apply to the submission of amendments, updates, or su...`

**Verdict: reject** — source frames the whole list as advice: "What special RECOMMENDATIONS apply ... The following RECOMMENDATIONS apply to amendments, updates, and supplements", and the item itself is "Begin each filename with the date (YYYY-MM-DD)". Answer: "must begin with the date". Changed from keep to reject at the Phase 8 code-review gate. My original keep was wrong twice over: the recorded reason only checked content (the YYYY-MM-DD form) and never addressed obligation level, and it was inconsistent with ::097 in the same document, which I rejected for exactly this. The detector had steered me there -- `recommend`/`recommendations` were missing from the modal lexicon, so the frame scored 0 and routed to the lenient modal_added_to_unmodalised_source branch at medium severity instead of high.

## `fda-78268::gold::097` — high

**Q.** Which types of submissions must always include a Table of Contents regardless of whether they are paper or electronic?

**A.** A Table of Contents is required for any submission regarding a GRAS notice, Biotechnology Final Consultation, or New Protein Consultation, including new submissions, amendments, or supplements.

**Closest source sentence.** • With any submission regarding a GRAS notice, Biotechnology Final Consultation, or New Protein Consultation (whether the submission is a new submission or an amendment or supplement to a previous submission), regardless of whether the submission is transmitted in paper or electronic format;

- **modal_strengthened** (high) — answer asserts "required" (mandatory); strongest in source is "should" (recommended)
  - answer: `A Table of Contents is required for any submission regarding a GRAS notice, Biotechnol...`
  - source: `D. Table of Contents 10. Should I include a Table of Contents with my regulatory submi...`

**Verdict: reject** — source: "You SHOULD include a Table of Contents: - With any submission regarding a GRAS notice...". Answer: "A Table of Contents IS REQUIRED for any submission". The question is leading in the same direction (must always include), so both halves assert an obligation the guidance does not.

## `fda-88114::gold::103` — high

**Q.** What action should an investigator take if a research subject is entered into a study with waived consent but dies before their family can be reached?

**A.** If feasible, the investigator must provide information regarding the clinical investigation to the subject's legally authorized representative or a family member, which includes a same-sex spouse.

**Closest source sentence.** If a subject is entered into a clinical investigation with waived consent and the subject dies before a legally authorized representative or family member, which can include a same-sex spouse, can be contacted, information about the clinical investigation is to be provided to the

- **modal_strengthened_in_context** (high) — against the sentence that addresses it, the answer asserts "must" (mandatory) where the source says "can" (permissive)
  - answer: `If feasible, the investigator must provide information regarding the clinical investigati...`
  - source: `If a subject is entered into a clinical investigation with waived consent and the subject dies before a legally authorized representative or family member, which can include a same-sex spouse, can be contacted, informati`
- **actor_not_in_source** (medium) — answer places an obligation on "investigator", which the cited source never names (source names: irb, subject)
  - answer: `If feasible, the investigator must provide information regarding the clinical invest...`

**Verdict: keep** — investigator duties under the exception-from-informed-consent regulation are requirements. The sentence-level check selected the clause about a subject dying before the representative was contacted rather than the obligation clause.

## `ctgov-NCT02717455-Prot_SAP_000::gold::016` — medium

**Q.** What are the required parameters for Axial T2 FLAIR images during a 1.5 T Brain MR?

**A.** For the Axial T2 FLAIR sequence, the slice thickness must be 5 mm with a skip of 0 mm, using a 20 cm field of view (FOV).

**Closest source sentence.** 03/26/2021 70 • Axial T2 FLAIR images (slice thickness 5 mm skip 0 mm, 20 cm FOV) • Axial DWI, 6 directions (slice thickness 5 mm skip 0 mm, 26 cm FOV) • Post gadolinium sagittal 3DFSPGR images (slice thickness 1.5 mm no skip, 24 cm FOV) • Axial T1 post gadolinium (slice thickness 3 mm no skip, 16 cm

- **modal_added_to_unmodalised_source** (medium) — answer asserts "must" (mandatory); strongest in source is no modal at all
  - answer: `For the Axial T2 FLAIR sequence, the slice thickness must be 5 mm with a skip of 0 mm, using a 20 cm field of vi...`

**Verdict: keep** — source is a bare specification: "Axial T2 FLAIR images (slice thickness 5 mm skip 0 mm, 20 cm FOV)". Every figure matches, and protocol-specified imaging parameters are requirements.

## `ctgov-NCT02717455-Prot_SAP_000::gold::017` — medium

**Q.** What criteria must be met for a patient to be considered as having a Complete Response (CR) based on MR imaging?

**A.** A Complete Response requires the total disappearance of all evaluable tumor and mass effect on MR, while the patient is on a stable or decreasing dose of corticosteroids and has a stable or improving neurologic exam.

**Closest source sentence.** 11.2 Tumor Response Criteria 11.2.1 Complete Response (CR) Complete disappearance on MR of all evaluable tumor and mass effect, on a stable or decreasing dose of corticosteroids (or receiving only adrenal replacement doses), accompanied by a stable or improving neurologic examination.

- **actor_not_in_source** (medium) — answer places an obligation on "patient", which the cited source never names (source names: no party)
  - answer: `...f all evaluable tumor and mass effect on MR, while the patient is on a stable or decreasing dose of corticosteroids a...`

**Verdict: keep** — A Complete Response REQUIRES the total disappearance of all evaluable tumor is definitional, not a duty owed by the patient. The attribution check reads a definition as an obligation.

## `fda-113923::gold::033` — medium

**Q.** What documentation must be maintained when an onsite audit is conducted as part of supplier verification?

**A.** Documentation must include the name of the supplier audited, the audit procedures, the dates conducted, audit conclusions, corrective actions taken for significant deficiencies, and evidence that the audit was performed by a qualified auditor.

**Closest source sentence.** (v) Corrective actions taken in response to significant deficiencies identified during the audit;

- **modal_added_to_unmodalised_source** (medium) — answer asserts "must" (mandatory); strongest in source is no modal at all
  - answer: `Documentation must include the name of the supplier audited, the audit pr...`

**Verdict: keep** — source is 21 CFR 507.175(c)(7), a binding regulation enumerating required audit documentation (i)-(vi). Must include is correct; the regulation carries the obligation in its heading rather than in each item.

## `fda-72414::gold::089` — medium

**Q.** To whom must an ANDA applicant provide notice of a paragraph IV certification?

**A.** The applicant is required to provide notice to the holder of the approved NDA and to every owner of the patent that is the subject of the certification.

**Closest source sentence.** A notice of the paragraph IV certification must be provided to each owner of the patent that is the subject of the certification and to the holder of the approved NDA to which the ANDA refers.

- **actor_not_in_source** (medium) — answer places an obligation on "applicant", which the cited source never names (source names: agency, fda, subject)
  - answer: `The applicant is required to provide notice to the holder of the app...`

**Verdict: keep** — the source states the duty in the passive ("A notice of the paragraph IV certification MUST be provided to each owner"), so it names no actor and the applicant -- who does provide it -- looks novel. Passive voice defeats the attribution check.
