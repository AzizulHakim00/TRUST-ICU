# Protocol Amendment 001: Landmark Unit and Follow-up Policy

**Date:** 2026-08-04  
**Stage:** Before credentialed cohort extraction and before any outcome-frequency or model-performance inspection  
**Reason:** Cross-database harmonization and explicit competing-event handling

## Change 1: sampling unit

The original wording, “first eligible ICU stay per patient,” is replaced with:

> **First eligible ICU unit stay per hospital admission.**

MIMIC-IV hospital admissions can be ordered within a patient, but eICU documentation states
that separate health-system stays for the same `uniquepid` cannot always be chronologically
ordered. eICU does provide `patientHealthSystemStayID` and `unitVisitNumber`, which support
ordering ICU units within one hospital admission. The revised unit is therefore reproducible
in both databases and avoids inventing cross-admission chronology.

This amendment does not use labels or performance results.

## Change 2: six-hour landmark eligibility

A stay enters the landmark cohort only if the patient:

1. is at least 18 years old;
2. remains in the ICU strictly after six hours from ICU admission; and
3. is alive strictly after the six-hour landmark.

The observation interval remains `[0 h, 6 h)`. A measurement at exactly six hours is not a
predictor. An organ-support start at exactly six hours belongs to the prediction interval.

## Change 3: follow-up

The administrative prediction boundary remains 18 hours after ICU admission. Follow-up ends
at the earliest of:

- ICU discharge;
- death; or
- the 18-hour administrative boundary.

For the primary in-unit binary estimand, discharge without organ-support initiation is a
negative outcome. Death before 18 hours without the organ support is a competing event and is
excluded from the primary binary model, while its count and characteristics are reported.
Sensitivity analyses must include alternative handling of this competing event before a
clinical claim is made.

## Change 4: identifiers

`patient_id`, `hospital_admission_id`, `stay_id`, `hospital_id`, `site_id` and `dataset_id` are
retained for linkage, grouped splitting and evaluation only. They are prohibited predictors.

## Consequence

No model development is authorized by this amendment. Outcome contracts must still be locked,
source mappings validated and synthetic boundary tests passed before credentialed model
training.
