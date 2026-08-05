# Protocol Amendment 002: Locked Phase 0 Baseline Split and Selection

**Status:** prospective, written before credentialed model execution  
**Reason:** the original Phase 0 protocol specified temporal development validation but did not fully operationalize patient isolation, calibration, or selection between the two prespecified baselines.

## Amendment

For every primary task, MIMIC-IV eligible admissions are ordered by ICU admission time and divided into 70% training, 15% calibration, and 15% temporal testing. When a patient appears in an earlier split, all of that patient's rows in later splits are removed. No row is moved backward or randomly reassigned.

Logistic Regression and CatBoost remain the only Phase 0 models. Both are trained on the training split. Platt calibration is fitted using the calibration split. The development-selected baseline is the model with the higher calibrated temporal-test PR-AUC; lower Brier score breaks an exact PR-AUC tie, and Logistic Regression is the final simplicity tie-breaker.

Both prespecified baselines may be reported on eICU for transparent comparison, but the go/no-go gate is applied only to the model selected using MIMIC-IV. eICU cannot be used to change models, features, calibration, thresholds, hyperparameters, or this selection rule.

External uncertainty is estimated with a hospital-cluster bootstrap. Hospital-level results are released only as aggregate summaries without site identifiers.

## Scientific effect

This amendment reduces patient-overlap leakage, separates calibration from temporal evaluation, and prevents external-cohort model selection. It does not use or respond to any credentialed outcome results.
