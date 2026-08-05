-- TRUST-ICU Phase 0: local reviewed eICU mapping tables.
-- PostgreSQL syntax. Run only inside the credentialed environment.
-- Populate from the public CSV templates after local frequency review and clinical approval.

CREATE SCHEMA IF NOT EXISTS trust_icu_local;

CREATE TABLE IF NOT EXISTS trust_icu_local.eicu_feature_map (
    source_table text NOT NULL,
    source_code_normalized text NOT NULL,
    source_unit_normalized text NOT NULL DEFAULT '',
    canonical_variable text NOT NULL,
    canonical_unit text NOT NULL,
    multiplier double precision NOT NULL DEFAULT 1.0,
    offset double precision NOT NULL DEFAULT 0.0,
    status text NOT NULL CHECK (status IN ('pending_local_review', 'locked')),
    reviewer text,
    notes text,
    PRIMARY KEY (source_table, source_code_normalized, source_unit_normalized)
);

CREATE TABLE IF NOT EXISTS trust_icu_local.eicu_outcome_map (
    task text NOT NULL,
    source_table text NOT NULL,
    source_code_normalized text NOT NULL,
    classification text NOT NULL CHECK (classification IN ('positive', 'negative', 'corroborating')),
    status text NOT NULL CHECK (status IN ('pending_local_review', 'locked')),
    reviewer text,
    notes text,
    PRIMARY KEY (task, source_table, source_code_normalized)
);
