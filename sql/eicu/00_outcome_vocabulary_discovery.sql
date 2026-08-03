-- TRUST-ICU Phase 0: eICU-CRD v2.0 outcome vocabulary discovery.
-- Dialect: PostgreSQL.
-- IMPORTANT: output may contain credentialed aggregate counts. Store it outside GitHub.
-- Text matching here discovers candidates only; it MUST NOT be used as a final outcome label.

SET search_path TO eicu_crd;

DROP TABLE IF EXISTS trust_icu_outcome_vocabulary_candidates;
CREATE TEMP TABLE trust_icu_outcome_vocabulary_candidates AS

-- Ventilation: structured airway and vent timing documentation.
SELECT
  'invasive_mechanical_ventilation'::text AS task,
  'respiratorycare.airwaytype'::text AS source_field,
  lower(trim(rc.airwaytype)) AS candidate_text,
  count(*)::bigint AS rows,
  count(DISTINCT rc.patientunitstayid)::bigint AS unit_stays,
  count(DISTINCT p.hospitalid)::bigint AS hospitals,
  min(COALESCE(rc.ventstartoffset, rc.respcarestatusoffset))::integer AS min_offset,
  max(COALESCE(rc.ventstartoffset, rc.respcarestatusoffset))::integer AS max_offset
FROM respiratorycare AS rc
JOIN patient AS p USING (patientunitstayid)
WHERE rc.airwaytype IS NOT NULL
GROUP BY lower(trim(rc.airwaytype))

UNION ALL

SELECT
  'invasive_mechanical_ventilation',
  'treatment.treatmentstring',
  lower(trim(t.treatmentstring)),
  count(*)::bigint,
  count(DISTINCT t.patientunitstayid)::bigint,
  count(DISTINCT p.hospitalid)::bigint,
  min(t.treatmentoffset)::integer,
  max(t.treatmentoffset)::integer
FROM treatment AS t
JOIN patient AS p USING (patientunitstayid)
WHERE t.treatmentstring ~* '(mechanical[[:space:]]+vent|invasive[[:space:]]+vent|intubat|endotracheal|ett)'
GROUP BY lower(trim(t.treatmentstring))

UNION ALL

SELECT
  'invasive_mechanical_ventilation',
  'respiratorycharting.label_value',
  lower(trim(rc.respchartvaluelabel || ' = ' || COALESCE(rc.respchartvalue, ''))),
  count(*)::bigint,
  count(DISTINCT rc.patientunitstayid)::bigint,
  count(DISTINCT p.hospitalid)::bigint,
  min(rc.respchartoffset)::integer,
  max(rc.respchartoffset)::integer
FROM respiratorycharting AS rc
JOIN patient AS p USING (patientunitstayid)
WHERE (rc.respchartvaluelabel || ' ' || COALESCE(rc.respchartvalue, ''))
  ~* '(ventilator|vent mode|endotracheal|ett|assist.control|volume control|pressure control|simv|prvc|aprv)'
GROUP BY lower(trim(rc.respchartvaluelabel || ' = ' || COALESCE(rc.respchartvalue, '')))

UNION ALL

-- Vasopressors: discover all local aliases and units for the five canonical agents.
SELECT
  'vasopressor_initiation',
  'infusiondrug.drugname',
  lower(trim(i.drugname)),
  count(*)::bigint,
  count(DISTINCT i.patientunitstayid)::bigint,
  count(DISTINCT p.hospitalid)::bigint,
  min(i.infusionoffset)::integer,
  max(i.infusionoffset)::integer
FROM infusiondrug AS i
JOIN patient AS p USING (patientunitstayid)
WHERE i.drugname ~* '(norepinephrine|noradrenaline|levophed|epinephrine|adrenaline|vasopressin|phenylephrine|neosynephrine|neo-synephrine|dopamine)'
GROUP BY lower(trim(i.drugname))

UNION ALL

-- RRT: active-treatment candidates from structured treatment paths.
SELECT
  'renal_replacement_therapy',
  'treatment.treatmentstring',
  lower(trim(t.treatmentstring)),
  count(*)::bigint,
  count(DISTINCT t.patientunitstayid)::bigint,
  count(DISTINCT p.hospitalid)::bigint,
  min(t.treatmentoffset)::integer,
  max(t.treatmentoffset)::integer
FROM treatment AS t
JOIN patient AS p USING (patientunitstayid)
WHERE t.treatmentstring ~* '(dialysis|hemodialysis|haemodialysis|crrt|cvvh|cvvhd|cvvhdf|sled|peritoneal)'
GROUP BY lower(trim(t.treatmentstring))

UNION ALL

-- RRT: fluid-balance labels can corroborate active therapy, but are not sufficient alone.
SELECT
  'renal_replacement_therapy',
  'intakeoutput.cellpath_label',
  lower(trim(io.cellpath || ' | ' || io.celllabel)),
  count(*)::bigint,
  count(DISTINCT io.patientunitstayid)::bigint,
  count(DISTINCT p.hospitalid)::bigint,
  min(io.intakeoutputoffset)::integer,
  max(io.intakeoutputoffset)::integer
FROM intakeoutput AS io
JOIN patient AS p USING (patientunitstayid)
WHERE (io.cellpath || ' ' || io.celllabel)
  ~* '(dialysis|hemodialysis|haemodialysis|crrt|cvvh|cvvhd|cvvhdf|sled|peritoneal)'
GROUP BY lower(trim(io.cellpath || ' | ' || io.celllabel));

-- Candidate vocabulary, ordered to prioritize high-coverage terms.
SELECT
  task,
  source_field,
  candidate_text,
  rows,
  unit_stays,
  hospitals,
  min_offset,
  max_offset
FROM trust_icu_outcome_vocabulary_candidates
ORDER BY task, source_field, unit_stays DESC, candidate_text;

-- Coverage summary. Low hospital coverage may reflect missing interfaces rather than no treatment.
SELECT
  task,
  source_field,
  sum(rows) AS rows,
  count(DISTINCT candidate_text) AS candidate_terms,
  max(hospitals) AS maximum_hospitals_for_any_term,
  sum(unit_stays) AS term_level_unit_stay_count_not_deduplicated
FROM trust_icu_outcome_vocabulary_candidates
GROUP BY task, source_field
ORDER BY task, source_field;
