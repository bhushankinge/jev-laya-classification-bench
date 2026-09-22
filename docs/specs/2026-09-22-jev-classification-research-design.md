# Jev classification research pipeline — design

Date: 2026-09-22. Status: approved design (chat approval 2026-09-22). Owner: research lead.
Companion documents: `Federal_Opportunity_Classification_Report.pdf` (Section 1 defines the class set),
`qwen_discovery/QWEN_FINDINGS.md` (text-grounded distributions), `~/<bench-repo>` (Laya/Jev harness).

## 1. Goal

Decide, with measured evidence, how Jev (TypeSafe System One API), Laya (its open-weights sibling, self-hosted)
and Qwen3.5-35B (PCAI) should share the job of classifying federal opportunities, what confidence cutoffs make an
answer safe to accept without a human, and where humans are still needed. Deliver a research pipeline that can be
re-run on any prompt or model change, a Classification Review page in the CRM that serves the research now and
production later, and a written production target with automation triggers keyed on the labels.

Out of scope for this spec: the production classifier implementation itself (a separate plan informed by the
results), ITES-4H labeling experiments (the schema covers ITES-4H; experiments run on SEWP, GSA MAS, GSA 2GIT).

## 2. Label schema (what every labeler must output)

Fixed by the report's Section 1. Every labeler (Jev, Laya, Qwen, human) emits this per opportunity:

| Field | Values | Cardinality |
|---|---|---|
| `primary_class` | Hardware · Software · Maintenance & Support · Services · Installation & Integration · Furniture / Facilities · Other | one |
| `components` | subset of the 12 subclasses: General hardware · Consumables/Supplies · Perpetual license · Subscription/SaaS · Cloud/Hosting · Support/Maintenance contract · Warranty · Professional/Consulting · Training · Installation/Integration · Furniture/Facilities · Other | one or more |
| `lifecycle` | new · renewal · upgrade/expansion · replacement/refresh · unknown | one (of the primary component) |
| `domain` | the 15 domains in the report | one (of the primary component) |
| `fulfillment_mode` | à la carte · configured build · mixed · not applicable | one; `not applicable` when no hardware component |
| `flags` | `rfi_market_research`, `text_insufficient`, `brand_name_only` | booleans |
| `confidence` | per field, 0–1 | from the model; humans emit 1.0 |
| `source` | jev · laya · qwen · human · quote | provenance |

Fulfillment mode definitions:
- **à la carte**: every hardware item is an orderable SKU with a list/contract price that a rep can add to a quote
  from a distributor or catalog (TD SYNNEX, Ingram, D&H, B&H, GSA Advantage-style). No OEM configurator step.
- **configured build**: at least one item must be configured in an OEM portal (Cisco CCW, HPE OCA/iQuote,
  Dell Premier/OSC, NetApp, Pure, similar) where the price depends on the configuration, deal registration and
  human input, and can change between iterations.
- **mixed**: both present (a configured core plus à la carte add-ons).

## 3. Roles of the three models and of quotes and humans

- **Jev** (`jev-latest`, `https://api.typesafe.ai/v1/systemone`): first pass. One request per opportunity,
  one state text, a bundle of typed questions (Section 4). Returns a top choice, a probability per option and a
  confidence per question. Measured: p50 ≈ 150 ms from Arizona, 1,500 req/min accepted, $0.042 per million input
  tokens (state counted once per request). Key lives in `~/<bench-repo>/.env` as `JEV_API_KEY`; never in this folder.
- **Laya** (`convaiinnovations/laya` pinned per `~/<bench-repo>/manifest.json`): identical request schema,
  run locally on the RTX 2000 Ada (8 GB) via the bench harness (`harness/sequences.py`, `harness/predict.py`,
  eager FP16). Answers the "pay Jev or self-host" question in the same units.
- **Qwen3.5-35B-A3B-FP8** on PCAI: second opinion and reference labeler. The labeler already exists
  (`qwen_discovery/qwen_classify.py`); its schema is extended with `fulfillment_mode` and the three flags.
- **Quotes** (the CRM database, read-only): gold for top-level composition (rep-selected Product Type on quote lines) and for
  fulfillment mode (configurator fingerprints vs distributor-only lines).
- **Humans**: adjudicate disagreements and label what no system source covers, through the Classification Review page.

## 4. Jev question bundle

Variant A (default, ~19 questions):

| name | type | criteria |
|---|---|---|
| `primary_class` | choice | 7 top-level classes with one-line definitions from the report |
| `has_<subclass>` × 12 | noul | "Does the solicitation include <definition>?" one per subclass |
| `lifecycle` | choice | 5 lifecycle values with the renewal rule spelled out |
| `domain` | choice | 15 domains |
| `fulfillment_mode` | choice | 4 values with the Section 2 definitions and OEM examples |
| `rfi_market_research`, `text_insufficient`, `brand_name_only` | noul | one sentence each |

Variant B: `primary_subclass` as a single 12-way choice replaces the 12 nouls (tests whether one choice beats
twelve booleans). Variant C: hierarchical, a 7-way class choice plus one subclass choice per class present.

State text variants: S1 title + boilerplate-stripped description; S2 = S1 + structured lines; S3 = S2 + attachment
excerpt (first three RAG chunks). Cap near 600 tokens; record the tokenizer count per request.

Laya receives the identical bundle. Qwen receives its structured-extraction prompt and its output is mapped onto
the Section 2 schema by a fixed function (`kind → subclass`, primary component's lifecycle and domain).

## 5. Gold sets (E0)

1. **Composition gold** (top level only). For every opportunity with at least one typed quote line, the latest typed
   quote's set of Product Types mapped to top-level classes (the mapping already in `collect_report_data.py`).
   Expected size: SEWP ≈ 7,700, GSA MAS ≈ 340, GSA 2GIT ≈ 330.
2. **Fulfillment gold**. Per quoted opportunity, over all its quote lines:
   - configured build if any line's `extracted_data` carries a configurator fingerprint (Cisco: `ccw_line_number`,
     `deal_id`, `cisco_quote_id`; HPE and Dell fingerprints to be catalogued in E0 by inspecting `extracted_data`
     keys and `backup_data`), or one manufacturer has ≥ 8 lines and no distributor partner on those lines;
   - à la carte if every line's partner is a distributor (TD SYNNEX, Ingram Micro, D&H, B&H) and no manufacturer
     has ≥ 8 lines;
   - mixed if both conditions hold; otherwise unlabeled.
   Observed on SEWP: 184 explicit Cisco CCW, 1,627 OEM-heavy, 4,489 distributor-only. The proxy rules are validated
   against 200 human-labeled hardware rows before they are trusted (E4).
3. **Human gold**: 600 rows (200 per contract) drawn from the existing 12,000-row sample, stratified by Qwen
   subclass so rare subclasses are represented, labeled in the review UI with the full Section 2 schema.
4. **Disagreement set**: 300 rows where Jev's accepted answer and Qwen's mapped answer differ on `primary_class`
   or `components`, stratified by question (E3).

Gold files are JSONL keyed by opportunity id, frozen with a build date, never overwritten (new builds get a new
date suffix). Solicitation text is not stored in gold files; it is re-read from the sample.

## 6. Experiments

| # | Question | Inputs | Metric | Pass criterion |
|---|---|---|---|---|
| E1 | Which Jev/Laya question variant and state text works best | 12k sample; variants A/B/C × S1/S2/S3 on a 2,000-row subset first, winner on all | Accuracy and per-class precision/recall vs composition gold; subclass accuracy vs human gold | Winner beats or is within 2 points of Qwen-vs-gold agreement |
| E2 | Where to set confidence cutoffs | Winner's labels | Reliability curves and ECE per question; precision-vs-coverage curve; also top-1 minus top-2 margin as an alternative gate | Cutoff achieving ≥ 95% precision on `primary_class`, ≥ 90% on subclass presence, ≥ 90% on `fulfillment_mode`; report coverage at each |
| E3 | Who is right when Jev and Qwen disagree | 300 adjudicated disagreements | Win rate per question and per subclass | Defines per-question escalation rule (accept Jev / require Qwen agreement / always human) |
| E4 | Can models read fulfillment mode from text | Fulfillment gold + 200 human hardware rows | Accuracy, confusion matrix, precision on `configured build` | ≥ 90% on hardware rows; proxy-rule precision reported separately |
| E5 | Cost, latency, self-host option | Timing and token logs from E1; Laya numbers from the bench | $/opportunity per path (Jev only, Jev+Qwen, Laya), p50/p95 | Blended cost < $0.01 per opportunity; Jev-only path p95 < 3 s |
| E6 | Regression harness | Frozen gold + `evaluate.py` | Deltas vs previous run | Runs in one command; used on every prompt/model change |

The gating policy fitted by E2/E3: accept Jev when the question's confidence clears its cutoff; otherwise call Qwen;
accept if Qwen agrees with Jev's top choice; else queue for review. Always queue when any flag is true or
`primary_class` is Other. Coverage (share auto-accepted) is a reported outcome, not a target, but the production
decision expects ≥ 70% on SEWP.

## 7. Pipeline components (this folder, `pipeline/`)

| Unit | Does | Depends on |
|---|---|---|
| `gold.py` | builds the composition and fulfillment gold from the CRM database read-only; catalogues configurator fingerprints | psycopg2, credential pointer file |
| `label_jev.py` | runs a question variant × state variant over sample rows; resumable JSONL; logs tokens and latency | typesafe-sdk or raw HTTP, key from `~/<bench-repo>/.env` |
| `label_laya.py` | same bundle through the bench harness on the local GPU | `~/<bench-repo>` venv and weights |
| `label_qwen.py` | the existing labeler, schema extended with fulfillment and flags | PCAI endpoint, key from ai-services `.env` |
| `mapping.py` | maps each labeler's raw output to the Section 2 schema | `schema.py`, `bundle.py` |
| `evaluate.py` | metrics, calibration, threshold curves, agreement matrix, cost; writes `results/<run>/metrics.json` | gold + labels |
| `review_queue.py` | pushes rows to the CRM review API and pulls verdicts back into `gold/human-<date>.jsonl` | CRM backend API, service token |
| `report.py` | HTML/PDF results report in the style of the existing builder | matplotlib |

Conventions: read-only the CRM database sessions; credentials read at runtime from protected paths; no solicitation text in
committed outputs beyond the existing sample; every run writes `results/<run>/env.json` with model names, variant
ids, sample hash and timestamps.

## 8. Classification Review page (CRM feature)

Purpose: human adjudication now, production review queue later. Bounded feature in <crm-repo>.

- **Backend** (`backend/src/modules/classification-reviews/`): routes, controller, service, repository, validators,
  following the `teams` module layout. Endpoints: list queue (filters: status, vehicle, reason), get one (returns
  opportunity text bundle and proposals), post verdict, bulk enqueue (used by `review_queue.py`), export verdicts.
  Gated by a new RBAC permission `classification:review`; enqueue and export require `classification:manage`.
- **Table** `classification_reviews` (prisma-migration skill; mirror rule for data-acquisition models applies):
  id, opportunity_id (FK), run_id, reason (gold_sample · disagreement · fulfillment · low_confidence · flag),
  text_bundle jsonb (title, description, lines, attachment excerpt captured at enqueue time, so the page needs no
  cross-schema read of `rag.chunks`), proposals jsonb (per source: schema fields + confidences), suggested jsonb (the
  highest-confidence proposal, pre-fills the form), verdict jsonb (Section 2 schema), reviewer_id, status
  (queued · confirmed · corrected · skipped), created_at, reviewed_at. Unique (opportunity_id, run_id) so enqueue is
  idempotent; index on (status, created_at) and opportunity_id. JSONB payloads use the Python schema's snake_case keys
  verbatim so no mapping exists between the pipeline and the CRM.
- **Frontend** (`frontend/src/features/classificationReview/`): queue table, detail view with the text bundle on
  the left, proposals side by side, an editable verdict form pre-filled from the highest-confidence proposal,
  buttons Confirm / Correct / Skip, keyboard-first. Registered under the app routes behind the permission.
- **Deployment**: lives in prod (the test DB is reset from prod weekly and would lose verdicts). Deploy via the
  deploy-crm-prod skill. Research scripts write only through the API, so no direct prod SQL is needed for the study.
- Human budget for the study: ≈ 1,100 rows (600 gold, 300 disagreements, 200 fulfillment) ≈ 18 reviewer-hours.

## 9. Production target (informs the follow-up plan, not built here)

A `classification` module in ai-services, triggered on opportunity ingest (event from the data-acquisition sink or a
short poll on `created_at`), that builds the state text, calls Jev, applies the fitted gate, calls Qwen when needed,
writes one row per component with subclass, lifecycle, domain, fulfillment mode, source, confidence and evidence
span (new `opportunity_components` table plus flags on the opportunity), sends gated rows to `classification_reviews`,
and evaluates an automation-trigger table (`classification_triggers`: condition over labels → action). Laya replaces
Jev if E5 shows equal accuracy at lower cost. Load: ≈ 163 opportunities per day for SEWP + GSA, 72 more for ITES-4H.

Initial triggers:

| Condition | Action |
|---|---|
| hardware component and `fulfillment_mode` = configured build or mixed | open a Portal Build task for the OEM (Cisco CCW / HPE / Dell), assign to configuration engineer, flag price-variance risk |
| hardware component and `fulfillment_mode` = à la carte | run distributor availability and price lookup, pre-fill quote lines |
| `lifecycle` = renewal | incumbent lookup (prior quotes/SOs for client + brand), co-term and PoP checklist |
| Subscription/SaaS or Perpetual license component | publisher authorization check against the authorized-brand list |
| Support/Maintenance or Warranty component | request serial/asset list template |
| Professional/Consulting component | SOW required, route to services lead |
| Installation/Integration component | site access, travel, schedule checklist |
| `rfi_market_research` | capture/BD queue, no quote workflow |
| `text_insufficient` | run attachment extraction, then reclassify |
| `brand_name_only` | authorized-reseller check |
| `primary_class` = Other or any field below cutoff | review queue |

## 10. Success criteria for the research phase

1. A gating policy that reaches ≥ 95% precision on `primary_class`, ≥ 90% on subclass presence and ≥ 90% on
   `fulfillment_mode` for auto-accepted rows, with reported coverage (expected ≥ 70% on SEWP).
2. Per-question escalation rules with measured win rates from adjudicated disagreements.
3. Cost and latency per path, including the Laya self-host row.
4. A one-command regression harness over frozen gold.
5. The Classification Review page live in prod with ≈ 1,100 verdicts captured.
6. A results report and a written production plan.

## 11. Risks and mitigations

- **Jev state-length limit unknown**: measure the largest accepted state in E1 and cap below it; S3 may need trimming.
- **Quote gold is biased toward quoted (won-interest) opportunities**: top-level only; subclass and lifecycle rely on
  human gold; state this in every report.
- **Fulfillment proxy rules may be wrong**: validated on 200 human rows before use; HPE/Dell fingerprints may be absent
  (then those OEMs rely on human gold).
- **Prod endpoint load**: Qwen batches run at ≤ 48 concurrent; Jev at ≤ 10 req/s; both off-hours where possible.
- **Review UI touches prod**: schema change through prisma-migration, deploy through deploy-crm-prod, verify-live after.
- **Credentials**: Jev key, PCAI key and DB password stay in their protected files; nothing in this folder.
