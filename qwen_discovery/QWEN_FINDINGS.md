# Data-driven classification check — SEWP, GSA MAS, GSA 2GIT

Generated 2026-09-22. ITES-4H deliberately excluded.

## Method

- Random sample per vehicle from the CRM database (read-only): 6,000 SEWP, 3,000 GSA MAS, 3,000 GSA 2GIT. Deterministic order (`md5(id)`), so it is reproducible.
- Text fed per opportunity: title, latest CO description (SEWP boilerplate stripped), structured line items (GSA), and up to 3 attachment chunks from the RAG store where present.
- Labeled by the deployed Qwen3.5-35B-A3B on PCAI with a strict JSON schema: per component a free-text `what`, a fine-grained `kind` (12 values), a `lifecycle` (new / renewal / upgrade / replacement / unknown), a `domain` (15 values), a brand, plus `brand_name_only`, `confidence`, and an `unusual` free-text escape hatch so anything outside the closed sets surfaces.
- 11,958 of 12,000 labeled (42 transport errors). Low-confidence: SEWP 2.6%, MAS 1.3%, 2GIT 5.1%.
- One systematic model error was corrected post hoc: 93 components (cables, cords, mounts) were labeled `cloud/hosting service`; regex-corrected to `physical product`. Qwen itself flagged this mistake in `unusual` on several rows.
- Margin of error at 95% CI: about ±1.3 points for SEWP, ±1.8 for each GSA vehicle.

Files: `qwen_classify.py` (sample + label), `analyze.py` (aggregate), `analysis.json` (all distributions), `labels.jsonl` (per-opportunity labels keyed by opportunity id, no text).

## 1. Component incidence (% of opportunities containing the kind; overlaps)

| Kind | SEWP | GSA MAS | GSA 2GIT |
|---|---|---|---|
| Physical product (hardware) | 42.2 | 42.5 | 70.3 |
| Software subscription / SaaS | 29.0 | 16.5 | 12.8 |
| Support / maintenance contract | 19.1 | 11.3 | 8.3 |
| Software license (perpetual) | 14.8 | 8.4 | 10.1 |
| Professional / consulting service | 3.6 | 11.1 | 2.0 |
| Consumables / supplies | 2.1 | 7.9 | 2.7 |
| Furniture / facilities | 0.3 | 13.5 | 0.4 |
| Installation / integration | 1.5 | 5.2 | 1.2 |
| Warranty / extended warranty | 1.5 | 1.4 | 1.9 |
| Cloud / hosting service | 0.9 | 1.3 | 0.2 |
| Training | 0.9 | 0.9 | 0.3 |
| Other | 1.3 | 1.6 | 2.5 |

Multi-component opportunities: SEWP 31.9%, MAS 36.7%, 2GIT 30.1%. The legacy tags showed 15–20%, so the old data under-counts mixed requirements by roughly half.

## 2. Primary class (dominant component, mutually exclusive)

| Top level | SEWP | GSA MAS | GSA 2GIT |
|---|---|---|---|
| Hardware | 42.6 | 47.6 | 70.3 |
| Software | 40.9 | 22.7 | 20.1 |
| Maintenance & Support | 13.1 | 7.3 | 5.5 |
| Services | 1.9 | 7.2 | 1.2 |
| Furniture / Facilities | 0.3 | 13.3 | 0.3 |
| Installation & Integration | 0.2 | 0.7 | 0.4 |
| Other | 1.1 | 1.2 | 2.2 |

Compare the legacy tags: SEWP legacy shows Software at 62% incidence vs Hardware 43%; the text-grounded read is Software 41% / Hardware 43% as primary, with software incidence at ~44% once perpetual and subscription are combined. The legacy SEWP sample is skewed toward software because tagging was applied more often to software opportunities, not because SEWP is software-dominant.

## 3. Software New vs Renewal — do the old labels stand?

Lifecycle, as read from the text, for the components that carry it:

| Kind | Renewal share (SEWP / MAS / 2GIT) |
|---|---|
| Software subscription / SaaS | 52 / 41 / 56 |
| Support / maintenance contract | 52 / 42 / 53 |
| Software license (perpetual) | 6 / 5 / 3 |
| Warranty | 18 / 0 / 3 |
| Physical product | ~0 |

Against the legacy tags on the same opportunities:

| Legacy tag | Qwen agrees | Qwen disagrees | Precision |
|---|---|---|---|
| software/new (all 3 vehicles) | 438 | 10 | 98% |
| software/renewal (all 3 vehicles) | 466 | 214 | 69% |

Spot-checking the disagreements: where legacy said `software/renewal` and the text does not say renewal, the item is usually a support/maintenance contract, a term subscription, or a plainly new license (Palo Alto sustainment, VanDyke 3-year bundle, Mastercam "new software licence"). The old "renewal" label was really "recurring/term-shaped thing", not "renewing an existing entitlement". Where legacy said `software/new` and Qwen said renewal, the text explicitly says renewal in every case checked.

**Verdict:** `software/new` stands. `software/renewal` does not: ~30% of it is not a renewal. More importantly, renewal is not a software subclass at all. Half of support/maintenance contracts are renewals too, and perpetual licenses almost never are. Lifecycle should be a separate attribute on any component, not a leaf under Software.

## 4. Is the current set the best we can get?

The seven top-level classes in the report are not wrong, but three of them hide material, operationally different sub-populations that the text supports cleanly:

1. **Software must split into Perpetual license vs Subscription/SaaS.** Subscription is 2x perpetual on SEWP (29% vs 15%) and ~1.5x on both GSA vehicles. Quoting workflow, publisher authorization, PoP, and renewal tracking all differ.
2. **Services must split into Professional/consulting vs Installation vs Training.** On GSA MAS, professional services appear on 11.1% of opportunities; the legacy `services` tag sits at 0.1% because the old set had no home for them and they were dropped or mis-tagged as maintenance.
3. **Consumables/supplies deserve their own class on GSA MAS** (7.9%: toner, paper, office supplies). Today they fall into Hardware. Immaterial on SEWP and 2GIT (2–3%), but the class costs nothing there.
4. **Warranty is a distinct, small subclass of Maintenance & Support** (1.4–1.9%) with a different lifecycle profile (mostly new, sold with hardware).
5. **Cloud/hosting is real but small** (~1%: Starlink, AWS GovCloud, Azure). Keep as a Software subclass, not a top level.
6. **Lifecycle (new / renewal / upgrade / replacement) becomes a component attribute** applicable to software, support, and warranty.

Two things surfaced that are not product classes but should be flags:

- **RFI / market research** solicitations: SEWP 2.9%, MAS 3.4%, 2GIT 2.1%. Not a buy; should be a solicitation-type flag so they never get routed as hardware or software.
- **Attachment-only text**: the description says "see attached BOM" and nothing else. 3.2% of 2GIT, 0.8% of SEWP. These need the attachment pipeline before any classifier can label them; they explain most of 2GIT's low-confidence rate.

The `other` bucket is 1.3–2.5% and its contents are generic phrases ("IT equipment and services", "items from BOM", "shipping"), not a missing category. Nothing else new emerged from the free-text `what` field or `unusual` notes. So: the taxonomy above is close to exhaustive for these three vehicles; what was missing was granularity inside Software, Services, and Hardware, plus the lifecycle axis.

## 5. Second axis: solution domain (primary component)

| Domain | SEWP | GSA MAS | GSA 2GIT |
|---|---|---|---|
| Business / productivity software | 20.1 | 12.7 | 7.5 |
| Infrastructure / platform software | 15.2 | 8.6 | 9.3 |
| Cybersecurity | 12.9 | 6.1 | 4.6 |
| End-user devices | 9.2 | 6.0 | 13.6 |
| Networking | 8.4 | 5.6 | 13.8 |
| Cables / accessories / peripherals | 4.5 | 6.6 | 13.4 |
| Storage / backup | 4.5 | — | 6.4 |
| Compute / servers | 4.1 | — | 5.8 |
| Audio-visual | 3.4 | 6.6 | 6.3 |
| Telecom / mobility | — | 4.8 | — |
| Furniture / facilities | — | 13.3 | — |
| Other | 11.1 | 18.7 | 11.2 |

The domain "other" share is high on MAS because the vehicle carries non-IT buys (office supplies, radios, lab gear). Cybersecurity at 13% of SEWP is a meaningful cross-cut that the current hierarchy cannot express.

Brand-name-only restriction: SEWP 66%, MAS 49%, 2GIT 60% of opportunities.

## 6. Recommended target set (SEWP / GSA)

Top level (primary + components): Hardware · Consumables/Supplies · Software–Perpetual · Software–Subscription/SaaS · Software–Cloud/Hosting · Support & Maintenance · Warranty · Professional Services · Installation/Integration · Training · Furniture/Facilities · Other.

Attributes on every component: `lifecycle` (new / renewal / upgrade-addon / replacement), `domain` (list above), `brand`.

Opportunity-level flags: `rfi_or_market_research`, `brand_name_only`, `text_insufficient` (attachment-only).

## Limits

- Qwen labels are model output, not ground truth. On the top-level class it agrees with the legacy tag's primary on 84–95% of tagged opportunities and the spot checks favored Qwen in the disagreements, but a human audit of ~200 rows per vehicle would put a hard number on it.
- Only 3 attachment chunks were used; opportunities whose description is "see attached" are under-read.
- Percentages are of the sample, not of all opportunities; sampling error is stated above.
