# Licence evidence — iWildCam v2.0 and the detection corpus

This file exists because one licence claim in `DATASET_CARD.md` was wrong, and it was wrong in a way that
mattered: it was read as prohibiting commercial use. The claim was corrected on 2026-10-24. Rather than
leave the correction as an assertion in the card, the underlying text is recorded here verbatim so a marker
can check our reading against the page without re-deriving it.

Method: the page was fetched with a browser and its DOM walked in document order, so each licence sentence
is attached to the heading it actually sits under. A search snippet cannot do that — that is exactly how the
first reading went wrong.

---

## 1. WILDS datasets page — per-dataset licence blocks in page order

Source: <https://wilds.stanford.edu/datasets/>
Retrieved: 2026-10-24T05:09:42Z and 2026-10-24T05:09:53Z (two passes; section order reproduced below).

Each dataset on that page has the same sub-structure: `# <Dataset>` → `# Motivation` → `# Problem setting`
→ `# Dataset citation` → `# License`. The licence sentence therefore belongs to the nearest preceding
dataset heading. Walking the headings in document order gives:

| Order | Dataset heading | Sentence under its `# License` heading |
|---|---|---|
| 1 | **iWildCam** | "Distributed under the Community Data License Agreement – Permissive – V1.0." |
| 2 | Camelyon17 | (no sentence matched; licence expressed as link text only) |
| 3 | RxRx1 | "This dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License." |
| 4 | OGB-MolPCBA | "Distributed under the MIT license." |
| 5 | GlobalWheat | "Distributed under the MIT license." |
| 6 | CivilComments | (no sentence matched) |
| 7 | FMoW | "Distributed under the FMoW Challenge Public License." |
| 8 | PovertyMap | (no sentence matched) |
| 9 | Amazon | (no sentence matched) |
| 10 | Py150 | "Distributed under the MIT license." |

The misread in the earlier revision of the card was **row 3 quoted as if it were row 1**. "Creative Commons
Attribution-NonCommercial-ShareAlike 4.0" is the RxRx1 licence, nine heading-boundaries below iWildCam.

## 2. iWildCam block, verbatim

Extracted 2026-10-24T05:09:53Z from `https://wilds.stanford.edu/datasets/#iwildcam`, element-level, with
the anchor `href` preserved.

`# Problem setting`:

> We consider the domain generalization setting, where the domains are camera traps, and we seek to learn
> models that generalize to photos taken from new camera deployments. The task is multi-class species
> classification. Concretely, the input x is a photo taken by a camera trap, the label y is one of 182
> different animal species, and the domain d is an integer that identifies the camera trap that took the
> photo.

This corroborates the audit's independent count (`dataset_audit` / notebook 01): 182 species are *present*
in the label table against 206 taxonomy entries.

`# Dataset citation` (a `<pre>` block):

```
@article{beery2020iwildcam,
  title={The iWildCam 2020 Competition Dataset},
  author={Beery, Sara and Cole, Elijah and Gjoka, Arvi},
  journal={arXiv preprint arXiv:2004.10340},
  year={2020}
}
```

`# License` (paragraph, plus the hyperlink it carries):

> Distributed under the Community Data License Agreement – Permissive – V1.0.

Link text "Community Data License Agreement – Permissive – V1.0" → `href`
`https://cdla.io/permissive-1-0/`, which redirects to `https://cdla.dev/permissive-1-0/` (observed at
2026-10-24T05:10:16Z). SPDX identifier on that page: `CDLA-Permissive-1.0`.

## 3. CDLA-Permissive 1.0 — the clauses that bind us

Retrieved 2026-10-24T05:10:24–05:10:35Z from `https://cdla.dev/permissive-1-0/`, verbatim.

> **1.10** "Results" means the outcomes or outputs that You obtain from Your Computational Use of Data.
> Results shall not include more than a de minimis portion of the Data on which the Computational Use is
> based.

> **1.12** "Use" means using Data (including accessing, copying, studying, reviewing, adapting, analyzing,
> evaluating, or making Computational Use of it), either by machines or humans, or a combination of both.

> **2.1** Subject to the conditions set forth in Section 3 of this Agreement, Data Provider(s) hereby
> grant(s) to You a worldwide, non-exclusive, irrevocable (except as provided in Section 5) right to:
> (a) Use Data; and (b) Publish Data.

> **3.1** If You Publish Data You Receive or Enhanced Data:
> (a) You may do so under a license of Your choice provided that You give anyone who Receives the Data from
> You the text of this Agreement, the name of this Agreement and/or a hyperlink or other method reasonably
> likely to provide a copy of the text of this Agreement; and
> (b) You must cause any Data files containing Enhanced Data to carry prominent notices that You have
> changed those files; and
> (c) If You Publish Data You Receive, You must preserve all credit or attribution to the Data Provider(s).
> Such retained credit or attribution includes any of the following to the extent they exist in Data as You
> have Received it: legal notices or metadata; identification of the Data Provider(s); or hyperlinks to Data
> to the extent it is practical to do so.

> **3.4** This Agreement imposes no obligations or restrictions on Your Use or Publication of Results.

### What this means for this project, concretely

* **Training is permitted without conditions.** Downloading, caching and running machine-learning
  experiments on the images is `Use` under §1.12, and §2.1 grants it worldwide. Nothing in §3 applies to
  Use alone — §3's conditions trigger on *Publishing Data*.
* **The classifier is a Result, so it is unencumbered.** Weights, metrics, confusion matrices and the alert
  rule are outputs of Computational Use (§1.10, §3.4). Submitting or publishing the trained model therefore
  adds no licence obligation, which is what `configs/experiments.yaml` and the model card should and do say.
* **Publishing our recarved split files would be Enhanced Data**, so if `data/splits/*.csv` ever ships
  publicly, §3.1(a) and (b) require a copy of the agreement (or its name/hyperlink) and a prominent notice
  that the files were changed. `make_splits.py` already writes a `manifest.json` describing exactly what was
  changed and with which seed; the remaining task is to attach the licence name and hyperlink to it at
  submission.
* **Attribution obligation** is §3.1(c): preserve the Beery, Cole and Gjoka citation and the WILDS page
  hyperlink. Both are carried in `DATASET_CARD.md` and `SOURCE_LOG.md` row 4a.
* **Commercial use is not restricted.** This is the specific claim the earlier card got wrong: there is no
  NonCommercial clause in CDLA-Permissive 1.0. The card should not say otherwise, and should not have said
  otherwise.

## 4. Detection corpus (separate dataset, separate licence)

The boxed detection set (`ankanghosh651/object-detection-wildlife-dataset-yolo-format`, recorded in
`DATASET_CARD.md` §4) reports its licence as MIT via the Kaggle API at download time. That is a
self-declared mirror label, not a signed statement from the image rights-holders, and camera-trap imagery
originates with third parties. It is kept as an open item, not resolved here.

## 5. Upstream imagery terms — Snapshot Safari

Resolved on 2026-10-24 rather than left as an assumption. Source: the Snapshot Safari organisation page on
Zooniverse, <https://www.zooniverse.org/organizations/meredithspalmer/snapshot-safari>, read from the rendered
DOM at 2026-10-24T05:15:08Z and 05:15:17Z (the page is a client-side app, so a plain fetch returns nothing —
the text below came from the executed page).

Verbatim, with the sentence immediately before it kept so the scope is not guessed:

> (…Kruger, Kgalagadi, Karoo, Mountain Zebra, Camdeboo, and Enonkishu) are also available on LILA for
> download. Ecologists who wish to access metadata and use data for publications should email
> snapshotsafari@gmail.com with details about your research questions.
>
> **Images captured by Snapshot Safari are licensed under a Creative Commons Attribution 4.0 International
> License.** To view a copy of this license, visit Creative Commons.
>
> Most of the images found in the species menus come directly from our camera traps, but occasionally we will
> display photos from other image databases so our volunteers get the clearest view of an animal while
> classifying. You can find a complete list of image attributions here.

The licence sentence's hyperlink resolves to `http://creativecommons.org/licenses/by/4.0`, i.e. **CC BY 4.0**:
attribution required, commercial use permitted, no ShareAlike and no NonCommercial clause.

### Combined effect for this project

Two layers apply to what we downloaded, and they point the same way:

| Layer | Instrument | Practical obligation here |
| --- | --- | --- |
| WILDS v2.0 release (the resized image bundle + label tables we actually use) | CDLA-Permissive-1.0 | Conditions bind on *publishing data*: pass on the Agreement (text, name or hyperlink), mark files we changed, keep provider attribution. No obligation on Results (§3.4) — the classifier is unencumbered. |
| Underlying camera-trap photographs | CC BY 4.0 | Credit Snapshot Safari wherever imagery is shown or redistributed. |

So the honest one-line statement for the report is: **iWildCam v2.0 is CDLA-Permissive-1.0 as a WILDS
release, over imagery that Snapshot Safari licenses CC BY 4.0; attribution to both is required and nothing
restricts commercial use.** The earlier "non-commercial only" claim in `DATASET_CARD.md` was wrong at both
layers.

Two things this does *not* prove, and should not be quoted as if it did:

1. **Per-image provenance.** We have not verified that every frame in the mirror is Snapshot Safari imagery —
   the network aggregates several projects, and the page above itself warns that some displayed photos come
   from other image databases. The CC BY 4.0 statement covers images "captured by Snapshot Safari"; anything
   else in the pool inherits its own contributor's terms.
2. **A metadata-use courtesy.** The same page asks ecologists who want metadata *for publications* to email
   `snapshotsafari@gmail.com`. This is coursework, not a publication, so no obligation is triggered — but the
   request is recorded here so it is visible if this work is ever extended into a research output.

## 6. Still open after this evidence pass

1. **The Kaggle mirror carries no licence file.** We rely on the upstream statements above; the mirror page
   should be re-checked at submission in case it changes.
2. **Detection corpus provenance** (section 4) — its MIT label is a mirror self-declaration, not a statement
   from the rights-holders of those photographs.
3. Nothing in this file is legal advice; it is a reading of public pages, recorded with timestamps and URLs
   so a reviewer can repeat it.

