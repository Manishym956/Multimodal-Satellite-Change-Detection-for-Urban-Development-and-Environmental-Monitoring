# Mentor briefing: full explanation of the progress since the 25 October draft

## Part 1. The problem (for context)

- Sentinel-2 gives optical images: easy to interpret, but clouds often block them.
- Sentinel-1 gives radar (SAR) images: they see through cloud but are harder to interpret.
- To detect change, a model compares a **before** image and an **after** image of the same place.
- The practical problem: the **after optical image is often missing or cloudy**. A common idea is to use an **older,
  cloud-free optical image** together with the radar.
- **Our question (RQ1):** does that older ("historical") optical image actually help a radar-based change detector?

**Data:** OSCD, a public benchmark of 24 cities with urban change marked pixel by pixel. We train on 10 cities, choose
settings on 4 validation cities, and test on 10 cities that are never used for any choice.

**Models:**

| name | what it is |
|---|---|
| A | DS-UNet, the published dual-stream model, trained only with complete inputs |
| B | the same DS-UNet, trained with **modality dropout**: during training the optical images are randomly removed (set to zero), so the model learns to work without them |
| SAR-only | a U-Net that sees only radar |
| E (new) | an **early-fusion** U-Net: radar and optical go into the same network from the first layer, trained with the same dropout as B |

**Why 10 seeds:** one training run can be lucky. Each model is trained 10 times with different random starts, and every
comparison is made run by run (paired), so the result is not one lucky run.

**Pre-registration:** before each experiment I wrote the plan (hypotheses, tests, file hashes) and committed it to GitHub,
and only then looked at test results. Anything decided after seeing results is labelled "exploratory".

**F1:** the main score (0 to 1) that balances missed changes and false alarms. A difference of 0.01 is one F1 point.

## Part 2. Where the 25 October draft stood

- **RQ1:** the historical image gave **no detectable gain** in model B (−0.010, not significant).
- **RQ2:** the age of the image (1–18 months) made "no detectable difference".
- **RQ3:** adding VH polarisation to VV lowered F1 for the radar-only model.
- **RQ4:** DS-UNet beat all simpler models when all images are present.
- **RQ5:** the historical image helped only in shrubland (1.3% of pixels).
- **Clouds:** a robustness study with synthetic (artificial) clouds.

## Part 3. The internal review and what it criticised

I ran a simulated peer review: five reviewer roles played by an AI assistant (so it is a self-check, not real peer
review). Verdict: **major revision**. They praised the method and statistics, and raised five main points:

1. **RQ2 overclaimed.** The test only checked for a *straight-line* trend, yet the draft said "no effect of age".
2. **RQ1 might be an artefact of the model.** In DS-UNet the optical and radar branches meet only at the very last layer,
   so the optical image cannot influence how the radar is read. Maybe the image did not help *because of this design*.
3. **Uncertainty from the choice of test cities was missing.** All error bars came from training runs on the same 10
   cities.
4. **The shrubland result (RQ5) was not attributable to the image.** It compared two different models.
5. **All clouds were artificial.** No real cloudy image was ever tested.

## Part 4. What I did: one re-analysis and four experiments

### Re-analysis (no new training)

- **RQ2** reworded to "no detectable *linear* trend". An extra analysis showed the 1-month-old image scores lowest, but
  that comes mostly from one city and is explained by the image already showing part of the change.
- **City uncertainty:** I resampled the test cities. Finding: DS-UNet's lead over radar-only comes **mostly from Las
  Vegas**; without Las Vegas the lead shrinks from 0.138 to 0.049.

### Experiment 1 (Stage 4B): the same model with and without the historical image

- **Why:** to measure the image's own contribution, not a comparison between two different models.
- **What:** model B evaluated twice, with the historical image and with no optical input at all.
- **Result:** the image adds **+0.007 F1, not significant**.
- **Extra finding:** B scored 0.060 above the radar-only model. Of that, 0.053 is there even with *no* optical input. So
  B's advantage comes from **how B was trained**, not from the image.
- **RQ5:** within B, the shrubland effect disappears, so it cannot be credited to the image.

### Experiment 2 (Step 9): a model where the image *can* influence the radar

- **Why:** to answer criticism 2 directly.
- **What:** new model E (early fusion) trained with the same dropout as B, 10 seeds.
- **Result:** the historical image **lowers F1 by 0.091**, in every one of the 10 runs (significant).
- **Why it hurts:** precision drops by 0.133 and false alarms rise. The model sees "optical image at the before date, empty
  at the after date" and appears to read the difference as change. This explanation is likely but not directly tested.
- **Meaning:** the RQ1 null was not caused by DS-UNet's design. **When the image is allowed to interact with the radar,
  it does harm.**

### Experiment 3 (Step 10): repeating on 10 new training runs

- **Why:** in Experiment 2 I noticed, after seeing the results, that model E with *no* optical input beat the radar-only
  model by 0.072. Because I spotted that after the fact, it needed a fresh, pre-registered test.
- **What:** E and the radar-only model retrained on 10 new seeds (11–20).
- **Results:**
  - E without optical beats radar-only: **+0.072, all 10 runs** (confirmed);
  - the image still hurts E: **−0.116** (replicated).
- **Caveat:** the +0.072 comes mostly from **Las Vegas**; without it the gain is +0.028. New seeds repeat the training,
  not the cities.
- **Meaning:** training with optical data that is randomly removed seems to produce a **better radar-only detector**.

### Experiment 4 (Step 11): real cloudy images

- **Why:** criticism 5; all earlier clouds were artificial.
- **What:**
  - downloaded from Google Earth Engine a real partly cloudy Sentinel-2 image for each test city, within ±30 days of the
    official after date (7 cities partly cloudy, 3 fully overcast);
  - added a clear same-day image as a control;
  - the scene-selection rule was fixed before downloading;
  - tested A, B and radar-only with clouds unmasked, masked by an automatic cloud detector (s2cloudless), or the image
    discarded.
- **Results (7 partly cloudy cities):**

| situation | F1 |
|---|---|
| radar-only model | 0.436 |
| A (normal fusion) with the real cloudy image | **0.162**, far below radar-only |
| B with the real cloudy image, no mask | 0.248 |
| B with cloud pixels masked | 0.456, slightly above radar-only (+0.020, significant) |
| B with the cloudy image simply discarded | 0.463 |

- **Meaning:**
  - feeding a cloudy image to a normal fusion model is worse than ignoring optical data altogether;
  - a dropout-trained model handles it;
  - in practice, the detector masks so much that **discarding the cloudy image works just as well as masking it**;
  - the clear control matched the official image (difference −0.002), so the downloaded data are consistent.

## Part 5. How the answers to the research questions changed

| RQ | 25 October | now |
|---|---|---|
| RQ1 | no detectable gain | **no gain with late fusion, a significant loss with early fusion**; the null is not a design artefact |
| RQ2 | no effect of age | **no detectable linear trend**; the 1-month low is confounded |
| RQ3 | VH lowers F1 (radar-only) | unchanged |
| RQ4 | DS-UNet best | unchanged, but the lead rests mainly on Las Vegas |
| RQ5 | helps in shrubland | **not attributable to the image**; no material benefit anywhere |
| clouds | synthetic only | **real clouds added**; practical rule: train with dropout, drop the cloudy image |
| new | — | dropout training gives a better radar-only detector (confirmed, city-concentrated) |

**The paper's message in one sentence:** an older optical image does not help radar change detection and can hurt it,
and when the new optical image is cloudy, a model trained with dropout should simply do without it.

## Part 6. The manuscript

- Rewritten around these findings; main text about 20% shorter (6,900 → 5,600 words) despite four new experiments.
- **New figures:** a summary plot of every pre-registered result, and a plot of F1 against image age.
- **Quality control:**
  - every one of 491 numbers and 752 table cells is checked by script against the stored results;
  - all 28 references verified (one dead DOI fixed);
  - the text was read line by line and 17 overstatements corrected.
- **Length:** 25 pages now, of which 8 are appendices that would go into supplementary material. In a journal's
  two-column format the main paper is about 10–12 pages.
- **Code:** public on GitHub under the GPL (version 2 or later, compatible with DS-UNet).

## Part 7. Honest limitations

- Only 10 test cities, and some effects rest largely on one city (Las Vegas).
- One real cloudy image per city; cloud amount not controlled.
- Two model designs and one training recipe; the simpler models were not tuned separately.
- The reviews and checks were done with the same AI assistant that helped write the code, so they are not independent.
  The real test is journal review. AI use is disclosed in the paper.

## Part 8. Decisions needed from the mentor

1. **Authorship:** co-author, or acknowledged for supervision?
2. **Journal:** suggested IEEE JSTARS; alternatives MDPI *Remote Sensing* or IEEE TGRS.
3. **Title:** keep "…for Urban Development and Environmental Monitoring", or add a subtitle, since environmental change
   is not tested?
4. **Conflict of interest and funding:** confirm none.

**Next steps after these decisions:** format the paper for the chosen journal, move the appendices to supplementary
material, finish the cover letter and submit.
