# Progress update: "Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring"

**Since the draft of 25 October 2026.** Updated manuscript attached (PDF, 25 pages).

## 1. What I did

1. **An internal review of the 25 October draft.** I put the draft through a simulated five-reviewer panel run with an
   AI assistant, not real journal reviewers. The verdict was *major revision*. The process and statistics were judged
   strong. The main criticisms:
   - RQ2 was worded more strongly than the test allowed;
   - RQ1 tested only one fusion design, the late-fusion DS-UNet;
   - there was no uncertainty across test cities;
   - all clouds were synthetic;
   - the paper was too long.
2. **Rewriting and re-analysis of stored results**, with no new training. RQ2 now claims only "no detectable *linear*
   trend". I added a test-city resampling analysis, and every analysis added after the results were seen is labelled
   exploratory.
3. **Four new experiments.** Each was pre-registered (plan and code hashes committed to GitHub before any test result
   existed):

| step | experiment | result |
|---|---|---|
| Stage 4B | B (DS-UNet with modality dropout) with vs without the historical image | **No detectable effect** (+0.007 F1). B's lead over the SAR-only model comes from the dropout-trained network itself, not from the image. |
| Step 9 | **New model:** early-fusion U-Net, where the historical image can interact with SAR | **The image lowers F1 by 0.091** (10/10 seeds), through false alarms. So the RQ1 null is not just because DS-UNet fuses late. |
| Step 10 | Replication on 10 new training seeds (11–20) | Both findings replicate: the image lowers F1 (−0.116), and the dropout model *without any optical input* beats a model trained on SAR alone (+0.072). The second gain is concentrated in one city (Las Vegas). |
| Step 11 | **Real cloudy Sentinel-2 scenes** near OSCD date 2 for all 10 test cities (Earth Engine) | A standard fusion model falls far below SAR alone on real clouds (−0.275). The dropout model with clouds masked recovers most of the loss (+0.208) and is slightly above SAR alone (+0.020). Discarding the cloudy image works at least as well as masking it. |

## 2. How the conclusions changed

| research question | 25 October draft | now |
|---|---|---|
| RQ1 (historical image helps?) | no detectable gain | **No gain with late fusion; a detectable loss with early fusion.** Stronger and more general. |
| RQ2 (age of the image) | "no detectable effect" | **No detectable linear trend**; the 1-month low is confounded and concentrated in one city |
| RQ3 (VV/VH) | VH lowers F1 for SAR alone | unchanged |
| RQ4 (comparison with baselines) | DS-UNet best with full input | unchanged, but the lead over SAR-only rests mainly on Las Vegas |
| RQ5 (where it helps) | small benefit in shrubland | **not attributable to the image** (Stage 4B); no material benefit anywhere |
| Clouds | synthetic and composite only | **adds real cloudy scenes** and a practical rule: train with modality dropout and discard the optical image when cloud is detected |
| New finding | — | modality-dropout training gives a better SAR-only detector (confirmed on new seeds) |

## 3. Changes to the manuscript

- Rewritten around the findings above; main text about 20% shorter than the 25 October draft (6,900 → 5,600 words,
  excluding tables) despite the four new experiments.
- New figures: a summary forest plot of all pre-registered results, and a plot of F1 against the historical image's age.
- New sections: §4.2 (late vs early fusion), §4.3 (dropout-trained SAR detector), §5.5 (real clouds). The detailed
  synthetic-cloud tables moved to Appendix E.
- Audit: every number and table is checked by script against the stored result files; all 28 references verified;
  one dead DOI fixed.
- Code released under the GPL (version 2 or later, compatible with DS-UNet). Everything is public on GitHub.

## 4. Decisions I need from you

1. **Authorship:** would you like to be a co-author, or acknowledged for supervision?
2. **Target journal:** my suggestion is *IEEE JSTARS*; the alternatives are MDPI *Remote Sensing* or *IEEE TGRS*.
3. **Title:** the reviewers noted that "Environmental Monitoring" is not tested (OSCD labels urban change only). Keep it,
   or add a subtitle such as "…: Historical Optical Imagery, Modality Dropout and Real Clouds on OSCD"?
4. Conflict of interest and funding statements (I believe none, to confirm).

## 5. Limitations to be aware of

One benchmark with 10 test cities (several effects rest on Las Vegas), one real cloudy scene per city, and two model
architectures. AI assistance in code and text is disclosed in the manuscript.
