# Cover letter (draft; fill in the bracketed fields once the journal is chosen)

Dear Editor,

Please consider the manuscript "Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring" for publication in [journal] as a
[regular paper / article].

SAR–optical change detectors are usually trained and tested with optical images at both dates, but the post-change
optical image is often missing or cloudy. A common response is to fill the gap with an older, cloud-free optical image.
We measured, on the public OSCD benchmark with the official DS-UNet code, whether that image helps. It did not: it made no
detectable difference in a late-fusion network and lowered F1 in an early-fusion network by raising false alarms, a loss
that replicated on new training seeds. We also report that modality-dropout training gave a better SAR-only detector than
SAR-only training, and we evaluate models on real cloudy Sentinel-2 acquisitions, where a model trained only with full
input fell far below SAR alone while a dropout-trained model with cloud discarded or masked recovered most of the loss.

Every result uses 10 paired training seeds, validation-only model and threshold selection, and a pre-registration
written before each test evaluation; all code, pre-registrations and analysis outputs are public
([repository URL]). We believe these findings are useful to [journal]'s readers who build or deploy SAR–optical
monitoring under cloud.

The manuscript is original, has not been published, and is not under consideration elsewhere. [All authors have approved
the submission.] An AI assistant was used for parts of the code and text, as disclosed in the manuscript; the author(s)
take responsibility for the content. [Suggested / excluded reviewers, if the journal asks.]

Sincerely,
[Name(s)]
Polaris School of Technology, Medhavi Skills University
manishym946@gmail.com

## Highlights (only if the journal asks for them; each under 85 characters)

- A historical optical image did not help SAR change detection on OSCD.
- In an early-fusion network it lowered F1 by raising false alarms.
- Modality-dropout training gave a better SAR-only detector.
- On real cloudy scenes, discarding the cloudy image beat passing it through.
- All results use 10 seeds, validation-only selection and pre-registration.
