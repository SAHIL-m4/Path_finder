# AI Disclosure

**Tool used:** Claude (Anthropic), via the claude.ai chat interface.

## What the AI did
* Proposed the overall pipeline (gravity alignment, quaternion attitude with gravity feedback,
  double integration with ZUPT, dropout coasting, damped blind extrapolation) from the
  Track 03 brief.
* Wrote the first full version of `src/main.py`, `tools/synth_and_eval.py`, this repo's README files
  and this disclosure.
* Built `web/pathfinder.html` (a JavaScript port of the pipeline with a plot) and filled the hackathon
  slide template with the Track 03 content.
* Ran the synthetic experiments and parameter sweeps that set the constants in `main.py`
  (gravity gain, bias gain, velocity leak, coasting and blind decay constants). It also found and fixed
  three of its own bugs during that process: a gyro bias learned from a moving start window, ZUPT bias
  refresh absorbing real rotation, and a leak driven by motion rather than sensor noise.

## What the AI did not do / what you still must do
* No real EuRoC, TUM-VI or organiser-degraded data was available to the AI, so **all reported
  accuracy numbers come from a synthetic generator** and are not real-data results.
* The AI did not record the walkthrough video.

## Human contribution (team to complete)
<!-- Be specific. Replace this block with what your team actually did, for example:
- Chose Track 03 and the approach; reviewed every line of main.py
- Ran main.py on the organiser's degraded EuRoC/TUM-VI files and scored with evo (results: ...)
- Made the slides and recorded the video
- Changes you made to the AI's code and why -->
