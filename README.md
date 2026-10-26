# Predicting internet outage duration from cause, type and scope

Code and data for: Tiwari, N. (2026). *Predicting Internet Outage Duration from Cause, Type, and Scope: Linear Regression Tested on Unseen Countries.* Preprint: https://doi.org/10.5281/zenodo.23653234

Data: Cloudflare Radar Outage Center (outage annotations API). Cloudflare Radar API data are licensed CC BY-NC 4.0 (https://radar.cloudflare.com/about); the files in repository root are non-commercial research copies with attribution to Cloudflare.

## Run order
1. `pull_outages.py` – downloads the outage records (needs a free Cloudflare API token with Radar read permission in the environment variable `CLOUDFLARE_API_TOKEN`). Writes `outages_raw.csv`. Optional: re-running may return newer records than those used in the paper.
2. `build_model_ready.py` – cleans the raw file, builds the episodes and flags, writes `outages_model_ready.csv`, `outages_episode_level.csv` and `cleaning_log.txt`.
3. `paper_analysis.py` – runs every analysis in the paper (fixed seed 42) and writes `paper_results.json`, the four figures and the printed summary (`paper_analysis_output.txt`).
4. `verify_pull.py` – optional completeness check that re-pulls the data year by year.

To reproduce the paper's numbers without the API, keep `outages_model_ready.csv` next to `paper_analysis.py` and run `python3 paper_analysis.py` (about one minute).

`data_dictionary.md` describes every column.

## Setup
`pip install -r requirements.txt`  (paper used Python 3.13, pandas 3.0, NumPy 2.5, scikit-learn 1.9, Matplotlib 3.11)

## License
Add your preferred licence for the code (for example MIT). Data: CC BY-NC 4.0, credit Cloudflare.
