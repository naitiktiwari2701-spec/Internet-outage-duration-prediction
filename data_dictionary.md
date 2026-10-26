# outages_model_ready.csv — data dictionary (862 rows, 26 columns)
Source: Cloudflare Radar Outages API (pulled Sep 2026, 866 rows). See cleaning_log.txt.

## Target
- duration_hours — endDate minus startDate
- log_duration_hours — natural log of the above (USE THIS as y)

## Features (X) — the research-question predictors
- cause_group — analyst-assigned cause (WEATHER+NATURAL_DISASTER merged; causes with <15 rows -> OTHER). Categorical, one-hot encode.
- outageCause — original un-merged cause label (do NOT use alongside cause_group)
- type_group — outageType with PLATFORM merged into NETWORK (PLATFORM had only 10 rows, 8 of them in the unknown-country group): NATIONWIDE / REGIONAL / NETWORK_OR_PLATFORM. Categorical. (Original label kept in outageType; do not use both.)
- n_asns, n_locations, n_countries — counts of affected networks / listed locations / resolved countries
- has_asn_record — 1 if an ASN list exists at all. (n_asns = 0 means "no ASN record", not "no networks affected"). has_location_record stays in the file for reference but is NOT a model feature: it is redundant with n_locations.

## Optional features (not in the research question; use for an extended model only)
- start_hour_local, start_dow_local — approximate local time (first timezone of primary country)
- start_year, start_month

## Grouping / weighting (for the train/test split — NOT features)
- primary_country — country code ('NA_NAM' = Namibia; 'UNK' = unresolved, 10 rows)
- episode_id — rows from the same country + same cause_group whose start dates chain within 7 days (487 episodes)
- episode_size, sample_weight (= 1/episode_size)

## Reference only (never features)
- id, description, linkedUrl, startDate, endDate

## Quality flags
- flag_long_over_30d — duration > 720 h (21 rows)
- flag_shared_endpoint — long AND shares an identical end timestamp with another row (7 rows); end date is probably an administrative close-out, so duration is unreliable

## Companion file
- outages_episode_level.csv — one row per episode (median duration, modal type). Use as a sensitivity check.
