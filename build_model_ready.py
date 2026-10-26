import pandas as pd, numpy as np, ast, pytz

raw = pd.read_csv('outages_raw.csv')
log = []
def note(s): log.append(s); print(s)

def parse(x, default):
    try: return ast.literal_eval(x) if isinstance(x, str) else default
    except Exception: return default

df = raw.copy()
note(f"Start: {len(df)} raw rows")

# 1. Parse nested fields
o = df['outage'].apply(lambda s: parse(s, {}))
df['outageCause'] = o.apply(lambda d: d.get('outageCause'))
df['outageType']  = o.apply(lambda d: d.get('outageType'))
df['loc_list']  = df['locations'].apply(lambda s: parse(s, []))
df['asn_list']  = df['asns'].apply(lambda s: parse(s, []))
df['asn_det']   = df['asnsDetails'].apply(lambda s: parse(s, []))

# 2. Drop rows with no cause/type label (not usable; earlier version mislabeled them UNKNOWN)
bad = df['outageCause'].isna() | df['outageType'].isna()
note(f"Dropped {bad.sum()} rows with missing cause/type label (ids {sorted(df.loc[bad,'id'].tolist())})")
df = df[~bad].copy()

# 3. Dates + target
df['startDate'] = pd.to_datetime(df['startDate'], utc=True)
df['endDate']   = pd.to_datetime(df['endDate'], utc=True)
df['duration_hours'] = (df['endDate'] - df['startDate']).dt.total_seconds()/3600
assert (df['duration_hours'] > 0).all()
df['log_duration_hours'] = np.log(df['duration_hours'])

# 4. Scope features, with explicit "record exists" flags
df['n_asns'] = df['asn_list'].apply(len)
df['n_locations'] = df['loc_list'].apply(len)
df['has_asn_record'] = (df['n_asns'] > 0).astype(int)
df['has_location_record'] = (df['n_locations'] > 0).astype(int)

# 5. Country: event locations first; if none, country of the affected ASNs
def countries(row):
    if row['loc_list']: return list(row['loc_list'])
    cs = [a.get('location', {}).get('code') for a in row['asn_det'] if isinstance(a, dict)]
    return sorted({c for c in cs if c})
df['country_list'] = df.apply(countries, axis=1)
df['n_countries'] = df['country_list'].apply(len)
df['primary_country'] = df['country_list'].apply(lambda l: l[0] if l else 'UNK').replace({'NA':'NA_NAM'})  # 'NA' (Namibia) is parsed as missing by pandas
note(f"Rows with no resolvable country: {(df.primary_country=='UNK').sum()}")

# 6. Cause grouping (documented). Original label kept in outageCause.
merge = {'WEATHER':'WEATHER_NATURAL','NATURAL_DISASTER':'WEATHER_NATURAL'}
df['cause_group'] = df['outageCause'].replace(merge)
vc = df['cause_group'].value_counts()
rare = vc[vc < 15].index
df['cause_group'] = df['cause_group'].where(~df['cause_group'].isin(rare), 'OTHER')
note("cause_group counts:\n" + df['cause_group'].value_counts().to_string())

# 6b. Type grouping: PLATFORM (10 rows, 8 with unknown country) merged with NETWORK (both are non-geographic scope)
df['type_group'] = df['outageType'].replace({'PLATFORM': 'NETWORK_OR_PLATFORM', 'NETWORK': 'NETWORK_OR_PLATFORM'})

# 7. Episodes: same primary country + same cause_group, consecutive starts <= 7 days apart
df = df.sort_values(['primary_country','cause_group','startDate']).reset_index(drop=True)
gap = df.groupby(['primary_country','cause_group'])['startDate'].diff().dt.total_seconds()/86400
new_ep = gap.isna() | (gap > 7)
df['episode_id'] = new_ep.cumsum()
sizes = df['episode_id'].map(df['episode_id'].value_counts())
df['episode_size'] = sizes
df['sample_weight'] = 1.0 / sizes   # each episode counts once in weighted analyses
note(f"Episodes: {df.episode_id.nunique()} from {len(df)} rows; largest = {sizes.max()} rows; rows in multi-row episodes: {(sizes>1).sum()}")

# 8. Duration quality flags (kept, not deleted)
df['flag_long_over_30d'] = (df['duration_hours'] > 720).astype(int)
same_end = df.groupby('endDate')['id'].transform('count')
df['flag_shared_endpoint'] = ((same_end > 1) & (df['duration_hours'] > 720)).astype(int)
note(f"Long (>30d): {df.flag_long_over_30d.sum()}; long AND sharing an identical end timestamp (likely administrative close-out): {df.flag_shared_endpoint.sum()}")

# 9. Local time (approximate: first timezone of the primary country; UTC if unknown)
def local(row):
    tz = 'UTC'
    try: tz = pytz.country_timezones[row['primary_country']][0]
    except Exception: pass
    t = row['startDate'].tz_convert(tz)
    return pd.Series({'start_hour_local': t.hour, 'start_dow_local': t.dayofweek})
df = pd.concat([df, df.apply(local, axis=1)], axis=1)
df['start_year'] = df['startDate'].dt.year
df['start_month'] = df['startDate'].dt.month

# 10. Final column order
meta = ['id','description','linkedUrl','startDate','endDate','primary_country','episode_id','episode_size','sample_weight']
target = ['duration_hours','log_duration_hours']
feats = ['outageCause','cause_group','outageType','type_group','n_asns','n_locations','n_countries','has_asn_record','has_location_record']
optional = ['start_hour_local','start_dow_local','start_year','start_month']
flags = ['flag_long_over_30d','flag_shared_endpoint']
out = df[meta+target+feats+optional+flags].copy()
out['startDate'] = out['startDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
out['endDate']   = out['endDate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
assert out['id'].is_unique and out[target+feats].notna().all().all()
out.to_csv('outages_model_ready.csv', index=False)

# 11. Episode-level version (one row per episode: median duration, modal categories)
g = out.groupby('episode_id')
ep = pd.DataFrame({
  'episode_id': g['episode_id'].first(),
  'primary_country': g['primary_country'].first(),
  'cause_group': g['cause_group'].first(),
  'outageType': g['outageType'].agg(lambda s: s.mode().iloc[0]),
  'n_rows': g.size(),
  'median_duration_hours': g['duration_hours'].median(),
  'n_asns': g['n_asns'].median(), 'n_locations': g['n_locations'].median(), 'n_countries': g['n_countries'].median(),
  'start_year': g['start_year'].first()})
ep['log_median_duration_hours'] = np.log(ep['median_duration_hours'])
ep.to_csv('outages_episode_level.csv', index=False)
note(f"Final: {len(out)} event rows, {len(ep)} episodes")
open('cleaning_log.txt','w').write("\n".join(log))
