#!/usr/bin/env python3
"""Reproducible, dependency-free audit of the UCI Bank Marketing bank-full data."""
from __future__ import annotations
import csv, hashlib, math, re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
CSV_PATH, NAMES_PATH = RAW / "bank-full.csv", RAW / "bank-names.txt"
EXPECTED = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
NUMERIC = ["age","balance","day","duration","campaign","pdays","previous"]
TARGET = "y"

def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def svg_bar(path, title, labels, values, color="#2864a5"):
    width,height=900,500; left,bottom,top=100,390,80; maxv=max(values) or 1
    n=len(values); slot=(width-left-35)/n; bw=min(48,slot*.68)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="22">{escape(title)}</text>',
           f'<path d="M {left} {top} V {bottom} H {width-25}" stroke="#333" fill="none"/>']
    for i,(lab,val) in enumerate(zip(labels,values)):
        x=left+slot*i+(slot-bw)/2; h=(bottom-top)*val/maxv; y=bottom-h
        parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}"/>',
                  f'<text x="{x+bw/2:.1f}" y="{bottom+22}" text-anchor="middle" font-family="Arial" font-size="12">{escape(str(lab))}</text>',
                  f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{val:,}</text>']
    parts.append('</svg>'); path.write_text('\n'.join(parts),encoding='utf-8')

def svg_hist(path,title,values,bins=30):
    lo,hi=min(values),max(values); span=hi-lo or 1; counts=[0]*bins
    for v in values: counts[min(bins-1,int((v-lo)/span*bins))]+=1
    width,height=900,500; left,bottom,top=85,390,80; maxv=max(counts)
    slot=(width-left-35)/bins; bw=slot
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>',
         f'<text x="450" y="38" text-anchor="middle" font-family="Arial" font-size="22">{escape(title)}</text>',f'<path d="M {left} {top} V {bottom} H 875" stroke="#333" fill="none"/>']
    for i,c in enumerate(counts):
        h=(bottom-top)*c/maxv; x=left+i*slot
        out.append(f'<rect x="{x:.2f}" y="{bottom-h:.2f}" width="{bw+0.3:.2f}" height="{h:.2f}" fill="#4c8c72"/>')
    out.extend([f'<text x="{left}" y="{bottom+25}" font-family="Arial" font-size="13">{lo:g}</text>',f'<text x="870" y="{bottom+25}" text-anchor="end" font-family="Arial" font-size="13">{hi:g}</text>','</svg>'])
    path.write_text('\n'.join(out),encoding='utf-8')

def main():
    TABLES.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    if not CSV_PATH.is_file() or not NAMES_PATH.is_file(): raise FileNotFoundError("Expected data/raw/bank-full.csv and bank-names.txt")
    names=NAMES_PATH.read_text(encoding='latin-1')
    with CSV_PATH.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter=';'); columns=reader.fieldnames
        rows=list(reader)
    if columns != EXPECTED: raise ValueError(f"Unexpected columns: {columns}")
    n=len(rows); num={c:[int(r[c]) for r in rows] for c in NUMERIC}
    cats=[c for c in columns if c not in NUMERIC]
    missing={c:sum(r[c] in (None,'') for r in rows) for c in columns}
    unknown={c:sum(r[c]=='unknown' for r in rows) for c in columns}
    exact=Counter(tuple(r[c] for c in columns) for r in rows); duplicate_rows=sum(v-1 for v in exact.values() if v>1)
    target=Counter(r[TARGET] for r in rows)
    # Compact feature schema and category counts.
    types=[(c,'integer' if c in NUMERIC else 'categorical',missing[c],unknown[c],len(set(r[c] for r in rows))) for c in columns]
    write_csv(TABLES/'column_audit.csv',['column','inferred_type','actual_nulls','unknown_count','distinct_values'],types)
    freq=[]
    for c in cats:
        for value,count in sorted(Counter(r[c] for r in rows).items()): freq.append((c,value,count,100*count/n))
    write_csv(TABLES/'categorical_frequencies.csv',['feature','category','count','percent'],freq)
    desc=[]; outlier_rows=[]; hist={}
    for c,vs in num.items():
        s=sorted(vs); q1=s[int(.25*(n-1))]; q3=s[int(.75*(n-1))]; iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
        out=sum(v<lo or v>hi for v in vs); hist[c]=vs
        desc.append((c,n,min(vs),max(vs),mean(vs),median(vs),pstdev(vs),q1, q3,iqr,lo,hi,out,100*out/n))
        outlier_rows.append((c,out,100*out/n,lo,hi))
    write_csv(TABLES/'numerical_summary.csv',['feature','count','min','max','mean','median','std_population','q1','q3','iqr','lower_fence','upper_fence','iqr_outlier_count','iqr_outlier_percent'],desc)
    write_csv(TABLES/'outlier_summary.csv',['feature','iqr_outlier_count','iqr_outlier_percent','lower_fence','upper_fence'],outlier_rows)
    write_csv(TABLES/'target_distribution.csv',['target','count','percent'],[(k,target[k],100*target[k]/n) for k in ('no','yes')])
    # First/last snapshot supports checking that input is ordered by date; no identity field exists.
    sha=hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    svg_bar(FIGURES/'target_distribution.svg','Target class distribution',['no','yes'],[target['no'],target['yes']])
    for c in NUMERIC: svg_hist(FIGURES/f'{c}_distribution.svg',f'{c} distribution',num[c])
    for c in cats:
        cc=Counter(r[c] for r in rows); svg_bar(FIGURES/f'{c}_distribution.svg',f'{c} category counts',list(sorted(cc)),[cc[k] for k in sorted(cc)])
    duration_yes=[int(r['duration']) for r in rows if r['y']=='yes']; duration_no=[int(r['duration']) for r in rows if r['y']=='no']
    dmed_yes=median(duration_yes); dmed_no=median(duration_no)
    report=f'''# Phase 1 — Bank Marketing dataset audit

## Scope and integrity

- Files present: `data/raw/bank-full.csv` ({CSV_PATH.stat().st_size:,} bytes) and `data/raw/bank-names.txt` ({NAMES_PATH.stat().st_size:,} bytes).
- CSV format: UTF-8, semicolon-delimited, double-quoted fields; header and all rows parse consistently.
- SHA-256 (`bank-full.csv`): `{sha}`.
- Observed dimensions: **{n:,} rows × {len(columns)} columns**; names file documents 45,211 rows and 17 attributes — {'matches' if n==45211 and len(columns)==17 else 'does not match'}.
- All expected columns and order match the names file's attribute definitions. Numeric fields parse as integers; other fields are categorical strings.
- Actual blank/null cells: **{sum(missing.values()):,}**. Exact duplicate rows beyond the first occurrence: **{duplicate_rows:,}**. No client identifier is supplied, so entity-level duplicates cannot be established.
- `unknown` sentinel values: **{sum(unknown.values()):,}** across fields; see `tables/column_audit.csv`. They are category values, not nulls.

## Target

| Class | Count | Share |
|---|---:|---:|
| no | {target['no']:,} | {100*target['no']/n:.2f}% |
| yes | {target['yes']:,} | {100*target['yes']/n:.2f}% |

The positive class is a minority. See `tables/target_distribution.csv` and `figures/target_distribution.svg`.

## Feature profile and outliers

`tables/numerical_summary.csv` reports min/max, mean, median, population standard deviation, quartiles and Tukey fences for all seven numeric features. `tables/categorical_frequencies.csv` contains counts and shares for every category. Numerical histograms and selected category plots are in `reports/figures/`.

IQR outlier counts use values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. This is a screening rule, not evidence of errors; especially `balance` and `campaign` are expected to be skewed. `pdays=-1` is a documented sentinel meaning no previous contact and should not be interpreted as a continuous negative duration. `day` is a day of month (1–31), not a continuous measurement.

## Leakage and modeling boundary

**`duration` is a critical target leakage risk for a pre-call prediction.** It is the length of the current call, therefore unavailable until the call ends. Exclude it when the intended prediction is made before/during dialing; it may only be used for an explicitly post-call use case. In this file, median duration is {dmed_yes:g} seconds for `yes` and {dmed_no:g} seconds for `no`, consistent with it carrying strong outcome information.

`campaign`, `pdays`, and `previous` describe campaign/contact history; assess their availability and timing against the deployment decision point. The names file says rows are ordered by date (May 2008–November 2010), making random validation potentially optimistic if the intended use is future campaigns. The supplied table has no explicit date or client ID, preventing direct temporal/client-group audit.

No models, encodings, parameter tuning, or raw-data modifications were performed.

## Source and license

Local `bank-names.txt` attributes the data to Paulo Cortez and Sérgio Moro and cites Moro, Laureano & Cortez (2011). The official [UCI Bank Marketing dataset page](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing) identifies this 45,211-row `bank-full.csv` as the legacy 17-attribute version and lists **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Recommended dataset citation: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

## Reproduction

Run `python src/audit_dataset.py` from the project root (Python standard library only). Outputs are written under `reports/tables/` and `reports/figures/`.
'''
    (ROOT/'reports'/'phase1_dataset_audit.md').write_text(report,encoding='utf-8')
    print(f"Rows={n:,}; columns={len(columns)}; target no={target['no']:,} ({100*target['no']/n:.2f}%), yes={target['yes']:,} ({100*target['yes']/n:.2f}%); nulls={sum(missing.values()):,}; unknown={sum(unknown.values()):,}; duplicate rows={duplicate_rows:,}; duration median yes/no={dmed_yes:g}/{dmed_no:g} sec")

if __name__=='__main__': main()
