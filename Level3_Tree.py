#!/usr/bin/env python3
"""Causal Level 3 Tree model loader using saved chronological snapshots."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

FEATURES=["ret1","ret3","ret7","ret30","volstd7","volstd30","range","dist7","dist30","volchg7","volchg30"]
BASE=Path(__file__).resolve().parent
SNAPSHOT_DIR=BASE/"level3_tree_weights"
MANIFEST=SNAPSHOT_DIR/"manifest.json"


def make_features(df):
 c=df["Close"]; v=df["VolNum"]; f=pd.DataFrame(index=df.index)
 f["ret1"]=c.pct_change(1)*100; f["ret3"]=c.pct_change(3)*100; f["ret7"]=c.pct_change(7)*100; f["ret30"]=c.pct_change(30)*100
 f["volstd7"]=c.pct_change().rolling(7).std()*100; f["volstd30"]=c.pct_change().rolling(30).std()*100
 f["range"]=(df["High"]-c)/c*100; f["dist7"]=(c-c.rolling(7).mean())/c.rolling(7).mean()*100; f["dist30"]=(c-c.rolling(30).mean())/c.rolling(30).mean()*100
 f["volchg7"]=v.pct_change(7)*100; f["volchg30"]=v.pct_change(30)*100
 return f[FEATURES]

def load_snapshot(decision_index, manifest=None):
 if manifest is None:
  with open(MANIFEST) as fh: manifest=json.load(fh)
 eligible=[x for x in manifest if x["decision_index"]<=decision_index]
 if not eligible: return None
 return joblib.load(BASE/eligible[-1]["file"])

def predict_probability(df, decision_index=None, model=None):
 if model is None:
  if decision_index is None: decision_index=len(df)-1
  model=load_snapshot(decision_index)
  if model is None: return float("nan")
 x=make_features(df).iloc[[-1]]
 if x.isna().any().any(): return float("nan")
 x=x.to_numpy()
 return float(model.predict_proba(x)[0,1])
