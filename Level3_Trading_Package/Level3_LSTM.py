#!/usr/bin/env python3
"""Causal Level 3 LSTM loader using saved chronological snapshots."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FEATURES=["ret1","ret3","ret7","ret30","volstd7","volstd30","range","dist7","dist30","volchg7","volchg30"]
SEQ_LEN=30; HIDDEN=32; LAYERS=5
BASE=Path(__file__).resolve().parent; SNAPSHOT_DIR=BASE/"level3_lstm_weights"; MANIFEST=SNAPSHOT_DIR/"manifest.json"

class LSTMClassifier(nn.Module):
 def __init__(self):
  super().__init__(); self.lstm=nn.LSTM(len(FEATURES),HIDDEN,num_layers=LAYERS,batch_first=True); self.fc=nn.Linear(HIDDEN,1)
 def forward(self,x): return self.fc(self.lstm(x)[0][:,-1,:]).squeeze(-1)

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
 return torch.load(BASE/eligible[-1]["file"],map_location="cpu",weights_only=False)

def predict_probability(df,decision_index=None,bundle=None):
 if decision_index is None: decision_index=len(df)-1
 if bundle is None: bundle=load_snapshot(decision_index)
 if bundle is None or len(df)<SEQ_LEN: return float("nan")
 x=make_features(df).iloc[-SEQ_LEN:].to_numpy(float)
 if not np.isfinite(x).all(): return float("nan")
 mean=np.asarray(bundle["scaler_mean"]); scale=np.asarray(bundle["scaler_scale"]); scale=np.where(scale==0,1,scale)
 z=(x-mean)/scale; m=LSTMClassifier(); m.load_state_dict(bundle["state_dict"]); m.eval()
 with torch.no_grad(): return float(torch.sigmoid(m(torch.tensor(z[None],dtype=torch.float32))).item())
