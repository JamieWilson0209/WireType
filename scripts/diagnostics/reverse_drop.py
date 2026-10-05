#!/usr/bin/env python3
"""Why MCNS -> FAFB transfer drops.

Refits the reverse linear read-out from saved embeddings (reproduces the reported
macro-F1 to within the MCNS training subsample), then: class counts in each volume,
the confusion matrix, where FAFB dopamine cells go by cell type, label coverage of
those types in MCNS, a label-free read-out scaler, and output connectivity of
correct against wrong dopamine calls. Runs locally; needs the reverse transfer's
FAFB embeddings and the MCNS training embeddings.

In the paper (Results, the reverse direction): the output synapses of
FAFB dopaminergic neurons labelled correctly against those missed (medians 368 and
152) and the one-sided Mann-Whitney test (p = 1.7e-12) come from the last section.

Names used below: M, F = harmonised MCNS and FAFB node tables; EM, EF = the reverse
model's embeddings of MCNS and FAFB; tr = MCNS brain neurons with an experimental label
(the read-out's training pool); te = connected FAFB neurons with an experimental label (the scored set); sc, clf = the read-out's
scaler and logistic regression; pr = its FAFB predictions; d, da = FAFB scored
neurons and their dopaminergic subset, with predictions attached.

    PYTHONPATH=src .venv/bin/python scripts/diagnostics/reverse_drop.py
"""
import numpy as np, pandas as pd, sys
import numpy as np, pandas as pd, sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
sys.path.insert(0,"src")
from wiretype.data.labels import harmonise
import contextlib, io

# --- Inputs: node tables (harmonise prints a summary; silenced) and embeddings
with contextlib.redirect_stdout(io.StringIO()):
    M=harmonise(pd.read_parquet("data/processed/mcns_nodes.parquet"),"mcns")
    F=harmonise(pd.read_parquet("data/processed/fafb_nodes.parquet"),"fafb")
EM=np.load("experiments/train/embeddings_mcns_displacement_refined_split_random_s24k_ntbest_brain.npy").astype(np.float32)
EF=np.load("experiments/transfer/transfer_mcns_to_fafb_checkpoint_mcns_displacement_refined_split_random_s24k_ntbest_brain_source_embeddings_fafb.npy").astype(np.float32)

# --- Populations: MCNS brain neurons with synapses (the reverse model's scope)
inscope=(M.region=="brain")&((M.n_in+M.n_out)>0)
tr=inscope&M.nt_known.notna()
te=(F.region=="brain")&F.nt_known.notna()&((F.n_in+F.n_out)>0)  # connected only, as the transfer scores
print("MCNS confident training cells (all in-scope, superset of the train split):",tr.sum(), "| FAFB scored:",te.sum())
print("\nclass counts  MCNS confident | FAFB confident")
print(pd.concat([M.loc[tr,"nt_known"].value_counts(),F.loc[te,"nt_known"].value_counts()],axis=1,keys=["mcns","fafb"]))

# --- Refit the reverse linear read-out on a 52,000-neuron subsample of MCNS
rng=np.random.default_rng(0)
idx=np.flatnonzero(tr)
sub=rng.choice(idx,min(len(idx),52000),replace=False)
sc=StandardScaler().fit(EM[sub])
clf=LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000).fit(sc.transform(EM[sub]),M.nt_known.values[sub])
yt=F.loc[te,"nt_known"].values
pr=clf.predict(sc.transform(EF[te.values]))
cls=clf.classes_
print("\nreplicated reverse macro-F1 (source scaler):",round(f1_score(yt,pr,labels=cls,average="macro"),3))
print(pd.DataFrame(confusion_matrix(yt,pr,labels=cls,normalize="true"),index=cls,columns=cls).round(2))

# --- Where FAFB dopamine cells go, by type
d=F.loc[te].assign(pred=pr)
da=d[d.nt_known=="dopamine"]
print("\nFAFB dopamine cells by type (n, share correct, top wrong call):")
g=da.groupby("cell_type").apply(lambda x: pd.Series({"n":len(x),"correct":(x.pred=="dopamine").mean(),"top_pred":x.pred.value_counts().index[0]})).sort_values("n",ascending=False)
print(g.head(15))
# Label coverage: do MCNS experimental labels include the same dopaminergic types?
mda=set(M.loc[tr&(M.nt_known=="dopamine"),"cell_type"].dropna())
print("\nFAFB DA cells whose type has confident DA cells in MCNS:",round(da.cell_type.isin(mda).mean(),3),"| MCNS confident DA types:",len(mda))
print("MCNS confident DA types, top:",M.loc[tr&(M.nt_known=="dopamine"),"cell_type"].value_counts().head(8).to_dict())

# --- Type coverage overall: scores for FAFB neurons whose type MCNS labels, and the rest
mt=set(M.loc[tr,"cell_type"].dropna())
print("\nFAFB scored cells whose type has confident cells in MCNS:",round(d.cell_type.isin(mt).mean(),3))
for inm in [True,False]:
    s=d[d.cell_type.isin(mt)==inm]
    print(f"  type in MCNS confident set={inm}: n={len(s)}, macro-F1 {f1_score(s.nt_known,s.pred,labels=cls,average='macro'):.3f}, acc {(s.nt_known==s.pred).mean():.3f}")

# --- Read-out scaler variant (label-free): target's own embedding statistics
sc2=StandardScaler().fit(EF[(F.region=="brain").values])
pr2=clf.predict(sc2.transform(EF[te.values]))
print("\nread-out scaled with FAFB's own embedding stats (label-free):",round(f1_score(yt,pr2,labels=cls,average="macro"),3))
print("per-class F1 source-scaler:",dict(zip(cls,np.round(f1_score(yt,pr,labels=cls,average=None),3))))
print("per-class F1 own-scaler:   ",dict(zip(cls,np.round(f1_score(yt,pr2,labels=cls,average=None),3))))

# --- Within FAFB dopamine cells: output connectivity of correct against wrong calls
#     (the paper's 368 against 152 median output synapses and the Mann-Whitney test)
print("\n=== within FAFB dopamine cells: output connectivity of correct vs wrong calls")
da=da.assign(ok=da.pred=="dopamine")
print(da.groupby("ok")[["syn_out","n_out","syn_in","n_in"]].median())
from scipy.stats import mannwhitneyu
print("Mann-Whitney syn_out correct>wrong p =", mannwhitneyu(da.loc[da.ok,"syn_out"],da.loc[~da.ok,"syn_out"],alternative="greater").pvalue)
q=pd.qcut(da.syn_out,4,labels=["q1 low","q2","q3","q4 high"])
print("share called dopamine by syn_out quartile:",da.groupby(q).ok.mean().round(2).to_dict())
mda_out=M.loc[tr&(M.nt_known=="dopamine"),"syn_out"]
print("median syn_out: MCNS confident DA",mda_out.median(),"| FAFB DA",da.syn_out.median(), "| MCNS confident ACh", M.loc[tr&(M.nt_known=="acetylcholine"),"syn_out"].median())
oa=d[d.nt_known=="octopamine"].assign(ok=lambda x:x.pred=="octopamine")
print("octopamine correct vs wrong median syn_out:",oa.groupby("ok").syn_out.median().to_dict())
