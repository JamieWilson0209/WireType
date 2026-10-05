#!/usr/bin/env python3
"""How FAFB -> MCNS transfer works.

From the seed-0 paper encoder's saved embeddings: (A) replicate the linear read-out
(0.821); (B) per-type accuracy against the MCNS/FAFB output-synapse ratio; (C)
accuracy within class by the MCNS cell's output-synapse quartile; (D) project out
the embedding directions that linearly predict log partner and synapse counts and
rescore, forward and reverse, against random directions; (E) share of MCNS cells
whose nearest FAFB training cell is the same named type, with chance and the
untrained encoder as controls. Runs locally.

In the paper (Results, cell-type matching):
section E gives the 81.9% same-type nearest neighbours (against 0.9% by chance and
52.7% for WireType (untrained)), the probe's 98.5% and 82.5% accuracy with and
without a same-type neighbour, and dopamine's 12% same-type neighbours at 99%
accuracy. The same search by region is in regions_and_floors.py.

Names used below: M, F = harmonised MCNS and FAFB node tables; S = FAFB's random
split; EF, EM = WireType's embeddings of FAFB and MCNS (seed 0); CF, CM = neurons with
at least one connection in each volume (only these are fitted or scored, as in
wiretype.eval.transfer.zero_shot); ftr = connected FAFB
training neurons with an experimental label (the probe's fit); fall = connected FAFB
training neurons; mte = connected MCNS brain neurons with an experimental label (the
scored set); fl, ml = the
experimental labels of each volume; pr0 = the replicated probe's MCNS predictions;
d = the scored MCNS neurons with those predictions. ER, EFR = the reverse model's
embeddings of MCNS and FAFB.

    PYTHONPATH=src .venv/bin/python scripts/diagnostics/forward_mechanism.py
"""
import numpy as np, pandas as pd, sys, contextlib, io, warnings
import numpy as np, pandas as pd, sys, contextlib, io, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from scipy.stats import spearmanr
sys.path.insert(0,"src")
from wiretype.data.labels import harmonise

# --- Inputs: node tables (harmonise prints a summary; silenced), split, embeddings
with contextlib.redirect_stdout(io.StringIO()):
    M=harmonise(pd.read_parquet("data/processed/mcns_nodes.parquet"),"mcns")
    F=harmonise(pd.read_parquet("data/processed/fafb_nodes.parquet"),"fafb")
S=pd.read_parquet("data/processed/fafb_splits.parquet").set_index("node_id").reindex(F.node_id)["split_random"].values
CF=((F.n_in+F.n_out)>0).values
CM=((M.n_in+M.n_out)>0).values
P="experiments/transfer/transfer_fafb_to_mcns_checkpoint_fafb_displacement_refined_split_random_s24k_ntbest_source_embeddings_"
EF=np.load(P+"fafb.npy").astype(np.float32)
EM=np.load(P+"mcns.npy").astype(np.float32)
DEG=["n_in","n_out","syn_in","syn_out"]

def deg(N):
    # log(1 + count) of partners and synapses, in and out: the "count" targets of section D
    return np.log1p(N[DEG].to_numpy(float))

ftr=(S=="train")&F.nt_known.notna().values&CF
mte=((M.region=="brain")&M.nt_known.notna()).values&CM
CL=None

def readout(Es,Et,src_mask,src_lab,tgt_mask,tgt_lab,C):
    # The linear probe: scaler and class-balanced logistic regression fitted on the
    # source, applied to the target. Returns target macro-F1, predictions, classes.
    sc=StandardScaler().fit(Es[src_mask])
    clf=LogisticRegression(C=C,class_weight="balanced",max_iter=2000).fit(sc.transform(Es[src_mask]),src_lab[src_mask])
    pr=clf.predict(sc.transform(Et[tgt_mask]))
    return f1_score(tgt_lab[tgt_mask],pr,labels=clf.classes_,average="macro"),pr,clf.classes_

# --- A. replicate the forward linear probe (C = 0.01, as the report chose)
fl=F.nt_known.values
ml=M.nt_known.values
f0,pr0,cls=readout(EF,EM,ftr,fl,mte,ml,0.01)
print(f"A. replicated forward macro-F1 (linear): {f0:.3f}  (compare the seed-0 report's nt_known/brain macro-F1)")
d=M.loc[mte].assign(pred=pr0,ok=lambda x:x.pred==x.nt_known)

# --- B. per-type accuracy vs connectivity-scale ratio
Fb=F[F.region=="brain"]
Mb=M[(M.region=="brain")&((M.n_in+M.n_out)>0)]
r=np.log2(Mb.groupby("cell_type").syn_out.median()/Fb.groupby("cell_type").syn_out.median()).replace([np.inf,-np.inf],np.nan).dropna()
ta=d.groupby("cell_type").agg(n=("ok","size"),acc=("ok","mean"),nt=("nt_known",lambda x:x.mode().iat[0]))
ta=ta.join(r.rename("ratio"),how="inner")
ta=ta[ta.n>=10]
rho,p=spearmanr(ta.ratio,ta.acc)
print(f"\nB. per-type accuracy vs log2(MCNS/FAFB output synapses): {len(ta)} types (>=10 cells), Spearman rho={rho:.2f} p={p:.1e}")
for nt,g in ta.groupby("nt"):
    if len(g)>=8:
        rr,pp=spearmanr(g.ratio,g.acc)
        print(f"   {nt:13s} types={len(g):4d} median ratio x{2**g.ratio.median():.1f}  mean acc {g.acc.mean():.3f}  rho={rr:+.2f} (p={pp:.2g})")
ta["rbin"]=pd.cut(ta.ratio,[-9,0,1,2,9],labels=["MCNS<=FAFB","1-2x","2-4x",">4x"])
print("   accuracy by ratio bin (types):",ta.groupby("rbin",observed=True).acc.agg(["size","mean"]).round(3).to_dict("index"))

# --- C. within class, MCNS cell's own output synapses
print("\nC. forward accuracy within class by the MCNS cell's output-synapse quartile:")
for nt,g in d.groupby("nt_known"):
    if len(g)>=40:
        q=pd.qcut(g.syn_out.rank(method="first"),4,labels=["q1","q2","q3","q4"])
        print(f"   {nt:13s}",g.groupby(q,observed=True).ok.mean().round(2).to_dict())

# --- D. remove count directions
def count_basis(E,mask,Y,rounds):
    # An orthonormal basis of embedding directions that linearly predict the log
    # counts, grown over `rounds` rounds of fitting on what the last round left.
    B=np.zeros((E.shape[1],0),np.float32)
    Er=E.copy()
    for _ in range(rounds):
        sc=StandardScaler().fit(Er[mask])
        X=sc.transform(Er[mask])
        W=Ridge(alpha=1.0).fit(X,Y[mask]).coef_.T/sc.scale_[:,None]      # directions in raw embedding space
        B=np.linalg.qr(np.hstack([B,W]))[0].astype(np.float32)
        Er=E-(E@B)@B.T
    return B

def null(E,B):
    # E with the span of B projected out
    return E-(E@B)@B.T

def r2(E,mask,Y,Et,Yt,mt):
    # How well the embeddings still predict log counts (ridge), in the source and the target
    sc=StandardScaler().fit(E[mask])
    m=Ridge(1.0).fit(sc.transform(E[mask]),Y[mask])
    return r2_score(Y[mask],m.predict(sc.transform(E[mask]))), r2_score(Yt[mt],m.predict(sc.transform(Et[mt])))

YF,YM=deg(F),deg(M)
fall=(S=="train")&CF
print("\nD. count directions removed (linear read-out macro-F1; R2 of log counts from embeddings, FAFB / MCNS)")
print(f"   none            F1 {f0:.3f}   R2 {r2(EF,fall,YF,EM,YM,mte)[0]:.2f} / {r2(EF,fall,YF,EM,YM,mte)[1]:.2f}")
rng=np.random.default_rng(0)   # shared by the random controls (D), the reverse subsample (D') and the sample (E)
for rounds in (1, 4, 16):
    B=count_basis(EF,fall,YF,rounds)
    f,pr,_=readout(null(EF,B),null(EM,B),ftr,fl,mte,ml,0.01)
    rf=r2(null(EF,B),fall,YF,null(EM,B),YM,mte)
    ctrl=[]
    for s in range(3):
        # control: remove as many random directions
        R=np.linalg.qr(rng.standard_normal((EF.shape[1],B.shape[1])))[0].astype(np.float32)
        ctrl.append(readout(null(EF,R),null(EM,R),ftr,fl,mte,ml,0.01)[0])
    pc=dict(zip(cls,np.round(f1_score(ml[mte],pr,labels=cls,average=None),2)))
    print(f"   {B.shape[1]:3d} dims        F1 {f:.3f}   R2 {rf[0]:.2f} / {rf[1]:.2f}   random-{B.shape[1]} control F1 {np.mean(ctrl):.3f}   per class {pc}")

# --- D'. same on the reverse model
ER=np.load("experiments/train/embeddings_mcns_displacement_refined_split_random_s24k_ntbest_brain.npy").astype(np.float32)
EFR=np.load("experiments/transfer/transfer_mcns_to_fafb_checkpoint_mcns_displacement_refined_split_random_s24k_ntbest_brain_source_embeddings_fafb.npy").astype(np.float32)
ins=((M.region=="brain")&((M.n_in+M.n_out)>0)).values
mtr=ins&M.nt_known.notna().values
sub=np.zeros(len(M),bool)
sub[rng.choice(np.flatnonzero(mtr),52000,replace=False)]=True
fte=((F.region=="brain")&F.nt_known.notna()).values&CF
fr0,prr,_=readout(ER,EFR,sub,ml,fte,fl,1.0)
print(f"\nD'. reverse, none      F1 {fr0:.3f}")
for rounds in (1, 4, 16):
    B=count_basis(ER,ins,YM,rounds)
    f,pr,_=readout(null(ER,B),null(EFR,B),sub,ml,fte,fl,1.0)
    pc=dict(zip(cls,np.round(f1_score(fl[fte],pr,labels=cls,average=None),2)))
    print(f"    reverse {B.shape[1]:3d} dims F1 {f:.3f}   per class {pc}")

# --- E. nearest FAFB training neighbour: same cell type?
#     Cosine similarity on embeddings standardised with FAFB training statistics.
sc=StandardScaler().fit(EF[fall])
A=sc.transform(EF[fall])
A/=np.linalg.norm(A,axis=1,keepdims=True)
ftypes=F.cell_type.values[fall]
shared=d.cell_type.isin(set(ftypes)).values
q=np.flatnonzero(mte)[shared]
q=rng.choice(q,min(15000,len(q)),replace=False)   # the random sample of 15,000 scored MCNS neurons
Q=sc.transform(EM[q])
Q/=np.linalg.norm(Q,axis=1,keepdims=True)
nn=np.concatenate([np.argmax(Q[i:i+1000]@A.T,axis=1) for i in range(0,len(Q),1000)])
same=ftypes[nn]==M.cell_type.values[q]
samet=F.nt_known.values[fall][nn]
print(f"\nE. MCNS cells (types shared with FAFB, n={len(q)}): nearest FAFB neighbour is the same cell type {same.mean():.3f}")
e=pd.DataFrame({"nt":M.nt_known.values[q],"same":same,"ok":(pr0[np.searchsorted(np.flatnonzero(mte),q)]==M.nt_known.values[q])})
print("   by class: same-type share / read-out accuracy")
print(e.groupby("nt").agg(n=("same","size"),same_type=("same","mean"),acc=("ok","mean")).round(3))
print("   accuracy when nearest neighbour is same type vs not:",e.groupby("same").ok.mean().round(3).to_dict())

# --- E controls: chance and the untrained encoder
vc=pd.Series(ftypes).value_counts(normalize=True)
qt=M.cell_type.values[q]
print(f"\nE-control chance (random FAFB training cell shares the type): {np.mean([vc.get(t,0) for t in qt]):.4f}")
U="experiments/transfer/transfer_fafb_to_mcns_checkpoint_fafb_untrained_refined_split_random_s24k_ntbest_source_embeddings_"
import os
if os.path.exists(U+"fafb.npy"):
    UF=np.load(U+"fafb.npy").astype(np.float32)
    UM=np.load(U+"mcns.npy").astype(np.float32)
    sc=StandardScaler().fit(UF[fall])
    A=sc.transform(UF[fall])
    A/=np.linalg.norm(A,axis=1,keepdims=True)
    Q=sc.transform(UM[q])
    Q/=np.linalg.norm(Q,axis=1,keepdims=True)
    nn=np.concatenate([np.argmax(Q[i:i+1000]@A.T,axis=1) for i in range(0,len(Q),1000)])
    print(f"E-control untrained encoder: nearest FAFB neighbour same cell type {(ftypes[nn]==qt).mean():.3f}")
