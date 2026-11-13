# Who absorbed the DDEP loss? Analysis script (Python 3; pandas, numpy, scipy, statsmodels).
# Writes tables to ./out/. Seed fixed for permutation and bootstrap results.
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy import stats
import sys,os
def load():
    # usage: python analysis.py path/to/ghana_banking_panel_2020_2025.xlsx
    f=sys.argv[1] if len(sys.argv)>1 else 'ghana_banking_panel_2020_2025.xlsx'
    return pd.read_excel(f,sheet_name='Panel_Long')
os.makedirs('out',exist_ok=True)
rng=np.random.default_rng(20261110)
P=load()
P['burden']=1-P.cir-P.pbt_margin
W=P.pivot(index='bank_code',columns='year')
def g(v,y): return W[(v,y)]
banks=W.index
# sample rule
base_year=pd.Series(2021,index=banks)
have=lambda b,y: not np.isnan(g('roa',y)[b]) and not np.isnan(g('pbt_margin',y)[b]) and not np.isnan(g('cir',y)[b])
def sample(bymap=None):
    rows=[]
    for b in banks:
        by=2021 if have(b,2021) else (2020 if have(b,2020) else None)
        if by and have(b,2022) and have(b,2025): rows.append((b,by))
    return rows
S=sample()
print(len(S),[b for b,_ in S if _!=2021], 'excluded:',[b for b in banks if b not in [x for x,_ in S]])
def build(basefun):
    R=[]
    for b in banks:
        by=basefun(b)
        if by is None or not (have(b,2022) and have(b,2025)): continue
        r=dict(bank=b)
        for v in ['roa','pbt_margin','cir','nim','burden']:
            r[v+'_base']=by(v) if callable(by) else None
        R.append(r)
    return R
def frame(mode='2021'):
    R=[]
    for b in banks:
        if not (have(b,2022) and have(b,2025)): continue
        if mode=='2021': ys=[2021] if have(b,2021) else ([2020] if have(b,2020) else None)
        elif mode=='2020': ys=[2020] if have(b,2020) else None
        elif mode=='avg':
            ys=[y for y in (2020,2021) if have(b,y)] or None
        elif mode=='strict2021': ys=[2021] if have(b,2021) else None
        if not ys: continue
        r=dict(bank=b,base_years='+'.join(map(str,ys)))
        for v in ['roa','pbt_margin','cir','nim','burden']:
            r[v+'_base']=np.mean([g(v,y)[b] for y in ys])
            r[v+'_22']=g(v,2022)[b]; r[v+'_25']=g(v,2025)[b]
        r['ln_size']=P[(P.bank_code==b)&(P.year==2020)].ln_assets_share.iloc[0]
        R.append(r)
    D=pd.DataFrame(R).set_index('bank')
    D['excess']=D.burden_22-D.burden_base
    for v in ['roa','pbt_margin']:
        D[v+'_rebound']=D[v+'_25']-D[v+'_22']
        D[v+'_scar']=D[v+'_25']-D[v+'_base']
        D[v+'_shock']=D[v+'_22']-D[v+'_base']
        D[v+'_recovered']=(D[v+'_25']>=D[v+'_base']).astype(int)
    return D
D=frame('2021'); D.to_csv('out/bank_measures.csv')
print(D.shape)
def ols_perm(x,y,B=20000):
    x=np.asarray(x,float); y=np.asarray(y,float)
    X=sm.add_constant(x); m=sm.OLS(y,X).fit(); h=m.get_robustcov_results('HC3')
    b=m.params[1]
    yy=np.array(y); xx=np.array(x)
    cnt=0
    for _ in range(B):
        yp=rng.permutation(yy); bp=np.polyfit(xx,yp,1)[0]
        if abs(bp)>=abs(b)-1e-12: cnt+=1
    return dict(n=len(x),slope=b,se_hc3=h.bse[1],ci_lo=h.conf_int()[1][0],ci_hi=h.conf_int()[1][1],p_hc3=h.pvalues[1],p_perm=(cnt+1)/(B+1),r2=m.rsquared)
def spear_boot(x,y,B=10000):
    rho,p=stats.spearmanr(x,y); n=len(x); bs=[]
    x=np.array(x);y=np.array(y)
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(set(x[i]))<2 or len(set(y[i]))<2: continue
        bs.append(stats.spearmanr(x[i],y[i])[0])
    return dict(n=n,rho=rho,p=p,lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5))
res={}
# 1 descriptives
d=D[['burden_base','burden_22','excess']].describe(percentiles=[.25,.5,.75]).T
print(d.round(3)); d.to_csv('out/t1_burden_describe.csv')
print('rank by excess:\n',D.excess.sort_values(ascending=False).round(3).to_string())
for nm,x in [('baseline ROA','roa_base'),('ln size','ln_size')]:
    r=spear_boot(D[x],D.excess,2000); print('Spearman excess vs',nm,r)
# 2
print('recovered ROA',D.roa_recovered.sum(),'of',len(D),' PBT',D.pbt_margin_recovered.sum())
print(D[['roa_base','roa_22','roa_25','roa_rebound','roa_scar','pbt_margin_base','pbt_margin_22','pbt_margin_25','pbt_margin_scar']].describe().round(4).T)
H1={}
for out_ in ['roa','pbt_margin']:
    H1[out_+'_scar']=ols_perm(D.excess,D[out_+'_scar'])
    H1[out_+'_rebound']=ols_perm(D.excess,D[out_+'_rebound'])
    r=stats.spearmanr(D.excess,D[out_+'_scar']); H1[out_+'_scar_spearman']=dict(rho=r[0],p=r[1])
H1df=pd.DataFrame(H1).T; print(H1df.round(4)); H1df.to_csv('out/t2_H1.csv')
# 3 decomposition (pbt margin = 1 - cir - burden)
D['d_pbt_25_22']=D.pbt_margin_25-D.pbt_margin_22
D['d_cir_25_22']=-(D.cir_25-D.cir_22); D['d_burden_25_22']=-(D.burden_25-D.burden_22)
D['d_pbt_25_b']=D.pbt_margin_25-D.pbt_margin_base
D['d_cir_25_b']=-(D.cir_25-D.cir_base); D['d_burden_25_b']=-(D.burden_25-D.burden_base)
D['d_nim_25_22']=D.nim_25-D.nim_22; D['d_nim_25_b']=D.nim_25-D.nim_base
assert np.allclose(D.d_pbt_25_22,D.d_cir_25_22+D.d_burden_25_22)
assert np.allclose(D.d_pbt_25_b,D.d_cir_25_b+D.d_burden_25_b)
dec=D[['d_pbt_25_22','d_cir_25_22','d_burden_25_22','d_nim_25_22','d_pbt_25_b','d_cir_25_b','d_burden_25_b','d_nim_25_b']]
print(dec.describe().round(3).T); dec.to_csv('out/t3_decomposition.csv')
share=(D.d_burden_25_22/D.d_pbt_25_22)
print('share of rebound from burden fall: median',share.median(), 'banks where burden>cost:',(D.d_burden_25_22.abs()>D.d_cir_25_22.abs()).sum())
print('baseline: burden contrib>0 count',(D.d_burden_25_b>0).sum(),'cost contrib>0',(D.d_cir_25_b>0).sum())
r=stats.spearmanr(D.d_nim_25_b,D.roa_scar); print('NIM change vs ROA scar spearman',r)
# 4 H2
H2=spear_boot(D.roa_base,D.roa_25); print('H2',H2)
H2b=spear_boot(D.pbt_margin_base,D.pbt_margin_25); print('H2 pbt',H2b)
q=lambda s: pd.qcut(s.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
tt=pd.crosstab(q(-D.roa_base),q(-D.roa_25),rownames=['baseline ROA quartile'],colnames=['2025 ROA quartile']); print(tt); tt.to_csv('out/t4_transition.csv')
# robustness
rob=[]
rob.append(dict(check='Main (baseline 2021; UMB 2020)',**{k:H1['roa_scar'][k] for k in ['n','slope','p_perm']},rho=H2['rho'],rho_lo=H2['lo'],rho_hi=H2['hi']))
for mode,label in [('2020','Baseline 2020 only'),('avg','Baseline 2020-21 average'),('strict2021','Excluding UMB (2021 only)')]:
    Dm=frame(mode); r1=ols_perm(Dm.excess,Dm.roa_scar,5000); r2=spear_boot(Dm.roa_base,Dm.roa_25,3000)
    rob.append(dict(check=label,n=r1['n'],slope=r1['slope'],p_perm=r1['p_perm'],rho=r2['rho'],rho_lo=r2['lo'],rho_hi=r2['hi']))
r1=H1['pbt_margin_scar']; rob.append(dict(check='Outcome: PBT margin',n=r1['n'],slope=r1['slope'],p_perm=r1['p_perm'],rho=H2b['rho'],rho_lo=H2b['lo'],rho_hi=H2b['hi']))
loo=[]
for b in D.index:
    Dm=D.drop(b); r1=ols_perm(Dm.excess,Dm.roa_scar,2000); r2=stats.spearmanr(Dm.roa_base,Dm.roa_25)
    loo.append(dict(dropped=b,slope=r1['slope'],p_perm=r1['p_perm'],p_hc3=r1['p_hc3'],rho=r2[0]))
loo=pd.DataFrame(loo); loo.to_csv('out/t6_leave_one_out.csv')
print(loo.round(4).to_string())
rob.append(dict(check='Leave-one-out: slope range',n=len(D)-1,slope=f"{loo.slope.min():.3f} to {loo.slope.max():.3f}",p_perm=f"max {loo.p_perm.max():.3f}",rho=f"{loo.rho.min():.2f} to {loo.rho.max():.2f}",rho_lo=None,rho_hi=None))
rb=pd.DataFrame(rob); print(rb.to_string()); rb.to_csv('out/t5_robustness.csv',index=False)
D.to_csv('out/bank_measures_full.csv')
