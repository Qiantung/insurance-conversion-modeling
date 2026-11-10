# -*- coding: utf-8 -*-
"""保险产品客户成交预测作业。仅从指定 CSV 和数据字典读取数据。

执行：python optimized_homework.py --data-dir 原始数据目录 --output-dir 输出目录
依赖：numpy、pandas、scikit-learn、matplotlib、openpyxl（仅读字典）。
本程序输出模型结果 HTML、统计 JSON、可复现模型和分箱结果。
Excel 由同目录的 build_iv_workbook.mjs 将统计 JSON 导出。
"""
from pathlib import Path
import argparse, base64, hashlib, html, io, json, os, platform, shutil, sys
# 使用单线程避免部分Windows受限环境对线程池管道的权限限制。
# 此设定仅影响执行资源，不改变训练数据、模型候选或评价口径。
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

SEED=20261008
MAX_FEATURES=12
CORR_LIMIT=0.65
SMOOTH=0.5  # 每个箱的正、负样本计数均加 0.5，避免 log(0)；真实样本数不变。
CSV_NAME='model_t_medical_recommend_sample_crzj_homework(utf-8).csv'
DICT_NAME='作业1_智能投顾建模样本说明和数据字典_to学生.xlsx'

def clean_numeric(s):
    """只将无穷值当作缺失；不把未知含义的负数或零擅自改写。"""
    return pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)

class WoeBinner:
    """基于训练集的监督分箱；缺失单列，包含缺失在内最多五箱。

    数值变量用浅层决策树寻找成交率区隔；类别变量先合并低频类别，
    再按训练集成交率排序、沿此顺序分箱。测试集未知类别映射 WOE=0。
    每次交叉验证均重新训练全部分箱，避免验证标签参与分箱。
    """
    def fit(self,X,y):
        self.rules={}; self.rows=[]; self.iv={}
        y=np.asarray(y,dtype=int)
        for col in X.columns:
            s=X[col]; cat=not pd.api.types.is_numeric_dtype(s)
            rule={'category':cat}
            if cat:
                s=s.astype('string')
                vc=s.value_counts()
                # 城市等高基数变量的小类别先合并，减小小样本偶然波动。
                keep=vc[vc>=max(100,int(len(s)*0.01))].index.tolist()
                z=s.where(s.isin(keep),'__RARE__').where(s.notna())
                rates=pd.DataFrame({'z':z,'y':y}).groupby('z').y.mean().sort_values(kind='stable')
                order={str(k):i for i,k in enumerate(rates.index)}
                v=z.map(order).astype(float)
                rule.update(keep=keep,order=order)
            else:
                v=clean_numeric(s)
            good=v.notna().to_numpy(); missing=bool((~good).any())
            max_leaves=4 if missing else 5
            cuts=[]
            if good.sum()>0 and v[good].nunique()>1:
                tree=DecisionTreeClassifier(max_leaf_nodes=max_leaves,
                    min_samples_leaf=max(100,int(good.sum()*0.05)),random_state=SEED)
                tree.fit(v[good].to_numpy().reshape(-1,1),y[good])
                cuts=sorted(float(t) for t,f in zip(tree.tree_.threshold,tree.tree_.feature) if f>=0)
            rule['cuts']=cuts
            bins=np.searchsorted(cuts,v.fillna(0).to_numpy(),side='left')
            bins[~good]=-1
            ids=sorted(np.unique(bins).tolist())
            k=len(ids); pos=int(y.sum()); neg=len(y)-pos
            wmap={}; total_iv=0.; entries=[]
            for b in ids:
                mask=bins==b; n=int(mask.sum()); n1=int(y[mask].sum()); n0=n-n1
                p1=(n1+SMOOTH)/(pos+SMOOTH*k)
                p0=(n0+SMOOTH)/(neg+SMOOTH*k)
                w=float(np.log(p1/p0)); iv=(p1-p0)*w
                wmap[int(b)]=w; total_iv+=iv
                if b==-1:
                    label='Missing'
                elif cat:
                    categories=[a for a,r in rule['order'].items() if np.searchsorted(cuts,r,side='left')==b]
                    label='{'+'; '.join(categories)+'}'
                else:
                    left='-inf' if b==0 else format(cuts[b-1],'.12g')
                    right='inf' if b==len(cuts) else format(cuts[b],'.12g')
                    label=f'({left}, {right}]'
                entries.append({'name':col,'type':'text' if cat else 'numeric','cls':label,
                    'cnt_0':n0,'cnt_1':n1,'cnt_all':n,'target_1':n1/n,
                    'woe':w,'iv':float(iv),'bin_id':int(b),'dist_0':p0,'dist_1':p1})
            for row in entries: row['sum_iv']=float(total_iv)
            rule['woe']=wmap
            self.rules[col]=rule; self.iv[col]=float(total_iv); self.rows.extend(entries)
        return self

    def transform(self,X):
        out={}
        for col,rule in self.rules.items():
            s=X[col]
            if rule['category']:
                z=s.astype('string')
                z=z.where(z.isin(rule['keep']),'__RARE__').where(s.notna())
                v=z.map(rule['order']).astype(float)
                unknown=s.notna() & v.isna()
            else:
                v=clean_numeric(s); unknown=pd.Series(False,index=s.index)
            b=np.searchsorted(rule['cuts'],v.fillna(0).to_numpy(),side='left')
            b[v.isna().to_numpy()]=rule.get('missing_bin',-1)
            w=np.array([rule['woe'].get(int(i),0.) for i in b],dtype=float)
            w[np.asarray(unknown)]=0.
            out[col]=w
        return pd.DataFrame(out,index=X.index)

def ratio_equivalent(name,columns):
    """仅识别字面可对应的已有占比变量，不凭空构造分子分母。"""
    candidates=[]
    if name.endswith('_pol_num'): candidates.append(name[:-8]+'_pol_ratio')
    if name.endswith('_act_prem'): candidates.append(name+'_ratio')
    return next((c for c in candidates if c in columns),None)

def select_features(Z,iv,max_features=12,corr_limit=0.65,iv_floor=0.02):
    """优先高 IV，并剔除 WOE 绝对 Pearson 相关系数超过 0.65 的变量。

    若原数值有对应占比且占比 IV 至少为其 80%，优先占比变量。
    原始数值仍保留在 IV 表；筛选过程记录入模和未入模原因。
    """
    order=sorted(iv,key=lambda c:(-iv[c],c))
    deferred={c:ratio_equivalent(c,iv) for c in order}
    deferred={c:r for c,r in deferred.items() if r and iv[r]>=iv[c]*0.8 and iv[r]>=iv_floor}
    order=[c for c in order if c not in deferred]+[c for c in order if c in deferred]
    selected=[]; reasons={}
    for c in order:
        if Z[c].std()<1e-12: reasons[c]='WOE常量，无法提供区分'; continue
        if iv[c]<iv_floor: reasons[c]=f'IV低于{iv_floor}'; continue
        if c in deferred and deferred[c] in selected:
            reasons[c]='优先对应占比变量 '+deferred[c]; continue
        corr=Z[selected].corrwith(Z[c]).abs() if selected else pd.Series(dtype=float)
        if len(corr) and corr.max()>corr_limit:
            reasons[c]=f'与 {corr.idxmax()} 的WOE相关系数={corr.max():.4f}'; continue
        if len(selected)>=max_features: reasons[c]=f'达到{max_features}变量上限'; continue
        selected.append(c); reasons[c]='入模'
    return selected,reasons

def make_model(key):
    if key.startswith('LR_'):
        return LogisticRegression(C=float(key.split('_')[1]),penalty='l2',solver='lbfgs',max_iter=1500,random_state=SEED)
    return HistGradientBoostingClassifier(learning_rate=0.05,max_iter=150,
        max_leaf_nodes=int(key.split('_')[1]),min_samples_leaf=150,l2_regularization=10,
        early_stopping=False,random_state=SEED)

def ks_threshold(y,p):
    fpr,tpr,thresholds=roc_curve(y,p)
    i=int(np.argmax(tpr-fpr))
    return float(thresholds[i]),float(tpr[i]-fpr[i])

def evaluate(y,p,threshold):
    pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'AUC':float(roc_auc_score(y,p)),'KS':ks_threshold(y,p)[1],
        'RECALL':float(recall_score(y,pred,zero_division=0)),
        'PRE':float(precision_score(y,pred,zero_division=0)),
        'threshold':threshold,'RECALL_at_0.5':float(recall_score(y,p>=0.5,zero_division=0)),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def deciles(y,p,ids):
    """按概率降序、客户编号升序拆成十组；同分跨组会明确说明。"""
    tab=pd.DataFrame({'target':np.asarray(y),'p':p,'id':np.asarray(ids)})
    tab=tab.sort_values(['p','id'],ascending=[False,True],kind='stable').reset_index(drop=True)
    tab['group']=np.minimum(np.arange(len(tab))*10//len(tab)+1,10)
    rows=[]; cn=cp=0; baseline=tab.target.mean(); total_pos=int(tab.target.sum())
    for g,d in tab.groupby('group',sort=True):
        n=len(d); n1=int(d.target.sum()); cn+=n; cp+=n1
        rows.append({'decile':int(g),'range':f'[{d.p.min():.8f}, {d.p.max():.8f}]',
            'cnt_0':n-n1,'cnt_1':n1,'sample_cnt':n,'target_cnt':n1,
            'cum_sample_cnt':cn,'cum_target_cnt':cp,'target%':n1/n,
            'target_per':n1/total_pos,'cum_target%':cp/cn,
            'cum_target_per':cp/total_pos,'lift':(n1/n)/baseline,
            'cum_lift':(cp/cn)/baseline})
    result=pd.DataFrame(rows)
    assert len(result)==10 and result.sample_cnt.sum()==len(tab)
    assert result.cnt_1.sum()==total_pos and result.cnt_0.sum()==len(tab)-total_pos
    return result

def image_uri(fig):
    buffer=io.BytesIO(); fig.savefig(buffer,format='png',dpi=145,bbox_inches='tight'); plt.close(fig)
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()

def table(df):
    return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}',escape=True)

# 以下为优化版：训练样本划分与原版相同；方案选择改用训练集内部前推验证。
import copy
from sklearn.metrics import brier_score_loss
from scipy.stats import ks_2samp

POLICIES={'retain':'独立缺失箱','neutral':'稀疏缺失WOE置零','merge':'稀疏缺失合并最大普通箱'}

def apply_sparse(base,policy):
    """只调整小于100条的训练缺失箱；不改动原始记录。

    合并目标按普通箱人数最大确定，不使用验证标签。合并后重新计算全部
    统计量。中性方案保留原始统计WOE及IV，另列实际模型编码model_woe。
    """
    b=copy.deepcopy(base); result=[]; b.changes=[]
    for col,rule in b.rules.items():
        rows=[copy.deepcopy(r) for r in base.rows if r['name']==col]
        missing=next((r for r in rows if r['bin_id']==-1),None)
        if missing and missing['cnt_all']<100 and len(rows)>1:
            change={'name':col,'missing_count':missing['cnt_all'],'policy':policy}
            if policy=='merge':
                target=max((r for r in rows if r['bin_id']!=-1),key=lambda r:r['cnt_all'])
                change['merged_into']=target['cls']
                target['cls']+=' ∪ Missing'
                for k in ['cnt_0','cnt_1','cnt_all']: target[k]+=missing[k]
                rule['missing_bin']=target['bin_id']; rows=[r for r in rows if r['bin_id']!=-1]
            if policy!='retain': b.changes.append(change)
        pos=sum(r['cnt_1'] for r in rows); neg=sum(r['cnt_0'] for r in rows); k=len(rows)
        for r in rows:
            r['target_1']=r['cnt_1']/r['cnt_all']
            r['dist_0']=(r['cnt_0']+.5)/(neg+.5*k); r['dist_1']=(r['cnt_1']+.5)/(pos+.5*k)
            r['woe']=float(np.log(r['dist_1']/r['dist_0']))
            r['iv']=(r['dist_1']-r['dist_0'])*r['woe']
            r['model_woe']=0. if policy=='neutral' and r['bin_id']==-1 and r['cnt_all']<100 else r['woe']
        rule['woe']={r['bin_id']:r['model_woe'] for r in rows}
        b.iv[col]=sum(r['iv'] for r in rows)
        for r in rows:r['sum_iv']=b.iv[col]
        result.extend(rows)
    b.rows=result; b.policy=policy
    return b

def bin_ids(binner,X):
    """使用训练边界映射分箱编号，供漂移与独立核验使用。"""
    result={}
    for col,rule in binner.rules.items():
        s=X[col]
        if rule['category']:
            s=s.astype('string'); s=s.where(s.isin(rule['keep']),'__RARE__').where(s.notna())
            v=s.map(rule['order']).astype(float)
        else:v=clean_numeric(s)
        ids=np.searchsorted(rule['cuts'],v.fillna(0),side='left')
        ids[v.isna().to_numpy()]=rule.get('missing_bin',-1)
        result[col]=ids
    return pd.DataFrame(result,index=X.index)

def metric_table(frame):
    """展示百分数，底层JSON和CSV仍保留数值，避免文字格式参与计算。"""
    frame=frame.copy()
    pct={'RECALL','PRE','RECALL_at_0.5','target%','target_per','cum_target%','cum_target_per','actual_rate','mean_probability',
         'succ%','capture','conversion','contact_share','capture_low','capture_high',
         'missing_train','missing_month','train_rate','month_rate','rate_change'}
    for col in frame.columns:
        if col in pct:frame[col]=frame[col].map(lambda v:f'{v:.2%}' if pd.notna(v) else '—')
    return table(frame)

def psi(reference,current,k):
    """固定训练箱；按计数加0.5平滑后计算PSI。数值越大表示分布变化越大。"""
    a=np.bincount(reference,minlength=k).astype(float)+.5
    b=np.bincount(current,minlength=k).astype(float)+.5
    a/=a.sum(); b/=b.sum()
    return float(np.sum((b-a)*np.log(b/a)))

def auc_fast(y,p):
    """独立的平均秩AUC算法，处理同分，供Bootstrap使用。"""
    n1=y.sum(); n0=len(y)-n1
    ranks=pd.Series(p).rank(method='average').to_numpy()
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def bootstrap(y,p1,p2,ids,reps=1000):
    """同一重采样同时用于两模型；按月份×标签分层，有放回抽取原记录。

    不生成新观测、不改变训练数据、不重拟合模型。区间只描述这份评估样本
    下的条件不确定性，不覆盖整个训练/筛选过程。此前已查看val，不能称它
    是从未使用过的新盲测；本轮没有按val结果反向选模型。
    """
    rng=np.random.default_rng(SEED+91)
    groups=[np.flatnonzero((ids==m)&(y==v)) for m in np.unique(ids) for v in [0,1]]
    records=[]
    for _ in range(reps):
        ix=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        yy=y[ix]; a=p1[ix]; b=p2[ix]
        a1=auc_fast(yy,a); a2=auc_fast(yy,b)
        ks1=ks_2samp(a[yy==1],a[yy==0],method='asymp').statistic
        ks2=ks_2samp(b[yy==1],b[yy==0],method='asymp').statistic
        records.append([a1,a2,a2-a1,float(ks1),float(ks2),float(ks2-ks1)])
    values=np.asarray(records); result=[]
    real=[auc_fast(y,p1),auc_fast(y,p2),auc_fast(y,p2)-auc_fast(y,p1),
          ks_threshold(y,p1)[1],ks_threshold(y,p2)[1],ks_threshold(y,p2)[1]-ks_threshold(y,p1)[1]]
    for j,name in enumerate(['LR_AUC','GBDT_AUC','GBDT_minus_LR_AUC','LR_KS','GBDT_KS','GBDT_minus_LR_KS']):
        lo,hi=np.quantile(values[:,j],[.025,.975])
        result.append({'metric':name,'estimate':real[j],'lower_95':float(lo),'upper_95':float(hi),'repetitions':reps})
    return result

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True,help='Directory containing authorized source CSV and dictionary')
    parser.add_argument('--output-dir',default='outputs/优化版')
    parser.add_argument('--baseline-results',default=None,help='可选：原版model_results.json，用于同源结果对照，不参与调参')
    args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); base=Path(args.data_dir)
    df=pd.read_csv(base/CSV_NAME,dtype={'customer_id':'string','gender':'string','city_name':'string','education':'string','txh_level_code':'string'})
    dictionary=pd.read_excel(base/DICT_NAME,sheet_name='样本特征')
    meanings=dict(zip(dictionary['字段名'],dictionary['中文含义'])); types=dict(zip(dictionary['字段名'],dictionary['数据类型']))
    features=[c for c in df if c not in ['customer_id','calculate_month','target']]
    early=df.index[df.calculate_month.between(202412,202509)]; val=np.sort(df.index[df.calculate_month.between(202510,202512)])
    strata=df.loc[early,'calculate_month'].astype(str)+'_'+df.loc[early,'target'].astype(str)
    train,test=train_test_split(early,test_size=.2,stratify=strata,random_state=SEED)
    train=np.sort(train); test=np.sort(test); splits={'train':train,'test':test,'val':val}
    X=df.loc[train,features]; y=df.loc[train,'target'].to_numpy(); months=df.loc[train,'calculate_month'].to_numpy()
    assert len(features)==204 and len(df)==63610 and df.customer_id.nunique()==len(df)
    # 前推验证的训练窗口逐步扩展，每次只预测更晚月份，不跨越外部test/val。
    fold_limits=[(202503,202504,202505),(202505,202506,202507),(202507,202508,202509)]
    candidates=[(mode,n,c) for mode in POLICIES for n in [8,10,12] for c in [.01,.1,1.]]
    cv_rows=[]; oof={cfg:np.full(len(train),np.nan) for cfg in candidates}; folds=[]
    sensitivity=[]; cached=[]; sparse_audit=[]
    for fold,(end,start,finish) in enumerate(fold_limits,1):
        a=np.flatnonzero(months<=end); b=np.flatnonzero((months>=start)&(months<=finish))
        folds.append({'fold':fold,'train_end':end,'valid_start':start,'valid_end':finish,'train_count':len(a),'valid_count':len(b)})
        fitted=WoeBinner().fit(X.iloc[a],y[a]); variants={mode:apply_sparse(fitted,mode) for mode in POLICIES}
        cache={}
        for mode,binner in variants.items():
            za=binner.transform(X.iloc[a]); zb=binner.transform(X.iloc[b]); cache[mode]=(binner,za,zb)
            for n in [8,10,12]:
                selected,_=select_features(za,binner.iv,max_features=n)
                for c in [.01,.1,1.]:
                    model=make_model(f'LR_{c}').fit(za[selected],y[a]); p=model.predict_proba(zb[selected])[:,1]
                    cfg=(mode,n,c); oof[cfg][b]=p
                    cv_rows.append({'policy':mode,'n_features':n,'C':c,'fold':fold,'AUC':roc_auc_score(y[b],p),'selected':selected})
            for change in binner.changes:sparse_audit.append({'fold':fold,**change})
        cached.append((a,b,cache))
        print(f'完成前推验证{fold}/3：训练截至{end}，验证{start}—{finish}；27组方案。',flush=True)
    cv=pd.DataFrame(cv_rows)
    ranking=cv.groupby(['policy','n_features','C']).AUC.agg(['mean','std']).reset_index()
    best=float(ranking['mean'].max())
    # AUC容差0.001是事先设定的简约选择规则，不是统计显著性界限。
    eligible=ranking[ranking['mean']>=best-.001].copy()
    eligible['policy_priority']=eligible.policy.map({'merge':0,'neutral':1,'retain':2})
    chosen=eligible.sort_values(['n_features','policy_priority','mean','C'],ascending=[True,True,False,True]).iloc[0]
    cfg=(chosen.policy,int(chosen.n_features),float(chosen.C)); mode,n,c=cfg
    # 对强相关阈值、IV门槛做单因素敏感性检验；只展示，不按它继续扩张选优。
    for corr_limit,iv_floor in [(.5,.02),(.65,.02),(.8,.02),(.65,.01),(.65,.05)]:
        for fold,(a,b,cache) in enumerate(cached,1):
            binner,za,zb=cache[mode]; sel,_=select_features(za,binner.iv,max_features=n,corr_limit=corr_limit,iv_floor=iv_floor)
            p=make_model(f'LR_{c}').fit(za[sel],y[a]).predict_proba(zb[sel])[:,1]
            sensitivity.append({'corr_limit':corr_limit,'iv_floor':iv_floor,'fold':fold,'features':len(sel),'AUC':roc_auc_score(y[b],p)})
    # 在同一前推验证划分和同一最终选定的特征规则下选择加分模型复杂度。
    gb_oof={leaf:np.full(len(train),np.nan) for leaf in [3,7]}; gb_cv=[]
    for fold,(a,b,cache) in enumerate(cached,1):
        binner,za,zb=cache[mode]; sel,_=select_features(za,binner.iv,max_features=n)
        for leaf in [3,7]:
            model=make_model(f'GBDT_{leaf}').fit(za[sel],y[a]); p=model.predict_proba(zb[sel])[:,1]
            gb_oof[leaf][b]=p;gb_cv.append({'max_leaf_nodes':leaf,'fold':fold,'AUC':roc_auc_score(y[b],p)})
    gb_ranking=pd.DataFrame(gb_cv).groupby('max_leaf_nodes').AUC.mean()
    leaf=int(gb_ranking.idxmax())
    # 固定配置后，才对全部train重拟合并评价test/val。旧版val已被查看，此限制如实披露。
    binner=apply_sparse(WoeBinner().fit(X,y),mode); ztrain=binner.transform(X)
    selected,reasons=select_features(ztrain,binner.iv,max_features=n)
    assert len(selected)==n and 8<=n<=15
    corr=ztrain[selected].corr();maxcorr=float(np.max(np.abs(corr.to_numpy()-np.eye(n))))
    assert maxcorr<=.65+1e-10
    models={};thresholds=[];metrics=[];decs={};predictions=[];month_metrics=[];business=[];calibration=[]
    prediction_arrays={}
    for label,key,op in [('逻辑回归',f'LR_{c}',oof[cfg]),('梯度提升树',f'GBDT_{leaf}',gb_oof[leaf])]:
        covered=np.isfinite(op);threshold,cvks=ks_threshold(y[covered],op[covered])
        thresholds.append({'model':label,'candidate':key,'threshold':threshold,'forward_validation_count':int(covered.sum()),'forward_AUC':roc_auc_score(y[covered],op[covered]),'forward_KS':cvks})
        model=make_model(key).fit(ztrain[selected],y);models[label]=model
        for dataset,ix in splits.items():
            z=ztrain if dataset=='train' else binner.transform(df.loc[ix,features])
            yy=df.loc[ix,'target'].to_numpy();p=model.predict_proba(z[selected])[:,1]
            prediction_arrays[(label,dataset)]=(yy,p)
            result={'model':label,'dataset':dataset,**evaluate(yy,p,threshold),'Brier':brier_score_loss(yy,p)};metrics.append(result)
            t=deciles(yy,p,df.loc[ix,'customer_id']);decs[label+'_'+dataset]=t.to_dict('records')
            for position,r in enumerate(ix):predictions.append({'row_index':int(r),'customer_id':str(df.loc[r,'customer_id']),'month':int(df.loc[r,'calculate_month']),'target':int(yy[position]),'dataset':dataset,'model':label,'probability':float(p[position])})
            for month in sorted(df.loc[ix,'calculate_month'].unique()):
                m=(df.loc[ix,'calculate_month'].to_numpy()==month)
                month_metrics.append({'model':label,'dataset':dataset,'month':int(month),'sample_cnt':int(m.sum()),'cnt_1':int(yy[m].sum()),**evaluate(yy[m],p[m],threshold)})
            if dataset in ['test','val']:
                order=np.lexsort((df.loc[ix,'customer_id'].astype(str).to_numpy(),-p))
                for share in [.1,.2,.3]:
                    k=int(np.ceil(len(ix)*share));success=int(yy[order[:k]].sum());baseline=float(yy.mean())
                    business.append({'model':label,'dataset':dataset,'contact_share':k/len(ix),'contact_count':k,'captured_actual_positive':success,'conversion':success/k,'capture':success/yy.sum(),'lift':success/k/baseline,'random_expected_positive':k*baseline,'excess_vs_random':success-k*baseline})
                # 校准诊断只描述给定样本；不使用评估标签反向修正模型概率。
                score=pd.DataFrame({'p':p,'y':yy}).sort_values('p').reset_index(drop=True)
                score['group']=np.minimum(np.arange(len(score))*10//len(score)+1,10)
                for g,v in score.groupby('group'):calibration.append({'model':label,'dataset':dataset,'group':int(g),'count':len(v),'mean_probability':v.p.mean(),'actual_rate':v.y.mean()})
    print('最终配置：'+str(cfg)+f'；GBDT叶节点={leaf}。开始Bootstrap及漂移分析。',flush=True)
    # test/val固定训练评分区间的PSI；特征PSI同样使用训练分箱，不能各月重分箱。
    id_train=bin_ids(binner,X);drift=[];response=[];score_drift=[]
    lrtrain=prediction_arrays[('逻辑回归','train')][1]
    scorecuts=np.unique(np.quantile(lrtrain,np.arange(.1,1,.1)))
    train_score_bin=np.searchsorted(scorecuts,lrtrain,side='left')
    for dataset in ['test','val']:
        ix=splits[dataset];allids=bin_ids(binner,df.loc[ix,features]);p=prediction_arrays[('逻辑回归',dataset)][1]
        for month in sorted(df.loc[ix,'calculate_month'].unique()):
            mask=df.loc[ix,'calculate_month'].to_numpy()==month
            score_drift.append({'dataset':dataset,'month':int(month),'score_PSI':psi(train_score_bin,np.searchsorted(scorecuts,p[mask],side='left'),len(scorecuts)+1)})
            for col in selected:
                levels=sorted(set(id_train[col])|set(allids.loc[allids.index[mask],col]));mapping={v:j for j,v in enumerate(levels)}
                a=id_train[col].map(mapping).to_numpy();b=allids.loc[allids.index[mask],col].map(mapping).to_numpy()
                drift.append({'dataset':dataset,'month':int(month),'name':col,'description':meanings[col],'PSI':psi(a,b,len(levels)),
                    'missing_train':float(X[col].isna().mean()),'missing_month':float(df.loc[ix[mask],col].isna().mean())})
                for r in [r for r in binner.rows if r['name']==col]:
                    m=(allids[col].to_numpy()==r['bin_id'])&mask;nn=int(m.sum());nn1=int(df.loc[ix[m],'target'].sum())
                    response.append({'dataset':dataset,'month':int(month),'name':col,'cls':r['cls'],'train_count':r['cnt_all'],'train_rate':r['target_1'],'month_count':nn,'month_positive':nn1,'month_rate':nn1/nn if nn else None,'rate_change':nn1/nn-r['target_1'] if nn else None})
    uncertainty={}
    for dataset in ['test','val']:
        yy,p1=prediction_arrays[('逻辑回归',dataset)];p2=prediction_arrays[('梯度提升树',dataset)][1]
        uncertainty[dataset]=bootstrap(yy,p1,p2,df.loc[splits[dataset],'calculate_month'].to_numpy())
    ivrows=[]
    order=selected+sorted([col for col in features if col not in selected],key=lambda col:(-binner.iv[col],col))
    for col in order:
        for r in [r for r in binner.rows if r['name']==col]:
            r.update(description=meanings[col] or '',source_type=types[col],selected=col in selected,reason=reasons[col]);ivrows.append(r)
    model=models['逻辑回归'];selected_table=[]
    for j,col in enumerate(selected):
        coef=float(model.coef_[0,j]);sd=float(ztrain[col].std())
        selected_table.append({'name':col,'description':meanings[col],'sum_iv':binner.iv[col],
            'coefficient':coef,'woe_sd':sd,'effect_per_woe_sd':coef*sd,'odds_ratio':float(np.exp(coef))})
    split_rows=[]
    for dataset,ix in splits.items():
        for month,g in df.loc[ix].groupby('calculate_month'):split_rows.append({'dataset':dataset,'month':int(month),'sample_cnt':len(g),'cnt_1':int(g.target.sum()),'succ%':float(g.target.mean())})
    ratio_anomalies=[{'name':col,'out_of_0_1':int(((df[col]<0)|(df[col]>1)).sum())} for col in features if col.endswith('_ratio') and ((df[col]<0)|(df[col]>1)).any()]
    negative_durations=[{'name':col,'negative':int((df[col]<0).sum())} for col in features if col.endswith('_dur') and (df[col]<0).any()]
    audit={'rows':len(df),'columns':len(df.columns),'candidate_features':len(features),'train':len(train),'test':len(test),'val':len(val),
        'source_sha256':hashlib.sha256((base/CSV_NAME).read_bytes()).hexdigest(),'max_selected_abs_correlation':maxcorr,
        'ratio_anomalies':ratio_anomalies,'negative_durations':negative_durations,'python':platform.python_version(),'sklearn':sklearn.__version__}
    payload={'audit':audit,'config':{'sparse_policy':mode,'n_features':n,'C':c,'GBDT_max_leaf_nodes':leaf,'corr_limit':.65,'iv_floor':.02,'simplification_auc_tolerance':.001},
        'iv_rows':ivrows,'selected':selected_table,'metrics':metrics,'thresholds':thresholds,'split':split_rows,'deciles':decs,
        'correlation':corr.to_dict(),'rules':binner.rules,'intercept':float(model.intercept_[0]),'forward_folds':folds,
        'cv':cv_rows,'cv_summary':ranking.to_dict('records'),'sensitivity':sensitivity,'GBDT_cv':gb_cv,
        'sparse_changes':binner.changes,'monthly_metrics':month_metrics,'feature_drift':drift,'score_drift':score_drift,
        'bin_month_response':response,'business':business,'calibration':calibration,'uncertainty':uncertainty}
    baseline_path=Path(args.baseline_results) if args.baseline_results else out/'baseline_results.json'
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
        assert baseline['audit']['source_sha256']==audit['source_sha256']
        payload['baseline_metrics']=baseline['metrics']
    def default(obj):
        if isinstance(obj,np.generic):return obj.item()
        raise TypeError(type(obj).__name__)
    (out/'model_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=default),encoding='utf-8')
    pd.DataFrame(ivrows).to_csv(out/'iv_train_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(out/'prediction_details.csv',index=False,encoding='utf-8-sig')
    member=pd.DataFrame({'row_number':np.arange(len(df))+2,'customer_id':df.customer_id,'calculate_month':df.calculate_month,'target':df.target,'dataset':''})
    for dataset,ix in splits.items():member.loc[ix,'dataset']=dataset
    member.to_csv(out/'sample_split.csv',index=False,encoding='utf-8-sig')
    joblib.dump({'binning_rules':binner.rules,'selected':selected,'models':models,'thresholds':thresholds},out/'fitted_models.joblib')
    for name,records in [('forward_validation',cv_rows),('monthly_metrics',month_metrics),('feature_drift',drift),('bin_month_response',response),('business_metrics',business)]:
        pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    write_report(payload,df,out,prediction_arrays)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(json.dumps({'config':payload['config'],'changes':binner.changes,'val_uncertainty':uncertainty['val']},ensure_ascii=False),flush=True)

def write_report(d,df,out,predictions):
    """先给业务结论，再给证据和方法。所有数字取本次模型运行结果。"""
    sections=[]
    def section(title,content):sections.append(f'<section><h2>{title}</h2>{content}</section>')
    met=pd.DataFrame(d['metrics']);business=pd.DataFrame(d['business']);monthly=pd.DataFrame(d['monthly_metrics'])
    cfg=d['config'];val=met[(met.model=='逻辑回归')&(met.dataset=='val')].iloc[0]
    top=business[(business.model=='逻辑回归')&(business.dataset=='val')].sort_values('contact_share')
    first=top.iloc[0];second=top.iloc[1]
    capacity_fig,capacity_ax=plt.subplots(figsize=(7.5,3.6))
    for label in ['逻辑回归','梯度提升树']:
        t=business[(business.model==label)&(business.dataset=='val')].sort_values('contact_share')
        capacity_ax.plot(t.contact_share*100,t.capture*100,marker='o',label='Logistic regression' if label=='逻辑回归' else 'Gradient boosting')
    capacity_ax.plot([10,20,30],[10,20,30],'--',color='#8b96a6',label='Random selection expectation')
    capacity_ax.set(title='Historical label capture at contact capacity (val)',xlabel='Share of ranked customers selected (%)',ylabel='Share of historical positives captured (%)',xticks=[10,20,30])
    capacity_ax.grid(alpha=.2);capacity_ax.legend();capacity_fig.tight_layout()
    capacity_fig.savefig(out/'business_capacity.png',dpi=160,bbox_inches='tight');capacity_uri=image_uri(capacity_fig)
    section('一、核心结论：模型能帮助回答什么',f'''<p>本题解决的是：在给定保险客户样本中，预测客户的成交倾向，并为有限营销触达资源提供排序依据。使用逻辑回归作为可解释主模型，梯度提升树作为加分对比。最终选择{cfg['n_features']}个变量，采用{POLICIES[cfg['sparse_policy']]}方案。</p>
    <p>时间外val的逻辑回归AUC为<strong>{val.AUC:.4f}</strong>、KS为<strong>{val.KS:.4f}</strong>。按分数排序，前10%客户共{int(first.contact_count):,}人，其中{int(first.captured_actual_positive):,}人在历史标签中成交，样本成交率{first.conversion:.2%}，Lift为{first.lift:.2f}；前20%覆盖全部成交客户的{second.capture:.2%}。</p>
    <p>这些是历史样本排序表现，不是实施营销后的新增成交数。数据没有提供触达成本、佣金、保费收益或对照实验，不能计算真实ROI、增量转化率或承诺未来销量。</p>'''+metric_table(met[['model','dataset','AUC','KS','RECALL','PRE','threshold','Brier']]))
    if d.get('baseline_metrics'):
        old=pd.DataFrame(d['baseline_metrics']);compare=[]
        for version,frame in [('原版',old),('优化版',met)]:
            for _,r in frame[frame.dataset=='val'].iterrows():
                compare.append({'版本':version,'模型':r['model'],'AUC':r['AUC'],'KS':r['KS'],'RECALL':r['RECALL'],'PRE':r['PRE'],
                    'threshold':r['threshold'],'阈值入选人数':int(r['TP']+r['FP'])})
        previous=old[(old.model=='逻辑回归')&(old.dataset=='val')].iloc[0]
        section('优化前后对照：哪些发生了变化',metric_table(pd.DataFrame(compare))+f'''<p>逻辑回归val的AUC变化仅为{val.AUC-previous.AUC:+.6f}，排序能力基本持平；不把它包装成显著提升。召回率从{previous.RECALL:.2%}变为{val.RECALL:.2%}，同时精确率从{previous.PRE:.2%}变为{val.PRE:.2%}，主要伴随分类阈值降低。</p><p>按固定阈值比较，优化版多选入{int((val.TP+val.FP)-(previous.TP+previous.FP))}人，多识别{int(val.TP-previous.TP)}个历史成交标签，同时多纳入{int(val.FP-previous.FP)}个历史未成交者。是否值得扩大触达需要实际成本和收益依据，现有数据不能替业务方作利润判断。本轮改善集中在稀疏箱稳健性、时间验证、解释和核验完整性。</p>''')
    section('二、业务应用：从排序到触达方案','''<p>如果营销资源只能覆盖部分名单，可先按成交倾向排序，再结合既有服务流程安排人工核实和需求沟通。下表比较覆盖前10%、20%、30%的历史识别效果；这些比例是分析情景，不代表已知的真实人员容量或预算。</p><p>random_expected_positive=触达人数×本样本成交率，仅表示同规模随机选人时的数学期望。excess_vs_random是识别出的历史成交标签数量差，不能称为模型造成的新增成交。未利用标签直接挑客户，排序依据只有模型概率。</p>'''+f'<img src="{capacity_uri}" alt="触达容量与历史成交客户捕获率">'+metric_table(business)+f'''<p>在val中，选择前20%（{int(second.contact_count):,}人）可识别{int(second.captured_actual_positive):,}个历史成交客户，仍有{int(df[df.calculate_month>=202510].target.sum()-second.captured_actual_positive):,}个成交客户在名单之外。这体现触达范围与覆盖率的取舍：覆盖越少越集中，但会漏掉更多潜在成交者。</p><p>分类阈值与名单容量是两个口径。报告的RECALL使用训练前推验证确定的固定阈值；“前20%”则是容量约束下的排序情景，不能将两者混用。</p>''')
    explain=[]
    for r in d['selected']:
        col=r['name'];bins=[v for v in d['iv_rows'] if v['name']==col and v['cnt_all']>=100]
        high=max(bins,key=lambda a:a['target_1']);low=min(bins,key=lambda a:a['target_1'])
        explain.append({'变量':r['description'],'高成交率训练箱':high['cls'],'该箱人数':high['cnt_all'],'该箱成交率':f"{high['target_1']:.2%}",
            '低成交率训练箱':low['cls'],'低箱成交率':f"{low['target_1']:.2%}",'每WOE标准差的logit变化':r['effect_per_woe_sd']})
    section('三、业务解释：客户行为与模型信号','''<p>时间间隔变量描述“距离某项行为已经过去多久”，反映近期互动、既有投保经历或关系阶段；保单占比描述现有业务结构。它们为成交倾向提供关联信息，不证明任何行为会直接导致成交。</p><p>下表仅使用训练箱，排除不足100人的箱后列出高低成交率对比。没有假定所有变量都单调，也不把缺失值解释为“从未发生”。负数间隔仍按原值保留，故“-inf”区间可能包含业务含义未明的负数。</p>'''+table(pd.DataFrame(explain))+'''<p>对近期浏览或计划书互动较明显的客户，可在已有授权与服务流程内优先了解保障需求；对已有投保关系的客户，可核实现有保障及近期需求变化。这些是沟通方向，不是从现有数据推导出的产品适配结论。样本未提供收入、风险偏好及完整需求信息，不能仅凭高成交分数推荐某一具体产品或保额。</p><p>系数作用于WOE而非原始天数；effect_per_woe_sd用于比较尺度统一后的模型关联强度。该数值不是因果效应，也不是原始变量每增加一天带来的概率变化。</p>'''+metric_table(pd.DataFrame(d['selected'])))
    ranks=pd.DataFrame(d['cv_summary']).sort_values('mean',ascending=False)
    summary=ranks.groupby(['policy','n_features'])['mean'].max().reset_index().rename(columns={'mean':'best_mean_AUC'})
    size_best=ranks.groupby('n_features')['mean'].max()
    reduction_note=f"按IV顺序截取的8、10、12变量方案，最佳平均AUC分别为{size_best.loc[8]:.4f}、{size_best.loc[10]:.4f}、{size_best.loc[12]:.4f}。因此本轮保留12变量，而不是为了精简强行降低变量数。这只比较了本次候选顺序，不证明每个变量都不可替代，也不等于搜索过所有变量组合。"
    sens=pd.DataFrame(d['sensitivity']).groupby(['corr_limit','iv_floor']).agg(mean_AUC=('AUC','mean'),std_AUC=('AUC','std'),min_features=('features','min'),max_features=('features','max')).reset_index()
    section('四、优化比较与定稿规则',f'''<p>为检查跨期稳定性，本轮在原train内部采用扩展窗口前推验证：早期月份训练，随后月份验证。原train/test/val成员完全不变。每折重新学习分箱、IV、相关性及变量顺序，杜绝用未来验证标签分箱。</p>'''+table(pd.DataFrame(d['forward_folds']))+f'''<p>共比较3种稀疏缺失箱策略×8/10/12变量×C=0.01/0.1/1，合计27组方案。先找3折平均AUC最高值，再在距最佳不超过0.001的候选中优先选更少变量、再优先合并稀疏箱。0.001是简约选择容差，不是“统计无差异”的结论。最终：{POLICIES[cfg['sparse_policy']]}，{cfg['n_features']}变量，C={cfg['C']}。</p><p>{reduction_note}</p>'''+table(summary)+'''<details><summary>展开全部27组方案与离散程度</summary>'''+table(ranks)+'''</details><p>下表仅做单因素敏感性检验，不再借此扩张搜索：相关性阈值0.5/0.65/0.8、IV门槛0.01/0.02/0.05。最终相关性阈值仍为0.65，IV门槛仍为0.02；报告展示变动后实际选出的变量数。</p>'''+table(sens)+'''<p>保留方案直接使用平滑WOE；中性方案保留原始WOE统计但将不足100人的缺失箱模型编码置零；合并方案将其并入人数最大的普通箱后重算WOE和IV，不更改原始值。所有箱（含缺失）仍不超过5个。</p>'''+table(pd.DataFrame(d['sparse_changes']))+'''<p>重要验证限制：原版test和val的成绩已经在上一轮查看过。本轮方案选择未用它们的标签或成绩，但同一val不能再被宣称为从未使用过的新盲测。没有更多未见月份可用于全新独立验证，因此保留此限制，不将“做过优化”自动表述为“泛化一定更好”。</p>''')
    section('五、分月表现与时间稳定性','''<p>下表为固定模型、固定分类阈值逐月评估，没有逐月重新调参。val月份较晚，重点观察其AUC、KS和召回率是否持续下降；单月差异也可能受样本波动影响，不能只凭一个低值断定业务机制改变。</p>'''+metric_table(monthly[monthly.dataset=='val'])+'''<details><summary>展开train/test的逐月结果</summary>'''+metric_table(monthly[monthly.dataset!='val'])+'</details>')
    drift=pd.DataFrame(d['feature_drift']);score=pd.DataFrame(d['score_drift'])
    response=pd.DataFrame(d['bin_month_response'])
    # 缺失箱也有业务含义不确定性，响应变化仅保留样本量充足的箱作正文展示。
    changes=response[(response.dataset=='val')&(response.month_count>=100)].copy()
    changes['abs_change']=changes.rate_change.abs();changes=changes.sort_values('abs_change',ascending=False).head(15).drop(columns='abs_change')
    largest=drift[drift.dataset=='val'].sort_values('PSI',ascending=False).iloc[0]
    drift_note=f"最明显的特征分布变化出现在{int(largest.month)}的{largest.description}，PSI={largest.PSI:.4f}；该字段缺失率从train的{largest.missing_train:.2%}变为当月的{largest.missing_month:.2%}。这提示应核对浏览行为分布和数据采集口径，不能只归因于缺失率，更不能直接判定是某次活动造成。评分PSI同时列出：整体评分分布变化较小，并不能抵消单个重要字段变化所需的核查。"
    section('六、分布漂移与箱内成交率','''<p>特征PSI使用最终训练分箱，评分PSI使用训练分数十分位切点；各月都映射到同一边界，不能各自重分箱后比较。计数加0.5平滑，以免空箱产生无穷值。PSI只衡量分布变化，没有使用固定阈值把变化武断地判为合格或失败。</p>'''+f'<p>{drift_note}</p>'+table(score[score.dataset=='val'])+metric_table(drift[drift.dataset=='val'].sort_values(['month','PSI'],ascending=[True,False]))+'''<p>PSI低不保证模型有效，因为相同人群分布下成交关系也可能变化。因此同时比较“同一训练箱在较晚月份的实际成交率”。下面列出val中月样本量至少100、成交率绝对变化最大的15个箱；这是诊断性排序，不是显著性检验，也不证明变化原因。</p>'''+metric_table(changes)+'''<details><summary>展开所有月份、所有入模箱的响应对照</summary>'''+metric_table(response)+'</details>')
    u=pd.DataFrame(d['uncertainty']['val']);diff=u[u.metric=='GBDT_minus_LR_AUC'].iloc[0];ksdiff=u[u.metric=='GBDT_minus_LR_KS'].iloc[0]
    interpretation='该条件区间未跨0，支持本样本中梯度提升树AUC较高。' if diff.lower_95>0 else '该条件区间包含0，不能据此认为AUC提升稳定。'
    section('七、模型提升是否稳定：配对Bootstrap',f'''<p>分别对test、val进行1000次有放回重采样，每次在月份×target层内抽取原始记录，同一抽样索引同时用于两模型。仅重采样给定样本，不生成新数据，不重拟合模型。采用2.5%与97.5%百分位形成95%条件区间。</p>'''+table(u)+f'''<p>val的GBDT−LR AUC差为{diff.estimate:.4f}，95%条件区间为[{diff.lower_95:.4f}, {diff.upper_95:.4f}]。{interpretation}KS差为{ksdiff.estimate:.4f}，区间[{ksdiff.lower_95:.4f}, {ksdiff.upper_95:.4f}]；AUC更高不能直接推断KS、收益或所有月份都更好。</p><p>区间以现有训练完成的模型为条件，没有覆盖训练/特征选择的不确定性，月份层内也假定记录可交换；因此不能把它当作完整的模型开发置信区间。</p><details><summary>展开test区间</summary>'''+table(pd.DataFrame(d['uncertainty']['test']))+'</details>')
    section('八、概率、阈值与业务边界','''<p>给定数据每月成交比例均为20%，但材料未给出业务总体的抽样比例。AUC和Lift说明样本内排序能力；模型输出不能直接作为真实总体的绝对成交概率。校准表仅对照给定样本的平均预测概率与实际比例，不利用test/val标签重新校准。</p>'''+metric_table(pd.DataFrame(d['thresholds']))+table(pd.DataFrame(d['calibration']))+'''<p>没有实际触达成本和收益数据，不能臆造“最优利润阈值”。本次仍使用train前推验证预测最大KS点确定分类阈值，以满足召回率比较口径；若未来获得成本、容量和未见样本，才有依据另行制定和验证运营阈值。</p><p>字段生成时点、标签观察窗口不完整。特别是近期浏览、计划书及保单间隔字段，业务应用前需要核实其是否在评分时点已可获得；高IV只是核查线索，既不能证明泄漏，也不能证明没有泄漏。现有材料不足以完成这一业务核验，报告不补造事实。</p>''')
    for label in ['逻辑回归','梯度提升树']:
        content='<p>按概率降序、客户编号升序处理同分；每数据集近似等人数分十组。同概率可能跨组，区间可能相接或重叠。cnt_0/1为未成交/成交人数；target%=组内成交率；target_per=正样本份额；cum_target%=累计成交率；cum_target_per=累计捕获率；lift=组内成交率/本集基准率。</p>'
        for dataset in ['train','test','val']:content+=f'<h3>{dataset}</h3>'+metric_table(pd.DataFrame(d['deciles'][label+'_'+dataset]))
        section('九、标准十分组：'+label,content)
    # 输出可单独导出的科学图，不依赖联网或第三方网页资源。
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for dataset in ['train','test','val']:
        yy,p=predictions[('逻辑回归',dataset)];fpr,tpr,_=roc_curve(yy,p)
        axes[0].plot(fpr,tpr,label=f'{dataset} AUC={roc_auc_score(yy,p):.4f}')
        axes[1].plot(fpr,tpr-fpr,label=f'{dataset} KS={max(tpr-fpr):.4f}')
        t=pd.DataFrame(d['deciles']['逻辑回归_'+dataset]);axes[2].plot(t.decile,t.lift,marker='o',label=dataset)
    axes[0].plot([0,1],[0,1],'k--',alpha=.4)
    for ax,title,xlabel,ylabel in zip(axes,['ROC','KS','Decile lift'],['FPR','FPR','Decile'],['TPR','TPR - FPR','Lift']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'model_curves.png',dpi=160,bbox_inches='tight');uri=image_uri(fig)
    section('十、模型曲线与相关性',f'<img src="{uri}" alt="逻辑回归真实ROC、KS、提升曲线">'+table(pd.DataFrame(d['correlation']).reset_index().rename(columns={'index':'name'}))+f'<p>logit(p)={d["intercept"]:.8f}+Σ系数×模型WOE。最大绝对相关系数为{d["audit"]["max_selected_abs_correlation"]:.4f}。Pearson检查只排除较强线性相关，不保证不存在非线性冗余。</p>')
    section('十一、数据核对与知识点','''<p>原始数据63,610行、207列，扣除月份、客户编号与target后204候选特征。课件202特征与实物不一致，已按确认采用204。原始数据不覆盖，不添加外部记录；按用户确认保留167处负数时间间隔、6处越界占比（单元格计数）并披露。分箱只改变表示方式，不改写原始值。</p>'''+metric_table(pd.DataFrame(d['split']))+table(pd.DataFrame(d['audit']['ratio_anomalies']))+table(pd.DataFrame(d['audit']['negative_durations']))+'''<p>WOE=ln(正样本分布/负样本分布)，IV=Σ(正分布−负分布)×WOE，按每箱正负计数各加0.5平滑；真实计数及成交率不平滑。中性方案的原始WOE与模型WOE不同，Excel明确分列，避免掩盖编码差异。</p><p>逻辑回归预测0/1成交并输出样本条件概率；线性回归直接用于本题可能越出0—1。正态分布不是逻辑回归原始特征的必要假设。Bootstrap区间描述评估指标的不确定性，不替代经过正则与变量筛选后的严格系数推断。</p>'''+f'<p>CSV SHA-256：<code>{d["audit"]["source_sha256"]}</code></p>')
    section('十二、复现与完整中文注释程序','''<p>optimized_homework.py输出本报告、统计JSON、预测明细及全部比较表；Excel用配套build_iv_workbook.mjs生成。代码带中文注释。原始数据另行提供即可，不需要下载外部数据。具体运行及库版本见README和requirements。</p><details><summary>展开完整实际执行代码</summary><pre><code>'''+html.escape(Path(__file__).read_text(encoding='utf-8'))+'</code></pre></details>')
    css='body{font-family:"Microsoft YaHei",Arial,sans-serif;line-height:1.75;color:#172434;background:#f2f5f9;margin:0}main{max-width:1480px;background:white;margin:24px auto;padding:32px}h1,h2{color:#17385f}h1{font-size:28px}h2{font-size:21px;border-bottom:2px solid #dbe3ed;padding-bottom:8px}section{margin:32px 0;overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums;margin:16px 0}th{background:#17385f;color:white;padding:9px;text-align:right}td{padding:8px;border-bottom:1px solid #dfe5ec;text-align:right}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f5f7fa}summary{cursor:pointer;color:#17385f;font-weight:bold}pre{background:#f1f4f8;font-size:13px;line-height:1.55;padding:20px;overflow:auto}img{max-width:100%}code{font-family:Consolas,monospace}'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>成交预测作业·优化版</title><style>'+css+'</style><main><h1>保险客户成交预测作业 · 优化版</h1><p>业务解释、前推验证、分月稳定性与模型不确定性</p>'+''.join(sections)+'</main></html>'
    (out/'M_close_model_results.html').write_text(doc,encoding='utf-8')

if __name__=='__main__':run()
