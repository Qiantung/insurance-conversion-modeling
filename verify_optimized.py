# -*- coding: utf-8 -*-
"""只读核验：原始数据、样本划分、分箱计数、Excel缓存值和模型指标。"""
from pathlib import Path
import os, json, hashlib, html, argparse
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import openpyxl
import joblib
from optimized_homework import WoeBinner, CSV_NAME

parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',default='E:/我的文档/课程文档/保险运营程序设计与实践')
parser.add_argument('--output-dir',default='outputs/优化版')
args=parser.parse_args()
out=Path(args.output_dir)
base=Path(args.data_dir)
d=json.loads((out/'model_results.json').read_text(encoding='utf-8'))
df=pd.read_csv(base/CSV_NAME,dtype={'customer_id':'string'})
split=pd.read_csv(out/'sample_split.csv',dtype={'customer_id':'string'})
assert split.customer_id.equals(df.customer_id)
assert split.target.equals(df.target)
assert set(split.dataset)=={'train','test','val'}
assert hashlib.sha256((base/CSV_NAME).read_bytes()).hexdigest()==d['audit']['source_sha256']
saved=joblib.load(out/'fitted_models.joblib')
binner=WoeBinner(); binner.rules=saved['binning_rules']
z=binner.transform(df[list(binner.rules)])
assert np.isfinite(z.to_numpy()).all()

# 独立按原始值与保存的边界重算每个训练箱计数，不依赖Excel公式结果。
trainmask=(split.dataset=='train').to_numpy()
for col,rule in binner.rules.items():
    s=df[col]
    if rule['category']:
        s=s.astype('string'); s=s.where(s.isin(rule['keep']),'__RARE__').where(s.notna())
        v=s.map(rule['order']).astype(float)
    else: v=pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)
    bins=np.searchsorted(rule['cuts'],v.fillna(0),side='left'); bins[v.isna()]=rule.get('missing_bin',-1)
    group=[r for r in d['iv_rows'] if r['name']==col]
    assert len(group)<=5 and sum(r['cnt_all'] for r in group)==int(trainmask.sum())
    assert abs(sum(r['iv'] for r in group)-group[0]['sum_iv'])<1e-12
    for r in group:
        m=(bins==r['bin_id'])&trainmask
        assert int(m.sum())==r['cnt_all']
        assert int(df.loc[m,'target'].sum())==r['cnt_1']

# 独立用平均秩求AUC，用同分概率聚合后的累计分布求KS。
for label,model in saved['models'].items():
    threshold=next(t['threshold'] for t in saved['thresholds'] if t['model']==label)
    for name in ['train','test','val']:
        mask=(split.dataset==name).to_numpy()
        p=model.predict_proba(z.loc[mask,saved['selected']])[:,1]
        y=df.loc[mask,'target'].to_numpy(); pos=y.sum(); neg=len(y)-pos
        auc=(pd.Series(p).rank(method='average').to_numpy()[y==1].sum()-pos*(pos+1)/2)/(pos*neg)
        g=pd.DataFrame({'p':p,'y':y}).groupby('p').y.agg(['sum','size']).sort_index(ascending=False)
        ks=max((g['sum'].cumsum()/pos-(g['size']-g['sum']).cumsum()/neg).max(),0)
        recall=((p>=threshold)&(y==1)).sum()/pos
        m=next(v for v in d['metrics'] if v['model']==label and v['dataset']==name)
        assert abs(auc-m['AUC'])<1e-12 and abs(ks-m['KS'])<1e-12 and abs(recall-m['RECALL'])<1e-12
        assert sum(m[k] for k in ['TN','FP','FN','TP'])==len(y)
        dec=d['deciles'][label+'_'+name]
        assert len(dec)==10 and sum(r['cnt_1'] for r in dec)==pos
        assert sum(r['cnt_0'] for r in dec)==neg
        assert abs(dec[-1]['cum_target_per']-1)<1e-12
        for r in dec:
            assert r['cnt_0']+r['cnt_1']==r['sample_cnt']
            assert abs(r['lift']-r['cnt_1']/r['sample_cnt']/(pos/len(y)))<1e-12

# Excel只读检查：真实导出内容、公式及缓存值均与独立统计一致。
book=openpyxl.load_workbook(out/'M_close_iv_train.xlsx',data_only=True)
ws=book['IV_train']; assert ws.max_row==len(d['iv_rows'])+1
for i,r in enumerate(d['iv_rows'],2):
    assert ws.cell(i,1).value==r['name']
    for col,key in [(2,'sum_iv'),(5,'cnt_0'),(6,'cnt_1'),(7,'cnt_all'),(8,'target_1'),(9,'woe'),(10,'iv'),(14,'dist_0'),(15,'dist_1'),(16,'model_woe')]:
        assert abs(ws.cell(i,col).value-r[key])<1e-10,(i,col,key)
form=openpyxl.load_workbook(out/'M_close_iv_train.xlsx',data_only=False)
assert form['IV_train']['I2'].data_type=='f' and form['IV_train']['B2'].data_type=='f'
assert len(form['IV_train'].tables)==1

# HTML完整收录本次执行的源码，且图片内嵌，无外部数据链接。
doc=(out/'M_close_model_results.html').read_text(encoding='utf-8')
source=Path(__file__).with_name('optimized_homework.py').read_text(encoding='utf-8')
assert html.escape(source) in doc
assert 'data:image/png;base64,' in doc
assert doc.count('<h3>train</h3>')==2 and doc.count('<h3>test</h3>')==2 and doc.count('<h3>val</h3>')==2
# 逐记录核对预测明细；按月份核对指标；按实际训练箱独立重算PSI。
details=pd.read_csv(out/'prediction_details.csv')
for (label,name),g in details.groupby(['model','dataset']):
    g=g.sort_values('row_index');ix=g.row_index.to_numpy()
    pp=saved['models'][label].predict_proba(z.loc[ix,saved['selected']])[:,1]
    assert np.allclose(pp,g.probability,atol=1e-12,rtol=0)
    assert np.array_equal(df.loc[ix,'target'],g.target)
    for month,t in g.groupby('month'):
        yy=t.target.to_numpy();p=t.probability.to_numpy();pos=yy.sum();neg=len(yy)-pos
        auc=(pd.Series(p).rank().to_numpy()[yy==1].sum()-pos*(pos+1)/2)/(pos*neg)
        reported=next(r for r in d['monthly_metrics'] if r['model']==label and r['dataset']==name and r['month']==month)
        assert abs(auc-reported['AUC'])<1e-12

def independent_ids(col,rule):
    s=df[col]
    if rule['category']:
        s=s.astype('string');s=s.where(s.isin(rule['keep']),'__RARE__').where(s.notna());v=s.map(rule['order']).astype(float)
    else:v=pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)
    ids=np.searchsorted(rule['cuts'],v.fillna(0),side='left');ids[v.isna()]=rule.get('missing_bin',-1)
    return ids
for r in d['feature_drift']:
    ids=independent_ids(r['name'],saved['binning_rules'][r['name']])
    mask=(split.dataset==r['dataset']).to_numpy()&(df.calculate_month==r['month']).to_numpy()
    levels=sorted(set(ids[trainmask])|set(ids[mask]))
    a=np.array([(ids[trainmask]==i).sum()+.5 for i in levels]);a/=a.sum()
    b=np.array([(ids[mask]==i).sum()+.5 for i in levels]);b/=b.sum()
    psi=float(np.sum((b-a)*np.log(b/a)))
    assert abs(psi-r['PSI'])<1e-12
for r in d['bin_month_response']:
    ids=independent_ids(r['name'],saved['binning_rules'][r['name']])
    binid=next(t['bin_id'] for t in d['iv_rows'] if t['name']==r['name'] and t['cls']==r['cls'])
    mask=(split.dataset==r['dataset']).to_numpy()&(df.calculate_month==r['month']).to_numpy()&(ids==binid)
    assert int(mask.sum())==r['month_count'] and int(df.loc[mask,'target'].sum())==r['month_positive']
for fold in d['forward_folds']:assert fold['train_end']<fold['valid_start']<=fold['valid_end']<202510
assert len(d['cv'])==81 and len(d['cv_summary'])==27
for r in d['uncertainty']['val']:
    assert r['repetitions']==1000 and r['lower_95']<=r['upper_95']
assert all(r['missing_count']<100 for r in d['sparse_changes'])
# 简约方案选择只依赖训练CV汇总，独立复核最终配置。
best=max(r['mean'] for r in d['cv_summary'])
eligible=[r for r in d['cv_summary'] if r['mean']>=best-.001]
chosen=sorted(eligible,key=lambda r:(r['n_features'],{'merge':0,'neutral':1,'retain':2}[r['policy']],-r['mean'],r['C']))[0]
assert chosen['policy']==d['config']['sparse_policy'] and chosen['n_features']==d['config']['n_features'] and chosen['C']==d['config']['C']

result={'status':'passed','variables':len(binner.rules),'bins':len(d['iv_rows']),
        'selected_variables':len(saved['selected']),'models':len(saved['models']),
        'metrics_recomputed':6,'decile_tables_checked':6,'xlsx_rows_checked':ws.max_row-1,
        'source_code_embedded':True,'forward_cv_candidates':27,'monthly_metrics_checked':len(d['monthly_metrics']),'feature_PSI_checked':len(d['feature_drift']),'bin_month_response_checked':len(d['bin_month_response']),'source_sha256':d['audit']['source_sha256']}
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
