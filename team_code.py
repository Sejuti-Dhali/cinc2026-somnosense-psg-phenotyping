import os,numpy as np,pandas as pd,joblib,warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from helper_code import (find_patients,load_demographics,load_edf_to_nparrays,
    load_label,load_age,load_sex,load_bmi,get_standardized_race,
    get_standardized_ethnicity,update_demographics_table,
    HEADERS,DEMOGRAPHICS_FILE,ALGORITHMIC_ANNOTATIONS_SUBFOLDER)

MODEL_FILE='xgb_model.joblib';IMPUTER_FILE='imputer.joblib'
SCALER_FILE='scaler.joblib';FEAT_FILE='feature_names.joblib'

def _safe_div(a,b,d=0.0):
    try:return d if(b==0 or np.isnan(b))else a/b
    except:return d

def _count_events(sig,code):
    c=0;inside=False
    for v in sig:
        if v==code and not inside:c+=1;inside=True
        elif v!=code:inside=False
    return c

def _find_annot(data_folder,site_id,bids_folder,session_id):
    d=os.path.join(data_folder,ALGORITHMIC_ANNOTATIONS_SUBFOLDER,str(site_id))
    p=os.path.join(d,f"{bids_folder}-ses{session_id}.edf")
    if os.path.isfile(p):return p
    if os.path.isdir(d):
        for f in os.listdir(d):
            if bids_folder in f and f.endswith('.edf'):return os.path.join(d,f)
    return ""

def _caisr_features(path):
    feats={}
    if not path or not os.path.isfile(path):return feats
    ch,fs=load_edf_to_nparrays(path)
    if not ch:return feats
    def get(name):
        for k in ch:
            if name.lower() in k.lower():return ch[k],fs.get(k,1.0)
        return None,None
    stage_sig,_=get('stage_caisr');tst_min=0.0
    if stage_sig is not None and len(stage_sig)>0:
        s=stage_sig.astype(int);v=s[s!=9]
        total=len(v);sleep=np.sum(v!=5);tst_min=_safe_div(sleep*30,60.0)
        feats['sleep_efficiency']=_safe_div(sleep,total)
        feats['pct_n3']=_safe_div(np.sum(v==1),sleep)
        feats['pct_n2']=_safe_div(np.sum(v==2),sleep)
        feats['pct_n1']=_safe_div(np.sum(v==3),sleep)
        feats['pct_rem']=_safe_div(np.sum(v==4),sleep)
        feats['pct_wake']=_safe_div(np.sum(v==5),total)
        feats['total_sleep_min']=tst_min
        onset=next((i for i,x in enumerate(v) if x!=5),None)
        if onset is not None:
            rem=next((i for i,x in enumerate(v[onset:]) if x==4),None)
            feats['rem_latency']=float(rem) if rem is not None else float(total)
            feats['waso']=float(np.sum(v[onset:]==5))
        else:
            feats['rem_latency']=float(total);feats['waso']=0.0
        tr=np.sum(np.diff(v)!=0)
        feats['stage_transitions']=float(tr);feats['fragmentation']=_safe_div(tr,sleep)
        for pc in ['caisr_prob_n3','caisr_prob_n2','caisr_prob_n1','caisr_prob_r','caisr_prob_w']:
            sg,_=get(pc)
            feats[f'mean_{pc}']=float(np.nanmean(sg)) if sg is not None else np.nan
            feats[f'std_{pc}']=float(np.nanstd(sg)) if sg is not None else np.nan
    ar_sig,_=get('arousal_caisr')
    if ar_sig is not None and len(ar_sig)>0:
        ac=_count_events(ar_sig.astype(int),1);hr=_safe_div(tst_min,60.0,1.0)
        feats['arousal_index']=_safe_div(ac,hr);feats['pct_arousal']=_safe_div(np.sum(ar_sig==1),len(ar_sig))
    resp_sig,_=get('resp_caisr')
    if resp_sig is not None and len(resp_sig)>0:
        r=resp_sig.astype(int);hr=_safe_div(tst_min,60.0,1.0)
        oa=_count_events(r,1);ca=_count_events(r,2);ma=_count_events(r,3)
        hy=_count_events(r,4);rera=_count_events(r,5);tot=oa+ca+ma+hy+rera
        feats['ahi']=_safe_div(oa+ca+ma+hy,hr);feats['cai']=_safe_div(ca,hr)
        feats['oai']=_safe_div(oa,hr);feats['rdi']=_safe_div(tot,hr)
        feats['central_fraction']=_safe_div(ca,tot);feats['rera_index']=_safe_div(rera,hr)
    limb_sig,_=get('limb_caisr')
    if limb_sig is not None and len(limb_sig)>0:
        l=limb_sig.astype(int);hr=_safe_div(tst_min,60.0,1.0)
        feats['plmi']=_safe_div(_count_events(l,2),hr);feats['lmi']=_safe_div(_count_events(l,1),hr)
    return feats

def _demo_features(demo):
    feats={}
    age=load_age(demo);feats['age']=age if age>0 else np.nan
    sex=load_sex(demo);feats['sex_female']=1.0 if sex=='Female' else(0.0 if sex=='Male' else np.nan)
    bmi=load_bmi(demo);feats['bmi']=bmi if bmi>0 else np.nan
    feats['obese']=1.0 if bmi>=30 else(0.0 if bmi>0 else np.nan)
    feats['overweight']=1.0 if 25<=bmi<30 else(0.0 if bmi>0 else np.nan)
    race=get_standardized_race(demo)
    for r in ['White','Black','Asian','Others']:feats[f'race_{r.lower()}']=1.0 if race==r else 0.0
    feats['hispanic']=1.0 if get_standardized_ethnicity(demo)=='Hispanic' else 0.0
    return feats

def _build_fv(demo,annot_path):
    f={};f.update(_demo_features(demo));f.update(_caisr_features(annot_path));return f

def _to_matrix(dicts,names):
    return np.array([[d.get(k,np.nan) for k in names] for d in dicts],dtype=np.float32)

def train_model(data_folder,model_folder,verbose=1):
    if verbose:print(f"[train] Loading from {data_folder}")
    os.makedirs(model_folder,exist_ok=True)
    demo_file=os.path.join(data_folder,DEMOGRAPHICS_FILE)
    patients=find_patients(demo_file)
    if verbose:print(f"[train] {len(patients)} patients")
    feat_dicts=[];labels=[]
    for i,p in enumerate(patients):
        bf=p[HEADERS['bids_folder']];si=p[HEADERS['site_id']];sess=p[HEADERS['session_id']]
        demo=load_demographics(demo_file,bf,sess);label=load_label(demo)
        if verbose and i%50==0:print(f"  [{i+1}/{len(patients)}] {bf} label={label}")
        feat_dicts.append(_build_fv(demo,_find_annot(data_folder,si,bf,sess)));labels.append(label)
    names=sorted(set(k for d in feat_dicts for k in d))
    X=_to_matrix(feat_dicts,names);y=np.array(labels,dtype=np.int32)
    if verbose:print(f"[train] X={X.shape} pos={int(np.sum(y))}/{len(y)}")
    imp=SimpleImputer(strategy='median');sc=StandardScaler()
    Xi=imp.fit_transform(X);Xs=sc.fit_transform(Xi)
    neg,pos=int(np.sum(y==0)),int(np.sum(y==1))
    model=xgb.XGBClassifier(n_estimators=500,max_depth=5,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.7,min_child_weight=5,gamma=1.0,
        reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=neg/max(pos,1),
        random_state=42,eval_metric='auc',n_jobs=-1)
    model.fit(Xs,y)
    joblib.dump(model,os.path.join(model_folder,MODEL_FILE))
    joblib.dump(imp,os.path.join(model_folder,IMPUTER_FILE))
    joblib.dump(sc,os.path.join(model_folder,SCALER_FILE))
    joblib.dump(names,os.path.join(model_folder,FEAT_FILE))
    if verbose:print(f"[train] Done.")

def load_model(model_folder,verbose=1):
    model=joblib.load(os.path.join(model_folder,MODEL_FILE))
    imp=joblib.load(os.path.join(model_folder,IMPUTER_FILE))
    sc=joblib.load(os.path.join(model_folder,SCALER_FILE))
    names=joblib.load(os.path.join(model_folder,FEAT_FILE))
    if verbose:print(f"[load] Model loaded.")
    return model,imp,sc,names

def run_model(data_folder,model_folder,output_folder,verbose=1):
    if verbose:print(f"[run] Inference on {data_folder}")
    model,imp,sc,names=load_model(model_folder,verbose)
    demo_file=os.path.join(data_folder,DEMOGRAPHICS_FILE)
    patients=find_patients(demo_file);os.makedirs(output_folder,exist_ok=True)
    results={}
    for i,p in enumerate(patients):
        bf=p[HEADERS['bids_folder']];si=p[HEADERS['site_id']];sess=p[HEADERS['session_id']]
        demo=load_demographics(demo_file,bf,sess)
        if verbose and i%50==0:print(f"  [{i+1}/{len(patients)}] {bf}")
        fv=_build_fv(demo,_find_annot(data_folder,si,bf,sess))
        x=np.array([[fv.get(k,np.nan) for k in names]],dtype=np.float32)
        prob=float(model.predict_proba(sc.transform(imp.transform(x)))[0,1])
        results[bf]=(int(prob>=0.5),round(prob,6))
    update_demographics_table(demo_file,output_folder,results)
    if verbose:print(f"[run] Done. {len(results)} predictions saved.")
