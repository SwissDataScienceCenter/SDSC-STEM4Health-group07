import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; REF=np.log(Ytr).mean(0)
def lowpass(d,w,passes):
    for _ in range(passes):
        e=np.concatenate([d[:,:1],d,d[:,-1:]],1); d=w*e[:,:-2]+(1-2*w)*e[:,1:-1]+w*e[:,2:]
    return d
def gp_s(ell,snr,n=13):
    k=np.arange(n); K=snr*np.exp(-0.5*((k[:,None]-k[None,:])/ell)**2)+snr*0.5; return K@np.linalg.inv(K+np.eye(n))
def sm(P,cfg,ref=REF):
    """cfg: {target_index: ('lp',w,passes) | ('gp',ell,snr) | ('const',shrink)}"""
    L=np.log(P); out=L.copy()
    for t,c in cfg.items():
        d=L[:,:,t]-ref[:,t]
        if c[0]=='lp': d=lowpass(d,c[1],c[2])
        elif c[0]=='gp': d=d@gp_s(c[1],c[2]).T
        elif c[0]=='mean': d=(1-c[1])*d+c[1]*d.mean(1,keepdims=True)
        out[:,:,t]=ref[:,t]+d
    return np.exp(out)
if __name__=='__main__':
    for n in sys.argv[1:]:
        P=np.load(f'{LAB}/oof/{n}.npz')['oof']; print(n,'raw',np.round(mape(P,Ytr).mean(0),2))
        for nm,c in [('lp .33x1',('lp',1/3,1)),('lp .33x3',('lp',1/3,3)),('lp .33x6',('lp',1/3,6)),('lp .33x12',('lp',1/3,12)),('gp 2,10',('gp',2,10)),('gp 3,10',('gp',3,10)),('gp 5,10',('gp',5,10)),('gp 3,3',('gp',3,3)),('mean .5',('mean',.5)),('mean .8',('mean',.8)),('mean 1',('mean',1.0))]:
            E=mape(sm(P,{0:c,1:c,2:c,3:c}),Ytr).mean(0); print('   %-10s'%nm,np.round(E,2))
