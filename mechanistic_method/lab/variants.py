import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from ode import *
V={}
def reg(cls): V[cls.name]=cls; return cls
@reg
class M01(Model): name='m01_base'; desc='ODE: growth slowed by IVCD, overflow byproduct, re-consumption; global parameters only'
@reg
class M02(Model): name='m02_cov7'; desc='m01 + 4 kinetic parameters scaled by 7 process signals'; zset='z7'; parent='m01_base'
@reg
class M03(Model):
    name='m03_feedD'; desc='m02 + feed D promotes byproduct re-consumption (decaying D state)'; zset='z7'; parent='m02_cov7'; states=('Xv','Xd','S','P','I','t','D')
    def spec(self): s=dict(BASE_SPEC); s.update(aD=(1.,1e-3,1e3,'log'),kD=(.5,.01,10,'log')); return s
    def init(self,p,Y0,V0,m0): s=super().init(p,Y0,V0,m0); s['D']=np.zeros(len(Y0)); return s
    def cons(self,p,s): return super().cons(p,s)*(1+p['aD']*s['D']*1e3)
    def extra(self,p,s,d,mu,kd,ov): d['D']=-p['kD']*s['D']
    def feed(self,p,s,fb,fs,fc,fd,Vn): super().feed(p,s,fb,fs,fc,fd,Vn); s['D']=s['D']+fd/Vn
@reg
class M04(Model):
    name='m04_Pinhib'; desc='m02 + byproduct inhibits growth'; zset='z7'; parent='m02_cov7'
    def spec(self): s=dict(BASE_SPEC); s.update(KP=(20.,.3,1e3,'log')); return s
    def mu(self,p,s): return super().mu(p,s)/(1+s['P']/p['KP'])
@reg
class M05(Model):
    name='m05_logistic'; desc='m02 with logistic capacity (1-Xv/Xmax) replacing IVCD inhibition'; zset='z7'; parent='m02_cov7'
    def spec(self): s=dict(BASE_SPEC); s.update(Xmax=(40.,25,200,'log')); s.pop('KI'); s.pop('nI'); return s
    def mu(self,p,s): return p['mumax']*s['S']/(p['Ks']+s['S'])*np.maximum(1-s['Xv']/p['Xmax'],0)
@reg
class M06(Model):
    name='m06_nutrient'; desc='m02 + unmeasured nutrient supplied by feed B limits growth'; zset='z7'; parent='m02_cov7'; states=('Xv','Xd','S','P','I','t','N')
    def spec(self): s=dict(BASE_SPEC); s.update(N0=(1.,.05,50,'log'),cN=(10.,.1,1e3,'log'),qN=(.02,1e-4,1,'log'),KN=(.2,.005,20,'log')); return s
    def init(self,p,Y0,V0,m0): s=super().init(p,Y0,V0,m0); s['N']=np.full(len(Y0),1.)*p['N0']; return s
    def mu(self,p,s): return super().mu(p,s)*np.maximum(s['N'],0)/(p['KN']+np.maximum(s['N'],0))
    def extra(self,p,s,d,mu,kd,ov): d['N']=-p['qN']*mu*s['Xv']
    def feed(self,p,s,fb,fs,fc,fd,Vn): super().feed(p,s,fb,fs,fc,fd,Vn); s['N']=s['N']+p['cN']*fb/Vn
KN_=np.array([0,2,4,6,8,10,13.])
def tv(p,pre,t): return np.exp(np.interp(t,KN_,np.array([0.]+[p[f'{pre}{i}'] for i in range(1,len(KN_))])))
@reg
class M07(Model):
    name='m07_tvmu'; desc='m02 + phase-dependent growth: piecewise-linear time multiplier on mu (knots every 2 d)'; zset='z7'; parent='m02_cov7'
    def spec(self):
        s=dict(BASE_SPEC)
        for i in range(1,len(KN_)): s[f'am{i}']=(0.,-1.5,1.5,'lin')
        return s
    def mu(self,p,s): return super().mu(p,s)*tv(p,'am',s['t'][0] if np.ndim(s['t']) else s['t'])
@reg
class M08(Model):
    name='m08_tvall'; desc='m02 + phase-dependent growth, substrate uptake and byproduct re-consumption (piecewise-linear in time)'; zset='z7'; parent='m07_tvmu'
    def spec(self):
        s=dict(BASE_SPEC)
        for pre in ['am','aq','ac']:
            for i in range(1,len(KN_)): s[f'{pre}{i}']=(0.,-2,2,'lin')
        return s
    def mu(self,p,s): return super().mu(p,s)*tv(p,'am',s['t'][0])
    def qS(self,p,s,mu,ov): return p['mS']*tv(p,'aq',s['t'][0])*s['S']/(p['Km']+s['S'])+ov
    def cons(self,p,s): return super().cons(p,s)*tv(p,'ac',s['t'][0])
@reg
class M09(Model):
    name='m09_death'; desc='m02 + death accelerated by low substrate and by byproduct'; zset='z7'; parent='m02_cov7'
    def spec(self): s=dict(BASE_SPEC); s.update(kdS=(.01,1e-5,1,'log'),KdS=(.5,.05,5,'log'),kdP=(.001,1e-6,.5,'log')); return s
    def kd(self,p,s): return super().kd(p,s)+p['kdS']*p['KdS']/(p['KdS']+s['S'])+p['kdP']*s['P']
@reg
class M10(Model):
    name='m10_ovHill'; desc='m02 with overflow flux driven by a Hill function of substrate as well as culture age'; zset='z7'; parent='m02_cov7'
    def spec(self): s=dict(BASE_SPEC); s.update(Kov=(1.,.05,30,'log'),nq=(2.,.5,8,'log')); return s
    def ov(self,p,s): return p['qov']*s['S']**p['nq']/(p['Kov']**p['nq']+s['S']**p['nq'])/(1+(s['I']/p['KIo'])**p['no'])
@reg
class M11(Model):
    name='m11_consHill'; desc='m02 with switch-like byproduct re-consumption (Hill in substrate, Monod in byproduct)'; zset='z7'; parent='m02_cov7'
    def spec(self): s=dict(BASE_SPEC); s.update(ng=(2.,.5,8,'log'),KPc=(5.,.05,100,'log')); return s
    def cons(self,p,s): return p['kc']*s['P']/(1+s['P']/p['KPc'])/(1+(s['S']/p['Kg'])**p['ng'])
@reg
class M12(Model): name='m12_cov10'; desc='m02 with 10 process signals (adds early/late feed-B dosing, early feed-C)'; zset='z10'; parent='m02_cov7'
@reg
class M13(Model): name='m13_cov4'; desc='m02 with 4 signals only (no substrate-rule flags as covariates)'; zset='z4'; parent='m02_cov7'
@reg
class M14(Model): name='m14_re6'; desc='m02 with covariates also on overflow flux and growth-inhibition constant'; zset='z7'; parent='m02_cov7'; RE=('mumax','mS','Yov','kc','qov','KI')
@reg
class M15(Model): name='m15_l2'; desc='m02 fitted with plain least squares on log errors'; zset='z7'; parent='m02_cov7'; loss='linear'
@reg
class M16(Model):
    name='m16_growthcoupled'; desc='m02 with growth-coupled uptake qS = mu/Yxs + maintenance + overflow'; zset='z7'; parent='m02_cov7'
    def spec(self): s=dict(BASE_SPEC); s.update(iYxs=(.02,1e-4,1,'log')); return s
    def qS(self,p,s,mu,ov): return super().qS(p,s,mu,ov)+p['iYxs']*mu
@reg
class M17(Model): name='m17_lam1'; desc='m02 with weak penalty on the covariate table'; zset='z7'; parent='m02_cov7'; lam=1.0
@reg
class M18(Model): name='m18_lam100'; desc='m02 with strong penalty on the covariate table'; zset='z7'; parent='m02_cov7'; lam=100.0
@reg
class M19(Model):
    name='m19_combo'; desc='ODE with feed-D-promoted and switch-like byproduct re-consumption, 6 kinetic parameters scaled by 7 process signals'; zset='z7'; parent='m14_re6'; RE=('mumax','mS','Yov','kc','qov','KI'); states=('Xv','Xd','S','P','I','t','D')
    def spec(self): s=dict(BASE_SPEC); s.update(aD=(1.,1e-3,1e3,'log'),kD=(.5,.01,10,'log'),ng=(2.,.5,8,'log'),KPc=(5.,.05,100,'log')); return s
    def init(self,p,Y0,V0,m0): s=super().init(p,Y0,V0,m0); s['D']=np.zeros(len(Y0)); return s
    def cons(self,p,s): return p['kc']*s['P']/(1+s['P']/p['KPc'])/(1+(s['S']/p['Kg'])**p['ng'])*(1+p['aD']*s['D']*1e3)
    def extra(self,p,s,d,mu,kd,ov): d['D']=-p['kD']*s['D']
    def feed(self,p,s,fb,fs,fc,fd,Vn): super().feed(p,s,fb,fs,fc,fd,Vn); s['D']=s['D']+fd/Vn
@reg
class M20(Model): name='m20_re6_cov10'; desc='m14 with 10 process signals'; zset='z10'; parent='m14_re6'; RE=('mumax','mS','Yov','kc','qov','KI')
@reg
class M21(Model): name='m21_re8'; desc='m02 with covariates on 8 parameters (adds death rate and substrate feed concentration)'; zset='z7'; parent='m14_re6'; RE=('mumax','mS','Yov','kc','qov','KI','kd1','cb')
@reg
class M22(Model):
    name='m22_bypfix'; desc='m02 + substrate-dependent byproduct consumption (S/(K+S) instead of Kg/(Kg+S)) + substrate-thresholded overflow'; zset='z7'; parent='m02_cov7'
    def spec(self):
        s=dict(BASE_SPEC); s.update(K_cons=(0.5,.01,20,'log'),So=(.1,.001,2.,'log'),Kov=(.2,.01,5,'log')); return s
    def ov(self,p,s):
        Seff=np.maximum(s['S']-p['So'],0.0)
        return p['qov']*Seff/(p['Kov']+Seff)/(1+(np.maximum(s['I'],1e-9)/p['KIo'])**p['no'])
    def cons(self,p,s):
        return p['kc']*s['P']*s['S']/(p['K_cons']+s['S'])
if __name__=='__main__':
    for nm in sys.argv[1:]: run(V[nm]())
