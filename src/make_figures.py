import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25})
BLUE='#1F4E79'; ORG='#E07B39'; GRN='#2E8B57'; RED='#C0392B'; GRY='#7F8C8D'; PUR='#6C5B9E'
R=json.load(open('data/results_v2.json')); P=pd.read_pickle('data/final_preds.pkl'); cur=json.load(open('data/curve.json'))['train_curve_mean']
def save(f,n): f.savefig(f'figures/{n}.png',dpi=220,bbox_inches='tight',facecolor='white'); plt.close(f)
order=['Naive (روز قبل)','Seasonal Naive (هفته قبل)','میانگین ۷ روزه','Gradient Boosting','MLP','Extra Trees','LSTM (NumPy)','Ridge','Ensemble (GB+LSTM+Ridge)']
order=sorted(order,key=lambda m:-R['avg_mae'][m])
# h1 walk-forward bars
fig,ax=plt.subplots(figsize=(7.4,3.6)); folds=['2008','2009','2010']; cols=[GRY,'#9CC3EA',BLUE]
yy=np.arange(len(order)); h=.26
for k,(f,c) in enumerate(zip(folds,cols)):
    ax.barh(yy+(k-1)*h,[R['folds'][f][m]['MAE'] for m in order],h,color=c,label=f'آزمون {f}')
for i,m in enumerate(order): ax.text(7.5,i,f"{R['avg_mae'][m]:.2f}",va='center',fontsize=8,fontweight='bold',color=RED if 'Ensemble' in m else '#222')
ax.set_yticks(yy); ax.set_yticklabels(order,fontsize=8); ax.set_xlabel('MAE (kWh در روز) — کمتر بهتر است'); ax.set_xlim(0,8.4); ax.text(7.5,len(order)-.35,'میانگین',ha='center',fontsize=8,color=GRY)
ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3); save(fig,'h1_walkforward')
# h2 training curve + grid
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.4,3.0),gridspec_kw={'width_ratios':[1.3,1]})
cu=np.array(cur); a1.plot(cu[:,0],color=BLUE,lw=2,label='آموزش'); a1.plot(cu[:,1],color=ORG,lw=2,label='اعتبارسنجی'); a1.set_xlabel('دوره آموزشی (epoch)'); a1.set_ylabel('MSE (مقیاس استاندارد)'); a1.legend(frameon=False)
g=R['lstm_grid']; lab=[f"H={x['hidden']}\nlr={x['lr']}" for x in g]; v=[x['val_MAE'] for x in g]
a2.bar(range(4),v,color=[RED if x==min(v) else BLUE for x in v]); a2.set_xticks(range(4)); a2.set_xticklabels(lab,fontsize=7.5); a2.set_ylim(4,5.2); a2.set_ylabel('MAE اعتبارسنجی')
for i,x in enumerate(v): a2.text(i,x+.03,f'{x:.2f}',ha='center',fontsize=8)
save(fig,'h2_lstm_training')
# h3 forecast + interval + anomalies
fig,ax=plt.subplots(figsize=(7.4,3.4)); ens=P['Ensemble (GB+LSTM+Ridge)']
ax.fill_between(P.index,P['lo'],P['hi'],color=ORG,alpha=.15,label='بازه اطمینان ۹۰٪ (Conformal)')
ax.plot(P.index,P['y'],color=GRY,lw=1.1,label='مقدار واقعی'); ax.plot(P.index,ens,color=ORG,lw=1.6,label='پیش‌بینی Ensemble')
an=P[P['z'].abs()>2.5]; ax.scatter(an.index,an['y'],color=RED,s=22,zorder=5,label='ناهنجاری (|z|>2.5)')
ax.set_ylabel('kWh/day'); ax.legend(frameon=False,fontsize=7.5,ncol=2,loc='upper right'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')); ax.set_ylim(-5,62)
save(fig,'h3_interval_anomaly')
# h4 architecture diagram
fig,ax=plt.subplots(figsize=(7.4,2.6)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,30)
def box(x,y,w,h,t,c,tc='white'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.5',fc=c,ec='none')); ax.text(x+w/2,y+h/2,t,ha='center',va='center',color=tc,fontsize=8,fontweight='bold',linespacing=1.5)
def arr(x1,x2,y=15): ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle='-|>',mutation_scale=12,color='#555',lw=1.4))
box(1,9,17,12,'ورودی:\n۲۸ روز اخیر\n(مصرف + تقویم)',GRY); arr(19,23)
box(24,9,19,12,'لایه LSTM\n۱۶ واحد\n(۴ دروازه)',BLUE); arr(44,48)
box(49,9,17,12,'بردار حالت\nh(T)',PUR); 
box(49,0,17,6,'تقویم روز هدف',ORG); ax.add_patch(FancyArrowPatch((57.5,6.4),(57.5,8.6),arrowstyle='-|>',mutation_scale=10,color='#555',lw=1.2)); arr(67,71)
box(72,9,13,12,'لایه\nخطی',GRN); arr(86,89); box(90,9,9,12,'ŷ\nkWh',RED)
save(fig,'h4_architecture')
print('ok')
