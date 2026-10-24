"""George Market Forecast Lab: five-model saved-evaluation dashboard (not live forecasting)."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='George Market | Five-Model Forecast Lab',page_icon='📊',layout='wide')
st.markdown('''<style>.block-container{max-width:1350px;padding-top:1.6rem}.hero{background:linear-gradient(120deg,#102A43,#146C70);padding:26px 30px;border-radius:16px;color:white}.hero h1{color:white;margin:0}.hero p{color:#E0F2F1;margin:6px 0 0}</style>''',unsafe_allow_html=True)
@st.cache_data
def load():
 p=pd.read_csv(ROOT/'combined_predictions.csv');p['month']=pd.to_datetime(p['month'])
 m=pd.read_csv(ROOT/'combined_metrics.csv')
 h=pd.read_csv(ROOT/'data/monthly_prices.csv');h['month']=pd.to_datetime(h['month'])
 return p,m,h
pred,metrics,history=load()
models=['Random Forest','XGBoost','SVR','LSTM','ANN']
colors={'Actual':'#243B53','Random Forest':'#2878B5','XGBoost':'#9B5DE5','SVR':'#D97757','LSTM':'#008F85','ANN':'#C24D8E','Last-month baseline':'#A28B22'}
st.markdown('<div class="hero"><h1>George Market · Forecast Lab</h1><p>IFS742 · Random Forest, XGBoost, SVR, LSTM and ANN · Historical price evaluation (R/kg)</p></div>',unsafe_allow_html=True)
st.caption('Saved results from five group experiments. This app does not retrain models or generate new future forecasts.')
with st.sidebar:
 st.header('Explore')
 commodity=st.selectbox('Commodity',['Potatoes','Brown onions','Tomatoes'])
 period=st.radio('Period',['Test','Validation','Training'],horizontal=False)
 visible=st.multiselect('Models on chart',models,default=models)
 baseline=st.checkbox('Show last-month baseline',True)
 st.divider()
 st.caption('George Fresh Produce Market · 3-month lookback · test: May–Aug 2026 · ANN uses 24 observed months; others use 25')
part={'Test':'test','Validation':'validation','Training':'train'}[period]
sel=pred[(pred.commodity==commodity)&(pred.partition==part)].copy()
sc=metrics[(metrics.commodity==commodity)&(metrics.partition==part)].sort_values('mae')
sc_models=sc[sc.model.isin(models)]
st.subheader(f'{commodity} · {period} results')
if len(sc_models):
 winner=sc_models.iloc[0]
 cols=st.columns(4)
 cols[0].metric('Lowest model MAE',f"R{winner.mae:.3f}/kg")
 cols[1].metric('Lowest-MAE model',winner.model)
 cols[2].metric('Model RMSE',f"R{winner.rmse:.3f}/kg")
 cols[3].metric('Target months',f"{int(winner.n)}")
if part=='test':st.warning('Only four test months per commodity. Small MAE differences do not establish a statistically reliable winner.')
left,right=st.columns([1.6,1],gap='large')
with left:
 st.markdown('#### Actual versus predicted prices')
 fig=go.Figure()
 actual=sel.groupby('month',as_index=False).actual.mean().sort_values('month')
 fig.add_trace(go.Scatter(x=actual.month,y=actual.actual,mode='lines+markers',name='Actual',line=dict(color=colors['Actual'],width=4)))
 for model in visible+(['Last-month baseline'] if baseline else []):
  ss=sel[sel.model==model].sort_values('month')
  if len(ss):fig.add_trace(go.Scatter(x=ss.month,y=ss.predicted,mode='lines+markers',name=model,line=dict(color=colors[model],width=2,dash='dash' if model=='Last-month baseline' else 'solid')))
 fig.update_layout(height=420,template='plotly_white',hovermode='x unified',yaxis_title='Price (R/kg)',legend=dict(orientation='h',y=1.22),margin=dict(l=10,r=10,t=30,b=10))
 st.plotly_chart(fig,use_container_width=True)
with right:
 st.markdown('#### Error comparison (MAE)')
 plot=sc[sc.model.isin(models+['Last-month baseline'])].sort_values('mae')
 fig2=px.bar(plot,x='mae',y='model',orientation='h',color='model',color_discrete_map=colors,text='mae')
 fig2.update_traces(texttemplate='%{text:.3f}',textposition='outside')
 fig2.update_layout(height=420,template='plotly_white',showlegend=False,xaxis_title='MAE (R/kg) · lower is better',yaxis_title='',margin=dict(l=10,r=35,t=10,b=10),yaxis=dict(autorange='reversed'))
 st.plotly_chart(fig2,use_container_width=True)
st.markdown('#### Metrics and model ranking')
tbl=sc[['model','n','mae','rmse','bias']].rename(columns={'model':'Model','n':'Months','mae':'MAE (R/kg)','rmse':'RMSE (R/kg)','bias':'Signed error (R/kg)'})
st.dataframe(tbl.style.format({'Months':'{:.0f}','MAE (R/kg)':'{:.4f}','RMSE (R/kg)':'{:.4f}','Signed error (R/kg)':'{:+.4f}'}),hide_index=True,use_container_width=True)
st.markdown('#### Monthly predictions')
pivot=sel.pivot_table(index='month',columns='model',values='predicted',aggfunc='first');pivot.insert(0,'Actual',actual.set_index('month').actual);pivot=pivot.reset_index().rename(columns={'month':'Month'})
st.dataframe(pivot.style.format({c:'{:.4f}' for c in pivot.columns if c!='Month'}),hide_index=True,use_container_width=True)
st.download_button('Download selected predictions (CSV)',sel.to_csv(index=False),file_name=f'{commodity.lower().replace(" ","_")}_{part}_predictions.csv',mime='text/csv')
st.divider()
st.subheader('Cross-commodity comparison')
comparison=metrics[(metrics.partition==part)&(metrics.model.isin(models+['Last-month baseline']))].pivot(index='commodity',columns='model',values='mae').reindex(['Potatoes','Brown onions','Tomatoes'])
st.dataframe(comparison.style.format('{:.3f}'),use_container_width=True)
st.caption('Compare MAE within each commodity. Price scales differ between commodities. ANN used 13 training targets versus 14 for the other models; testing uses the same four months.')
with st.expander('Model methodology and selection details'):
 st.markdown('**LSTM:** 8- and 16-unit candidates, seeds 42/43/44, validation-MAE selection, 3-month lookback. Candidate results below are from the original archived runs.')
 key=commodity.lower().replace(' ','_')
 try:
  cfg=json.loads((ROOT/'results'/key/'run_config.json').read_text())
  candidates=pd.read_csv(ROOT/'results'/key/'validation_candidates.csv')
  st.write(f"LSTM selected units: **{cfg['chosen_units']}** · max epochs: {cfg['epochs_cap']} · patience: {cfg['patience']} · batch size: {cfg['batch_size']}")
  candidates['Selection']=candidates['units'].apply(lambda n:'Selected' if n==cfg['chosen_units'] else 'Alternative')
  st.dataframe(candidates[['units','mae','rmse','mean_signed_error','Selection']].rename(columns={'units':'Units','mae':'Validation MAE','rmse':'Validation RMSE','mean_signed_error':'Signed error'}),hide_index=True,use_container_width=True)
 except (OSError,KeyError) as exc:st.info(f'LSTM candidate details unavailable: {exc}')
 st.markdown('**Random Forest:** selected design was reportedly refitted on training and validation data before test forecasting; the original HTML includes model tuning and a browser-based model demonstration.')
 st.markdown('**XGBoost:** depth 1–3, 50/100/200 estimators and learning rates 0.05/0.1; selected using validation RMSE, according to its source dashboard.')
 st.markdown('**SVR:** linear and RBF kernels with tuning of C, epsilon and gamma; selected using validation RMSE, according to its source dashboard.')
 st.markdown('**ANN:** scikit-learn MLPRegressor with three lagged-price inputs, one tanh hidden layer (2 or 4 neurons), L-BFGS optimiser and L2 alpha 0.1, 1 or 10. Six candidates per commodity; lowest validation RMSE selected. Trained on 13 target months (Dec 2024–Dec 2025) using the latest 24-month window; not refitted on validation data. Seed 42.')
 st.caption('Training protocols and selection metrics were not identical across models. Interpret test comparisons as descriptive, not a controlled architecture-only experiment.')
with st.expander('Historical monthly prices and commodity coverage'):
 cmap={'Potatoes':'potatoes_r_per_kg','Brown onions':'brown_onions_r_per_kg','Tomatoes':'tomatoes_r_per_kg'}
 col=cmap[commodity]
 if col in history:
  hist=history[['month',col]].dropna()
  fig3=px.line(hist,x='month',y=col,markers=True)
  fig3.add_vrect(x0='2026-01-01',x1='2026-04-30',fillcolor='#E4B34C',opacity=.12,line_width=0)
  fig3.add_vrect(x0='2026-05-01',x1='2026-08-31',fillcolor='#4D99CC',opacity=.12,line_width=0)
  fig3.update_layout(template='plotly_white',yaxis_title='Price (R/kg)',xaxis_title='Month')
  st.plotly_chart(fig3,use_container_width=True)
 st.write('Potatoes, brown onions and tomatoes each have 25 monthly observations (Aug 2024–Aug 2026). Avocados have 23 and beetroot 4; these were excluded from the common three-commodity comparison without inventing missing months.')
with st.expander('Data provenance and limitations'):
 st.write('Random Forest, XGBoost and SVR predictions were extracted from their submitted HTML dashboards; ANN results were extracted from the original full-precision CSV embedded in its HTML. LSTM predictions and model-selection results come from the uploaded LSTM project ZIP. Some HTML sources round values to four decimal places. Actual prices agree across sources to rounding tolerance.')
 st.write('Evaluation is one-month-ahead with observed prior prices, not a single four-month-ahead forecast. The test sample contains only four observations per commodity. Baseline values use the LSTM CSV, which agrees with the common actual price series.')
 st.write('ANN uses September 2024–August 2026 (24 observed months, 13 training targets), whereas the other model datasets include August 2024 (25 months, generally 14 training targets). All five share the January–April validation and May–August 2026 test months. Compare test metrics descriptively; different fitting protocols and selection rules limit causal model-to-model claims.')
st.info('Academic prototype: model rankings describe this limited historical test set. Do not use these results as evidence of dependable future market-price forecasts.')
