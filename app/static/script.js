/* ══════════════════════════════════════════════════════════
   CIPLA StockML — Interactive Script
   EDA fix: no state guard — backend auto-loads from disk
   ══════════════════════════════════════════════════════════ */

const TITLES = {
  0: ['Dashboard', 'Project Overview'],
  1: ['Step 1', 'Data Preprocessing'],
  2: ['Step 2', 'Exploratory Data Analysis'],
  3: ['Step 3', 'Feature Engineering'],
  4: ['Step 4', 'XGBoost Training'],
  5: ['Step 5', 'Predictions & Results'],
};

// ── Navigation ────────────────────────────────────────────────────────────────
function activateStep(n) {
  document.querySelectorAll('.sb-item').forEach(el => {
    el.classList.toggle('active', +el.dataset.step === n);
  });
  document.querySelectorAll('.panel').forEach(el => {
    el.classList.toggle('active', el.id === `panel-${n}`);
  });
  const [crumb, title] = TITLES[n];
  document.getElementById('tb-step-label').textContent = crumb;
  document.getElementById('tb-title').textContent = title;
  // Scroll content to top
  document.querySelector('.panel.active').scrollTop = 0;
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _tt;
function toast(msg, type = 'ok') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(_tt);
  _tt = setTimeout(() => el.classList.remove('show'), 4500);
}

// ── Spinner ───────────────────────────────────────────────────────────────────
function spin(n, on) { document.getElementById(`spin-${n}`).classList.toggle('hidden', !on); }
function showResult(n) { document.getElementById(`result-${n}`).classList.remove('hidden'); }
function markDone(n) {
  const el = document.querySelector(`.sb-item[data-step="${n}"]`);
  el.classList.add('done');
  const chk = document.getElementById(`check-${n}`);
  if (chk) { chk.classList.remove('hidden'); }
}

// ── API ───────────────────────────────────────────────────────────────────────
async function api(endpoint) {
  try {
    const res = await fetch(endpoint, { method: 'POST' });
    return await res.json();
  } catch(e) {
    return { ok: false, error: e.message };
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function kpi(label, val, sub = '', cls = '') {
  return `<div class="kpi ${cls}"><div class="kpi-l">${label}</div><div class="kpi-v">${val}</div>${sub ? `<div class="kpi-s">${sub}</div>` : ''}</div>`;
}
function tag(text, cls = 'tb') {
  return `<span class="tag ${cls}">${text}</span>`;
}
function table(el, rows) {
  if (!rows?.length) return;
  const cols = Object.keys(rows[0]);
  el.innerHTML = `<table>
    <thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c] ?? '—'}</td>`).join('')}</tr>`).join('')}</tbody>
  </table>`;
}
function animateBars(selector, dataAttr = 'w') {
  setTimeout(() => {
    document.querySelectorAll(selector).forEach(el => {
      el.style.width = el.dataset[dataAttr] + '%';
    });
  }, 120);
}

// ── STEP 1 – PREPROCESSING ───────────────────────────────────────────────────
document.getElementById('btn-preprocess').addEventListener('click', async () => {
  spin(1, true);
  const data = await api('/api/preprocess');
  spin(1, false);
  if (!data.ok) { toast('Preprocessing failed: ' + data.error, 'err'); return; }

  const r = data.report;

  document.getElementById('pre-kpis').innerHTML = [
    kpi('Rows Before', r.shape_before[0].toLocaleString(), 'raw records'),
    kpi('Columns Before', r.shape_before[1], 'raw features'),
    kpi('Rows After', r.shape_after[0].toLocaleString(), 'clean records'),
    kpi('Columns After', r.shape_after[1], 'kept features'),
    kpi('Dropped', r.dropped_columns.length, 'removed columns'),
  ].join('');

  document.getElementById('dropped-cols').innerHTML =
    r.dropped_columns.map(c => tag(c, 'tr')).join('');

  document.getElementById('kept-cols').innerHTML =
    Object.keys(r.dtypes || {}).map(c => tag(c, 'tg')).join('');

  const maxM = Math.max(...Object.values(r.missing_before), 1);
  document.getElementById('missing-bars').innerHTML =
    Object.entries(r.missing_before).map(([col, cnt]) =>
      `<div class="miss-row">
        <div class="miss-lbl">${col}</div>
        <div class="miss-bg"><div class="miss-fill" style="width:0%" data-w="${(cnt/maxM*100).toFixed(1)}"></div></div>
        <div class="miss-cnt">${cnt}</div>
      </div>`
    ).join('');
  animateBars('.miss-fill');

  table(document.getElementById('pre-table'), r.sample_rows);
  showResult(1); markDone(1);
  toast('✓ Preprocessing complete — dataset ready', 'ok');
});

// ── STEP 2 – EDA ─────────────────────────────────────────────────────────────
// NOTE: No state guard here. Backend auto-loads processed_data.csv from disk.
document.getElementById('btn-eda').addEventListener('click', async () => {
  spin(2, true);
  const data = await api('/api/eda');
  spin(2, false);
  if (!data.ok) { toast('EDA failed: ' + data.error, 'err'); return; }

  const r = data.report;
  const s = r.stats;

  document.getElementById('eda-kpis').innerHTML = [
    kpi('Records', (+s.total_records).toLocaleString(), '21 years of data'),
    kpi('Date Range', s.date_range.split(' to ')[0], 'from'),
    kpi('To', s.date_range.split(' to ')[1], 'until'),
    kpi('Min Close', '₹' + s.close_min, 'lowest price'),
    kpi('Max Close', '₹' + s.close_max, 'highest price'),
    kpi('Mean', '₹' + s.close_mean, '± ₹' + s.close_std),
  ].join('');

  document.getElementById('eda-plots').innerHTML = [
    ['close_history_plot', '📈 Closing Price History (2000–2021)', true],
    ['volume_plot',        '📊 Trading Volume Over Time',           true],
    ['ohlc_plot',          '🕯 OHLC Candlestick — Last 60 Days',   true],
    ['corr_plot',          '🔥 Feature Correlation Heatmap',        false],
    ['dist_plot',          '📉 Close Price Distribution',           false],
  ].map(([k, title, wide], i) => `
    <div class="plot-card ${wide ? 'wide' : ''}" style="animation-delay:${i * .08}s">
      <div class="plot-title">${title}</div>
      <img src="data:image/png;base64,${r[k]}" alt="${title}" loading="lazy"/>
    </div>`).join('');

  showResult(2); markDone(2);
  toast('✓ EDA complete — 5 charts generated', 'ok');
});

// ── STEP 3 – FEATURES ────────────────────────────────────────────────────────
document.getElementById('btn-features').addEventListener('click', async () => {
  spin(3, true);
  const data = await api('/api/features');
  spin(3, false);
  if (!data.ok) { toast('Feature engineering failed: ' + data.error, 'err'); return; }

  const r = data.report;

  document.getElementById('feat-kpis').innerHTML = [
    kpi('Total Features', r.total_features, 'engineered'),
    kpi('After Windowing', r.shape_after?.[0]?.toLocaleString() || '—', 'records'),
    kpi('MA Windows', '5 / 10 / 20 / 50', '4 periods'),
    kpi('Lag Periods', '1, 2, 3, 5, 10', '5 features'),
    kpi('RSI', '14-period', 'momentum'),
    kpi('MACD', 'EMA 12 & 26', 'trend signal'),
  ].join('');

  const groups = [
    { title: 'Moving Averages', icon: '📈', feats: r.moving_averages || [], cls: 'tb',  desc: 'Smoothed price over N rolling periods — captures trend direction' },
    { title: 'Lag Features',    icon: '⏪', feats: r.lag_features    || [], cls: 'tp',  desc: "Previous N days' close used directly as input features" },
    { title: 'EMA & MACD',      icon: '⚡', feats: r.ema_features    || [], cls: 'tt',  desc: 'Exponential MA with MACD momentum crossover signal' },
    { title: 'RSI & Volatility',icon: '🔥', feats: [...(r.rsi_features||[]), ...(r.volatility_features||[])], cls: 'tgo', desc: 'Relative strength index + rolling volatility (std dev)' },
    { title: 'Derived Prices',  icon: '🔢', feats: r.derived_features || [], cls: 'tg', desc: 'Intraday range, daily return %, open-to-close difference' },
    { title: 'Date Features',   icon: '📅', feats: r.date_features   || [], cls: 'tr',  desc: 'Year, month, day-of-week for capturing seasonality' },
  ];

  document.getElementById('feat-groups').innerHTML = groups.map((g, i) => `
    <div class="feat-group" style="animation-delay:${i * .07}s">
      <div class="fg-hd">
        <div class="fg-title">${g.icon} ${g.title}</div>
        <span class="fg-count">${g.feats.length} feature${g.feats.length !== 1 ? 's' : ''}</span>
      </div>
      <div class="fg-desc">${g.desc}</div>
      <div class="tag-wrap">${g.feats.map(f => tag(f, g.cls)).join('')}</div>
    </div>`).join('');

  showResult(3); markDone(3);
  toast('✓ 26 features engineered successfully', 'ok');
});

// ── STEP 4 – TRAINING ────────────────────────────────────────────────────────
document.getElementById('btn-train').addEventListener('click', async () => {
  toast('Training XGBoost model… (~20 seconds)', 'ok');
  spin(4, true);
  const data = await api('/api/train');
  spin(4, false);
  if (!data.ok) { toast('Training failed: ' + data.error, 'err'); return; }

  const m = data.meta;

  document.getElementById('train-kpis').innerHTML = [
    kpi('Algorithm', 'XGBoost', 'Gradient Boosting'),
    kpi('Estimators', m.n_estimators, 'boosted trees'),
    kpi('Train Size', (+m.train_size).toLocaleString(), '80% chronological'),
    kpi('Test Size',  (+m.test_size).toLocaleString(),  '20% held-out'),
    kpi('Features', m.features?.length || '—', 'input dimensions'),
    kpi('Max Depth', m.max_depth, 'per tree'),
  ].join('');

  document.getElementById('hyperparams').innerHTML = [
    ['Algorithm',     m.model_name || 'GradientBoosting'],
    ['n_estimators',  m.n_estimators],
    ['learning_rate', m.learning_rate],
    ['max_depth',     m.max_depth],
    ['subsample',     '0.8'],
    ['objective',     'reg:squarederror'],
    ['split',         '80 / 20 chronological'],
    ['scaler',        'StandardScaler'],
  ].map(([k,v]) => `<div class="hp-row"><span class="hp-k">${k}</span><span class="hp-v">${v}</span></div>`).join('');

  const maxI = Math.max(...m.feature_importance.map(x => x[1]));
  document.getElementById('imp-bars').innerHTML = m.feature_importance.map(([f, v]) => `
    <div class="imp-row">
      <div class="imp-lbl">${f}</div>
      <div class="imp-bg"><div class="imp-fill" style="width:0%" data-w="${(v/maxI*100).toFixed(1)}"></div></div>
      <div class="imp-val">${(v*100).toFixed(2)}%</div>
    </div>`).join('');
  animateBars('.imp-fill');

  showResult(4); markDone(4);
  toast('✓ Model trained & saved — xgboost_model.pkl', 'ok');
});

// ── STEP 5 – EVALUATE ────────────────────────────────────────────────────────
document.getElementById('btn-evaluate').addEventListener('click', async () => {
  spin(5, true);
  const data = await api('/api/evaluate');
  spin(5, false);
  if (!data.ok) { toast('Evaluation failed: ' + data.error, 'err'); return; }

  const r = data.report;

  document.getElementById('eval-kpis').innerHTML = [
    kpi('RMSE',  '₹' + r.rmse, 'root mean sq error', 'rmse'),
    kpi('MAE',   '₹' + r.mae,  'mean absolute error', 'mae'),
    kpi('R²',    r.r2,          'variance explained',  'r2'),
    kpi('MAPE',  r.mape + '%',  'mean abs % error',   'mape'),
    kpi('Test N', (+r.test_size).toLocaleString(), 'held-out days'),
  ].join('');

  document.getElementById('eval-plots').innerHTML = [
    ['pred_plot',       '📈 Actual vs Predicted Close Price', true],
    ['residual_plot',   '📊 Residuals Analysis',              true],
    ['importance_plot', '⭐ Feature Importances (XGBoost)',    true],
  ].map(([k, title, wide], i) => `
    <div class="plot-card ${wide ? 'wide' : ''}" style="animation-delay:${i*.1}s">
      <div class="plot-title">${title}</div>
      <img src="data:image/png;base64,${r[k]}" alt="${title}" loading="lazy"/>
    </div>`).join('');

  // Prediction table with colour-coded error
  if (r.predictions?.length) {
    const rows = r.predictions;
    document.getElementById('pred-table').innerHTML = `<table>
      <thead><tr><th>Date</th><th>Actual (₹)</th><th>Predicted (₹)</th><th>Error (₹)</th><th>Error %</th></tr></thead>
      <tbody>${rows.map(row => {
        const err = +(row.predicted - row.actual).toFixed(2);
        const pct = ((Math.abs(err)/row.actual)*100).toFixed(2);
        const c = Math.abs(err) < 10 ? 'var(--green)' : Math.abs(err) < 30 ? 'var(--gold)' : 'var(--red)';
        return `<tr><td>${row.date}</td><td style="color:var(--text)">${row.actual}</td><td style="color:var(--blue)">${row.predicted}</td><td style="color:${c}">${err > 0 ? '+' : ''}${err}</td><td style="color:${c}">${pct}%</td></tr>`;
      }).join('')}</tbody></table>`;
  }

  // Update sidebar
  updateSidebarMetrics(r.rmse, r.mae, r.r2, r.mape);
  showResult(5); markDone(5);
  toast(`✓ R² = ${r.r2} · RMSE ₹${r.rmse} · MAPE ${r.mape}%`, 'ok');
});

function updateSidebarMetrics(rmse, mae, r2, mape) {
  document.getElementById('sb-metrics').classList.remove('hidden');
  document.getElementById('sbm-r2').textContent   = r2;
  document.getElementById('sbm-rmse').textContent = rmse;
  document.getElementById('sbm-mae').textContent  = mae;
  document.getElementById('sbm-mape').textContent = mape + '%';
}

// ── RUN ALL ───────────────────────────────────────────────────────────────────
async function runAll() {
  const overlay = document.getElementById('overlay');
  overlay.classList.remove('hidden');
  const ovsIds = ['ovs-1','ovs-2','ovs-3','ovs-4','ovs-5'];

  function setOvs(i, status) {
    ovsIds.forEach((id, idx) => {
      if (idx < i)      document.getElementById(id).className = 'ovs done';
      else if (idx === i) document.getElementById(id).className = `ovs ${status}`;
      else              document.getElementById(id).className = 'ovs';
    });
    const pct = status === 'done' ? ((i+1)/5*100) : (i/5*100);
    document.getElementById('ov-bar').style.width = pct + '%';
    document.getElementById('ov-pct').textContent = Math.round(pct) + '%';
  }

  setOvs(0, 'active');
  const data = await api('/api/run_all');

  // Animate through remaining steps
  for (let i = 0; i < 5; i++) {
    setOvs(i, 'done');
    if (i < 4) setOvs(i+1, 'active');
    await new Promise(r => setTimeout(r, 200));
  }
  document.getElementById('ov-bar').style.width = '100%';
  document.getElementById('ov-pct').textContent = '100%';

  await new Promise(r => setTimeout(r, 600));
  overlay.classList.add('hidden');

  if (!data.ok) { toast('Pipeline error: ' + data.error, 'err'); return; }

  const m = data.metrics;
  updateSidebarMetrics(m.rmse, m.mae, m.r2, m.mape);

  // Populate eval panel
  const r = data.report;
  document.getElementById('eval-plots').innerHTML = [
    ['pred_plot',       '📈 Actual vs Predicted Close Price', true],
    ['residual_plot',   '📊 Residuals Analysis',              true],
    ['importance_plot', '⭐ Feature Importances (XGBoost)',    true],
  ].map(([k,title,wide]) => `
    <div class="plot-card ${wide ? 'wide' : ''}">
      <div class="plot-title">${title}</div>
      <img src="data:image/png;base64,${r[k]}" alt="${title}" loading="lazy"/>
    </div>`).join('');

  document.getElementById('eval-kpis').innerHTML = [
    kpi('RMSE', '₹'+m.rmse, 'root mean sq error', 'rmse'),
    kpi('MAE',  '₹'+m.mae,  'mean absolute error', 'mae'),
    kpi('R²',   m.r2,        'variance explained',  'r2'),
    kpi('MAPE', m.mape+'%',  'mean abs % error',   'mape'),
  ].join('');

  document.getElementById('result-5').classList.remove('hidden');

  [0,1,2,3,4,5].forEach(markDone);
  activateStep(5);
  toast(`✅ Full pipeline complete! R² = ${m.r2} · MAPE = ${m.mape}%`, 'ok');
}

document.getElementById('btn-run-all').addEventListener('click', runAll);
document.getElementById('dash-run-all').addEventListener('click', runAll);
