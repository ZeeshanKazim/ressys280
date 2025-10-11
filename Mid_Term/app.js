/* app.js — Food Recommender (Two-Tower + optional Deep tower + Graph)
   Works on GitHub Pages (relative /data/*.csv, no leading slash).
   Requires: two-tower.js (exports TwoTowerModel, TwoTowerDeepModel)
   Author: you :)
*/

/* ------------------------- tiny DOM helpers ------------------------- */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const pick = (...idsOrSelectors) => {
  for (const id of idsOrSelectors) {
    const el = id.startsWith('#') || id.startsWith('.') ? $(id) : $(`#${id}`);
    if (el) return el;
  }
  return null;
};
const setText = (el, txt) => { if (el) el.textContent = txt; };

/* ------------------------- status & wiring -------------------------- */
const statusEl = pick('status', 'statusText', 'edaStatus', '#status', '#edaStatus', '[data-role="status"]');
function setStatus(msg) { setText(statusEl, `Status: ${msg}`); }

/* --------------------------- global state --------------------------- */
let users = new Set();
let items = new Map();         // itemId -> { id, name, tags, minutes, n_steps, n_ingredients }
let interactions = [];         // { userId, itemId, rating, ts }

let userIdToIdx = new Map();
let itemIdToIdx = new Map();
let idxToUserId = [];
let idxToItemId = [];

let baseline = null;           // TwoTowerModel
let deepModel = null;          // TwoTowerDeepModel
let tensors = null;            // cached training tensors

// EDA caches
let topTagCounts = [];
let userActivityHist = null;
let itemPopularityHist = null;
let ratingsHist = null;

// UI elements (try multiple common IDs to avoid 404 wiring)
const btnLoad   = pick('btnLoadData', 'loadDataBtn', 'loadBtn', '#btnLoadData', '#loadBtn');
const btnTrainB = pick('btnTrainBase', 'trainBaselineBtn', '#btnTrainBase');
const btnTrainD = pick('btnTrainDeep', 'trainDeepBtn', '#btnTrainDeep');
const btnTest   = pick('btnTest', 'testBtn', '#btnTest');

const cvRatings = pick('histRatings', 'eda_ratings_hist', '#histRatings');
const cvTags    = pick('barTopTags', 'eda_top_tags', '#barTopTags');
const cvUserA   = pick('histUserActivity', 'eda_user_activity', '#histUserActivity');
const cvItemP   = pick('histItemPopularity', 'eda_item_popularity', '#histItemPopularity');
const cvBaseLoss= pick('baseLossCanvas', 'lossBaseline', '#baseLossCanvas');
const cvDeepLoss= pick('deepLossCanvas', 'lossDeep', '#deepLossCanvas');
const projCanvas= pick('projCanvas', 'pcaCanvas', '#projCanvas');

const tblDemoHistory = pick('demoHistTable', '#demoHistTable');
const tblDemoBase    = pick('demoBaseTable', '#demoBaseTable');
const tblDemoDeep    = pick('demoDeepTable', '#demoDeepTable');

const inpEmbB   = pick('embDimBase', '#embDimBase');
const inpEmbD   = pick('embDimDeep', '#embDimDeep');
const inpEpochB = pick('epochsBase', '#epochsBase');
const inpEpochD = pick('epochsDeep', '#epochsDeep');
const inpBatchB = pick('batchBase', '#batchBase');
const inpBatchD = pick('batchDeep', '#batchDeep');
const inpLrB    = pick('lrBase', '#lrBase');
const inpLrD    = pick('lrDeep', '#lrDeep');
const inpMaxInt = pick('maxInteractions', '#maxInteractions');
const inpTagFe  = pick('tagFeatures', '#tagFeatures');
const chkGraphRerank = pick('useGraph', '#useGraph'); // optional; ignored if absent

/* --------------------------- CSV utilities -------------------------- */

// Robust CSV line splitter (handles commas in quotes)
function splitCSVLine(line) {
  const out = [];
  let cur = '', inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQ && line[i + 1] === '"') { cur += '"'; i++; }
      else inQ = !inQ;
    } else if (c === ',' && !inQ) {
      out.push(cur); cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
}

async function fetchTextOrThrow(path) {
  const res = await fetch(`${path}?v=${Date.now()}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`fetch ${path} → ${res.status}`);
  return res.text();
}

// Try multiple candidates until one succeeds (for GH Pages + case sensitivity)
async function loadFirstAvailable(paths) {
  let lastErr = null;
  for (const p of paths) {
    try {
      const txt = await fetchTextOrThrow(p);
      return { path: p, text: txt };
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('No file found');
}

/* --------------------------- data loading --------------------------- */
function parseInteractionsCSV(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  const head = splitCSVLine(lines[0]).map(s => s.trim().toLowerCase());
  const colUser = head.indexOf('user_id') >= 0 ? 'user_id' : head.indexOf('user') >= 0 ? 'user' : null;
  const colItem = head.indexOf('recipe_id') >= 0 ? 'recipe_id' : head.indexOf('item_id') >= 0 ? 'item_id' : 'item';
  const colRating = head.indexOf('rating') >= 0 ? 'rating' : null;
  const colDate = head.indexOf('date') >= 0 ? 'date' : head.indexOf('timestamp') >= 0 ? 'timestamp' : null;

  const idxU = head.indexOf(colUser);
  const idxI = head.indexOf(colItem);
  const idxR = colRating ? head.indexOf(colRating) : -1;
  const idxT = colDate ? head.indexOf(colDate) : -1;

  const out = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = splitCSVLine(lines[i]);
    if (idxU < 0 || idxI < 0 || idxU >= cols.length || idxI >= cols.length) continue;
    const u = cols[idxU].trim();
    const it = cols[idxI].trim();
    const r = (idxR >= 0 && idxR < cols.length) ? Number(cols[idxR]) : 1;
    const ts = (idxT >= 0 && idxT < cols.length) ? (Date.parse(cols[idxT]) || 0) : 0;
    if (!u || !it) continue;
    out.push({ userId: u, itemId: it, rating: isFinite(r) ? r : 1, ts });
  }
  return out;
}

function parseRecipesCSV(text) {
  // We only need: id, name/title, tags (as array), minutes, n_steps, n_ingredients
  const lines = text.split(/\r?\n/).filter(Boolean);
  const head = splitCSVLine(lines[0]).map(s => s.trim().toLowerCase());
  const idxId = head.indexOf('id');
  const idxName = head.indexOf('name') >= 0 ? head.indexOf('name') : head.indexOf('title');
  const idxTags = head.indexOf('tags');
  const idxMin = head.indexOf('minutes');
  const idxSteps = head.indexOf('n_steps');
  const idxIngr = head.indexOf('n_ingredients');

  const map = new Map();
  for (let i = 1; i < lines.length; i++) {
    const cols = splitCSVLine(lines[i]);
    const id = cols[idxId]?.trim();
    if (!id) continue;
    const name = (idxName >= 0 && cols[idxName]) ? cols[idxName].trim() : `Recipe ${id}`;
    const rawTags = (idxTags >= 0 && cols[idxTags]) ? cols[idxTags].trim() : '[]';
    const minutes = (idxMin >= 0) ? Number(cols[idxMin]) : NaN;
    const n_steps = (idxSteps >= 0) ? Number(cols[idxSteps]) : NaN;
    const n_ing = (idxIngr >= 0) ? Number(cols[idxIngr]) : NaN;

    // tags look like "['tag1','tag2']" → make JSON safe, then parse
    let tags = [];
    if (rawTags && rawTags !== '[]') {
      try {
        const safe = rawTags
          .replace(/^u?"/, '"') // some dumps add unicode prefix
          .replace(/'/g, '"')
          .replace(/,\s*]/g, ']');
        const arr = JSON.parse(safe);
        if (Array.isArray(arr)) tags = arr.map(t => String(t).toLowerCase());
      } catch {
        // fallback: naive split
        tags = rawTags.replace(/[\[\]']/g, '').split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
      }
    }
    map.set(id, { id, name, tags, minutes, n_steps, n_ingredients: n_ing });
  }
  return map;
}

async function loadData() {
  try {
    setStatus('loading…');

    // 1) interactions (train)
    const interCandidates = [
      'data/interactions_train.csv',
      'data/interactions.csv',
      'data/RAW_interactions.csv'
    ];
    const { path: interPath, text: interText } = await loadFirstAvailable(interCandidates);
    interactions = parseInteractionsCSV(interText);

    // 2) recipes metadata
    const recipeCandidates = [
      'data/RAW_recipes.csv',
      'data/recipes.csv'
    ];
    const { path: recipePath, text: recipText } = await loadFirstAvailable(recipeCandidates);
    items = parseRecipesCSV(recipText);

    // users/items sets
    users = new Set(interactions.map(x => x.userId));
    const presentItemIds = new Set(interactions.map(x => x.itemId));
    // ensure we have item objects for items present in interactions (even if missing in recipes.csv)
    for (const it of presentItemIds) {
      if (!items.has(it)) items.set(it, { id: it, name: `Recipe ${it}`, tags: [] });
    }

    // indexers
    idxToUserId = Array.from(users);
    userIdToIdx = new Map(idxToUserId.map((u, i) => [u, i]));
    idxToItemId = Array.from(items.keys());
    itemIdToIdx = new Map(idxToItemId.map((it, i) => [it, i]));

    // EDA metrics
    computeEDA();

    setStatus(`loaded: users=${users.size}, items=${items.size} (file: ${recipePath}), interactions=${interactions.length} (file: ${interPath})`);
    drawEDA();
  } catch (err) {
    console.error(err);
    setStatus('fetch failed. Ensure /data/*.csv exist (case-sensitive). (Tip: path must be data/… without a leading slash on GitHub Pages)');
  }
}

/* ------------------------------- EDA -------------------------------- */
function histogram(arr, buckets) {
  const counts = new Array(buckets.length).fill(0);
  for (const v of arr) {
    for (let b = 0; b < buckets.length; b++) {
      const [lo, hi] = buckets[b];
      if (v >= lo && v <= hi) { counts[b]++; break; }
    }
  }
  return counts;
}

function computeEDA() {
  // ratings hist (1..5 common in Food.com)
  const r = interactions.map(x => Number(x.rating)).filter(Number.isFinite);
  const br = [[1,1],[2,2],[3,3],[4,4],[5,5]];
  ratingsHist = { buckets: ['1','2','3','4','5'], counts: histogram(r, br) };

  // user activity = ratings per user
  const perUser = new Map();
  for (const it of interactions) perUser.set(it.userId, (perUser.get(it.userId) || 0) + 1);
  const ua = Array.from(perUser.values());
  const bU = [[1,1],[2,2],[3,3],[4,5],[6,10],[11,20],[21,50],[51,100],[101,200],[201,500],[501, 1e9]];
  const uLabels = ['1','2','3','4–5','6–10','11–20','21–50','51–100','101–200','201–500','>500'];
  userActivityHist = { buckets: uLabels, counts: histogram(ua, bU) };

  // item popularity = ratings per item
  const perItem = new Map();
  for (const it of interactions) perItem.set(it.itemId, (perItem.get(it.itemId) || 0) + 1);
  const ip = Array.from(perItem.values());
  const bI = [[1,1],[2,2],[3,3],[4,5],[6,10],[11,20],[21,50],[51,100],[101,200],[201,500],[501, 2000]];
  const iLabels = ['1','2','3','4–5','6–10','11–20','21–50','51–100','101–200','201–500','>500'];
  itemPopularityHist = { buckets: iLabels, counts: histogram(ip, bI) };

  // tags frequency (top 20)
  const tagCount = new Map();
  for (const it of items.values()) {
    if (!it.tags) continue;
    for (const t of it.tags) tagCount.set(t, (tagCount.get(t) || 0) + 1);
  }
  topTagCounts = Array.from(tagCount.entries())
    .sort((a,b)=>b[1]-a[1])
    .slice(0, 20);
}

/* ------------------------------ drawing ------------------------------ */
function drawBars(canvas, labels, counts) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 600;
  const h = canvas.height = canvas.clientHeight || 220;
  ctx.clearRect(0,0,w,h);

  const max = Math.max(1, ...counts);
  const pad = 24;
  const barW = (w - pad*2) / counts.length;
  ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  for (let i=0;i<counts.length;i++) {
    const x = pad + i*barW + 2;
    const val = counts[i];
    const bh = Math.round((h - 48) * (val / max));
    const y = h - 24 - bh;
    ctx.fillStyle = '#4da3ff';
    ctx.fillRect(x, y, barW-4, bh);
    ctx.fillStyle = '#9aa4b2';
    // label (rotate if too many)
    if (labels.length <= 14) {
      ctx.fillText(labels[i], x + (barW-4)/2, h - 20);
    } else {
      ctx.save();
      ctx.translate(x + (barW-4)/2, h - 4);
      ctx.rotate(-Math.PI/3);
      ctx.fillText(labels[i], 0, 0);
      ctx.restore();
    }
  }
}

function drawEDA() {
  drawBars(cvRatings, ratingsHist.buckets, ratingsHist.counts);
  drawBars(cvUserA, userActivityHist.buckets, userActivityHist.counts);
  drawBars(cvItemP, itemPopularityHist.buckets, itemPopularityHist.counts);
  if (cvTags) {
    const labels = topTagCounts.map(([k])=>k);
    const values = topTagCounts.map(([,v])=>v);
    drawBars(cvTags, labels, values);
  }
}

/* --------------------------- tensors & model ------------------------- */
function makeTrainingTensors(maxInteractions = 80000) {
  // cap for memory in browser
  const N = Math.min(interactions.length, maxInteractions);
  const U = idxToUserId.length;
  const I = idxToItemId.length;

  const uIdx = new Int32Array(N);
  const iIdx = new Int32Array(N);
  const y    = new Float32Array(N);

  for (let k=0;k<N;k++) {
    const r = interactions[k];
    uIdx[k] = userIdToIdx.get(r.userId) ?? 0;
    iIdx[k] = itemIdToIdx.get(r.itemId) ?? 0;
    y[k] = Number(r.rating) || 1;
  }

  const t = {
    users: tf.tensor2d(uIdx, [N,1], 'int32'),
    items: tf.tensor2d(iIdx, [N,1], 'int32'),
    ratings: tf.tensor2d(y, [N,1], 'float32'),
    U, I, N
  };
  return t;
}

function disposeAllModels() {
  try { baseline?.dispose?.(); } catch {}
  try { deepModel?.dispose?.(); } catch {}
  try { tf.disposeVariables(); } catch {}
}

/* ------------------------------- train ------------------------------- */
async function trainBaseline() {
  if (!interactions.length) return setStatus('load data first');
  disposeAllModels();

  const embDim = Number(inpEmbB?.value || 32);
  const epochs = Number(inpEpochB?.value || 5);
  const batch  = Number(inpBatchB?.value || 256);
  const lr     = Number(inpLrB?.value || 1e-3);
  const maxInt = Number(inpMaxInt?.value || 80000);

  setStatus(`baseline: preparing tensors…`);
  tensors?.users?.dispose(); tensors?.items?.dispose(); tensors?.ratings?.dispose();
  tensors = makeTrainingTensors(maxInt);

  baseline = new TwoTowerModel(tensors.U, tensors.I, embDim);
  baseline.compile(lr);

  const ctx = cvBaseLoss?.getContext('2d');
  if (ctx) { ctx.clearRect(0,0,cvBaseLoss.width, cvBaseLoss.height); }

  setStatus(`baseline: training…`);
  await baseline.fit(
    [tensors.users, tensors.items],
    tensors.ratings,
    {
      epochs, batchSize: batch,
      onBatchEnd: (step, loss) => {
        drawLoss(cvBaseLoss, step, loss);
        if (step % 30 === 0) setStatus(`baseline: step ${step}, loss ${loss.toFixed(4)}`);
      }
    }
  );

  setStatus(`baseline: training done`);
  // optional: projection (after baseline)
  drawProjection(baseline.getItemEmbeddings());
}

async function trainDeep() {
  if (!interactions.length) return setStatus('load data first');
  disposeAllModels();

  const embDim = Number(inpEmbD?.value || 32);
  const epochs = Number(inpEpochD?.value || 5);
  const batch  = Number(inpBatchD?.value || 256);
  const lr     = Number(inpLrD?.value || 1e-3);
  const maxInt = Number(inpMaxInt?.value || 80000);
  const tagFeaturesMax = Math.max(0, Number(inpTagFe?.value || 200));

  setStatus(`deep: building features…`);
  // build tag bag-of-words (top K tags)
  const tagCounts = new Map();
  for (const it of items.values()) for (const t of (it.tags || [])) tagCounts.set(t, (tagCounts.get(t)||0)+1);
  const topTags = Array.from(tagCounts.entries()).sort((a,b)=>b[1]-a[1]).slice(0, tagFeaturesMax).map(([t])=>t);
  const tagIndex = new Map(topTags.map((t,idx)=>[t,idx]));

  const I = idxToItemId.length;
  const tagFeat = tf.buffer([I, topTags.length], 'float32');
  for (let i=0;i<I;i++) {
    const itemId = idxToItemId[i];
    const meta = items.get(itemId);
    if (!meta?.tags) continue;
    for (const t of meta.tags) {
      const j = tagIndex.get(t);
      if (j != null) tagFeat.set(1, i, j);
    }
  }
  const itemSideInfo = tagFeat.toTensor(); // [I, T]

  setStatus(`deep: preparing tensors…`);
  tensors?.users?.dispose(); tensors?.items?.dispose(); tensors?.ratings?.dispose();
  tensors = makeTrainingTensors(maxInt);

  deepModel = new TwoTowerDeepModel(tensors.U, tensors.I, embDim, itemSideInfo.shape[1]);
  deepModel.compile(lr);

  const ctx = cvDeepLoss?.getContext('2d');
  if (ctx) { ctx.clearRect(0,0,cvDeepLoss.width, cvDeepLoss.height); }

  setStatus(`deep: training…`);
  await deepModel.fit(
    [tensors.users, tensors.items],
    tensors.ratings,
    {
      epochs, batchSize: batch,
      itemSideInfo,
      onBatchEnd: (step, loss) => {
        drawLoss(cvDeepLoss, step, loss);
        if (step % 30 === 0) setStatus(`deep: step ${step}, loss ${loss.toFixed(4)}`);
      }
    }
  );

  setStatus(`deep: training done`);
  // optional: projection (after deep)
  drawProjection(deepModel.getItemEmbeddings());
  itemSideInfo.dispose();
}

/* -------------------------- loss mini-plot --------------------------- */
function drawLoss(canvas, step, loss) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 600;
  const h = canvas.height = canvas.clientHeight || 160;
  if (!canvas.__points) canvas.__points = [];
  canvas.__points.push([step, loss]);
  const pts = canvas.__points;
  const maxX = Math.max(...pts.map(p=>p[0]), 1);
  const minY = Math.min(...pts.map(p=>p[1]));
  const maxY = Math.max(...pts.map(p=>p[1]));

  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#5ecfff';
  ctx.beginPath();
  for (let i=0;i<pts.length;i++) {
    const [x,y] = pts[i];
    const px = 8 + (w-16) * (x / maxX);
    const py = 8 + (h-16) * ((maxY - y) / (maxY - minY + 1e-6));
    if (i===0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

/* ------------------------------ demo ------------------------------- */
function pickUserForDemo(minRatings = 20) {
  // if no user satisfies, relax to 10/5/2
  const steps = [minRatings, 10, 5, 2, 1];
  const byUser = new Map();
  for (const it of interactions) {
    if (!byUser.has(it.userId)) byUser.set(it.userId, []);
    byUser.get(it.userId).push(it);
  }
  for (const k of steps) {
    const cand = Array.from(byUser.entries()).filter(([,arr])=>arr.length >= k);
    if (cand.length) {
      const [userId, arr] = cand[Math.floor(Math.random()*cand.length)];
      // sort: rating desc, then recency
      arr.sort((a,b)=> (b.rating - a.rating) || (b.ts - a.ts));
      return { userId, history: arr.slice(0, 50) };
    }
  }
  return null;
}

function renderDemoTables(hist, baseTop, deepTop) {
  const render = (tbl, rows, showScore=false) => {
    if (!tbl) return;
    const thead = tbl.querySelector('thead');
    const tbody = tbl.querySelector('tbody');
    if (thead && !thead.__init) {
      thead.innerHTML = `
        <tr><th>#</th><th>Recipe</th>${showScore?'<th>score</th>':''}</tr>
      `;
      thead.__init = true;
    }
    if (tbody) {
      tbody.innerHTML = '';
      rows.forEach((r, idx) => {
        const tr = document.createElement('tr');
        const name = items.get(r.itemId)?.name || `Recipe ${r.itemId}`;
        tr.innerHTML = `<td>${idx+1}</td><td>${name}</td>${showScore?`<td>${r.score.toFixed(4)}</td>`:''}`;
        tbody.appendChild(tr);
      });
    }
  };

  render(tblDemoHistory, hist.map(h => ({ itemId: h.itemId })), false);
  render(tblDemoBase, baseTop, true);
  render(tblDemoDeep, deepTop, true);
}

async function runDemo() {
  if (!interactions.length) return setStatus('load data first');
  if (!baseline && !deepModel) return setStatus('train at least one model first');

  const pick = pickUserForDemo(20);
  if (!pick) { setStatus('No user with ≥20 ratings in this split. (Relaxed threshold automatically)'); return; }
  const { userId, history } = pick;

  // scores: dot(u, i)
  const exclude = new Set(history.map(h => h.itemId));
  const uid = userIdToIdx.get(userId);
  const allItemsIdx = idxToItemId.map((_,i)=>i);

  let baseTop = [];
  if (baseline) {
    const scores = await baseline.scoreUserAgainstAll(uid);
    baseTop = allItemsIdx
      .filter(i => !exclude.has(idxToItemId[i]))
      .map(i => ({ itemId: idxToItemId[i], score: scores[i] }))
      .sort((a,b)=>b.score-a.score)
      .slice(0, 10);
  }

  let deepTop = [];
  if (deepModel) {
    const scores = await deepModel.scoreUserAgainstAll(uid);
    deepTop = allItemsIdx
      .filter(i => !exclude.has(idxToItemId[i]))
      .map(i => ({ itemId: idxToItemId[i], score: scores[i] }))
      .sort((a,b)=>b.score-a.score)
      .slice(0, 10);
  }

  renderDemoTables(history.slice(0,10), baseTop, deepTop);
  setStatus('recommendations generated successfully!');
}

/* ------------------------ simple 2D projection ----------------------- */
function drawProjection(itemEmb) {
  if (!projCanvas || !itemEmb) return;
  const ctx = projCanvas.getContext('2d');
  const w = projCanvas.width = projCanvas.clientWidth || 780;
  const h = projCanvas.height = projCanvas.clientHeight || 260;
  ctx.clearRect(0,0,w,h);

  // quick PCA (SVD of cov) for small sample
  const E = itemEmb.arraySync(); // [I, d]
  const I = Math.min(1000, E.length);
  const d = E[0]?.length || 0;
  if (!I || !d) return;
  const X = E.slice(0, I);

  // center
  const mean = new Array(d).fill(0);
  for (const v of X) for (let j=0;j<d;j++) mean[j]+=v[j];
  for (let j=0;j<d;j++) mean[j]/=I;
  for (const v of X) for (let j=0;j<d;j++) v[j]-=mean[j];

  // power method to get first 2 components
  function powerVec(A, iters=40) {
    let v = new Array(A[0].length).fill(0).map(()=>Math.random()-0.5);
    for (let t=0;t<iters;t++) {
      const Av = new Array(v.length).fill(0);
      for (const row of A) {
        const dot = row.reduce((s,xi,idx)=>s+xi*v[idx],0);
        for (let j=0;j<v.length;j++) Av[j]+=row[j]*dot;
      }
      const norm = Math.sqrt(Av.reduce((s,a)=>s+a*a,0))+1e-9;
      v = Av.map(a=>a/norm);
    }
    return v;
  }
  const pc1 = powerVec(X);
  // deflate
  for (const row of X) {
    const dot = row.reduce((s,xi,idx)=>s+xi*pc1[idx],0);
    for (let j=0;j<d;j++) row[j]-=dot*pc1[j];
  }
  const pc2 = powerVec(X);

  const pts = E.slice(0, I).map(v => {
    const x = v.reduce((s,xi,idx)=>s+xi*pc1[idx],0);
    const y = v.reduce((s,xi,idx)=>s+xi*pc2[idx],0);
    return [x,y];
  });

  // scale to canvas
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  ctx.fillStyle = '#8bd1ff';
  for (const [x,y] of pts) {
    const px = 8 + (w-16) * ((x - minX) / (maxX - minX + 1e-9));
    const py = h-8 - (h-16) * ((y - minY) / (maxY - minY + 1e-9));
    ctx.fillRect(px-1, py-1, 2, 2);
  }
}

/* ----------------------------- wiring ------------------------------- */
if (btnLoad)   btnLoad.addEventListener('click', loadData);
if (btnTrainB) btnTrainB.addEventListener('click', trainBaseline);
if (btnTrainD) btnTrainD.addEventListener('click', trainDeep);
if (btnTest)   btnTest.addEventListener('click', runDemo);

// Make canvases crisp on HiDPI
function crispCanvas(c) {
  if (!c) return;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = c.clientWidth || c.width, h = c.clientHeight || c.height;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
}
[cvRatings, cvTags, cvUserA, cvItemP, cvBaseLoss, cvDeepLoss, projCanvas].forEach(crispCanvas);

// Optional: auto-load on first tab visit (comment out if undesired)
// document.addEventListener('DOMContentLoaded', () => setTimeout(()=> btnLoad?.click(), 150));

/* -------------------------------------------------------------------- */
