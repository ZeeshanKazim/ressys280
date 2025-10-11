/* app.js — Food Recommender (Two-Tower + Graph)
   - Parses ./data/RAW_recipes.csv (or recipes.csv) & ./data/interactions_train.csv
   - EDA plots
   - Train baseline & deep two-tower (in-batch softmax)
   - Demo compare baseline vs deep; optional graph re-rank
*/

import { TwoTowerModel } from './two-tower.js';
import { buildCoVisGraph, personalizedPageRank } from './graph.js';

/* --------------------------- Globals & State --------------------------- */
const state = {
  // data
  users: [],                   // array of raw user ids
  items: [],                   // [ {id, title, tagsIdx:Int32Array} ]
  userIdToIdx: new Map(),
  itemIdToIdx: new Map(),
  interactions: [],            // [{user, item, rating, ts?}]
  // tags
  tagToIdx: new Map(),         // string -> 0..T-1
  idxToTag: [],
  // training artifacts
  base: null,                  // TwoTowerModel
  deep: null,                  // TwoTowerModel
  itemTagRagged: null,         // per-item ragged for deep [{indices, splits}] (sliceable)
  itemEmbBase: null,           // tf.Tensor [I,D]
  itemEmbDeep: null,           // tf.Tensor [I,D]
  // graph
  graph: null,
  // metrics
  metrics: {
    loaded: false, users:0, items:0, inter:0,
    density: 0, coldUsers:0, coldItems:0, top1pctItems:0, paretoPct:0
  }
};

/* --------------------------- Tiny UI helpers --------------------------- */
const $ = (id)=>document.getElementById(id);
function setStatus(msg){ $('status').textContent = `Status: ${msg}`; }
function setStatus2(msg){ $('status2').textContent = msg; }
function fmt(n){ return n.toLocaleString(); }
function clamp(n,a,b){ return Math.max(a,Math.min(b,n)); }

/* --------------------------- CSV parsing --------------------------- */
// Robust enough for typical MovieLens/RAW_recipes shape
function parseCSV(text) {
  const out = [];
  let i=0, field='', row=[], inQ=false;
  while (i<text.length) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i+1] === '"'){ field+='"'; i+=2; continue; }
        inQ = false; i++; continue;
      }
      field += c; i++; continue;
    }
    if (c === '"'){ inQ = true; i++; continue; }
    if (c === ','){ row.push(field); field=''; i++; continue; }
    if (c === '\n' || c === '\r'){
      if (field.length>0 || row.length>0){ row.push(field); out.push(row); }
      field=''; row=[]; i++; if (c==='\r' && text[i]==='\n') i++; continue;
    }
    field += c; i++;
  }
  if (field.length>0 || row.length>0){ row.push(field); out.push(row); }
  return out;
}

/* --------------------------- Data loading --------------------------- */
async function fetchTextSafe(url) {
  const r = await fetch(url, {cache:'no-store'});
  if (!r.ok) throw new Error(`fetch failed: ${url}`);
  return await r.text();
}

function parseRecipesCSV(rows){
  // Expect columns: id,name,minutes,contributor_id,submitted,tags, ...
  const head = rows[0].map(h=>h.toLowerCase());
  const idxId   = head.indexOf('id');
  const idxName = head.indexOf('name');
  const idxTags = head.indexOf('tags'); // "['tag a','tag b', ...]"
  const items = [];
  for (let k=1;k<rows.length;k++){
    const r = rows[k]; if (!r || !r[idxId]) continue;
    const id = Number(r[idxId]);
    const title = (r[idxName]||`Recipe ${id}`).trim();
    // parse tags
    let tags = [];
    const cell = r[idxTags] || '';
    if (cell) {
      // Try to extract 'word' tokens inside brackets/quotes
      const m = cell.match(/[A-Za-z0-9_\-+ ]{2,}/g);
      if (m) tags = m.map(s=>s.trim().toLowerCase()).slice(0, 12);
    }
    items.push({id, title, tags});
  }
  return items;
}

function parseInteractionsCSV(rows){
  // Expect: user_id,recipe_id,rating(1..5), date(optional)
  const head = rows[0].map(h=>h.toLowerCase());
  const iu = head.indexOf('user_id');
  const ii = head.indexOf('recipe_id');
  const ir = head.indexOf('rating');
  const its= head.findIndex(x=>x.includes('date')||x.includes('timestamp'));
  const out = [];
  for (let k=1;k<rows.length;k++){
    const r = rows[k]; if (!r || !r[iu] || !r[ii] || !r[ir]) continue;
    out.push({
      user: Number(r[iu]),
      item: Number(r[ii]),
      rating: Number(r[ir]),
      ts: its>=0 ? Date.parse(r[its])||0 : 0
    });
  }
  return out;
}

function buildIndexers(items, interactions){
  const userSet = new Set(interactions.map(x=>x.user));
  const itemSet = new Set(items.map(x=>x.id));                 // only items we have metadata for
  const users = [...userSet].sort((a,b)=>a-b);
  const itemsArr = items.filter(it=>itemSet.has(it.id));
  const userIdToIdx = new Map(users.map((u,i)=>[u,i]));
  const itemIdToIdx = new Map(itemsArr.map((it,i)=>[it.id,i]));
  return { users, items: itemsArr, userIdToIdx, itemIdToIdx };
}

function buildTags(items, maxTagsKeep){
  // Count tags then keep top-K to cap memory
  const cnt = new Map();
  for (const it of items) for (const t of it.tags||[]) cnt.set(t, (cnt.get(t)||0)+1);
  const top = [...cnt.entries()].sort((a,b)=>b[1]-a[1]).slice(0, maxTagsKeep);
  const tagToIdx = new Map(top.map(([t],i)=>[t,i]));
  const idxToTag = top.map(([t])=>t);
  // For each item produce Int32Array of tag indices
  for (const it of items) {
    const idxs = (it.tags||[]).map(t=>tagToIdx.get(t)).filter(x=>x!=null);
    it.tagsIdx = new Int32Array(idxs);
  }
  return {tagToIdx, idxToTag};
}

function computeEDA(interactions, users, items){
  // Ratings histogram (1..5)
  const ratBins = [0,0,0,0,0];
  for (const r of interactions) if (r.rating>=1 && r.rating<=5) ratBins[r.rating-1]++;

  // Top tags
  const tagCnt = new Map();
  for (const it of items) for (const t of it.tags||[]) tagCnt.set(t, (tagCnt.get(t)||0)+1);
  const topTags = [...tagCnt.entries()].sort((a,b)=>b[1]-a[1]).slice(0, 20);

  // User activity (ratings per user)
  const byUser = new Map();
  for (const x of interactions) byUser.set(x.user, (byUser.get(x.user)||0)+1);
  const userDeg = [...byUser.values()];
  const userHist = binsFromArray(userDeg, [1,2,3,4,6,11,21,51,101,201,501]);

  // Item popularity (ratings per item)
  const byItem = new Map();
  for (const x of interactions) byItem.set(x.item, (byItem.get(x.item)||0)+1);
  const itemDeg = [...byItem.values()];
  const itemHist = binsFromArray(itemDeg, [1,2,3,4,6,11,21,51,101,201,501]);

  // Lorenz / Gini for item popularity
  const sorted = itemDeg.sort((a,b)=>a-b);
  const cum = []; let s=0, S = sorted.reduce((a,b)=>a+b,0)||1;
  for (let i=0;i<sorted.length;i++){ s += sorted[i]; cum.push(s/S); }
  const gini = 1 - 2*avg(cum);

  // Cold-start counts
  const coldUsers = userDeg.filter(x=>x<5).length;
  const coldItems = itemDeg.filter(x=>x<5).length;

  // Pareto: top 1% items cover what % of interactions?
  const itemCountsSortedDesc = itemDeg.slice().sort((a,b)=>b-a);
  const topK = Math.max(1, Math.floor(itemCountsSortedDesc.length*0.01));
  const cover = (itemCountsSortedDesc.slice(0, topK).reduce((a,b)=>a+b,0) / S) * 100;

  return { ratBins, topTags, userHist, itemHist, lorenz: cum, gini, coldUsers, coldItems, pareto: cover, top1pctItems: topK };
}

function binsFromArray(arr, edges){
  // edges like [1,2,3,4,6,11,21,...] as starts; produce ["1","2–2","3–3",...,"≥last"]
  const labels = [];
  for (let i=0;i<edges.length;i++){
    if (i===0) labels.push(`${edges[i]}`);
    else labels.push(`${edges[i-1]}–${edges[i]-1}`);
  }
  labels.push(`>${edges[edges.length-1]-1}`);
  const counts = new Array(labels.length).fill(0);
  for (const v of arr){
    let placed=false;
    for (let i=0;i<edges.length;i++){
      const lo = (i===0?edges[0]:edges[i-1]);
      const hi = (i===0?edges[0]:edges[i])-1;
      if (v>=lo && v<=hi){ counts[i]++; placed=true; break; }
    }
    if (!placed) counts[counts.length-1]++;
  }
  return {labels, counts};
}

/* --------------------------- Simple canvas charts --------------------------- */
function drawBars(canvasId, labels, values, opts={}){
  const c = $(canvasId), ctx = c.getContext('2d');
  const W = c.clientWidth, H = c.clientHeight; c.width=W; c.height=H;
  ctx.clearRect(0,0,W,H);
  const pad = 36, n = values.length, maxV = Math.max(1, Math.max(...values));
  const barW = (W - pad*2) / n * 0.8, gap = (W - pad*2)/n * 0.2;
  ctx.font='12px system-ui'; ctx.fillStyle='#cbd5e1'; ctx.textAlign='center';
  for (let i=0;i<n;i++){
    const x = pad + i*((W-pad*2)/n) + gap/2;
    const h = Math.max(2, (H-pad*2) * (values[i]/maxV));
    ctx.fillStyle = '#60a5fa';
    ctx.fillRect(x, H-pad-h, barW, h);
    ctx.fillStyle = '#9ca3af';
    const lab = (labels[i]??'').toString().slice(0,10);
    ctx.save(); ctx.translate(x+barW/2, H-8); ctx.rotate(-Math.PI/8); ctx.fillText(lab, 0, 0); ctx.restore();
  }
  // y-axis baseline
  ctx.strokeStyle='#374151'; ctx.beginPath(); ctx.moveTo(pad, H-pad); ctx.lineTo(W-pad, H-pad); ctx.stroke();
}

function drawLorenz(canvasId, cum){
  const c = $(canvasId), ctx = c.getContext('2d');
  const W = c.clientWidth, H = c.clientHeight; c.width=W; c.height=H;
  ctx.clearRect(0,0,W,H);
  const pad=36, n=cum.length;
  ctx.strokeStyle='#374151'; ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.lineTo(W-pad,pad); ctx.stroke();
  // line of equality
  ctx.strokeStyle='#4b5563'; ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,pad); ctx.stroke();
  // lorenz
  ctx.strokeStyle='#22d3ee'; ctx.beginPath(); ctx.moveTo(pad,H-pad);
  for (let i=0;i<n;i++){
    const x = pad + (W-pad*2) * (i/(n-1));
    const y = H - pad - (H-pad*2)*(cum[i]);
    ctx.lineTo(x,y);
  }
  ctx.stroke();
}

/* --------------------------- Training utils --------------------------- */
function makeBatches(interactions, batch, maxInter){
  const N = Math.min(maxInter || interactions.length, interactions.length);
  const order = [...Array(N).keys()];
  // shuffle
  for (let i=order.length-1;i>0;i--){ const j=(Math.random()* (i+1))|0; [order[i],order[j]]=[order[j],order[i]]; }
  const out=[];
  for (let k=0;k<N;k+=batch){
    const slice = order.slice(k, Math.min(N, k+batch));
    out.push(slice);
  }
  return out;
}

function raggedSliceAllItems(items, tagToIdx){
  // Build per-item ragged for DEEP inference/training
  // We store once globally as an array [{indices, splits}] for each per-batch build.
  const ragged = [];
  for (const it of items) {
    const idxs = it.tagsIdx || new Int32Array(0);
    ragged.push({indices: idxs, splits: new Int32Array([0, idxs.length])});
  }
  return ragged;
}

function buildRaggedForBatch(itemIdxArr, perItemRagged){
  // Concatenate ragged rows for a batch of item indices
  let total=0;
  for (const i of itemIdxArr){ total += perItemRagged[i].indices.length; }
  const indices = new Int32Array(total);
  const splits  = new Int32Array(itemIdxArr.length + 1);
  let off=0;
  for (let b=0;b<itemIdxArr.length;b++){
    const r = perItemRagged[itemIdxArr[b]];
    indices.set(r.indices, off);
    off += r.indices.length;
    splits[b+1] = off;
  }
  return {indices, splits};
}

/* --------------------------- PCA (SVD) for 2D projection --------------------------- */
function pca2D(tfMatrix){ // [N,D] -> [N,2]
  const X = tfMatrix.sub(tfMean(tfMatrix, 0));
  // covariance via SVD of X (faster than cov for tall matrices in JS)
  const {s, u, v} = tf.linalg.svd(X, true); // X = U * diag(s) * V^T
  const V2 = v.slice([0,0],[v.shape[0],2]); // [D,2]
  const proj = X.matMul(V2);                // [N,2]
  return proj;
}
function tfMean(t, axis){ return t.mean(axis, true); }

/* --------------------------- Demo helpers --------------------------- */
function topHistoryForUser(uRawId, k=10){
  const arr = state.interactions.filter(r=>r.user===uRawId)
               .sort((a,b)=> (b.rating-a.rating) || (b.ts-a.ts)).slice(0, k);
  return arr;
}
function pickRandomQualifiedUser(min=20){
  const byU = new Map();
  for (const r of state.interactions) byU.set(r.user,(byU.get(r.user)||0)+1);
  const qualified = [...byU.entries()].filter(([u,c])=>c>=min).map(([u])=>u);
  if (!qualified.length) return null;
  return qualified[(Math.random()*qualified.length)|0];
}
function excludeSeenAndSort(scores, seenSet, k=10){
  const arr=[];
  for (let i=0;i<scores.length;i++){
    if (!seenSet.has(i)) arr.push([i, scores[i]]);
  }
  arr.sort((a,b)=>b[1]-a[1]);
  return arr.slice(0,k);
}

/* --------------------------- Rendering helpers --------------------------- */
function fillTable(tbodyId, rows){
  const tb=$(tbodyId); tb.innerHTML='';
  rows.forEach((r,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>${i+1}</td><td>${r.title}</td><td class="small">${r.meta||''}</td>`;
    tb.appendChild(tr);
  });
}

/* --------------------------- Actions --------------------------- */
async function loadData(){
  try{
    $('btn-load').disabled = true;
    setStatus('loading…');
    // Recipes
    let recipesText = null;
    try { recipesText = await fetchTextSafe('./data/RAW_recipes.csv'); }
    catch { recipesText = await fetchTextSafe('./data/recipes.csv'); }
    const recipesRows = parseCSV(recipesText);
    const itemsRaw = parseRecipesCSV(recipesRows);

    // Interactions
    const interText = await fetchTextSafe('./data/interactions_train.csv');
    const interRows = parseCSV(interText);
    const interactionsRaw = parseInteractionsCSV(interRows);

    // Indexing
    const {users, items, userIdToIdx, itemIdToIdx} = buildIndexers(itemsRaw, interactionsRaw);

    // Keep only interactions whose items exist
    const inter = interactionsRaw.filter(x=>itemIdToIdx.has(x.item));

    // Tags (top-K only)
    const maxTagsKeep = 5000; // raw vocab cap before UI "maxTags" further trims deep features
    const {tagToIdx, idxToTag} = buildTags(items, maxTagsKeep);

    // Save
    Object.assign(state, {users, items, userIdToIdx, itemIdToIdx, interactions: inter,
                          tagToIdx, idxToTag});
    state.metrics.loaded = true;
    state.metrics.users = users.length;
    state.metrics.items = items.length;
    state.metrics.inter  = inter.length;

    // EDA
    const eda = computeEDA(inter, users, items);
    // density
    state.metrics.density = (inter.length / (users.length*items.length));
    state.metrics.coldUsers = eda.coldUsers;
    state.metrics.coldItems = eda.coldItems;
    state.metrics.top1pctItems = eda.top1pctItems;
    state.metrics.paretoPct = eda.pareto;

    // UI
    $('kv-users').textContent = `users: ${fmt(users.length)}`;
    $('kv-items').textContent = `items: ${fmt(items.length)}`;
    $('kv-inter').textContent = `interactions: ${fmt(inter.length)}`;
    setStatus2(`Users: ${fmt(users.length)}  Items: ${fmt(items.length)}  Interactions: ${fmt(inter.length)}  Density: ${eda ? eda.gini ? state.metrics.density.toExponential(2) : '—' : '—'}  Ratings present: yes  Cold users (<5): ${fmt(eda.coldUsers)}  Cold items (<5): ${fmt(eda.coldItems)}`);

    drawBars('chart-ratings', ['1','2','3','4','5'], eda.ratBins);
    drawBars('chart-tags', eda.topTags.map(x=>x[0]), eda.topTags.map(x=>x[1]));
    drawBars('chart-user-activity', eda.userHist.labels, eda.userHist.counts);
    drawBars('chart-item-activity', eda.itemHist.labels, eda.itemHist.counts);
    drawLorenz('chart-lorenz', eda.lorenz);
    $('gini').textContent = `Gini ≈ ${eda.gini.toFixed(3)}`;

    const coldHTML = `
      <tr><td>Cold users (&lt;5)</td><td>${fmt(eda.coldUsers)}</td></tr>
      <tr><td>Cold items (&lt;5)</td><td>${fmt(eda.coldItems)}</td></tr>
      <tr><td>Top 1% items (by interactions)</td><td>${fmt(eda.top1pctItems)}</td></tr>
      <tr><td>Pareto: % train covered by top 1%</td><td>${eda.pareto.toFixed(1)}%</td></tr>`;
    $('cold-table').innerHTML = coldHTML;

    setStatus(`loaded: users=${fmt(users.length)}, items=${fmt(items.length)} (file: data/RAW_recipes.csv), interactions=${fmt(inter.length)} (file: data/interactions_train.csv)`);

    // Precompute per-item ragged (deep)
    state.itemTagRagged = raggedSliceAllItems(state.items, state.tagToIdx);

    // Build co-vis graph (light)
    state.graph = buildCoVisGraph(state.interactions, state.itemIdToIdx, {alpha:0.8, maxNeighbors:64});

  } catch (e){
    console.error(e);
    setStatus('fetch failed. Ensure /data/*.csv exist (case-sensitive).');
  } finally {
    $('btn-load').disabled = false;
  }
}

async function train(which='baseline'){
  if (!state.metrics.loaded){ setStatus('load data first.'); return; }
  const embDim = clamp(parseInt($('embDim').value,10)||32, 8, 128);
  const epochs = clamp(parseInt($('epochs').value,10)||5, 1, 50);
  const batch  = clamp(parseInt($('batch').value,10)||256, 32, 2048);
  const lr     = clamp(Number($('lr').value)||0.001, 1e-4, 0.02);
  const maxInter = clamp(parseInt($('maxInter').value,10)||80000, 5000, state.interactions.length);
  const maxTagFeatures = clamp(parseInt($('maxTags').value,10)||200, 10, state.idxToTag.length);

  const canvas = (which==='baseline') ? $('loss-base') : $('loss-deep');
  const note   = (which==='baseline') ? $('base-note') : $('deep-note');
  canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
  const ctx = canvas.getContext('2d');
  function plotLoss(step, total, loss){
    const W=canvas.width, H=canvas.height, pad=24;
    const x = pad + (W-pad*2)* (step/Math.max(1,total-1));
    const y = H-pad - (H-pad*2)* Math.min(1, loss / 8); // scale
    if (step===0){ ctx.clearRect(0,0,W,H); ctx.strokeStyle='#374151'; ctx.strokeRect(pad,pad,W-pad*2,H-pad*2); ctx.beginPath(); ctx.moveTo(x,y); }
    else { ctx.strokeStyle='#22d3ee'; ctx.lineTo(x,y); ctx.stroke(); }
  }

  // Build model
  if (which==='baseline') {
    // Dispose old variables if any
    if (state.base) { state.base.userEmbedding.dispose(); state.base.itemEmbedding.dispose(); }
    state.base = new TwoTowerModel({
      numUsers: state.users.length, numItems: state.items.length,
      embDim, lr, mode:'baseline'
    });
  } else {
    if (state.deep) {
      // dispose previous deep
      state.deep.userEmbedding.dispose(); state.deep.itemEmbedding.dispose();
      if (state.deep.tagEmbedding) state.deep.tagEmbedding.dispose();
      if (state.deep.outW) state.deep.outW.dispose();
      if (state.deep.outB) state.deep.outB.dispose();
      if (state.deep.mlpWeights) for (const {w,b} of state.deep.mlpWeights){ w.dispose(); b.dispose(); }
    }
    // Remap tag ids to 0..maxTagFeatures-1 by frequency (they already are in idx order)
    const maxTagId = Math.min(maxTagFeatures-1, state.idxToTag.length-1);
    state.deep = new TwoTowerModel({
      numUsers: state.users.length, numItems: state.items.length,
      embDim, lr, mode:'deep', maxTagId, tagDim: Math.max(8, Math.floor(embDim/2)), mlpHidden:[Math.max(32, embDim)]
    });
  }

  // Prepare index tensors per interaction once
  const toIdx = (rawU, rawI) => ({
    u: state.userIdToIdx.get(rawU),
    i: state.itemIdToIdx.get(rawI)
  });

  const batches = makeBatches(state.interactions, batch, maxInter);
  const totalSteps = epochs * batches.length;
  let step = 0; const t0 = performance.now();

  for (let e=0; e<epochs; e++){
    for (const slice of batches){
      const uIdx = new Int32Array(slice.length);
      const iIdx = new Int32Array(slice.length);
      for (let b=0;b<slice.length;b++){
        const r = state.interactions[slice[b]];
        const ids = toIdx(r.user, r.item);
        uIdx[b]=ids.u; iIdx[b]=ids.i;
      }
      let loss;
      if (which==='baseline') {
        loss = state.base.trainStep({uIdx, iIdx});
      } else {
        const ragged = buildRaggedForBatch(iIdx, state.itemTagRagged);
        loss = state.deep.trainStep({uIdx, iIdx, tagBag: ragged});
      }
      plotLoss(step, totalSteps, loss);
      step++;
      if (step%15===0) await tf.nextFrame(); // keep UI responsive
    }
    note.textContent = `epoch ${e+1}/${epochs} done`;
  }

  // Cache item embeddings for fast scoring + projection
  if (which==='baseline') {
    if (state.itemEmbBase) state.itemEmbBase.dispose();
    state.itemEmbBase = state.base.getAllItemEmbeddings();
  } else {
    if (state.itemEmbDeep) state.itemEmbDeep.dispose();
    state.itemEmbDeep = state.deep.getAllItemEmbeddings(state.itemTagRagged);
  }

  // PCA projection (sample up to 1000 items)
  const mat = (which==='baseline') ? state.itemEmbBase : state.itemEmbDeep;
  const sampleN = Math.min(1000, mat.shape[0]);
  const proj = pca2D(mat.slice([0,0],[sampleN, mat.shape[1]]));
  drawScatter2D('proj2d', await proj.array(), state.items.slice(0,sampleN).map(x=>x.title));
  proj.dispose();

  const ms = (performance.now()-t0)|0;
  note.textContent = `Training done in ${(ms/1000).toFixed(1)}s.`;
  $('metrics').innerHTML =
    `<div>Model: <b>${which==='baseline'?'Baseline (ID embeddings)':'Deep (MLP + tags)'}</b></div>
     <div class="muted small">embDim=${embDim}, epochs=${epochs}, batch=${batch}, lr=${lr}, maxInter=${fmt(maxInter)}</div>
     <div>Items embedded: ${fmt(state.items.length)} · Users embedded: ${fmt(state.users.length)}</div>`;
}

function drawScatter2D(canvasId, pts, labels){
  const c=$(canvasId), ctx=c.getContext('2d');
  const W=c.clientWidth, H=c.clientHeight; c.width=W; c.height=H;
  ctx.clearRect(0,0,W,H);
  if (!pts.length) return;
  // normalize to [pad,W-pad]
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
  const pad=30;
  for (let i=0;i<pts.length;i++){
    const x = pad + (W-pad*2)*( (pts[i][0]-minX)/(maxX-minX+1e-9) );
    const y = H - pad - (H-pad*2)*( (pts[i][1]-minY)/(maxY-minY+1e-9) );
    ctx.beginPath(); ctx.arc(x,y,2.2,0,Math.PI*2); ctx.fillStyle='#93c5fd'; ctx.fill();
  }
}

/* --------------------------- Demo (Test) --------------------------- */
async function runTest(){
  if (!state.base && !state.deep){ $('test-note').textContent = 'Train at least one model first.'; return; }
  const uRaw = pickRandomQualifiedUser(20);
  if (!uRaw){ $('test-note').textContent = 'No user with ≥20 ratings in this split.'; return; }
  const uIdx  = state.userIdToIdx.get(uRaw);
  const seen  = new Set(state.interactions.filter(r=>r.user===uRaw).map(r=>state.itemIdToIdx.get(r.item)));

  // History table
  const history = topHistoryForUser(uRaw, 10).map(x=>{
    const it = state.items[state.itemIdToIdx.get(x.item)];
    return {title: it?.title || `Recipe ${x.item}`, meta: `★${x.rating}`};
  });
  fillTable('hist-body', history);

  // Baseline scores
  let baseRecs=[];
  if (state.base) {
    const uEmb = state.base.getUserEmbedding(uIdx);                     // [D]
    const scores = await tf.matMul(state.itemEmbBase, uEmb.reshape([state.itemEmbBase.shape[1],1]), false, false).reshape([state.itemEmbBase.shape[0]]).array();
    baseRecs = excludeSeenAndSort(scores, seen, 10).map(([i,s])=>({title: state.items[i].title, meta: s.toFixed(3)}));
    uEmb.dispose();
  }
  fillTable('base-recs', baseRecs);

  // Deep scores
  let deepRecs=[];
  if (state.deep) {
    const uEmb = state.deep.getUserEmbedding(uIdx);
    const scores = await tf.matMul(state.itemEmbDeep, uEmb.reshape([state.itemEmbDeep.shape[1],1]), false, false).reshape([state.itemEmbDeep.shape[0]]).array();
    deepRecs = excludeSeenAndSort(scores, seen, 10).map(([i,s])=>({title: state.items[i].title, meta: s.toFixed(3)}));
    uEmb.dispose();
  }
  const useGraph = $('use-graph').checked;
  if (useGraph && state.graph) {
    // Re-rank deep (if exists) or baseline using PPR from user's history seeds
    const seeds = history.map(h => state.itemIdToIdx.get(state.items.find(it=>it.title===h.title).id)).filter(x=>x!=null).slice(0,8);
    const rank = personalizedPageRank(state.graph, seeds, {iters:30, d:0.85});
    function rerank(list){
      return list.map(r=>{
        const idx = state.itemIdToIdx.get(state.items.find(it=>it.title===r.title).id);
        const boost = rank[idx] || 0;
        return {...r, meta: `${r.meta} • g=${boost.toFixed(3)}`, _score: Number(r.meta) + 0.2*boost};
      }).sort((a,b)=>b._score - a._score).map(({_score, ...rest})=>rest);
    }
    if (deepRecs.length) deepRecs = rerank(deepRecs);
    else if (baseRecs.length) baseRecs = rerank(baseRecs);
  }
  fillTable('deep-recs', deepRecs);

  $('test-note').textContent = 'Recommendations generated successfully!';
}

/* --------------------------- Tabs & Wire-up --------------------------- */
function switchTab(id){
  document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active', b.dataset.tab===id));
  document.querySelectorAll('.tabpage').forEach(p=>p.style.display = (p.id===id)?'block':'none');
}

window.addEventListener('DOMContentLoaded', ()=>{
  // Tabs
  document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click', ()=>switchTab(b.dataset.tab)));
  // Buttons
  $('btn-load').addEventListener('click', loadData);
  $('btn-train-base').addEventListener('click', ()=>train('baseline'));
  $('btn-train-deep').addEventListener('click', ()=>train('deep'));
  $('btn-test').addEventListener('click', runTest);
  setStatus('—');
});

/* --------------------------- Small helpers --------------------------- */
function avg(a){ return a.reduce((x,y)=>x+y,0)/Math.max(1,a.length); }
