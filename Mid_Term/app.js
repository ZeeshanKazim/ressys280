/* app.js — All UI, data loading, EDA, training orchestration, testing.
   IMPORTANT: This file assumes it's served from the SAME folder as:
   - RAW_recipes.csv (or recipes.csv)
   - interactions_train.csv
   - two-tower.js (this file uses the TwoTowerModel class defined there)
*/

/* ===========================
   Helpers & DOM shortcuts
=========================== */
const $ = (id) => document.getElementById(id);
const fmt = (n, d=0) => Number(n).toFixed(d);
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

function setActive(tab) {
  document.querySelectorAll('.tab').forEach(t=>{
    t.classList.toggle('active', t.dataset.tab===tab);
  });
  document.querySelectorAll('.view').forEach(v=>{
    v.classList.toggle('active', v.id===tab);
  });
}

/* Tabs */
window.addEventListener('DOMContentLoaded', ()=>{
  document.querySelectorAll('.tab').forEach(t=>{
    t.addEventListener('click', ()=> setActive(t.dataset.tab));
  });
});

/* ===========================
   Global state
=========================== */
let items = [];              // [{id,title,tags:Array<string>}]
let interactions = [];       // [{user,item,rating,ts}]
let numUsers=0, numItems=0;

const userIdToIdx = new Map(), itemIdToIdx = new Map();
let idxToUserId = [], idxToItemId = [];

const ratingsByUser = new Map(); // userId -> [{item, rating, ts}]
const ratingsByItem = new Map(); // itemId -> count

let baselineModel = null;
let deepModel = null;
let itemTagVocab = [];            // Array of strings (top K)
let itemTagMultiHot = null;       // Float32Array length = numItems * K (filled after vocab built)

/* ===========================
   Robust local fetch
=========================== */
async function smartFetch(candidates) {
  for (const path of candidates) {
    try {
      const r = await fetch(path, {cache:'no-cache'});
      if (r.ok) return await r.text();
    } catch(e){ /* ignore */ }
  }
  throw new Error(`fetch failed. Ensure ${candidates.join(' OR ')} exist (case-sensitive).`);
}

/* Basic CSV parser (handles quoted fields) */
function parseCSV(text) {
  const rows = [];
  let row = [], val = '', inQ = false;
  for (let i=0;i<text.length;i++){
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i+1] === '"'){ val+='"'; i++; }
        else inQ=false;
      } else val+=c;
    } else {
      if (c === '"') inQ = true;
      else if (c === ','){ row.push(val); val=''; }
      else if (c === '\n' || c === '\r'){
        // handle \r\n
        if (c === '\r' && text[i+1]==='\n') i++;
        row.push(val); val='';
        if (row.some(x=>x.length)) rows.push(row);
        row = [];
      } else val+=c;
    }
  }
  if (val.length||row.length) { row.push(val); rows.push(row); }
  return rows;
}

/* ===========================
   Load & Parse
=========================== */
async function loadData() {
  const status = $('loadStatus');
  status.textContent = 'Status: loading…';

  // 1) Items
  const itemText = await smartFetch(['./RAW_recipes.csv','./recipes.csv']);
  // 2) Interactions
  const interText = await smartFetch(['./interactions_train.csv']);

  // Parse items
  const itemRows = parseCSV(itemText);
  const header = itemRows[0].map(h=>h.trim().toLowerCase());
  const idIdx = Math.max(header.indexOf('id'), header.indexOf('recipe_id'));
  const nameIdx = Math.max(header.indexOf('name'), header.indexOf('title'));
  const tagsIdx = header.indexOf('tags');

  const idSeen = new Set();
  items = [];
  for (let r=1; r<itemRows.length; r++){
    const row = itemRows[r];
    const id = parseInt(row[idIdx],10);
    if (!Number.isFinite(id) || idSeen.has(id)) continue;
    idSeen.add(id);
    const title = (nameIdx>=0? row[nameIdx] : `Recipe ${id}`) || `Recipe ${id}`;
    let tags = [];
    if (tagsIdx>=0 && row[tagsIdx]){
      // RAW_recipes has python-literal list: "['tag1','tag2', ...]"
      const s = row[tagsIdx].trim();
      if (s.startsWith('[') && s.endsWith(']')) {
        const inner = s.slice(1,-1);
        tags = inner.split(',').map(x=>x.replace(/^[\s'"]+|[\s'"]+$/g,'').trim()).filter(Boolean);
      } else {
        // already comma separated
        tags = s.split('|').map(t=>t.trim()).filter(Boolean);
      }
    }
    items.push({id, title, tags});
  }

  // Parse interactions
  const interRows = parseCSV(interText);
  const h2 = interRows[0].map(h=>h.trim().toLowerCase());
  const uIdx = Math.max(h2.indexOf('user_id'), h2.indexOf('user'));
  const iIdx = Math.max(h2.indexOf('recipe_id'), h2.indexOf('item'), h2.indexOf('item_id'));
  const rIdx = Math.max(h2.indexOf('rating'), h2.indexOf('score'));
  const tIdx = Math.max(h2.indexOf('date'), h2.indexOf('timestamp'), h2.indexOf('ts'));

  interactions = [];
  for (let r=1; r<interRows.length; r++){
    const row = interRows[r];
    const user = parseInt(row[uIdx],10);
    const item = parseInt(row[iIdx],10);
    const rating = parseFloat(row[rIdx] ?? '0');
    if (!Number.isFinite(user) || !Number.isFinite(item)) continue;
    const tsRaw = row[tIdx] || '';
    const ts = Date.parse(tsRaw) || 0;
    interactions.push({user, item, rating: Math.max(1, Math.min(5, rating || 0)), ts});
  }

  // Indexers
  const users = [...new Set(interactions.map(d=>d.user))].sort((a,b)=>a-b);
  const itemIds = [...new Set(items.map(d=>d.id))].sort((a,b)=>a-b);

  userIdToIdx.clear(); itemIdToIdx.clear();
  users.forEach((u,idx)=> userIdToIdx.set(u, idx));
  itemIds.forEach((i,idx)=> itemIdToIdx.set(i, idx));
  idxToUserId = users; idxToItemId = itemIds;
  numUsers = users.length; numItems = itemIds.length;

  // Ratings per user/item
  ratingsByUser.clear(); ratingsByItem.clear();
  for (const it of interactions){
    if (!ratingsByUser.has(it.user)) ratingsByUser.set(it.user, []);
    ratingsByUser.get(it.user).push({item: it.item, rating: it.rating, ts: it.ts});
    ratingsByItem.set(it.item, (ratingsByItem.get(it.item) || 0) + 1);
  }

  // Compose status line
  $('datasetLine').textContent =
    `Users: ${numUsers.toLocaleString()}  Items: ${numItems.toLocaleString()}  Interactions: ${interactions.length.toLocaleString()}`;

  $('loadStatus').textContent = `Status: loaded. users=${numUsers}, items=${numItems} (file: ${header.includes('name')?'RAW_recipes.csv':'recipes.csv'}), interactions=${interactions.length} (file: interactions_train.csv)`;

  // Draw EDA
  drawEDA();

  // Enable model buttons
  $('btnTrainBaseline').disabled = false;
  $('btnTrainDeep').disabled = false;
  $('btnTest').disabled = false;
}

/* ===========================
   EDA (simple canvas charts)
=========================== */
function drawBars(canvas, labels, values, yLabel='') {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0,0,canvas.clientWidth, canvas.clientHeight);
  const w = canvas.clientWidth, h = canvas.clientHeight;

  const maxV = Math.max(1, ...values);
  const left = 40, right = 10, top = 10, bottom = 26;
  const chartW = w - left - right;
  const chartH = h - top - bottom;

  // Axes
  ctx.strokeStyle = '#1f2937';
  ctx.beginPath();
  ctx.moveTo(left, top); ctx.lineTo(left, top+chartH); ctx.lineTo(left+chartW, top+chartH);
  ctx.stroke();

  const n = values.length;
  const bw = chartW / n * 0.7;
  for (let i=0;i<n;i++){
    const x = left + (i+0.5) * (chartW/n) - bw/2;
    const y = top + chartH - (values[i] / maxV) * chartH;
    ctx.fillStyle = '#60a5fa';
    ctx.fillRect(x, y, bw, top+chartH - y);

    // x tick
    ctx.fillStyle = '#9ca3af';
    ctx.textAlign = 'center';
    ctx.font = '12px system-ui';
    ctx.fillText(String(labels[i]), left + (i+0.5)*(chartW/n), h-8);
  }

  // y max
  ctx.fillStyle = '#9ca3af';
  ctx.textAlign = 'right';
  ctx.fillText(maxV.toLocaleString(), left-6, top+10);
  if (yLabel){
    ctx.save();
    ctx.translate(10, top+chartH/2); ctx.rotate(-Math.PI/2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
}

function topTags(items, k=20){
  const map = new Map();
  for (const it of items){
    for (const t of it.tags || []) map.set(t, (map.get(t)||0)+1);
  }
  return [...map.entries()].sort((a,b)=> b[1]-a[1]).slice(0,k);
}

function histogram(vals, bins){
  const counts = new Array(bins.length).fill(0);
  for (const v of vals){
    for (let i=0;i<bins.length;i++){
      const [lo, hi] = bins[i];
      if (v>=lo && v<hi){ counts[i]++; break; }
    }
  }
  return counts;
}

function lorenzAndGini(freqs){
  const total = freqs.reduce((a,b)=>a+b,0);
  const sorted = [...freqs].sort((a,b)=>a-b);
  let cum = 0, area = 0;
  for (let i=0;i<sorted.length;i++){
    const prev = cum/total;
    cum += sorted[i];
    const next = cum/total;
    area += (prev + next)/2 * (1/sorted.length);
  }
  const gini = 1 - 2*area;
  return {gini};
}

function drawEDA(){
  // Ratings histogram
  const ratings = interactions.map(d=>d.rating);
  const rCounts = [1,2,3,4,5].map(x => ratings.filter(v=>v===x).length);
  drawBars($('cRatings'), [1,2,3,4,5], rCounts, 'count');

  // Top tags
  const tagTop = topTags(items, 20);
  drawBars($('cTags'), tagTop.map(d=>d[0]), tagTop.map(d=>d[1]), 'count');

  // User activity bins
  const byUserCounts = [...ratingsByUser.values()].map(a=>a.length);
  const uBins = [[1,2],[2,3],[3,4],[4,6],[6,11],[11,21],[21,51],[51,101],[101,201],[201,501],[501,Infinity]];
  const uLab = ['1','2–2','3–3','4–5','6–10','11–20','21–50','51–100','101–200','201–500','>500'];
  drawBars($('cUserActivity'), uLab, histogram(byUserCounts, uBins), 'users');

  // Item popularity bins
  const byItemCounts = [...ratingsByItem.values()];
  const iBins = uBins, iLab = uLab;
  drawBars($('cItemPopularity'), iLab, histogram(byItemCounts, iBins), 'items');

  // Lorenz (use simple Gini from item popularity)
  const {gini} = lorenzAndGini(byItemCounts.length? byItemCounts : [1]);
  $('gini').textContent = `Gini ≈ ${fmt(gini,3)}`;

  // Cold-start counts
  const coldUsers = byUserCounts.filter(c=>c<5).length;
  const coldItems = byItemCounts.filter(c=>c<5).length;
  const top1pctCut = Math.max(1, Math.floor(0.01 * numItems));
  const sortedByInteractions = [...ratingsByItem.entries()].sort((a,b)=>b[1]-a[1]);
  const topK = sortedByInteractions.slice(0, top1pctCut);
  const covered = new Set();
  for (const [iid] of topK){
    for (const r of interactions){ if (r.item===iid) covered.add(`${r.user}-${r.item}-${r.ts}`); }
  }
  const pareto = (covered.size / interactions.length) * 100;

  $('coldTbl').innerHTML = `
    <tr><td>Cold users (&lt;5)</td><td>${coldUsers.toLocaleString()}</td></tr>
    <tr><td>Cold items (&lt;5)</td><td>${coldItems.toLocaleString()}</td></tr>
    <tr><td>Top 1% items (by interactions)</td><td>${top1pctCut.toLocaleString()}</td></tr>
    <tr><td>Pareto: % train covered by top 1%</td><td>${fmt(pareto,1)}%</td></tr>
  `;
}

/* ===========================
   Item tag vocabulary (for Deep tower)
=========================== */
function buildTagVocab(K=200){
  const counts = new Map();
  for (const it of items){ for (const t of it.tags||[]) counts.set(t, (counts.get(t)||0)+1); }
  itemTagVocab = [...counts.entries()].sort((a,b)=>b[1]-a[1]).slice(0, K).map(d=>d[0]);

  // Multi-hot per item
  const Kdim = itemTagVocab.length;
  const map = new Map(itemTagVocab.map((t,i)=>[t,i]));
  itemTagMultiHot = new Float32Array(numItems * Kdim);
  for (let idx=0; idx<numItems; idx++){
    const itemId = idxToItemId[idx];
    const it = items.find(x=>x.id===itemId);
    if (!it || !it.tags) continue;
    for (const t of it.tags){
      const j = map.get(t);
      if (j!=null) itemTagMultiHot[idx*Kdim + j] = 1;
    }
  }
  return Kdim;
}

/* ===========================
   Batching utilities
=========================== */
function* makeBatches(maxN, batchSize){
  // choose up to maxN interactions, shuffled
  const N = Math.min(maxN, interactions.length);
  const idx = [...Array(N).keys()];
  for (let i=idx.length-1;i>0;i--){ const j=(Math.random()* (i+1))|0; [idx[i],idx[j]]=[idx[j],idx[i]]; }
  for (let s=0; s<idx.length; s+=batchSize){
    const slice = idx.slice(s, s+batchSize);
    const u = new Int32Array(slice.length);
    const v = new Int32Array(slice.length);
    for (let k=0;k<slice.length;k++){
      const r = interactions[slice[k]];
      u[k] = userIdToIdx.get(r.user);
      v[k] = itemIdToIdx.get(r.item);
    }
    yield {u, v};
  }
}

/* ===========================
   Loss plotting
=========================== */
function drawLoss(canvas, values){
  const labels = values.map((_,i)=>i+1);
  drawBars(canvas, labels, values, 'loss');
}

/* ===========================
   Training orchestration
=========================== */
async function trainBaseline() {
  if (!numUsers || !numItems){ alert('Load data first'); return; }
  $('bMsg').textContent = 'Building baseline…';
  baselineModel?.dispose();
  baselineModel = new TwoTowerModel({mode:'baseline', numUsers, numItems, embDim: parseInt($('bEmb').value,10), lr: parseFloat($('bLR').value)});
  const epochs = parseInt($('bEp').value,10);
  const batch = parseInt($('bBatch').value,10);
  const maxInter = parseInt($('bMax').value,10);

  const losses = [];
  for (let ep=0; ep<epochs; ep++){
    let step=0;
    for (const {u,v} of makeBatches(maxInter, batch)){
      const loss = await baselineModel.trainStep(u, v);
      if (step%10===0){ $('bMsg').textContent = `Training… epoch ${ep+1}/${epochs}, step ${step}, loss ${fmt(loss,4)}`; await sleep(0); }
      step++;
      losses.push(loss);
    }
  }
  $('bMsg').textContent = `Baseline done. Final loss ${fmt(losses.at(-1),4)}`;
  drawLoss($('cLossBaseline'), losses);

  // PCA projection of item embeddings
  await projectItems(baselineModel);
}

async function trainDeep() {
  if (!numUsers || !numItems){ alert('Load data first'); return; }
  $('dMsg').textContent = 'Preparing tags…';
  const K = buildTagVocab(parseInt($('dTags').value,10));

  deepModel?.dispose();
  deepModel = new TwoTowerModel({
    mode:'deep', numUsers, numItems,
    embDim: parseInt($('dEmb').value,10),
    lr: parseFloat($('dLR').value),
    itemFeatDim: K
  });
  deepModel.setItemFeatureMatrix(itemTagMultiHot); // Float32Array length numItems*K

  const epochs = parseInt($('dEp').value,10);
  const batch = parseInt($('dBatch').value,10);
  const maxInter = parseInt($('bMax').value,10); // reuse cap

  const losses = [];
  for (let ep=0; ep<epochs; ep++){
    let step=0;
    for (const {u,v} of makeBatches(maxInter, batch)){
      const loss = await deepModel.trainStep(u, v);
      if (step%10===0){ $('dMsg').textContent = `Training deep… epoch ${ep+1}/${epochs}, step ${step}, loss ${fmt(loss,4)}`; await sleep(0); }
      step++;
      losses.push(loss);
    }
  }
  $('dMsg').textContent = `Deep done. Final loss ${fmt(losses.at(-1),4)}`;
  drawLoss($('cLossDeep'), losses);

  await projectItems(deepModel);
}

/* ===========================
   PCA projection using tf.svd
=========================== */
async function projectItems(model){
  try {
    const all = tf.tidy(()=> model.getAllItemEmbeddings()); // [numItems, embDim]
    const sample = 1000;
    const idx = tf.util.createShuffledIndices(numItems).slice(0, Math.min(sample, numItems));
    const X = tf.gather(all, idx);
    const { u, s, v } = tf.linalg.svd(X, true); // X ≈ U S V^T
    const comps = v.slice([0,0],[2, v.shape[1]]); // top-2 rows
    const proj = tf.matMul(X, comps.transpose()); // [n,2]
    const pts = await proj.array();
    all.dispose(); X.dispose(); u.dispose(); s.dispose(); v.dispose(); comps.dispose(); proj.dispose();

    const cv = $('cPCA');
    const ctx = cv.getContext('2d');
    const W = cv.width = cv.clientWidth * devicePixelRatio;
    const H = cv.height = cv.clientHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.clearRect(0,0,cv.clientWidth, cv.clientHeight);

    // normalize
    const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
    const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
    function mapx(x){ return ((x-minX)/(maxX-minX))* (cv.clientWidth-40) + 20; }
    function mapy(y){ return ((y-minY)/(maxY-minY))* (cv.clientHeight-40) + 20; }

    ctx.fillStyle = '#60a5fa';
    for (const p of pts){
      const x = mapx(p[0]), y = mapy(p[1]);
      ctx.fillRect(x-1.5, cv.clientHeight-y-1.5, 3, 3);
    }
  } catch(e){
    console.warn('PCA projection failed:', e);
  }
}

/* ===========================
   Demo: recommendations
=========================== */
function renderRows(tbody, rows, score=false){
  tbody.innerHTML = rows.map((r,i)=>{
    const s = score? `<td>${fmt(r.score,4)}</td>` : `<td>${fmt(r.rating,1)}</td>`;
    return `<tr><td>${i+1}</td><td>${r.title}</td>${s}</tr>`;
  }).join('');
}

async function testOnce(){
  if (!baselineModel && !deepModel){ alert('Train at least one model first.'); return; }
  $('demoMsg').textContent = 'Picking user…';

  const eligible = idxToUserId.filter(u => (ratingsByUser.get(u)?.length || 0) >= 20);
  if (!eligible.length){ $('demoMsg').textContent = 'No user with ≥20 ratings in this split.'; return; }
  const userId = eligible[(Math.random()*eligible.length)|0];
  const seen = new Set(ratingsByUser.get(userId).map(r=>r.item));

  // History top-10
  const hist = ratingsByUser.get(userId)
    .slice().sort((a,b)=> b.rating - a.rating || b.ts - a.ts)
    .slice(0,10)
    .map(x=>({ title: items.find(it=>it.id===x.item)?.title || `Recipe ${x.item}`, rating:x.rating }));

  renderRows($('tblHistory'), hist, false);

  const useGraph = $('useGraph').checked && typeof window.personalizedPageRank === 'function';
  $('demoMsg').textContent = useGraph? 'Scoring (with graph re-rank)…' : 'Scoring…';

  // Compute scores per model
  const uIdx = userIdToIdx.get(userId);
  const candidateIdx = [];
  for (let i=0;i<numItems;i++){
    const iid = idxToItemId[i];
    if (!seen.has(iid)) candidateIdx.push(i);
  }

  const results = [];
  if (baselineModel){
    const scores = await baselineModel.scoreUserAgainstItems(uIdx, candidateIdx);
    results.push({kind:'base', scores});
  }
  if (deepModel){
    const scores = await deepModel.scoreUserAgainstItems(uIdx, candidateIdx);
    results.push({kind:'deep', scores});
  }

  // Optional: graph re-rank (PPR) — requires graph.js
  if (useGraph){
    try{
      const userSeenList = ratingsByUser.get(userId).map(r=>r.item);
      const graphTop = (arr) => {
        const top = window.personalizedPageRank
          ? window.personalizedPageRank(userSeenList, candidateIdx.map(i=>idxToItemId[i]))
          : null;
        if (top){
          const boost = new Map(top.map((id,rankIdx)=> [id, 1.0/(rankIdx+1)]));
          for (const r of results){
            for (let k=0;k<r.scores.length;k++){
              const itemId = idxToItemId[candidateIdx[k]];
              const b = boost.get(itemId) || 0;
              r.scores[k] += 0.05 * b; // tiny re-rank boost
            }
          }
        }
      };
      graphTop(results[0]?.scores || []);
    } catch(e){ console.warn('Graph re-rank failed:', e); }
  }

  // Render tables
  const topK = 10;
  for (const r of results){
    const scored = candidateIdx.map((idx, i)=> ({ idx, score: r.scores[i] }));
    scored.sort((a,b)=> b.score - a.score);
    const rows = scored.slice(0, topK).map(s=>{
      const itemId = idxToItemId[s.idx];
      return { title: items.find(it=>it.id===itemId)?.title || `Recipe ${itemId}`, score: s.score };
    });
    if (r.kind==='base') renderRows($('tblBase'), rows, true);
    if (r.kind==='deep') renderRows($('tblDeep'), rows, true);
  }

  $('demoMsg').textContent = 'Status: recommendations generated successfully!';
}

/* ===========================
   Wire up buttons
=========================== */
window.addEventListener('DOMContentLoaded', ()=>{
  $('btnLoad').addEventListener('click', loadData);
  $('btnTrainBaseline').addEventListener('click', ()=>trainBaseline().catch(e=>{console.error(e); $('bMsg').textContent = `Error: ${e.message}`;}));
  $('btnTrainDeep').addEventListener('click', ()=>trainDeep().catch(e=>{console.error(e); $('dMsg').textContent = `Error: ${e.message}`;}));
  $('btnTest').addEventListener('click', ()=>testOnce().catch(e=>{console.error(e); $('demoMsg').textContent = `Error: ${e.message}`;}));

  // Initially disable actions until data is loaded
  $('btnTrainBaseline').disabled = true;
  $('btnTrainDeep').disabled = true;
  $('btnTest').disabled = true;
});
