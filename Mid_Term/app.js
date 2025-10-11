// app.js — glue code (loading, EDA, UI, training, demo)

///////////////////////////
// Globals (shared state)
///////////////////////////
let RECIPES = new Map(); // itemId -> {title, tags:Set<string>}
let INTERACTIONS = [];   // {user, item, rating}
let USERS = new Map();   // userId -> [{item,rating}]
let ITEMS = new Map();   // itemId -> ratings count

let idxUser = new Map(); // id -> 0..U-1
let idxItem = new Map(); // id -> 0..I-1
let revUser = [];        // index -> id
let revItem = [];        // index -> id

let Baseline = null;     // TwoTower baseline instance
let Deep = null;         // TwoTower deep instance
let lastItemProj = null; // [[x,y], ...] for plotting

const $ = (q) => document.querySelector(q);

///////////////////////////
// Tiny tab router
///////////////////////////
document.querySelectorAll('.tab').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const which = btn.dataset.tab;
    document.querySelectorAll('main > section').forEach(s=>s.style.display='none');
    $('#tab-'+which).style.display='block';
  });
});

///////////////////////////
// CSV utilities
///////////////////////////
function parseCSV(text) {
  // Simple CSV parser that handles quoted fields with commas
  const rows = [];
  let i=0, field='', row=[], inQ=false;
  while (i <= text.length) {
    const c = text[i] ?? '\n';
    if (inQ) {
      if (c === '"') {
        if (text[i+1] === '"') { field += '"'; i++; }
        else inQ = false;
      } else field += c;
    } else {
      if (c === '"') inQ = true;
      else if (c === ',') { row.push(field); field=''; }
      else if (c === '\r') { /* skip */ }
      else if (c === '\n') { row.push(field); rows.push(row); field=''; row=[]; }
      else field += c;
    }
    i++;
  }
  // trim trailing empty row if any
  if (rows.length && rows[rows.length-1].length === 1 && rows[rows.length-1][0] === '') rows.pop();
  return rows;
}

function parseTagList(raw) {
  // RAW_recipes.csv tags look like "['weeknight','vegetarian', ...]"
  if (!raw || raw === '[]') return [];
  let s = raw.trim();
  // Make it JSON-parseable
  s = s.replaceAll("'", '"').replaceAll('None', 'null');
  try {
    const arr = JSON.parse(s);
    return Array.isArray(arr) ? arr.map(t=>String(t).toLowerCase().trim()).filter(Boolean) : [];
  } catch {
    // Fallback: split on commas inside brackets
    return s.replace(/^\[|\]$/g,'').split(',').map(x=>x.replace(/["']/g,'').trim().toLowerCase()).filter(Boolean);
  }
}

///////////////////////////
// Drawing helpers
///////////////////////////
function drawBars(canvas, bins, maxVal, labels=null) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  const pad=24, innerW=W-pad*2, innerH=H-pad*2;
  const n=bins.length, bw = innerW/n;
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, innerW, innerH);
  ctx.fillStyle='#7aa2ff';
  const scale = maxVal>0? innerH/maxVal : 0;
  bins.forEach((v,i)=>{
    const x=pad+i*bw+2, h=v*scale, y=pad+innerH-h;
    ctx.fillRect(x, y, Math.max(2,bw-4), h);
  });
  // x labels minimal
  if (labels) {
    ctx.fillStyle='#93a3bf'; ctx.font='12px ui-monospace';
    labels.forEach((t,i)=>{ ctx.fillText(String(t), pad+i*bw+4, H-6); });
  }
}

function drawLorenz(canvas, cdf) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H=canvas.height, pad=24, iw=W-pad*2, ih=H-pad*2;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#1e293b'; ctx.strokeRect(pad,pad,iw,ih);
  // baseline diagonal
  ctx.strokeStyle='#98a2b3'; ctx.beginPath();
  ctx.moveTo(pad,pad+ih); ctx.lineTo(pad+iw,pad); ctx.stroke();
  // curve
  ctx.strokeStyle='#60a5fa'; ctx.beginPath();
  cdf.forEach((p,j)=>{
    const x = pad + iw * p[0];
    const y = pad + ih * (1 - p[1]);
    j? ctx.lineTo(x,y) : ctx.moveTo(x,y);
  });
  ctx.stroke();
}

///////////////////////////
// EDA computations
///////////////////////////
function computeSummaries() {
  USERS.clear(); ITEMS.clear();
  for (const r of INTERACTIONS) {
    if (!USERS.has(r.user)) USERS.set(r.user, []);
    USERS.get(r.user).push({item:r.item, rating:r.rating});
    ITEMS.set(r.item, (ITEMS.get(r.item)||0)+1);
  }
}

function buildIndexers() {
  idxUser.clear(); idxItem.clear(); revUser.length=0; revItem.length=0;
  [...USERS.keys()].sort((a,b)=>a-b).forEach((u,i)=>{idxUser.set(u,i); revUser[i]=u;});
  [...ITEMS.keys()].sort((a,b)=>a-b).forEach((m,i)=>{idxItem.set(m,i); revItem[i]=m;});
}

function fillEDA() {
  // Ratings hist
  const hist=[0,0,0,0,0]; // 1..5
  INTERACTIONS.forEach(r=>{ const i=Math.min(5,Math.max(1, r.rating))|0; hist[i-1]++; });
  drawBars($('#c_hist'), hist, Math.max(...hist), ['1','2','3','4','5']);

  // Top tags
  const tagCount = new Map();
  for (const r of RECIPES.values()) r.tags.forEach(t=>tagCount.set(t,(tagCount.get(t)||0)+1));
  const top = [...tagCount.entries()].sort((a,b)=>b[1]-a[1]).slice(0,20);
  drawBars($('#c_tags'), top.map(x=>x[1]), top[0]?.[1]||0, top.map(x=>x[0].slice(0,6)));

  // User activity buckets
  const userSizes = [...USERS.values()].map(arr=>arr.length);
  const uBins = new Array(7).fill(0), uEdges=[1,'2–2','3–3','4–5','6–10','11–20','>20'];
  userSizes.forEach(n=>{
    if (n<=1) uBins[0]++; else if (n<=2) uBins[1]++; else if (n<=3) uBins[2]++;
    else if (n<=5) uBins[3]++; else if (n<=10) uBins[4]++; else if (n<=20) uBins[5]++; else uBins[6]++;
  });
  drawBars($('#c_user'), uBins, Math.max(...uBins), uEdges);

  // Item popularity buckets
  const itemSizes = [...ITEMS.values()];
  const iBins = new Array(7).fill(0), iEdges=[1,'2–2','3–3','4–10','11–20','21–50','>50'];
  itemSizes.forEach(n=>{
    if (n<=1) iBins[0]++; else if (n<=2) iBins[1]++; else if (n<=3) iBins[2]++;
    else if (n<=10) iBins[3]++; else if (n<=20) iBins[4]++; else if (n<=50) iBins[5]++; else iBins[6]++;
  });
  drawBars($('#c_item'), iBins, Math.max(...iBins), iEdges);

  // Lorenz & Gini
  const sorted = itemSizes.sort((a,b)=>a-b);
  const total = sorted.reduce((a,b)=>a+b,0) || 1;
  let acc=0; const cdf=[[0,0]];
  sorted.forEach((v,i)=>{acc+=v; cdf.push([(i+1)/sorted.length, acc/total]);});
  drawLorenz($('#c_lorenz'), cdf);
  // numeric Gini via trapezoid
  let area=0; for(let i=1;i<cdf.length;i++){ const x1=cdf[i-1][0], y1=cdf[i-1][1], x2=cdf[i][0], y2=cdf[i][1]; area += (y1+y2)*(x2-x1)/2; }
  const gini = 1 - 2*area;
  $('#gini').textContent = `Gini ≈ ${gini.toFixed(3)}`;

  // Cold start
  const coldUsers = [...USERS.values()].filter(arr=>arr.length<5).length;
  const coldItems = [...ITEMS.values()].filter(n=>n<5).length;
  const top1pctCount = Math.max(1, Math.round(ITEMS.size*0.01));
  const top1pctItems = [...ITEMS.entries()].sort((a,b)=>b[1]-a[1]).slice(0, top1pctCount);
  const coverage = top1pctItems.reduce((s,kv)=>s+kv[1],0) / (INTERACTIONS.length||1) * 100;
  $('#coldBody').innerHTML = `
    <tr><td>Cold users (&lt;5)</td><td>${coldUsers}</td></tr>
    <tr><td>Cold items (&lt;5)</td><td>${coldItems}</td></tr>
    <tr><td>Top 1% items (by interactions)</td><td>${top1pctCount}</td></tr>
    <tr><td>Pareto: % train covered by top 1%</td><td>${coverage.toFixed(1)}%</td></tr>`;
}

///////////////////////////
// Data loading
///////////////////////////
async function loadData() {
  const status = $('#status');
  const datasetLine = $('#datasetLine');
  try {
    status.textContent = 'loading…';
    // fetch from current folder
    const [recTxt, intTxt] = await Promise.all([
      fetch('./RAW_recipes.csv', {cache:'no-store'}).then(r=>{ if(!r.ok) throw new Error('RAW_recipes.csv not found'); return r.text(); }),
      fetch('./interactions_train.csv', {cache:'no-store'}).then(r=>{ if(!r.ok) throw new Error('interactions_train.csv not found'); return r.text(); })
    ]);

    // Parse recipes
    const recRows = parseCSV(recTxt);
    const headR = recRows[0].map(h=>h.trim().toLowerCase());
    const idIdx = headR.indexOf('id');
    const nameIdx = headR.indexOf('name');
    const tagsIdx = headR.indexOf('tags');
    if (idIdx<0 || nameIdx<0 || tagsIdx<0) throw new Error('RAW_recipes.csv missing id/name/tags columns');

    RECIPES.clear();
    for (let i=1;i<recRows.length;i++){
      const r = recRows[i];
      const id = +r[idIdx]; if (!Number.isFinite(id)) continue;
      const title = r[nameIdx] || `Recipe ${id}`;
      const tags = new Set(parseTagList(r[tagsIdx]).slice(0, 10)); // limit per item
      RECIPES.set(id, {title, tags});
    }

    // Parse interactions
    const intRows = parseCSV(intTxt);
    const headI = intRows[0].map(h=>h.trim().toLowerCase());
    const uIdx = headI.indexOf('user_id');
    const iIdx = headI.indexOf('recipe_id') >= 0 ? headI.indexOf('recipe_id') : headI.indexOf('item_id');
    const rIdx = headI.indexOf('rating');
    if (uIdx<0 || iIdx<0 || rIdx<0) throw new Error('interactions_train.csv missing user_id/recipe_id/rating');
    INTERACTIONS = [];
    for (let i=1;i<intRows.length;i++){
      const r = intRows[i];
      const u = +r[uIdx]; const it = +r[iIdx]; const y = +r[rIdx];
      if (Number.isFinite(u) && Number.isFinite(it) && Number.isFinite(y)) {
        INTERACTIONS.push({user:u, item:it, rating: y});
      }
    }

    computeSummaries(); buildIndexers(); fillEDA();

    datasetLine.textContent = `Users: ${USERS.size}  Items: ${RECIPES.size}  Interactions: ${INTERACTIONS.length}`;
    status.textContent = 'loaded: users='+USERS.size+', items='+RECIPES.size+', interactions='+INTERACTIONS.length+
      ' (file: ./RAW_recipes.csv, ./interactions_train.csv)';

    // precompute for models
    window.DATASET = {RECIPES, INTERACTIONS, USERS, ITEMS, idxUser, idxItem, revUser, revItem};
    $('#metricsBox').textContent = `Ready. Users=${USERS.size}, Items=${RECIPES.size}, Interactions=${INTERACTIONS.length}.`;
    $('#b-msg').textContent = 'Baseline: ready to train.';
    $('#d-msg').textContent = 'Deep: ready to train.';
  } catch (e) {
    console.error(e);
    status.textContent = `fetch failed: ${e.message}. Ensure ./RAW_recipes.csv and ./interactions_train.csv exist (case-sensitive).`;
  }
}

$('#loadBtn').addEventListener('click', loadData);

///////////////////////////
// Training — Baseline
///////////////////////////
function getMaxInteractionsCap() {
  const n = parseInt($('#b-maxint').value,10) || 80000;
  return Math.min(n, INTERACTIONS.length);
}

$('#trainBaseline').addEventListener('click', async ()=>{
  if (!window.DATASET) { $('#b-msg').textContent='Load data first.'; return; }

  // Dispose previous
  if (Baseline) { Baseline.dispose(); Baseline=null; }

  // sample first N interactions (shuffled)
  const cap = getMaxInteractionsCap();
  const sample = INTERACTIONS.slice(0, cap).sort(()=>Math.random()-.5);

  const params = {
    embDim: parseInt($('#b-emb').value,10)||32,
    epochs: parseInt($('#b-epochs').value,10)||5,
    batchSize: parseInt($('#b-batch').value,10)||256,
    lr: parseFloat($('#b-lr').value)||0.001
  };

  Baseline = new TwoTowerBaseline(idxUser.size, idxItem.size, params.embDim, params.lr);
  const canvas = $('#c_loss_b');
  $('#b-msg').textContent = 'Training…';
  await Baseline.train(sample, idxUser, idxItem, {
    epochs: params.epochs,
    batchSize: params.batchSize,
    onStep: (step, total, loss)=> {
      drawBars(canvas, [loss], Math.max(1, loss), ['loss']); // quick spark
      $('#b-msg').textContent = `Training… step ${step}/${total}, loss ${loss.toFixed(4)}`;
    }
  });
  $('#b-msg').textContent = 'Training done.';
  // projection
  lastItemProj = await Baseline.projectItems2D(500);
  plotProjection(lastItemProj);
});

///////////////////////////
// Training — Deep
///////////////////////////
$('#trainDeep').addEventListener('click', async ()=>{
  if (!window.DATASET) { $('#d-msg').textContent='Load data first.'; return; }
  if (Deep) { Deep.dispose(); Deep=null; }

  const embDim = parseInt($('#d-emb').value,10)||32;
  const epochs = parseInt($('#d-epochs').value,10)||5;
  const batchSize = parseInt($('#d-batch').value,10)||256;
  const lr = parseFloat($('#d-lr').value)||0.001;
  const maxTagFeatures = parseInt($('#d-tags').value,10)||200;

  // Build tag vocabulary (top K)
  const tagCount = new Map();
  for (const rec of RECIPES.values()) rec.tags.forEach(t=>tagCount.set(t,(tagCount.get(t)||0)+1));
  const vocab = [...tagCount.entries()].sort((a,b)=>b[1]-a[1]).slice(0, maxTagFeatures).map(x=>x[0]);
  const tag2idx = new Map(vocab.map((t,i)=>[t,i]));

  // item -> multihot vector
  const itemFeat = new Array(idxItem.size).fill(0).map(()=>new Float32Array(vocab.length));
  for (const [id, rec] of RECIPES.entries()) {
    const j = idxItem.get(id); if (j==null) continue;
    rec.tags.forEach(t=>{ const k=tag2idx.get(t); if (k!=null) itemFeat[j][k]=1; });
  }

  const cap = getMaxInteractionsCap();
  const sample = INTERACTIONS.slice(0,cap).sort(()=>Math.random()-.5);

  Deep = new TwoTowerDeep(idxUser.size, idxItem.size, embDim, vocab.length, lr, itemFeat);
  const canvas = $('#c_loss_d');
  $('#d-msg').textContent = 'Training deep…';
  await Deep.train(sample, idxUser, idxItem, {
    epochs, batchSize,
    onStep: (step,total,loss)=>{
      drawBars(canvas, [loss], Math.max(1, loss), ['loss']);
      $('#d-msg').textContent = `Training deep… step ${step}/${total}, loss ${loss.toFixed(4)}`;
    }
  });
  $('#d-msg').textContent = 'Training done.';
  lastItemProj = await Deep.projectItems2D(500);
  plotProjection(lastItemProj);
});

///////////////////////////
// Projection plot
///////////////////////////
function plotProjection(points) {
  const c = $('#c_proj'), ctx=c.getContext('2d');
  const W=c.width, H=c.height;
  ctx.clearRect(0,0,W,H);
  if (!points || !points.length) return;
  const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
  const minx=Math.min(...xs), maxx=Math.max(...xs), miny=Math.min(...ys), maxy=Math.max(...ys);
  const pad=20;
  for (const [x,y] of points) {
    const px = pad + (x-minx)/(maxx-minx+1e-6)*(W-2*pad);
    const py = pad + (1-(y-miny)/(maxy-miny+1e-6))*(H-2*pad);
    ctx.fillStyle='#7aa2ff'; ctx.fillRect(px,py,3,3);
  }
}

///////////////////////////
// Demo (recommendations)
///////////////////////////
function pickUserForDemo() {
  // try ≥20, else ≥10, else most active
  const eligible20 = [...USERS.entries()].filter(([,arr])=>arr.length>=20);
  if (eligible20.length) return eligible20[Math.floor(Math.random()*eligible20.length)][0];
  const eligible10 = [...USERS.entries()].filter(([,arr])=>arr.length>=10);
  if (eligible10.length) return eligible10[Math.floor(Math.random()*eligible10.length)][0];
  // fallback: user with max
  return [...USERS.entries()].sort((a,b)=>b[1].length-a[1].length)[0][0];
}

function renderList(tbody, rows, withScore=false) {
  tbody.innerHTML = rows.map((r,i)=>`
    <tr><td>${i+1}</td><td>${r.title}</td><td>${withScore? r.score.toFixed(3) : (r.rating ?? '')}</td></tr>
  `).join('');
}

$('#testBtn').addEventListener('click', async ()=>{
  if (!window.DATASET) { $('#demoMsg').textContent='Load data and train at least one model first.'; return; }
  const userId = pickUserForDemo();
  const history = (USERS.get(userId)||[]).sort((a,b)=> b.rating-a.rating).slice(0,10)
    .map(x=>({title:RECIPES.get(x.item)?.title||('Recipe '+x.item), rating:x.rating}));
  renderList($('#histBody'), history);

  const useGraph = $('#useGraph').checked;

  let baseRecs = [];
  if (Baseline) baseRecs = await Baseline.recommend(userId, USERS, idxUser, idxItem, RECIPES, 50);
  let deepRecs = [];
  if (Deep) deepRecs = await Deep.recommend(userId, USERS, idxUser, idxItem, RECIPES, 50);

  if (useGraph) {
    // optional re-rank via PPR using item co-vis graph centered on top history items
    const seeds = (USERS.get(userId)||[]).slice(0,50).map(x=>x.item);
    baseRecs = reRankWithPPR(baseRecs, seeds, 0.15, 10);
    deepRecs = reRankWithPPR(deepRecs, seeds, 0.15, 10);
  }

  renderList($('#baseBody'), baseRecs.slice(0,10), true);
  renderList($('#deepBody'), deepRecs.slice(0,10), true);
  $('#demoMsg').textContent = 'Recommendations generated successfully!';
});
