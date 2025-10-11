/* app.js – data loading, charts, training, demo, metrics (robust to / or /data/) */

// --------- tiny DOM helpers ----------
const $ = (id) => document.getElementById(id);
const fmt = (n) => (typeof n === 'number' ? n.toLocaleString() : n);

// --------- global state ----------
let users = new Set();
let items = new Map();     // itemId -> { title, tags: string[] }
let train = [];            // [{u,i,r,ts}]
let valid = [];
let user2items = new Map();// userId -> [{i,r,ts}]
let item2users = new Map();// itemId -> Set(userId)

let userIndex = new Map(), itemIndex = new Map();
let idx2user = [], idx2item = [];

let tag2idx = new Map(), idx2tag = [];
let topTagK = 200;

let baseline = null;
let deep = null;

let baseLossTrace = [];
let deepLossTrace = [];

let lastItemEmb = null;
let lastItemEmbDeep = null;

// --------- tabs ----------
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tabpane").forEach(p=>p.classList.add("hidden"));
    $(btn.dataset.tab).classList.remove("hidden");
  });
});

// --------- loading utils ----------
async function fetchFirstExisting(candidates) {
  for (const path of candidates) {
    try {
      const resp = await fetch(path, { cache: 'no-store' });
      if (resp.ok) return { path, text: await resp.text() };
    } catch (_) { /* ignore and try next */ }
  }
  return null;
}

function splitLines(text){ return text.split(/\r?\n/).filter(Boolean); }

// Recipes: id + title/name + tags (python-list style string)
function parseRecipes(csvText){
  const lines = splitLines(csvText);
  const header = lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
  const idIdx = header.findIndex(h => /^id$|(^|_)id$/i.test(h));
  const nameIdx = header.findIndex(h => /(name|title)/i.test(h));
  const tagsIdx = header.findIndex(h => /tags/i.test(h));
  for (const ln of lines){
    const cols = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
    const id = parseInt(cols[idIdx],10);
    if (Number.isNaN(id)) continue;
    const rawTitle = cols[nameIdx] ?? `Recipe ${id}`;
    // Remove outer quotes only; keep commas inside quoted names
    const title = rawTitle.replace(/^"|"$/g,'');
    let tags = [];
    if (tagsIdx >= 0 && cols[tagsIdx]){
      let raw = cols[tagsIdx].trim();
      // handle "['a', 'b']" or JSON-like
      raw = raw.replace(/^\s*\[|\]\s*$/g,"");
      tags = raw.split(/['"]\s*,\s*['"]|,\s*/g)
                .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
                .filter(Boolean)
                .slice(0, 24);
    }
    items.set(id, {title, tags});
  }
}

// Interactions: supports various headers
function parseInteractions(csvText, sink){
  const lines = splitLines(csvText);
  if (!lines.length) return;
  const header = lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
  // NB: 'user' is intentionally case-sensitive lower here (food.com files are lowercase)
  const uIdx = header.findIndex(h=>/user/i.test(h));
  const iIdx = header.findIndex(h=>/(item|recipe)_?id/i.test(h));
  const rIdx = header.findIndex(h=>/rating/i.test(h));
  const tIdx = header.findIndex(h=>/(time|date)/i.test(h));
  for (const ln of lines){
    const cols = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
    const u = parseInt(cols[uIdx],10);
    const i = parseInt(cols[iIdx],10);
    if (!Number.isInteger(u) || !Number.isInteger(i)) continue;
    const r = rIdx>=0 && cols[rIdx] !== '' ? parseFloat(cols[rIdx]) : 1;
    const ts = tIdx>=0 ? (Date.parse(cols[tIdx]) || 0) : 0;
    sink.push({u,i,r,ts});
    users.add(u);
    if (!user2items.has(u)) user2items.set(u,[]);
    user2items.get(u).push({i,r,ts});
    if (!item2users.has(i)) item2users.set(i,new Set());
    item2users.get(i).add(u);
  }
}

function buildIndexers(){
  idx2user = Array.from(users).sort((a,b)=>a-b);
  idx2item = Array.from(items.keys()).sort((a,b)=>a-b);
  userIndex = new Map(idx2user.map((u,idx)=>[u,idx]));
  itemIndex = new Map(idx2item.map((i,idx)=>[i,idx]));
}

function buildTagVocab(k){
  topTagK = k;
  const freq = new Map();
  for (const it of items.values()){
    for (const t of (it.tags||[])){ freq.set(t,(freq.get(t)||0)+1); }
  }
  const top = Array.from(freq.entries()).sort((a,b)=>b[1]-a[1]).slice(0,k);
  tag2idx = new Map(top.map(([t],i)=>[t,i]));
  idx2tag = top.map(([t])=>t);
}

async function handleLoad(){
  try{
    $('status').textContent = 'Status: loading…';

    // Try local folder first (same folder as index.html), then a nested ./data/
    const tried = {
      recipes: ['./PP_recipes.csv','./RAW_recipes.csv','PP_recipes.csv','RAW_recipes.csv','data/PP_recipes.csv','data/RAW_recipes.csv'],
      train:   ['./interactions_train.csv','./interactions.csv','interactions_train.csv','interactions.csv','data/interactions_train.csv','data/interactions.csv'],
      valid:   ['./interactions_validation.csv','interactions_validation.csv','data/interactions_validation.csv']
    };

    const rec = await fetchFirstExisting(tried.recipes);
    const tr  = await fetchFirstExisting(tried.train);
    const va  = await fetchFirstExisting(tried.valid); // optional

    if (!rec || !tr) throw new Error('recipes or interactions_train not found near index.html');

    // Reset state
    items.clear(); users.clear(); train.length=0; valid.length=0;
    user2items.clear(); item2users.clear();

    parseRecipes(rec.text);
    parseInteractions(tr.text, train);
    if (va) parseInteractions(va.text, valid);

    // If no validation split, make a fast 80/20 time-aware split
    if (!valid.length && train.length){
      const sorted = train.slice().sort((a,b)=>a.ts-b.ts);
      valid = sorted.filter((_,ix)=> ix%5===0);       // 20%
      train = sorted.filter((_,ix)=> ix%5!==0);
    }

    buildIndexers();
    buildTagVocab(parseInt(($('dK')?.value||'200'),10));

    const density = (train.length / (users.size * Math.max(1, items.size))).toExponential(2);
    // cold users/items <5, computed over all interactions we loaded (user2items / item2users)
    const coldUsers = Array.from(user2items.values()).filter(arr=>arr.length<5).length;
    const itemCounts = new Map();
    for (const r of train){ itemCounts.set(r.i,(itemCounts.get(r.i)||0)+1); }
    const coldItems = Array.from(itemCounts.values()).filter(c=>c<5).length;

    $('datasetLine').textContent =
      `Users: ${fmt(users.size)} Items: ${fmt(items.size)} Interactions (train): ${fmt(train.length)} `+
      `Density: ${density} Ratings present: ${train.some(x=>x.r!=null) ? 'yes':'no'} `+
      `Cold users (<5): ${fmt(coldUsers)} Cold items (<5): ${fmt(coldItems)} `+
      `(files: ${rec.path}, ${tr.path}${va?`, ${va.path}`:''})`;
    $('status').textContent = 'Status: loaded.';
    drawAllEDA();
  }catch(err){
    console.error(err);
    $('status').textContent = 'Status: fetch failed. Ensure CSVs sit next to index.html (or in ./data/).';
  }
}

// --------- EDA drawing ----------
function clearCanvas(ctx){
  const c = ctx.canvas;
  const dpr = window.devicePixelRatio || 1;
  // reset transform by reassigning width/height
  const w = c.clientWidth, h = c.clientHeight;
  c.width = Math.max(1, w * dpr);
  c.height = Math.max(1, h * dpr);
  ctx.setTransform(1,0,0,1,0,0);
  ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
}

function drawBars(id, buckets, maxVal){
  const ctx = $(id).getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight;
  const pad = 28, bw = (W-pad*2)/buckets.length-6, base = H-pad;
  const scale = (v)=> (maxVal? (v/maxVal) : 0) * (H-pad*2);
  ctx.strokeStyle = "#223047"; ctx.beginPath(); ctx.moveTo(pad,base+0.5); ctx.lineTo(W-pad,base+0.5); ctx.stroke();
  ctx.fillStyle = "#ffffff";
  buckets.forEach((v,i)=>{ const h = scale(v); const x = pad + i*(bw+6); ctx.fillRect(x, base - h, bw, h); });
}

function drawLine(id, points){
  const ctx = $(id).getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight;
  if (!points.length) return;
  const pad = 20, base = H-pad;
  const maxY = Math.max(...points.map(p=>p.y)) || 1;
  const maxX = Math.max(...points.map(p=>p.x)) || 1;
  const sx = (x)=> pad + (x/maxX)*(W-pad*2);
  const sy = (y)=> base - (y/maxY)*(H-pad*2);
  ctx.strokeStyle = "#7dd3fc"; ctx.beginPath();
  ctx.moveTo(sx(points[0].x), sy(points[0].y));
  for (let k=1;k<points.length;k++){ ctx.lineTo(sx(points[k].x), sy(points[k].y)); }
  ctx.stroke();
}

function drawAllEDA(){
  if (!train.length){ return; }
  // ratings
  const hist = [0,0,0,0,0];
  for (const r of train){ const v = Math.round(Math.max(1,Math.min(5,r.r||1))); hist[v-1]++; }
  drawBars('histRatings', hist, Math.max(...hist,1));

  // user activity
  const ucnt = new Map(); for (const r of train){ ucnt.set(r.u,(ucnt.get(r.u)||0)+1); }
  const userBuckets = [0,0,0,0,0,0,0,0];
  for (const v of ucnt.values()){
    const idx = v===1?0: v<=2?1: v<=3?2: v<=5?3: v<=10?4: v<=20?5: v<=50?6:7;
    userBuckets[idx]++;
  }
  drawBars('histUser', userBuckets, Math.max(...userBuckets,1));

  // item popularity
  const icnt = new Map(); for (const r of train){ icnt.set(r.i,(icnt.get(r.i)||0)+1); }
  const itemBuckets = [0,0,0,0,0,0,0,0,0];
  for (const v of icnt.values()){
    const idx = v===1?0: v<=2?1: v<=3?2: v<=5?3: v<=10?4: v<=20?5: v<=100?6: v<=500?7:8;
    itemBuckets[idx]++;
  }
  drawBars('histItem', itemBuckets, Math.max(...itemBuckets,1));

  // top tags
  const freq = new Map();
  for (const it of items.values()){
    for (const t of it.tags||[]) if (tag2idx.has(t)) freq.set(t, (freq.get(t)||0)+1);
  }
  const top20 = Array.from(freq.entries()).sort((a,b)=>b[1]-a[1]).slice(0,20);
  drawBars('topTags', top20.map(([,c])=>c), Math.max(1,...top20.map(([,c])=>c)));

  // lorenz + gini
  const counts = Array.from(icnt.values()).sort((a,b)=>a-b);
  const total = counts.reduce((s,v)=>s+v,0) || 1;
  let acc=0; const lor = counts.map(v=>{ acc+=v; return acc/total; });
  const ctx = $('lorenz').getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight, pad=24;
  ctx.strokeStyle="#334155"; ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,pad); ctx.stroke();
  ctx.strokeStyle="#22d3ee"; ctx.beginPath();
  counts.forEach((_,i)=>{
    const x = pad + (i/(counts.length-1||1))*(W-pad*2);
    const y = H-pad - lor[i]*(H-pad*2);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  const gini = 1 - 2 * lor.reduce((s,y,i)=> s + y/(counts.length||1), 0) / (counts.length||1);
  $('giniLine').textContent = `Gini ≈ ${gini.toFixed(3)}`;

  // cold-start table
  const coldUsers = userBuckets[0] + userBuckets[1];
  const coldItems = itemBuckets[0] + itemBuckets[1];
  const top1pctCut = Math.max(1, Math.floor(0.01*idx2item.length));
  const topItems = Array.from(icnt.entries()).sort((a,b)=>b[1]-a[1]).slice(0, top1pctCut);
  const covered = topItems.reduce((s,[,c])=>s+c,0)/(train.length||1) * 100;
  $('coldTbl').innerHTML =
    `<tr><td>Cold users (&lt;5)</td><td>${fmt(coldUsers)}</td></tr>
     <tr><td>Cold items (&lt;5)</td><td>${fmt(coldItems)}</td></tr>
     <tr><td>Top 1% items (by interactions)</td><td>${fmt(top1pctCut)}</td></tr>
     <tr><td>Pareto: % train covered by top 1%</td><td>${covered.toFixed(1)}%</td></tr>`;
}

// --------- TF helpers ----------
function buildBatchTensors(batch){
  const u = tf.tensor1d(batch.map(x=>userIndex.get(x.u)), 'int32');
  const i = tf.tensor1d(batch.map(x=>itemIndex.get(x.i)), 'int32');
  const r = tf.tensor1d(batch.map(x=>x.r??1), 'float32');
  return {u,i,r};
}

function itemTagVector(iIdx){
  const itemId = idx2item[iIdx];
  const obj = items.get(itemId);
  const vec = new Array(topTagK).fill(0);
  if (obj && obj.tags){
    for (const t of obj.tags){
      let k = tag2idx.get(t);
      if (k===undefined){ k = Math.abs(hashStr(t)) % topTagK; }
      vec[k] = 1;
    }
  }
  return vec;
}
function hashStr(s){ let h=0; for (let i=0;i<s.length;i++){ h=(h*31 + s.charCodeAt(i))|0; } return h>>>0; }

function makeShuffled(arr, maxN){
  const A = maxN ? arr.slice(0, maxN) : arr.slice();
  for (let i=A.length-1;i>0;i--){ const j = (Math.random()* (i+1))|0; [A[i],A[j]]=[A[j],A[i]]; }
  return A;
}

// --------- Training: Baseline ----------
async function trainBaseline(){
  if (!users.size) return;
  if (baseline) { baseline.dispose(); baseline=null; }
  lastItemEmb?.dispose?.(); lastItemEmb=null;
  baseLossTrace = []; drawLine('baseLoss', baseLossTrace);

  const emb = parseInt($('bEmb').value,10);
  const epochs = parseInt($('bEp').value,10);
  const batch = parseInt($('bBa').value,10);
  const lr = parseFloat($('bLr').value);
  const maxI = parseInt($('bMax').value,10);

  $('baseLine').textContent = 'Training baseline…';
  const data = makeShuffled(train, maxI);

  baseline = new TwoTowerModel(idx2user.length, idx2item.length, emb, {learningRate: lr});
  await baseline.compile();

  let step=0;
  for (let ep=0; ep<epochs; ep++){
    for (let b=0; b<data.length; b+=batch){
      const slice = data.slice(b, b+batch);
      const {u,i} = buildBatchTensors(slice);
      const loss = await baseline.trainStep(u,i);
      u.dispose(); i.dispose();
      baseLossTrace.push({x: ++step, y: loss});
      if (step%5===0) drawLine('baseLoss', baseLossTrace);
      await tf.nextFrame();
    }
  }
  $('baseLine').innerHTML = `Baseline done. Final loss <b>${baseLossTrace.at(-1).y.toFixed(4)}</b>`;
  lastItemEmb = baseline.itemEmbedding.read();
  drawProjection(lastItemEmb);
  computeAndShowMetrics().catch(console.error);
}

// --------- Training: Deep (tags -> MLP) ----------
async function trainDeepModel(){
  if (!users.size) return;
  if (deep) { deep.dispose(); deep=null; }
  lastItemEmbDeep?.dispose?.(); lastItemEmbDeep=null;
  deepLossTrace = []; drawLine('deepLoss', deepLossTrace);

  const emb = parseInt($('dEmb').value,10);
  const epochs = parseInt($('dEp').value,10);
  const batch = parseInt($('dBa').value,10);
  const lr = parseFloat($('dLr').value);
  const K = parseInt($('dK').value,10);
  buildTagVocab(K);

  $('deepLine').textContent = 'Training deep…';
  const data = makeShuffled(train, parseInt($('bMax').value,10));

  const itemTagMat = tf.tensor2d(
    idx2item.map((_,iIdx)=> itemTagVector(iIdx) ),
    [idx2item.length, topTagK],
    'float32'
  );

  deep = new DeepTwoTowerModel(idx2user.length, idx2item.length, emb, topTagK, {learningRate: lr});
  await deep.compile(itemTagMat);

  let step=0;
  for (let ep=0; ep<epochs; ep++){
    for (let b=0; b<data.length; b+=batch){
      const slice = data.slice(b, b+batch);
      const {u,i} = buildBatchTensors(slice);
      const loss = await deep.trainStep(u,i);
      u.dispose(); i.dispose();
      deepLossTrace.push({x: ++step, y: loss});
      if (step%5===0) drawLine('deepLoss', deepLossTrace);
      await tf.nextFrame();
    }
  }
  $('deepLine').innerHTML = `Deep done. Final loss <b>${deepLossTrace.at(-1).y.toFixed(4)}</b>`;
  lastItemEmbDeep = deep.getFrozenItemEmb();
  drawProjection(lastItemEmbDeep);
  itemTagMat.dispose();
  computeAndShowMetrics().catch(console.error);
}

// --------- Projection (cheap 2D) ----------
function powerIter(M, v, iters=20){
  let x = v;
  for (let t=0;t<iters;t++){ const y = M.matMul(x); const n = y.norm(); x = y.div(n); }
  return x;
}
function drawProjection(itemEmb){
  if (!itemEmb) return;
  const X = itemEmb;                 // [N,D]
  const XT = X.transpose();          // [D,N]
  const C = XT.matMul(X);            // [D,D]
  const v0 = tf.randomNormal([C.shape[0],1]);
  const v1 = powerIter(C, v0, 20);
  const w0 = tf.randomNormal([C.shape[0],1]);
  const C2 = C.sub(v1.matMul(v1.transpose()).mul(C.matMul(v1).transpose().matMul(v1)));
  const v2 = powerIter(C2, w0, 20);
  const P = tf.concat([v1,v2],1);    // [D,2]
  const Y = X.matMul(P);             // [N,2]

  const arr = Array.from(Y.dataSync());
  const coords = []; for (let i=0;i<arr.length;i+=2) coords.push({x:arr[i], y:arr[i+1]});
  const minx = Math.min(...coords.map(p=>p.x)), maxx = Math.max(...coords.map(p=>p.x));
  const miny = Math.min(...coords.map(p=>p.y)), maxy = Math.max(...coords.map(p=>p.y));
  const norm = coords.map(p=>({x:(p.x-minx)/(maxx-minx+1e-6), y:(p.y-miny)/(maxy-miny+1e-6)}));

  const ctx = $('proj').getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight, pad=10;
  ctx.fillStyle = "#cbd5e1";
  norm.slice(0,1000).forEach(p=>{
    const x = pad + p.x*(W-pad*2);
    const y = pad + (1-p.y)*(H-pad*2);
    ctx.fillRect(x,y,2,2);
  });

  v0.dispose(); v1.dispose(); w0.dispose(); v2.dispose(); P.dispose(); XT.dispose(); C.dispose(); C2.dispose(); Y.dispose();
}

// --------- Demo (hardened) ----------

// pick a user meeting minRatings from the full user2items map,
// fall back to the user with the maximum history if none meets it.
function pickUserForDemo(minRatings){
  const eligible = [];
  let bestU = null, bestC = 0;
  for (const [u, arr] of user2items.entries()){
    const c = arr.length;
    if (c >= minRatings) eligible.push(u);
    if (c > bestC){ bestC = c; bestU = u; }
  }
  if (eligible.length){
    return { user: eligible[Math.floor(Math.random()*eligible.length)], usedMin: minRatings };
  }
  // fallback — show the best we can find
  return { user: bestU, usedMin: bestC };
}

function setEmpty(tblId, msg='—'){
  $(tblId).innerHTML = `<tr><td class="muted" colspan="3">${msg}</td></tr>`;
}

async function demoOnce(){
  if (!baseline && !deep) {
    $('demoLine').textContent = 'Train a model first (baseline or deep).';
    setEmpty('histTbl','—');
    setEmpty('baseTbl','—');
    setEmpty('deepTbl','—');
    return;
  }
  const reqMin = parseInt($('minRatings').value,10) || 1;

  // Try a few times to find a user who both has history AND at least one unseen item
  let picked = pickUserForDemo(reqMin);
  let tries = 8;
  while (tries-- > 0) {
    const histArr = user2items.get(picked.user) || [];
    const seen = new Set(histArr.map(x=>x.i));
    if (histArr.length && seen.size < idx2item.length) break; // ok
    // else pick again
    picked = pickUserForDemo(picked.usedMin);
  }

  const u = picked.user;
  $('demoLine').textContent = `Testing with user ${u} (threshold used: ${picked.usedMin}).`;

  const histArr = (user2items.get(u)||[]).slice();
  if (!histArr.length){
    setEmpty('histTbl','no history for chosen user');
    setEmpty('baseTbl','no unseen items / no history');
    setEmpty('deepTbl','no unseen items / no history');
    return;
  }

  const hist = histArr.sort((a,b)=> (b.r-a.r) || (b.ts-a.ts)).slice(0,10);
  $('histTbl').innerHTML = hist.map((row,idx)=>(
    `<tr><td>${idx+1}</td><td>${escapeHtml(items.get(row.i)?.title||String(row.i))}</td><td>${row.r??''}</td></tr>`
  )).join('');

  const seen = new Set(histArr.map(x=>x.i));
  const candIdx = idx2item.map((iid,ii)=> ({iid,ii})).filter(x=>!seen.has(x.iid)).map(x=>x.ii);

  // If there are no candidates, show placeholders and stop early
  if (!candIdx.length){
    setEmpty('baseTbl','no unseen items for this user');
    setEmpty('deepTbl','no unseen items for this user');
    return;
  }

  const uIdx = tf.tensor1d([userIndex.get(u)], 'int32');
  let baseScores=[], deepScores=[];

  try {
    if (baseline){
      const s = await baseline.scoreUserAgainstAll(uIdx);
      baseScores = Array.from(s.dataSync());
      s.dispose();
    }
    if (deep){
      const s = await deep.scoreUserAgainstAll(uIdx);
      deepScores = Array.from(s.dataSync());
      s.dispose();
    }
  } finally {
    uIdx.dispose();
  }

  if (!baseScores.length) setEmpty('baseTbl','baseline not trained');
  if (!deepScores.length) setEmpty('deepTbl','deep model not trained');

  // Optional graph re-rank
  if ($('useGraph').checked && candIdx.length){
    const pr = personalizedPageRankForUser(u, user2items, item2users, {alpha:0.15, iters:20});
    const lambda = 0.15;
    if (baseline && baseScores.length) for (const ii of candIdx){ baseScores[ii] += lambda*(pr.get(idx2item[ii])||0); }
    if (deep && deepScores.length)    for (const ii of candIdx){ deepScores[ii]  += lambda*(pr.get(idx2item[ii])||0); }
  }

  const render = (tblId, scoresArr, emptyMsg) => {
    if (!scoresArr.length){ setEmpty(tblId, emptyMsg); return; }
    if (!candIdx.length){ setEmpty(tblId, 'no unseen items for this user'); return; }
    const picked = candIdx.map(ii=>({ii, s:scoresArr[ii]})).sort((a,b)=>b.s-a.s).slice(0,10);
    if (!picked.length){ setEmpty(tblId, 'no candidates'); return; }
    $(tblId).innerHTML = picked.map((row,idx)=>(
      `<tr><td>${idx+1}</td><td>${escapeHtml(items.get(idx2item[row.ii])?.title||String(idx2item[row.ii]))}</td><td>${row.s.toFixed(3)}</td></tr>`
    )).join('');
  };

  render('baseTbl', baseScores, 'baseline not trained');
  render('deepTbl', deepScores, 'deep model not trained');
  $('demoLine').textContent += '  — recommendations generated successfully!';
}

// --------- Metrics (Recall@10 / nDCG@10 with sampled negatives) ----------
async function computeAndShowMetrics(){
  const body = $('metricsBody');
  if (!valid.length){ body.textContent = 'No validation split found.'; return; }

  const sampleNeg = 800;
  const K = 10;
  const usersV = Array.from(new Set(valid.map(x=>x.u)));
  const uPick = usersV.slice(0, Math.min(120, usersV.length));

  async function evalModel(model){
    if (!model) return null;
    let hits=0, dcg=0, ideal=0;
    for (const u of uPick){
      const seen = new Set((user2items.get(u)||[]).map(x=>x.i));
      const testPos = valid.filter(x=>x.u===u).map(x=>x.i);
      if (!testPos.length) continue;

      const neg = [];
      while (neg.length<sampleNeg){
        const ii = idx2item[(Math.random()*idx2item.length)|0];
        if (!seen.has(ii) && !testPos.includes(ii)) neg.push(ii);
      }
      const candidates = testPos.slice(0, Math.min(testPos.length,5)).concat(neg);

      const uIdx = tf.tensor1d([userIndex.get(u)], 'int32');
      const scores = await model.scoreItems(uIdx, candidates.map(i=>itemIndex.get(i)));
      uIdx.dispose();
      const arr = Array.from(scores.dataSync());
      scores.dispose();

      const zipped = candidates.map((iid,ix)=>({iid, s:arr[ix], rel: testPos.includes(iid)?1:0}))
                               .sort((a,b)=>b.s-a.s).slice(0,K);
      hits += zipped.some(z=>z.rel>0) ? 1 : 0;
      dcg += zipped.reduce((s,z,idx)=> s + (z.rel>0 ? 1/Math.log2(idx+2) : 0), 0);
      ideal += 1;
    }
    return { recallAt10: hits/Math.max(1,uPick.length), ndcgAt10: dcg/Math.max(1,ideal) };
  }

  const baseM = await evalModel(baseline);
  const deepM = await evalModel(deep);

  body.innerHTML =
    `<p>Validation users sampled: ${fmt(uPick.length)} (sampled negatives=${sampleNeg})</p>`+
    `<p>Baseline — Recall@10: <b>${baseM?baseM.recallAt10.toFixed(3):'—'}</b> · nDCG@10: <b>${baseM?baseM.ndcgAt10.toFixed(3):'—'}</b></p>`+
    `<p>Deep (MLP+tags) — Recall@10: <b class="${deepM && baseM && deepM.recallAt10>=baseM.recallAt10?'ok':'bad'}">${deepM?deepM.recallAt10.toFixed(3):'—'}</b> · nDCG@10: <b>${deepM?deepM.ndcgAt10.toFixed(3):'—'}</b></p>`;
}

// --------- misc ----------
function escapeHtml(s){ return (s??'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;' }[m])); }

$('loadBtn').addEventListener('click', handleLoad);
$('trainBase').addEventListener('click', ()=>trainBaseline().catch(console.error));
$('trainDeep').addEventListener('click', ()=>trainDeepModel().catch(console.error));
$('btnTest').addEventListener('click', ()=>demoOnce().catch(console.error));
