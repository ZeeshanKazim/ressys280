/* app.js */
const $ = (id)=>document.getElementById(id);
const fmt = (n)=> (typeof n==='number'? n.toLocaleString(): n);

let items = new Map();       // itemId -> {title, tags[]}
let users = new Set();
let train = [];              // [{u,i,r,ts}]
let user2items = new Map();  // user -> [{i,r,ts}]
let item2users = new Map();  // item -> Set(users)

let userIndex = new Map(), itemIndex = new Map();
let idx2user = [], idx2item = [];

let tag2idx = new Map(); let topKTags = 200;

let retriever = null;  // TwoTower
let sasrec = null;     // SASRec

/* ---------- tiny UI helpers ---------- */
function setLog(el, msg){ el.value = msg+'\n'; el.scrollTop = el.scrollHeight; }
function pushLog(el, msg){ el.value += msg+'\n'; el.scrollTop = el.scrollHeight; }
function sleepFrame(){ return tf.nextFrame(); }

/* ---------- tabs ---------- */
document.querySelectorAll('.tab').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tabpane').forEach(p=>p.classList.add('hidden'));
    $(btn.dataset.tab).classList.remove('hidden');
  });
});

/* ---------- CSV loader ---------- */
async function fetchFirst(paths){
  for (const p of paths){
    try{
      const r = await fetch(p, {cache:'no-store'}); if (r.ok) return {path:p, text: await r.text()};
    }catch(_){}
  }
  return null;
}
function splitLines(t){ return t.split(/\r?\n/).filter(Boolean); }

function parseRecipes(csv){
  const lines = splitLines(csv); const header=lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/);
  const idIdx = header.findIndex(h=>/^id$|(^|_)id$/i.test(h));
  const nameIdx = header.findIndex(h=>/(name|title)/i.test(h));
  const tagsIdx = header.findIndex(h=>/tags/i.test(h));
  items.clear();
  for (const ln of lines){
    const c = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/);
    const id = parseInt(c[idIdx],10); if (!Number.isInteger(id)) continue;
    let title = (c[nameIdx]||`Recipe ${id}`).replace(/^"|"$/g,'');
    let ts = (c[tagsIdx]||'').trim();
    let tags=[];
    if (ts){
      ts = ts.replace(/^\s*\[|\]\s*$/g,'');
      tags = ts.split(/['"]\s*,\s*['"]|,\s*/g).map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim()).filter(Boolean).slice(0,24);
    }
    items.set(id,{title,tags});
  }
}
function parseInteractions(csv){
  const lines = splitLines(csv); const header=lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/);
  const uIdx = header.findIndex(h=>/user/i.test(h));
  const iIdx = header.findIndex(h=>/(item|recipe)_?id/i.test(h));
  const rIdx = header.findIndex(h=>/rating/i.test(h));
  const tIdx = header.findIndex(h=>/(time|date)/i.test(h));
  train.length = 0; users.clear(); user2items.clear(); item2users.clear();
  for (const ln of lines){
    const c = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/);
    const u = parseInt(c[uIdx],10), it = parseInt(c[iIdx],10); if(!Number.isInteger(u)||!Number.isInteger(it)) continue;
    const r = rIdx>=0 ? parseFloat(c[rIdx]) : 1;
    const ts = tIdx>=0 ? (Date.parse(c[tIdx])||0) : 0;
    train.push({u,i:it,r,ts});
    users.add(u);
    if (!user2items.has(u)) user2items.set(u,[]);
    user2items.get(u).push({i:it,r,ts});
    if (!item2users.has(it)) item2users.set(it,new Set());
    item2users.get(it).add(u);
  }
}
function buildIndexers(){
  idx2user = Array.from(users).sort((a,b)=>a-b);
  idx2item = Array.from(new Set(train.map(x=>x.i))).sort((a,b)=>a-b);
  userIndex = new Map(idx2user.map((u,ix)=>[u,ix]));
  itemIndex = new Map(idx2item.map((i,ix)=>[i,ix]));
}
function buildTagVocab(K){
  topKTags = K;
  const f = new Map();
  for (const it of items.values()) for (const t of (it.tags||[])) f.set(t,(f.get(t)||0)+1);
  const arr = Array.from(f.entries()).sort((a,b)=>b[1]-a[1]).slice(0, K);
  tag2idx = new Map(arr.map(([t],i)=>[t,i]));
}
function itemTagVec(ii){
  const itemId = idx2item[ii];
  const obj = items.get(itemId);
  const vec = new Array(topKTags).fill(0);
  if (obj && obj.tags){
    for (const t of obj.tags){
      const k = tag2idx.get(t);
      if (k!==undefined) vec[k]=1;
    }
  }
  return vec;
}

async function handleLoad(){
  try{
    $('loadLine').textContent = 'Status: loading…';
    const rec = await fetchFirst(['./data/PP_recipes.csv','./data/RAW_recipes.csv','PP_recipes.csv','RAW_recipes.csv']);
    const tr  = await fetchFirst(['./data/interactions_train.csv','interactions_train.csv']);
    if (!rec || !tr) throw new Error('CSV files not found (recipes and interactions_train)');
    parseRecipes(rec.text);
    parseInteractions(tr.text);
    buildIndexers();
    buildTagVocab(parseInt(($('ttK')?.value||'200'),10));

    const density = (train.length / (users.size * Math.max(1, idx2item.length)) ) || 0;
    $('counters').textContent =
      `Users: ${fmt(users.size)} · Items: ${fmt(idx2item.length)} · Interactions: ${fmt(train.length)} · `
      + `Density: ${density.toExponential(2)} · Ratings: ${train.some(x=>x.r!=null)?'yes':'no'}`;
    $('loadLine').textContent = `Status: loaded (recipes: ${rec.path}, interactions: ${tr.path}).`;
  }catch(err){
    console.error(err);
    $('loadLine').textContent = 'Status: load failed. Put CSVs in the repo root or /data/.';
  }
}

/* ---------- Training: Two-Tower ---------- */
function buildPairs(maxRows){
  // one row per positive (user,item). We'll down-sample if requested.
  const rows = train.slice().sort((a,b)=>a.ts-b.ts);
  if (maxRows && rows.length>maxRows) rows.length = maxRows;
  const u = new Int32Array(rows.length);
  const i = new Int32Array(rows.length);
  for (let k=0;k<rows.length;k++){ u[k]=userIndex.get(rows[k].u); i[k]=itemIndex.get(rows[k].i); }
  return {u,i};
}
async function trainTwoTower(){
  if (!users.size || !idx2item.length){ setLog($('ttLog'),'Load data first.'); return; }
  retriever?.dispose?.(); retriever=null;

  const emb = parseInt($('ttEmb').value,10);
  const ep  = parseInt($('ttEp').value,10);
  const ba  = parseInt($('ttBa').value,10);
  const lr  = parseFloat($('ttLr').value);
  const maxR = parseInt($('ttMax').value,10);
  buildTagVocab(parseInt($('ttK').value,10));

  const {u,i} = buildPairs(maxR);
  if (u.length===0){ setLog($('ttLog'),'No rows to train.'); return; }

  setLog($('ttLog'),`Training Two-Tower… rows=${fmt(u.length)} batch=${ba} emb=${emb} lr=${lr}`);
  await sleepFrame();

  retriever = new TwoTower(idx2user.length, idx2item.length, emb, {learningRate:lr});

  // mini-batches
  let step=0; const stepsPerEpoch = Math.ceil(u.length/ba);
  for (let epoch=0; epoch<ep; epoch++){
    for (let b=0; b<u.length; b+=ba){
      const uB = tf.tensor1d(u.slice(b, b+ba),'int32');
      const iB = tf.tensor1d(i.slice(b, b+ba),'int32');
      const loss = await retriever.trainStep(uB, iB);
      uB.dispose(); iB.dispose();
      step++;
      if (step%5===0) pushLog($('ttLog'),`epoch ${epoch+1}/${ep} · step ${step%stepsPerEpoch||stepsPerEpoch}/${stepsPerEpoch} · loss=${loss.toFixed(4)}`);
      await sleepFrame();
    }
  }
  pushLog($('ttLog'),`✅ Two-Tower done. Steps=${step}.`);
  drawProjection(retriever.itemEmb.read());
}

/* ---------- Training: SASRec ---------- */
function buildSequences(maxRows, L){
  // For each user, create rolling sequences; target is next item.
  const rows=[];
  for (const [u, arr] of user2items.entries()){
    const seq = arr.slice().sort((a,b)=>a.ts-b.ts).map(x=> itemIndex.get(x.i)).filter(x=>x!==undefined);
    if (seq.length<3) continue;
    for (let t=1; t<seq.length; t++){
      const start = Math.max(0, t-L);
      const context = seq.slice(start, t);
      const pad = Array(Math.max(0, L - context.length)).fill(0);
      const inp = pad.concat(context.map(x=>x+1)); // use 1..ni tokens, 0 pad
      const tgt = seq[t]+1; // 1..ni
      rows.push({inp, tgt});
      if (maxRows && rows.length>=maxRows) break;
    }
    if (maxRows && rows.length>=maxRows) break;
  }
  return rows;
}
async function trainSASRec(){
  if (!users.size || !idx2item.length){ setLog($('saLog'),'Load data first.'); return; }
  const d  = parseInt($('saD').value,10);
  const L  = parseInt($('saL').value,10);
  const ep = parseInt($('saEp').value,10);
  const ba = parseInt($('saBa').value,10);
  const lr = parseFloat($('saLr').value);
  const maxR = parseInt($('saMax').value,10);

  const rows = buildSequences(maxR, L);
  if (!rows.length){ setLog($('saLog'),'No sequences could be built (need users with ≥3 events).'); return; }

  sasrec?.dispose?.(); sasrec=null;
  setLog($('saLog'),`Training SASRec… seqRows=${fmt(rows.length)} batch=${ba} d=${d} L=${L} lr=${lr}`);
  await sleepFrame();

  sasrec = new SASRec(idx2item.length, d, L, {learningRate:lr});

  let step=0; const stepsPerEpoch = Math.ceil(rows.length/ba);
  for (let epoch=0; epoch<ep; epoch++){
    for (let b=0; b<rows.length; b+=ba){
      const batch = rows.slice(b, b+ba);
      const X = tf.tensor2d(batch.flatMap(r=>r.inp), [batch.length, L], 'int32');
      const y = tf.tensor1d(batch.map(r=>r.tgt), 'int32');
      const loss = await sasrec.trainStep(X, y);
      X.dispose(); y.dispose();
      step++;
      if (step%5===0) pushLog($('saLog'),`epoch ${epoch+1}/${ep} · step ${step%stepsPerEpoch||stepsPerEpoch}/${stepsPerEpoch} · loss=${loss.toFixed(4)}`);
      await sleepFrame();
    }
  }
  pushLog($('saLog'),`✅ SASRec done. Steps=${step}.`);
  drawProjection( sasrec.itemEmb.read().slice([1,0],[idx2item.length, sasrec.d]) ); // skip pad row
}

/* ---------- projection (quick SVD-ish) ---------- */
function drawProjection(itemEmb){
  if (!itemEmb) return;
  const X = itemEmb; // [N,d]
  const XT = X.transpose();
  const C = XT.matMul(X); // [d,d]
  function power(M){
    let v = tf.randomNormal([M.shape[0],1]);
    for (let t=0;t<25;t++){ v = M.matMul(v); v = v.div(v.norm()); }
    return v;
  }
  const v1 = power(C);
  const C2 = C.sub( v1.matMul(v1.transpose()).mul( C.matMul(v1).transpose().matMul(v1) ) );
  const v2 = power(C2);
  const P = tf.concat([v1,v2],1); // [d,2]
  const Y = X.matMul(P);          // [N,2]
  const arr = Array.from(Y.dataSync()); Y.dispose(); v1.dispose(); v2.dispose(); C.dispose(); C2.dispose(); XT.dispose();
  const pts = []; for (let i=0;i<arr.length;i+=2) pts.push({x:arr[i], y:arr[i+1]});
  const minx=Math.min(...pts.map(p=>p.x)), maxx=Math.max(...pts.map(p=>p.x));
  const miny=Math.min(...pts.map(p=>p.y)), maxy=Math.max(...pts.map(p=>p.y));
  const norm = pts.map(p=>({x:(p.x-minx)/(maxx-minx+1e-6), y:(p.y-miny)/(maxy-miny+1e-6)}));
  const ctx = $('proj').getContext('2d'); const c=ctx.canvas;
  const W=c.clientWidth, H=c.clientHeight, pad=8;
  c.width=W*devicePixelRatio; c.height=H*devicePixelRatio; ctx.scale(devicePixelRatio,devicePixelRatio);
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#cbd5e1';
  norm.slice(0,1200).forEach(p=>{
    const x = pad + p.x*(W-pad*2), y = pad + (1-p.y)*(H-pad*2);
    ctx.fillRect(x,y,2,2);
  });
}

/* ---------- demo ---------- */
function pickUser(){
  const cand = Array.from(user2items.keys()).filter(u => (user2items.get(u)||[]).length>=3);
  return cand[(Math.random()*cand.length)|0];
}
async function runDemo(){
  if (!retriever && !sasrec){ $('demoLine').textContent='Train at least one model first.'; return; }
  const u = pickUser(); if (!u){ $('demoLine').textContent='Need users with ≥3 events.'; return; }
  $('demoLine').textContent = `User ${u} — generating…`;
  const hist = (user2items.get(u)||[]).slice().sort((a,b)=>a.ts-b.ts);
  $('histTbl').innerHTML = hist.slice(-15).map((row,ix)=> `<tr><td>${hist.length-15+ix+1}</td><td>${escape(items.get(row.i)?.title||row.i)}</td></tr>`).join('');

  // Build sequence for SASRec if available; else fallback to Two-Tower scoring against all.
  let scores = null; const seen = new Set(hist.map(x=>x.i));
  if (sasrec){
    const L = parseInt($('saL').value,10);
    const seq = hist.slice(-L).map(x=> itemIndex.get(x.i)+1).filter(v=>v>0);
    const pad = Array(Math.max(0, L - seq.length)).fill(0);
    const X = tf.tensor2d([pad.concat(seq)], [1,L], 'int32');
    const s = await sasrec.scoreNext(X); X.dispose();
    const arr = Array.from(s.dataSync()); s.dispose();
    scores = arr; // [ni]
  } else if (retriever){
    const uIdx = tf.tensor1d([userIndex.get(u)], 'int32');
    const s = await retriever.scoreUserAgainstAll(uIdx); uIdx.dispose();
    const arr = Array.from(s.dataSync()); s.dispose();
    scores = arr;
  }

  // apply PPR re-rank if requested
  if ($('usePPR').checked){
    const pr = PPR.personalizedPageRank(u, user2items, item2users, {alpha:0.15, iters:20});
    const lam = 0.15;
    for (let ii=0; ii<idx2item.length; ii++){
      const itemId = idx2item[ii];
      scores[ii] += lam * (pr.get(itemId)||0);
    }
  }

  // assemble candidates excluding seen
  const cand = [];
  for (let ii=0; ii<idx2item.length; ii++){
    const itemId = idx2item[ii];
    if (seen.has(itemId)) continue;
    cand.push({ii, s:scores[ii]});
  }
  cand.sort((a,b)=>b.s-a.s);
  const top = cand.slice(0,10);
  $('recTbl').innerHTML = top.map((r,ix)=> `<tr><td>${ix+1}</td><td>${escape(items.get(idx2item[r.ii])?.title||idx2item[r.ii])}</td><td>${r.s.toFixed(3)}</td></tr>`).join('');
  $('demoLine').textContent += ' done.';
}

function escape(s){ return (s??'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;' }[m])); }

/* ---------- wiring ---------- */
$('btnLoad').addEventListener('click', ()=>handleLoad().catch(console.error));
$('btnTrainTT').addEventListener('click', ()=>trainTwoTower().catch(err=>{console.error(err);pushLog($('ttLog'),'❌ '+err.message);}));
$('btnTrainSA').addEventListener('click', ()=>trainSASRec().catch(err=>{console.error(err);pushLog($('saLog'),'❌ '+err.message);}));
$('btnDemo').addEventListener('click', ()=>runDemo().catch(console.error));
