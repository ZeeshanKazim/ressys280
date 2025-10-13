/* app.js — data + training + demo glue */

const $ = id => document.getElementById(id);
const esc = s => (s??'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;' }[m]));
const fmt = n => (typeof n==='number' ? n.toLocaleString() : n);

// Global state
let users = new Set();
let items = new Map();           // itemId -> {title, tags[]}
let train = [];                  // [{u,i,r,ts}]
let user2items = new Map();      // userId -> [{i,r,ts}]
let item2users = new Map();      // itemId -> Set(user)
let userIndex = new Map(), itemIndex = new Map(), idx2user=[], idx2item=[];
let tag2idx = new Map(), topTagK = 200;

let retriever = null;   // Two-Tower
let sasrec = null;      // SASRec
let itemProjSource = null;

let seqRows = [];       // built sequence training rows

// Tabs
document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tabpane").forEach(p=>p.classList.add("hidden"));
    $(btn.dataset.tab).classList.remove("hidden");
  });
});

// Fetch helpers
async function fetchFirstExisting(paths){
  for(const p of paths){
    try{
      const r = await fetch(p, {cache:'no-store'});
      if(r.ok){ return {path:p, text: await r.text()}; }
    }catch(_){}
  }
  return null;
}
const splitLines = t => t.split(/\r?\n/).filter(Boolean);

// Parse recipes (RAW_recipes.csv or PP_recipes.csv)
function parseRecipes(csv){
  const lines = splitLines(csv);
  const header = lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
  const idIdx = header.findIndex(h=>/^id$|(^|_)id$/i.test(h));
  const nameIdx = header.findIndex(h=>/(name|title)/i.test(h));
  const tagsIdx = header.findIndex(h=>/tags/i.test(h));
  for(const ln of lines){
    const cols = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
    const id = parseInt(cols[idIdx],10); if(!Number.isInteger(id)) continue;
    const title = (cols[nameIdx]||`Recipe ${id}`).replace(/^"|"$/g,'');
    let tags=[];
    if(tagsIdx>=0 && cols[tagsIdx]){
      let raw = cols[tagsIdx].trim();
      raw = raw.replace(/^\s*\[|\]\s*$/g,"");
      tags = raw.split(/['"]\s*,\s*['"]|,\s*/g)
        .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
        .filter(Boolean).slice(0,24);
    }
    items.set(id, {title, tags});
  }
}

// Interactions: user,item,rating?,time?
function parseInteractions(csv, sink){
  const lines = splitLines(csv);
  if(!lines.length) return;
  const header = lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
  const uIdx = header.findIndex(h=>/user/i.test(h));
  const iIdx = header.findIndex(h=>/(item|recipe)_?id/i.test(h));
  const rIdx = header.findIndex(h=>/rating/i.test(h));
  const tIdx = header.findIndex(h=>/(time|date)/i.test(h));
  let tFallback=0;
  for(const ln of lines){
    const c = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
    const u = parseInt(c[uIdx],10);
    const i = parseInt(c[iIdx],10);
    if(!Number.isInteger(u)||!Number.isInteger(i)) continue;
    const r = rIdx>=0 && c[rIdx]!=='' ? parseFloat(c[rIdx]) : 1;
    const ts = tIdx>=0 ? (Date.parse(c[tIdx])||(++tFallback)) : (++tFallback);
    sink.push({u,i,r,ts});
    users.add(u);
    if(!user2items.has(u)) user2items.set(u,[]);
    user2items.get(u).push({i,r,ts});
    if(!item2users.has(i)) item2users.set(i,new Set());
    item2users.get(i).add(u);
  }
}

function buildIndexers(){
  idx2user = Array.from(users).sort((a,b)=>a-b);
  idx2item = Array.from(items.keys()).sort((a,b)=>a-b);
  userIndex = new Map(idx2user.map((u,ix)=>[u,ix]));
  // reserve 0 as PAD for SASRec -> shift items by +1
  itemIndex = new Map(); itemIndex.set(0,0); // PAD
  for(let k=0;k<idx2item.length;k++) itemIndex.set(idx2item[k], k+1);
}

function buildTagVocab(k=200){
  topTagK = k;
  const freq = new Map();
  for(const v of items.values()){
    for(const t of (v.tags||[])) freq.set(t,(freq.get(t)||0)+1);
  }
  const top = Array.from(freq.entries()).sort((a,b)=>b[1]-a[1]).slice(0,k);
  tag2idx = new Map(top.map(([t],i)=>[t,i]));
}

// item tag vector (multi-hot, hashed fallback)
function itemTagVector(iIdx){
  const iid = idx2item[iIdx];
  const obj = items.get(iid);
  const v = new Array(topTagK).fill(0);
  if(obj?.tags){
    for(const t of obj.tags){
      let k = tag2idx.get(t);
      if(k===undefined){ k = Math.abs(hashStr(t)) % topTagK; }
      v[k]=1;
    }
  }
  return v;
}
function hashStr(s){ let h=0; for(let i=0;i<s.length;i++){ h=(h*31 + s.charCodeAt(i))|0; } return h>>>0; }

// ---------- Load
async function handleLoad(){
  $('dataLine').textContent = 'Loading…';
  try{
    // reset
    users.clear(); items.clear(); train.length=0; user2items.clear(); item2users.clear();

    const rec = await fetchFirstExisting(['data/RAW_recipes.csv','data/PP_recipes.csv']);
    const tr  = await fetchFirstExisting(['data/interactions_train.csv']);
    if(!rec || !tr) throw new Error('missing CSVs in ./data/');

    parseRecipes(rec.text);
    parseInteractions(tr.text, train);

    buildIndexers();
    buildTagVocab(parseInt($('kTags').value,10));

    $('dataLine').innerHTML =
      `Users: <b>${fmt(users.size)}</b> · Items: <b>${fmt(items.size)}</b> · Interactions: <b>${fmt(train.length)}</b> `+
      `(files: ${rec.path}, ${tr.path})`;

    // clean old models + projection
    retriever?.dispose(); retriever=null;
    sasrec?.dispose(); sasrec=null;
    itemProjSource?.dispose?.(); itemProjSource=null;
    clearCanvas($('proj').getContext('2d'));
  }catch(e){
    console.error(e);
    $('dataLine').textContent = 'Load failed. Ensure CSVs are in ./data/.';
  }
}

// ---------- Canvas utils
function clearCanvas(ctx){
  const c = ctx.canvas, dpr=window.devicePixelRatio||1;
  const w=c.clientWidth, h=c.clientHeight;
  c.width=Math.max(1,w*dpr); c.height=Math.max(1,h*dpr);
  ctx.setTransform(1,0,0,1,0,0); ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
}
function powerIter(M, v, iters=20){ let x=v; for(let t=0;t<iters;t++){ const y=M.matMul(x); const n=y.norm(); x=y.div(n); } return x; }
function drawProjection(itemEmb){
  if(!itemEmb) return;
  const X = itemEmb;            // [N,D]
  const XT = X.transpose();     // [D,N]
  const C = XT.matMul(X);       // [D,D]
  const v0 = tf.randomNormal([C.shape[0],1]);
  const v1 = powerIter(C, v0, 20);
  const w0 = tf.randomNormal([C.shape[0],1]);
  const C2 = C.sub(v1.matMul(v1.transpose()).mul(C.matMul(v1).transpose().matMul(v1)));
  const v2 = powerIter(C2, w0, 20);
  const P = tf.concat([v1,v2],1);          // [D,2]
  const Y = X.matMul(P);                   // [N,2]
  const arr = Array.from(Y.dataSync());
  const pts = []; for(let i=0;i<arr.length;i+=2) pts.push({x:arr[i], y:arr[i+1]});
  const minx=Math.min(...pts.map(p=>p.x)), maxx=Math.max(...pts.map(p=>p.x));
  const miny=Math.min(...pts.map(p=>p.y)), maxy=Math.max(...pts.map(p=>p.y));
  const norm = pts.map(p=>({x:(p.x-minx)/(maxx-minx+1e-6), y:(p.y-miny)/(maxy-miny+1e-6)}));
  const ctx = $('proj').getContext('2d'); clearCanvas(ctx);
  const W=ctx.canvas.clientWidth,H=ctx.canvas.clientHeight,pad=8;
  ctx.fillStyle="#cbd5e1";
  norm.slice(0,1200).forEach(p=>{
    const x=pad+p.x*(W-pad*2), y=pad+(1-p.y)*(H-pad*2);
    ctx.fillRect(x,y,2,2);
  });
  v0.dispose(); v1.dispose(); w0.dispose(); v2.dispose(); P.dispose(); XT.dispose(); C.dispose(); C2.dispose(); Y.dispose();
}

// ---------- Build SASRec training rows (≥2 events; sliding windows)
function buildSequenceRows(maxLen, limitRows){
  const rows=[];
  for(const [u,h] of user2items.entries()){
    const seq = h.slice().sort((a,b)=>a.ts-b.ts).map(r=>itemIndex.get(r.i)||0);
    if(seq.length<2) continue;
    for(let t=1;t<seq.length;t++){
      const prefix = seq.slice(Math.max(0,t-maxLen), t); // <= maxLen
      const lastPos = prefix.length-1;                   // last valid index after pad
      const x = Array(maxLen).fill(0);
      x.splice(maxLen-prefix.length, prefix.length, ...prefix);
      rows.push({u, seq:x, lastPos: maxLen-1, pos: seq[t]}); // we set lastPos to end; causal mask still ok
      if(rows.length>=limitRows) return rows;
    }
  }
  return rows;
}
function sampleNegatives(batchPos, numItems, k){
  // sample from [1..numItems] excluding batchPos elements (rough but OK for tiny data)
  const out=new Int32Array(batchPos.length*k);
  for(let b=0;b<batchPos.length;b++){
    let j=0;
    while(j<k){
      const cand = 1 + Math.floor(Math.random()*numItems);
      if(cand!==batchPos[b]){ out[b*k+j]=cand; j++; }
    }
  }
  return out;
}

function shuffle(arr){ for(let i=arr.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [arr[i],arr[j]]=[arr[j],arr[i]]; } return arr; }

// ---------- Train Two-Tower
async function trainTT(){
  if(!users.size || !items.size){ $('logTT').value = 'Load data first.'; return; }
  retriever?.dispose(); retriever=null;
  $('logTT').value = 'Training Two-Tower…\n';

  const emb = parseInt($('embTT').value,10);
  const epochs = parseInt($('epTT').value,10);
  const batch  = parseInt($('bsTT').value,10);
  const lr     = parseFloat($('lrTT').value);
  const maxR   = parseInt($('maxRowsTT').value,10);
  buildTagVocab(parseInt($('kTags').value,10));

  // Build small shuffled dataset (u,i) from train
  const rows = shuffle(train.slice(0, maxR));
  retriever = new TwoTowerModel(idx2user.length, idx2item.length+1 /* PAD shift */, emb, {learningRate: lr});

  // We do ID towers only (no extra features) – content comes from tag-aware SASRec. (Simpler + fast)
  // If you want content in retriever, project tags into item tower here and add to itemEmbedding.

  let step=0, totalSteps=Math.ceil(rows.length/batch)*epochs;
  for(let ep=0; ep<epochs; ep++){
    for(let b=0; b<rows.length; b+=batch){
      const chunk = rows.slice(b, b+batch);
      const u = tf.tensor1d(chunk.map(x=>userIndex.get(x.u)),'int32');
      const i = tf.tensor1d(chunk.map(x=> (itemIndex.get(x.i)??0) ),'int32'); // shifted
      const loss = await retriever.trainStep(u,i);
      u.dispose(); i.dispose();
      $('logTT').value += `epoch ${ep+1}/${epochs} · step ${++step}/${totalSteps} · loss=${loss.toFixed(4)}\n`;
      if(step%8===0) $('logTT').scrollTop = $('logTT').scrollHeight;
      await tf.nextFrame();
    }
  }
  $('logTT').value += '✔ Two-Tower done.\n';

  // projection from retriever’s item embeddings (skip PAD at 0)
  const itemEmb = retriever.getItemEmbeddingTensor().slice([1,0]); // drop PAD row
  itemProjSource?.dispose?.(); itemProjSource=itemEmb;
  drawProjection(itemEmb);
}

// ---------- Train SASRec
async function trainSAS(){
  if(!users.size || !items.size){ $('logS').value='Load data first.'; return; }
  $('logS').value = 'Building sequences…\n';
  const d = parseInt($('dS').value,10);
  const L = parseInt($('lenS').value,10);
  const negK = parseInt($('negS').value,10);
  const epochs = parseInt($('epS').value,10);
  const batch = parseInt($('bsS').value,10);
  const lr = parseFloat($('lrS').value);
  const maxRows = parseInt($('maxSeqRows').value,10);

  seqRows = buildSequenceRows(L, maxRows);
  if(!seqRows.length){
    $('logS').value += 'No sequences could be built (need users with ≥2 events).\n';
    return;
  }
  $('logS').value += `Sequences: ${fmt(seqRows.length)} rows.\n`;

  sasrec?.dispose(); sasrec=null;
  sasrec = new SASRecModel(idx2item.length+1 /* PAD shift */, d, L, {learningRate:lr, numNeg: negK});

  const data = shuffle(seqRows.slice());
  let step=0, totalSteps=Math.ceil(data.length/batch)*epochs;

  for(let ep=0; ep<epochs; ep++){
    for(let b=0; b<data.length; b+=batch){
      const chunk = data.slice(b, b+batch);
      const B = chunk.length;

      const seq = tf.tensor2d(chunk.map(r=>r.seq), [B,L], 'int32');
      const lastPos = tf.tensor1d(new Int32Array(B).fill(L-1), 'int32');
      const pos = tf.tensor1d(chunk.map(r=>r.pos), 'int32');

      const neg = sampleNegatives(chunk.map(r=>r.pos), idx2item.length, negK);
      const negT = tf.tensor3d(Array.from(neg), [B,negK,1], 'int32').squeeze([-1]); // [B,k]

      const loss = await sasrec.trainStep(seq,lastPos,pos,negT);
      seq.dispose(); lastPos.dispose(); pos.dispose(); negT.dispose();

      $('logS').value += `epoch ${ep+1}/${epochs} · step ${++step}/${totalSteps} · loss=${loss.toFixed(4)}\n`;
      if(step%8===0) $('logS').scrollTop = $('logS').scrollHeight;
      await tf.nextFrame();
    }
  }
  $('logS').value += '✔ SASRec done.\n';
}

// ---------- Demo
function pickUserWithHistory(){
  const cand = Array.from(user2items.entries()).filter(([,h])=>h.length>=2).map(([u])=>u);
  if(!cand.length) return null;
  return cand[(Math.random()*cand.length)|0];
}
async function demoOnce(){
  if(!sasrec){ $('demoLine').textContent = 'Train SASRec first.'; return; }
  const u = pickUserWithHistory();
  if(!u){ $('demoLine').textContent = 'No user with ≥2 events.'; return; }

  const hist = user2items.get(u).slice().sort((a,b)=>a.ts-b.ts);
  const last15 = hist.slice(-15).reverse();
  $('histTbl').innerHTML = last15.map((r,ix)=>(
    `<tr><td>${ix+1}</td><td>${esc(items.get(r.i)?.title || r.i)}</td><td>${r.r??''}</td></tr>`
  )).join('');

  // Build model input from ALL history (cap at L)
  const L = sasrec.L;
  const seq = hist.map(r=> itemIndex.get(r.i)||0);
  const trimmed = seq.slice(Math.max(0, seq.length - L));
  const x = Array(L).fill(0); x.splice(L-trimmed.length, trimmed.length, ...trimmed);

  const seqT = tf.tensor2d([x], [1,L], 'int32');
  const lastPos = tf.tensor1d([L-1], 'int32');
  let scores = await sasrec.scoreAll(seqT,lastPos); // [N]
  seqT.dispose(); lastPos.dispose();

  // Exclude seen items
  const seen = new Set(hist.map(r=>r.i));
  const arr = Array.from(scores.dataSync()); scores.dispose();
  let pairs = [];
  for(let k=1;k<=idx2item.length;k++){
    const iid = idx2item[k-1]; // shift back
    if(!seen.has(iid)) pairs.push({iid, s: arr[k]}); // index k corresponds to item k (1..N)
  }

  // Optional PPR blend
  if($('usePPR').checked){
    const pr = personalizedPageRankForUser(u, user2items, item2users, {alpha:0.15, iters:20});
    const lambda = 0.20;
    pairs = pairs.map(p=>({iid:p.iid, s: p.s + lambda*(pr.get(p.iid)||0)}));
  }

  pairs.sort((a,b)=>b.s-a.s);
  const top = pairs.slice(0,10);
  $('recsTbl').innerHTML = top.map((p,ix)=>(
    `<tr><td>${ix+1}</td><td>${esc(items.get(p.iid)?.title||p.iid)}</td><td>${p.s.toFixed(3)}</td></tr>`
  )).join('');
  $('demoLine').textContent = `User ${u} · ${top.length} recs computed.`;
}

// ---------- Metrics text
function refreshMetrics(){
  const lines = [];
  lines.push(`Users: ${fmt(users.size)} · Items: ${fmt(items.size)} · Interactions: ${fmt(train.length)}`);
  if(retriever) lines.push(`Two-Tower emb=${retriever.embDim}`);
  if(sasrec) lines.push(`SASRec d=${sasrec.d} L=${sasrec.L}`);
  $('metricsBody').innerHTML = lines.join('<br>');
}

// ---------- Wire
$('btnLoad').addEventListener('click', async()=>{ await handleLoad(); refreshMetrics(); });
$('btnTrainTT').addEventListener('click', async()=>{ await trainTT(); refreshMetrics(); });
$('btnTrainS').addEventListener('click', async()=>{ await trainSAS(); refreshMetrics(); });
$('btnDemo').addEventListener('click', ()=>demoOnce());

// Auto-attach tab behavior
document.addEventListener('DOMContentLoaded', ()=>{ /* nothing else */ });
