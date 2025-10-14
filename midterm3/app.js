/* app.js — orchestrates data, training, demo */

const $ = id => document.getElementById(id);
const fmt = n => (typeof n==='number' ? n.toLocaleString() : n);

let users = new Set();
let items = new Map();           // itemId -> {title, tags[]}
let rows = [];                   // [{u,i,r,ts}]
let user2items = new Map();      // u -> [{i,r,ts}]
let item2users = new Map();      // i -> Set(u)

let userIndex = new Map(), itemIndex = new Map();
let idx2user = [], idx2item = [];

let tag2idx = new Map(), idx2tag = [];
let topKTags = 200;

let retriever = null;
let sasrec = null;

let itemMatProjection = null; // tf.Tensor2d cached for plotting

// ====== UI tabs ======
document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tabpane").forEach(p=>p.classList.add("hidden"));
    $(btn.dataset.tab).classList.remove("hidden");
  });
});

// ====== loading ======
async function fetchFirst(paths){
  for (const p of paths){
    try{
      const r = await fetch(p, {cache:'no-store'});
      if (r.ok) return {path:p, text:await r.text()};
    }catch{}
  }
  return null;
}
function splitCSVLine(s){ return s.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g); }
function splitLines(s){ return s.split(/\r?\n/).filter(Boolean); }

function parseRecipes(csv){
  items.clear();
  const lines = splitLines(csv);
  const header = splitCSVLine(lines.shift());
  const idIdx = header.findIndex(h=>/^id$|(^|_)id$/i.test(h));
  const nameIdx = header.findIndex(h=>/(name|title)/i.test(h));
  const tagsIdx = header.findIndex(h=>/tags/i.test(h));
  for (const ln of lines){
    const cols = splitCSVLine(ln);
    const id = parseInt(cols[idIdx], 10);
    if (!Number.isInteger(id)) continue;
    const title = (cols[nameIdx]||`Recipe ${id}`).replace(/^"|"$/g,'');
    let tags = [];
    if (tagsIdx >= 0 && cols[tagsIdx]){
      let raw = cols[tagsIdx].trim();
      raw = raw.replace(/^\s*\[|\]\s*$/g,''); // strip []
      tags = raw.split(/['"]\s*,\s*['"]|,\s*/g)
                .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
                .filter(Boolean).slice(0,24);
    }
    items.set(id, {title, tags});
  }
}

function parseInteractions(csv){
  rows.length=0; users.clear(); user2items.clear(); item2users.clear();
  const lines = splitLines(csv);
  const header = splitCSVLine(lines.shift());
  const uIdx = header.findIndex(h=>/user/i.test(h));
  const iIdx = header.findIndex(h=>/(item|recipe)_?id/i.test(h));
  const rIdx = header.findIndex(h=>/rating/i.test(h));
  const tIdx = header.findIndex(h=>/(time|date)/i.test(h));
  for (const ln of lines){
    const cols = splitCSVLine(ln);
    const u = parseInt(cols[uIdx],10);
    const i = parseInt(cols[iIdx],10);
    if (!Number.isInteger(u) || !Number.isInteger(i)) continue;
    const r = rIdx>=0 && cols[rIdx]!=='' ? parseFloat(cols[rIdx]) : 5;
    const ts = tIdx>=0 ? (Date.parse(cols[tIdx]) || 0) : 0;
    rows.push({u,i,r,ts});
    users.add(u);
    if (!user2items.has(u)) user2items.set(u,[]);
    user2items.get(u).push({i,r,ts});
    if (!item2users.has(i)) item2users.set(i,new Set());
    item2users.get(i).add(u);
  }
  // sort by time per user
  user2items.forEach(arr=>arr.sort((a,b)=>a.ts-b.ts));
}

function buildIndexers(){
  idx2user = Array.from(users).sort((a,b)=>a-b);
  idx2item = Array.from(items.keys()).sort((a,b)=>a-b);
  userIndex = new Map(idx2user.map((u,ix)=>[u,ix]));
  itemIndex = new Map(idx2item.map((i,ix)=>[i,ix]));
}

function buildTagVocab(K){
  topKTags = K;
  const freq = new Map();
  items.forEach(obj=> (obj.tags||[]).forEach(t=>freq.set(t,(freq.get(t)||0)+1)));
  const top = Array.from(freq.entries()).sort((a,b)=>b[1]-a[1]).slice(0,K);
  tag2idx = new Map(top.map(([t],ix)=>[t,ix]));
  idx2tag = top.map(([t])=>t);
}

async function handleLoad(){
  try{
    $('status').textContent = 'Status: loading…';
    const rec = await fetchFirst(['./data/PP_recipes.csv','./data/RAW_recipes.csv','PP_recipes.csv','RAW_recipes.csv']);
    const tr  = await fetchFirst(['./data/interactions_train.csv','interactions_train.csv']);
    if (!rec || !tr) throw new Error('CSV files not found in ./data/');
    parseRecipes(rec.text);
    parseInteractions(tr.text);
    buildIndexers();
    buildTagVocab(parseInt(($('rtK').value||'200'),10));

    const density = (rows.length/(users.size*Math.max(1,items.size))).toExponential(2);
    $('dsLine').textContent =
      `Users: ${fmt(users.size)} · Items: ${fmt(items.size)} · Interactions: ${fmt(rows.length)} · Density: ${density} · Ratings: yes`;
    $('status').textContent = 'Status: loaded.';
    drawHist();
    drawTopTags();
  }catch(e){
    console.error(e); $('status').textContent = 'Status: failed to load.';
  }
}

function drawBars(id, arr){
  const ctx = $(id).getContext('2d');
  const c = ctx.canvas;
  const dpr = window.devicePixelRatio||1;
  c.width = c.clientWidth*dpr; c.height = c.clientHeight*dpr; ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,c.clientWidth,c.clientHeight);
  const W=c.clientWidth,H=c.clientHeight,p=26; const bw=(W-p*2)/arr.length-6, base=H-p;
  ctx.strokeStyle="#263247"; ctx.beginPath(); ctx.moveTo(p,base+0.5); ctx.lineTo(W-p,base+0.5); ctx.stroke();
  ctx.fillStyle="#e5e7eb";
  arr.forEach((v,i)=>{ const h=(H-p*2)*v/Math.max(1,Math.max(...arr)); const x=p+i*(bw+6); ctx.fillRect(x,base-h,bw,h);});
}
function drawHist(){
  const hist=[0,0,0,0,0];
  rows.forEach(r=>{ const v=Math.min(5,Math.max(1,Math.round(r.r||5))); hist[v-1]++; });
  drawBars('hist', hist);
}
function drawTopTags(){
  const f=new Map(); items.forEach(it=> (it.tags||[]).forEach(t=>f.set(t,(f.get(t)||0)+1)));
  const top = Array.from(f.entries()).sort((a,b)=>b[1]-a[1]).slice(0,30);
  drawBars('topTags', top.map(([,c])=>c));
}

// ====== training: retriever ======
function itemTagVectorOfIndex(iIdx){
  const iid = idx2item[iIdx]; const obj = items.get(iid); const K=topKTags;
  const v = new Array(K).fill(0);
  if (obj && obj.tags){
    for (const t of obj.tags){
      if (tag2idx.has(t)) v[tag2idx.get(t)]=1;
    }
  }
  return v;
}

function makeShuffled(arr,maxN){ const A=maxN?arr.slice(0,maxN):arr.slice(); for(let i=A.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [A[i],A[j]]=[A[j],A[i]];} return A;}

async function trainRetriever(){
  if (!users.size){ $('logRetriever').value += `Load data first.\n`; return; }
  retriever?.dispose?.(); retriever=null; itemMatProjection?.dispose?.(); itemMatProjection=null;

  const emb=parseInt($('rtEmb').value,10);
  const ep =parseInt($('rtEp').value,10);
  const ba =parseInt($('rtBa').value,10);
  const lr =parseFloat($('rtLr').value);
  const K  =parseInt($('rtK').value,10);
  const max=parseInt($('rtMax').value,10);

  buildTagVocab(K);

  const data = makeShuffled(rows, max);
  retriever = new TwoTowerRetriever(idx2user.length, idx2item.length, emb, K, {learningRate:lr});

  const tagMat = tf.tensor2d(idx2item.map((_,ix)=> itemTagVectorOfIndex(ix)), [idx2item.length, K], 'float32');
  await retriever.compile(tagMat);

  $('logRetriever').value = '';
  let step=0;
  for (let e=0;e<ep;e++){
    for (let i=0;i<data.length;i+=ba){
      const slice = data.slice(i, i+ba);
      const u = tf.tensor1d(slice.map(r=> userIndex.get(r.u) ), 'int32');
      const it= tf.tensor1d(slice.map(r=> itemIndex.get(r.i) ), 'int32');
      const loss = await retriever.trainStep(u,it);
      u.dispose(); it.dispose();
      step++;
      if (step%5===0){ $('logRetriever').value += `epoch ${e+1}/${ep} · step ${i/ba+1}/${Math.ceil(data.length/ba)} · loss=${loss.toFixed(4)}\n`; $('logRetriever').scrollTop = $('logRetriever').scrollHeight; await tf.nextFrame(); }
    }
  }
  $('logRetriever').value += `✅ Two-Tower done. Steps=${step}.\n`;
  drawProjection(retriever.getItemEmbMatrix());
  $('metricsBody').textContent = `Retriever trained: emb=${emb}, epochs=${ep}, batch=${ba}, lr=${lr}`;
}

// ====== SASRec ======
function buildUserSeqsPlus1(){
  // Map u -> [itemIdx+1] with ≥3 events
  const m = new Map();
  user2items.forEach((arr,u)=>{
    if (arr.length<3) return;
    const seq = arr.map(e=> 1 + itemIndex.get(e.i)).filter(x=>Number.isInteger(x));
    if (seq.length>=3) m.set(u, seq);
  });
  return m;
}

async function trainSASRec(){
  if (!users.size){ $('logSAS').value += `Load data first.\n`; return; }
  const d=parseInt($('srD').value,10);
  const L=parseInt($('srL').value,10);
  const neg=parseInt($('srNeg').value,10);
  const ep=parseInt($('srEp').value,10);
  const ba=parseInt($('srBa').value,10);
  const lr=parseFloat($('srLr').value);
  const max=parseInt($('srMax').value,10);

  sasrec?.dispose?.(); sasrec=null;

  const userSeqs = buildUserSeqsPlus1();
  if (userSeqs.size===0){ $('logSAS').value = 'No sequences could be built (need users with ≥3 events).'; return; }

  sasrec = new SASRec(idx2item.length+1, d, L, {learningRate:lr});
  await sasrec.compile();

  const samples = sasrec.buildSamples(userSeqs, max, neg);
  if (!samples){ $('logSAS').value = 'Could not form samples.'; return; }

  $('logSAS').value = '';
  sasrec.onStep = ({ep, step, loss})=>{
    $('logSAS').value += `epoch ${ep}/${ep} · step ${step} · loss=${loss.toFixed(4)}\n`;
    $('logSAS').scrollTop = $('logSAS').scrollHeight;
  };
  await sasrec.train(samples, ep, ba);
  $('logSAS').value += `✅ SASRec done.\n`;
  $('metricsBody').textContent += ` | SASRec trained: d=${d}, L=${L}, neg=${neg}, epochs=${ep}`;
}

// ====== projection (PCA-ish via power-iter SVD on X^T X) ======
function powerIter(M, v, iters=18){ let x=v; for(let t=0;t<iters;t++){ const y=M.matMul(x); const n=y.norm(); x=y.div(n);} return x;}
function drawProjection(itemMat){
  if (!itemMat) return;
  const X=itemMat; const XT=X.transpose(); const C=XT.matMul(X);
  const v0=tf.randomNormal([C.shape[0],1]), v1=powerIter(C,v0,20);
  const w0=tf.randomNormal([C.shape[0],1]);
  const P1=v1; const C2=C.sub(P1.matMul(P1.transpose()).matMul(C));
  const v2=powerIter(C2,w0,20);
  const P=tf.concat([P1,v2],1);
  const Y=X.matMul(P); const arr=Array.from(Y.dataSync());
  const pts=[]; for(let i=0;i<arr.length;i+=2) pts.push({x:arr[i],y:arr[i+1]});
  const minx=Math.min(...pts.map(p=>p.x)), maxx=Math.max(...pts.map(p=>p.x));
  const miny=Math.min(...pts.map(p=>p.y)), maxy=Math.max(...pts.map(p=>p.y));
  const norm=pts.slice(0,1500).map(p=>({x:(p.x-minx)/(maxx-minx+1e-6), y:(p.y-miny)/(maxy-miny+1e-6)}));

  const ctx=$('proj').getContext('2d');
  const c=ctx.canvas, dpr=window.devicePixelRatio||1; c.width=c.clientWidth*dpr; c.height=c.clientHeight*dpr; ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,c.clientWidth,c.clientHeight);
  const W=c.clientWidth,H=c.clientHeight,p=8;
  ctx.fillStyle="#cbd5e1";
  norm.forEach(p=>{
    const x=p+''; // noop
  });
  norm.forEach(p=>{
    const x=p.x*(W-p*2)+p, y=(1-p.y)*(H-p*2)+p;
    ctx.fillRect(x,y,2,2);
  });

  v0.dispose(); v1.dispose(); w0.dispose(); v2.dispose(); P.dispose(); XT.dispose(); C.dispose(); C2.dispose(); Y.dispose();
}

// ====== demo ======
function pickUserWithK(least=3){
  const cand = Array.from(user2items.entries()).filter(([,arr])=>arr.length>=least).map(([u])=>u);
  if (!cand.length) return null;
  return cand[(Math.random()*cand.length)|0];
}

async function runDemo(){
  $('demoLine').textContent = '—';
  if (!retriever || !sasrec){ $('demoLine').textContent = 'Train both models first.'; return; }
  const u = pickUserWithK(3);
  if (!u){ $('demoLine').textContent = 'Need users with ≥3 events.'; return; }

  const hist = user2items.get(u).slice(-15);
  $('histTbl').innerHTML = hist.map((h,ix)=> `<tr><td>${ix+1}</td><td>${escape(items.get(h.i)?.title || h.i)}</td></tr>`).join('');

  // 1) Retriever: top 200 (exclude seen)
  const uIdx = tf.tensor1d([userIndex.get(u)], 'int32');
  const s = await retriever.scoreUserAgainstAll(uIdx); const arr = Array.from(s.dataSync()); s.dispose(); uIdx.dispose();
  const seen = new Set((user2items.get(u)||[]).map(x=> itemIndex.get(x.i)));
  const candIdx = arr.map((v,ii)=>({ii,v})).filter(x=>!seen.has(x.ii)).sort((a,b)=>b.v-a.v).slice(0,200).map(x=>x.ii);

  // 2) SASRec score on candidates
  const seqPlus1 = user2items.get(u).slice(-parseInt($('srL').value,10)).map(e=> 1+itemIndex.get(e.i));
  const pad = Array(Math.max(0, parseInt($('srL').value,10)-seqPlus1.length)).fill(0).concat(seqPlus1);
  const seqT = tf.tensor2d([pad], [1, parseInt($('srL').value,10)], 'int32');
  const candPlus1 = tf.tensor2d([candIdx.map(ii=>1+ii)], [1, candIdx.length], 'int32');
  let logits = sasrec.scoreNext(seqT, candPlus1).squeeze(); // [C]
  seqT.dispose(); candPlus1.dispose();

  // 3) PPR (optional) — blend
  if ($('chkPPR').checked){
    const pr = personalizedPageRankForUser(u, user2items, item2users, {alpha:0.15, iters:20});
    const prv = candIdx.map(ii=> pr.get(idx2item[ii])||0 );
    const prT = tf.tensor1d(prv,'float32');
    logits = logits.add(prT.mul(0.15)); // blend
    prT.dispose();
  }

  const sc = Array.from(logits.dataSync());
  logits.dispose();
  const top10 = candIdx.map((ii,ix)=>({ii,score:sc[ix]})).sort((a,b)=>b.score-a.score).slice(0,10);

  $('nextTbl').innerHTML = top10.map((r,ix)=>(
    `<tr><td>${ix+1}</td><td>${escape(items.get(idx2item[r.ii])?.title || idx2item[r.ii])}</td><td>${r.score.toFixed(3)}</td></tr>`
  )).join('');
  $('demoLine').textContent = `User ${u} — generated successfully.`;
}

function escape(s){ return (s??'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m])); }

// ====== wire up ======
$('btnLoad').addEventListener('click', handleLoad);
$('btnTrainRetriever').addEventListener('click', ()=>trainRetriever().catch(console.error));
$('btnTrainSASRec').addEventListener('click', ()=>trainSASRec().catch(console.error));
$('btnDemo').addEventListener('click', ()=>runDemo().catch(console.error));
