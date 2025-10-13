/* app.js — data, EDA, training loops, demo, metrics */

const $ = (id)=>document.getElementById(id);
const fmt = (n)=> typeof n==='number' ? n.toLocaleString() : n;

// ---------- global state ----------
let users = new Set();
let items = new Map();   // itemId -> {title, tags:[]}
let train = [];          // [{u,i,r,ts}]
let valid = [];
let user2items = new Map(); // userId -> [{i,r,ts}]
let item2users = new Map(); // itemId -> Set(userId)

let userIndex = new Map(), itemIndex = new Map();
let idx2user = [], idx2item = [];

// tag vocab
let tag2idx = new Map(), idx2tag = [];
let topTagK = 200;
let itemTagMat = null; // tf.Tensor2d after build

// models
let tw = null;          // TwoTowerModel
let sa = null;          // SASRec
let lastItemEmb = null; // for projection

// traces
let twTrace=[], saTrace=[];

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tabpane").forEach(p=>p.classList.add("hidden"));
    $(btn.dataset.tab).classList.remove("hidden");
  });
});

// ---------- loader ----------
async function fetchFirstExisting(paths){
  for(const p of paths){
    try{
      const r = await fetch(p,{cache:'no-store'}); if(r.ok){ return {path:p,text:await r.text()};}
    }catch(_){}
  }
  return null;
}
function splitLines(t){ return t.split(/\r?\n/).filter(Boolean); }

function parseRecipes(csv){
  const lines = splitLines(csv); if(!lines.length) return;
  const header = lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
  const idI = header.findIndex(h=>/^id$|(^|_)id$/i.test(h));
  const nameI = header.findIndex(h=>/(name|title)/i.test(h));
  const tagsI = header.findIndex(h=>/tags/i.test(h));
  for(const ln of lines){
    const c = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
    const id = parseInt(c[idI],10); if(!Number.isInteger(id)) continue;
    const title = (c[nameI]||`Recipe ${id}`).replace(/^"|"$/g,'');
    let tags=[];
    if(tagsI>=0 && c[tagsI]){
      let raw = c[tagsI].trim().replace(/^\[/,'').replace(/\]$/,'');
      tags = raw.split(/['"]\s*,\s*['"]|,\s*/g)
                .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
                .filter(Boolean).slice(0,24);
    }
    items.set(id,{title, tags});
  }
}
function parseInteractions(csv, sink){
  const lines = splitLines(csv); if(!lines.length) return;
  const header = lines.shift().split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
  const uI = header.findIndex(h=>/user/.test(h));
  const iI = header.findIndex(h=>/(item|recipe)_?id/i.test(h));
  const rI = header.findIndex(h=>/rating/i.test(h));
  const tI = header.findIndex(h=>/(time|date)/i.test(h));
  for(const ln of lines){
    const c = ln.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
    const u = parseInt(c[uI],10), i = parseInt(c[iI],10);
    if(!Number.isInteger(u)||!Number.isInteger(i)) continue;
    const r = rI>=0 && c[rI]!=='' ? parseFloat(c[rI]) : 1;
    const ts = tI>=0 ? (Date.parse(c[tI]) || parseInt(c[tI],10) || 0) : 0;
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
  userIndex = new Map(idx2user.map((u,k)=>[u,k]));
  itemIndex = new Map(idx2item.map((i,k)=>[i,k]));
}
function buildTagVocab(k){
  topTagK = k;
  const f = new Map();
  for(const it of items.values()){ for(const t of (it.tags||[])){ f.set(t,(f.get(t)||0)+1); } }
  const top = Array.from(f.entries()).sort((a,b)=>b[1]-a[1]).slice(0,k);
  tag2idx = new Map(top.map(([t],i)=>[t,i])); idx2tag = top.map(([t])=>t);
}

function itemTagVector(itemId){
  const it = items.get(itemId);
  const v = new Array(topTagK).fill(0);
  if(it && it.tags){ for(const t of it.tags){ const k = tag2idx.get(t); if(k!==undefined) v[k]=1; } }
  return v;
}

async function handleLoad(){
  try{
    $('status').textContent='Status: loading…';

    const rec = await fetchFirstExisting(['data/PP_recipes.csv','data/RAW_recipes.csv','./PP_recipes.csv','./RAW_recipes.csv']);
    const tr  = await fetchFirstExisting(['data/interactions_train.csv','./interactions_train.csv']);
    const va  = await fetchFirstExisting(['data/interactions_validation.csv','./interactions_validation.csv']);

    if(!rec || !tr) throw new Error('CSV files not found next to index.html (in ./data/)');

    // reset
    items.clear(); users.clear(); train.length=0; valid.length=0;
    user2items.clear(); item2users.clear();

    parseRecipes(rec.text);
    parseInteractions(tr.text, train);
    if(va) parseInteractions(va.text, valid);

    // optional 80/20 by time if no valid
    if(!valid.length && train.length){
      const sorted = train.slice().sort((a,b)=>a.ts-b.ts);
      valid = sorted.filter((_,ix)=>ix%5===0);
      train = sorted.filter((_,ix)=>ix%5!==0);
    }

    buildIndexers();
    buildTagVocab(parseInt(($('twK').value||'200'),10));

    // build itemTagMat tensor (in item index space)
    if(itemTagMat) { itemTagMat.dispose(); itemTagMat=null; }
    itemTagMat = tf.tensor2d(idx2item.map(id=>itemTagVector(id)), [idx2item.length, topTagK], 'float32');

    const density = (train.length / (users.size * Math.max(1,items.size))).toExponential(2);
    const coldU = Array.from(user2items.values()).filter(a=>a.length<5).length;
    const coldI = Array.from(items.keys()).filter(i=>!item2users.has(i) || item2users.get(i).size<5).length;

    $('dsLine').textContent = `Users: ${fmt(users.size)} Items: ${fmt(items.size)} Train interactions: ${fmt(train.length)} · Density: ${density} · Ratings: ${train.some(x=>x.r!=null)?'yes':'no'} · Cold users<5: ${fmt(coldU)} · Cold items<5: ${fmt(coldI)} (files: ${rec.path}, ${tr.path}${va?`, ${va.path}`:''})`;
    $('status').textContent='Status: loaded.';

    drawEDA();
  }catch(err){
    console.error(err);
    $('status').textContent='Status: load failed. Ensure CSVs are in ./data/.';
  }
}

// ---------- EDA (lightweight) ----------
function clearCanvas(ctx){
  const c = ctx.canvas, dpr = window.devicePixelRatio||1;
  const w=c.clientWidth,h=c.clientHeight; c.width=w*dpr; c.height=h*dpr; ctx.setTransform(1,0,0,1,0,0); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
}
function drawBars(id, arr){
  const ctx=$(id).getContext('2d'); clearCanvas(ctx);
  const W=ctx.canvas.clientWidth,H=ctx.canvas.clientHeight,p=26;
  const bw = (W-p*2)/arr.length-6, base=H-p, max=Math.max(...arr,1);
  ctx.strokeStyle="#1f2937"; ctx.beginPath(); ctx.moveTo(p,base+0.5); ctx.lineTo(W-p,base+0.5); ctx.stroke();
  ctx.fillStyle="#e5e7eb";
  arr.forEach((v,i)=>{ const h=(v/max)*(H-p*2); const x=p+i*(bw+6); ctx.fillRect(x,base-h,bw,h); });
}
function drawEDA(){
  if(!train.length) return;
  const hist=[0,0,0,0,0];
  for(const r of train){ const v=Math.round(Math.max(1,Math.min(5,r.r||1))); hist[v-1]++; }
  drawBars('histRatings',hist);

  const ucnt=new Map(); for(const r of train){ ucnt.set(r.u,(ucnt.get(r.u)||0)+1); }
  const ub=[0,0,0,0,0,0,0,0]; for(const v of ucnt.values()){ const i=v===1?0: v<=2?1: v<=3?2: v<=5?3: v<=10?4: v<=20?5: v<=50?6:7; ub[i]++; }
  drawBars('histUser',ub);

  const icnt=new Map(); for(const r of train){ icnt.set(r.i,(icnt.get(r.i)||0)+1); }
  const ib=[0,0,0,0,0,0,0,0,0]; for(const v of icnt.values()){ const i=v===1?0: v<=2?1: v<=3?2: v<=5?3: v<=10?4: v<=20?5: v<=100?6: v<=500?7:8; ib[i]++; }
  drawBars('histItem',ib);

  // top tags
  const f=new Map(); for(const it of items.values()){ for(const t of it.tags||[]) if(tag2idx.has(t)) f.set(t,(f.get(t)||0)+1); }
  const top20=Array.from(f.entries()).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>x[1]);
  drawBars('topTags',top20);
}

// ---------- helpers ----------
function buildBatchTensors(batch){
  const u = tf.tensor1d(batch.map(x=>userIndex.get(x.u)),'int32');
  const i = tf.tensor1d(batch.map(x=>itemIndex.get(x.i)),'int32');
  const r = tf.tensor1d(batch.map(x=>x.r??1),'float32');
  return {u,i,r};
}
function makeShuffled(arr, maxN){
  const A = maxN ? arr.slice(0, maxN) : arr.slice();
  for(let i=A.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [A[i],A[j]]=[A[j],A[i]]; }
  return A;
}
function escapeHtml(s){ return (s??'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;' }[m])); }

// ---------- Two-Tower training ----------
async function trainTwoTower(){
  if(!users.size) return;
  tw?.dispose?.(); tw=null; lastItemEmb?.dispose?.(); lastItemEmb=null; twTrace=[]; drawLine('twLoss',twTrace);

  const emb=parseInt($('twEmb').value,10);
  const epochs=parseInt($('twEp').value,10);
  const batch=parseInt($('twBa').value,10);
  const lr=parseFloat($('twLr').value);
  const K=parseInt($('twK').value,10);
  const maxR=parseInt($('twMax').value,10);

  $('twLine').textContent='Training Two-Tower…';
  buildTagVocab(K);
  itemTagMat?.dispose?.();
  itemTagMat = tf.tensor2d(idx2item.map(id=>itemTagVector(id)), [idx2item.length, topTagK],'float32');

  tw = new TwoTowerModel(idx2user.length, idx2item.length, emb, topTagK, {learningRate:lr});
  await tw.compile(); tw.setItemTagMatrix(itemTagMat);

  const data = makeShuffled(train, maxR);
  let step=0;
  for(let ep=0; ep<epochs; ep++){
    for(let b=0; b<data.length; b+=batch){
      const slice = data.slice(b, b+batch);
      const {u,i} = buildBatchTensors(slice);
      const loss = await tw.trainStep(u,i);
      u.dispose(); i.dispose();
      twTrace.push({x:++step, y:loss}); if(step%5===0) drawLine('twLoss',twTrace);
      await tf.nextFrame();
    }
  }
  $('twLine').innerHTML=`Done. Final loss <b>${twTrace.at(-1).y.toFixed(4)}</b>`;
  lastItemEmb = tw.itemEmbedding.read();
  drawProjection(lastItemEmb);
  computeMetrics().catch(console.error);
}

// ---------- Build sequences for SASRec ----------
function buildSequences(maxLen){
  // per-user sorted by time -> sliding windows
  const seqs = []; // {seq: [L], pos: i}
  for(const [u, arr] of user2items.entries()){
    const sorted = arr.slice().sort((a,b)=>a.ts-b.ts);
    const idxs = sorted.map(x=> itemIndex.get(x.i)+1 ); // +1 shift; 0 reserved as PAD
    for(let t=1; t<idxs.length; t++){
      const start = Math.max(0, t-maxLen);
      const hist = idxs.slice(start, t);
      const pad = Array(Math.max(0, maxLen-hist.length)).fill(0).concat(hist); // left pad
      const pos = idxs[t];
      seqs.push({u, seq: pad, pos});
    }
  }
  return seqs;
}

// ---------- SASRec training ----------
async function trainSASRec(){
  if(!users.size) return;
  sa?.dispose?.(); sa=null; saTrace=[]; drawLine('saLoss',saTrace);

  const d=parseInt($('saDim').value,10);
  const L=parseInt($('saLen').value,10);
  const K=parseInt($('saNeg').value,10);
  const epochs=parseInt($('saEp').value,10);
  const batch=parseInt($('saBa').value,10);
  const lr=parseFloat($('saLr').value);
  const maxRows=parseInt($('saMax').value,10);

  $('saLine').textContent='Preparing sequences…';
  let seqs = buildSequences(L);
  if(!seqs.length){ $('saLine').textContent='No sequences available (need timestamps).'; return; }
  seqs = makeShuffled(seqs, maxRows);

  sa = new SASRec(idx2item.length+1, d, L, {negatives:K, learningRate:lr});

  // fast negative sampler avoiding the positive id
  const sampler = (posTensor, K)=>{
    const B = posTensor.shape[0];
    let rnd = tf.randomUniform([B,K], 1, sa.numItems, 'int32');
    // (light check) If equals pos, replace with 1 (rare)
    const eq = rnd.equal(posTensor.expandDims(1));
    rnd = tf.where(eq, tf.onesLike(rnd), rnd);
    eq.dispose(); return rnd;
  };

  $('saLine').textContent='Training SASRec…';
  let step=0;
  for(let ep=0; ep<epochs; ep++){
    for(let b=0; b<seqs.length; b+=batch){
      const slice = seqs.slice(b, b+batch);
      const S = tf.tensor2d(slice.map(x=>x.seq), [slice.length, L], 'int32');
      const P = tf.tensor1d(slice.map(x=>x.pos), 'int32');
      const loss = await sa.trainStep(S,P,sampler);
      S.dispose(); P.dispose();
      saTrace.push({x:++step, y:loss}); if(step%5===0) drawLine('saLoss',saTrace);
      await tf.nextFrame();
    }
  }
  $('saLine').innerHTML=`Done. Final loss <b>${saTrace.at(-1).y.toFixed(4)}</b>`;
  computeMetrics().catch(console.error);
}

// ---------- Projection (PCA-ish power method) ----------
function powerIter(M, v, iters=20){ let x=v; for(let t=0;t<iters;t++){ const y=M.matMul(x); const n=y.norm(); x=y.div(n); } return x; }
function drawProjection(itemEmb){
  if(!itemEmb) return;
  const X=itemEmb, XT=X.transpose(), C=XT.matMul(X);
  const v0=tf.randomNormal([C.shape[0],1]); const v1=powerIter(C,v0,20);
  const w0=tf.randomNormal([C.shape[0],1]); const C2=C.sub(v1.matMul(v1.transpose()).mul(C.matMul(v1).transpose().matMul(v1)));
  const v2=powerIter(C2,w0,20), P=tf.concat([v1,v2],1), Y=X.matMul(P);
  const arr=Array.from(Y.dataSync()); const pts=[]; for(let i=0;i<arr.length;i+=2) pts.push({x:arr[i],y:arr[i+1]});
  const minx=Math.min(...pts.map(p=>p.x)),maxx=Math.max(...pts.map(p=>p.x)); const miny=Math.min(...pts.map(p=>p.y)),maxy=Math.max(...pts.map(p=>p.y));
  const norm=pts.map(p=>({x:(p.x-minx)/(maxx-minx+1e-6), y:(p.y-miny)/(maxy-miny+1e-6)}));
  const ctx=$('proj').getContext('2d'); clearCanvas(ctx);
  const W=ctx.canvas.clientWidth,H=ctx.canvas.clientHeight,p=8; ctx.fillStyle="#cbd5e1";
  norm.slice(0,1500).forEach(p=>{ const x=p*(0); }); // no-op to avoid linter
  norm.slice(0,1500).forEach(p=>{ const x=p.x*(W-2*p)+p, y=(1-p.y)*(H-2*p)+p; ctx.fillRect(x,y,2,2); });
  v0.dispose(); v1.dispose(); w0.dispose(); v2.dispose(); P.dispose(); XT.dispose(); C.dispose(); C2.dispose(); Y.dispose();
}

// ---------- small chart util ----------
function drawLine(id, pts){
  const ctx=$(id).getContext('2d'); clearCanvas(ctx);
  const W=ctx.canvas.clientWidth,H=ctx.canvas.clientHeight,p=18,base=H-p;
  const maxX=Math.max(...pts.map(d=>d.x),1), maxY=Math.max(...pts.map(d=>d.y),1);
  ctx.strokeStyle="#7dd3fc"; ctx.beginPath();
  pts.forEach((d,j)=>{ const x=p+(d.x/maxX)*(W-p*2), y=base-(d.y/maxY)*(H-p*2); j?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke();
}

// ---------- Demo ----------
function pickUser(minR){
  const c=new Map(); for(const r of train) c.set(r.u,(c.get(r.u)||0)+1);
  const cand = Array.from(c.entries()).filter(([,n])=>n>=minR).map(([u])=>u);
  if(!cand.length){ const max=Math.max(0,...c.values()); return {user: Array.from(c.entries()).filter(([,n])=>n===max)[0][0], usedMin:max}; }
  return {user: cand[(Math.random()*cand.length)|0], usedMin:minR};
}

async function demoOnce(){
  if(!users.size){ $('demoLine').textContent='Load data first.'; return; }
  if(!tw && !sa){ $('demoLine').textContent='Train at least Two-Tower or SASRec.'; return; }

  const minR=parseInt($('minR').value,10);
  const pick=pickUser(minR); const u=pick.user;
  $('demoLine').textContent = `Testing with user ${u} (threshold used: ${pick.usedMin}).`;

  const hist=(user2items.get(u)||[]).slice().sort((a,b)=> (b.r-a.r)||(b.ts-a.ts)).slice(0,10);
  $('histTbl').innerHTML = hist.map((row,ix)=>`<tr><td>${ix+1}</td><td>${escapeHtml(items.get(row.i)?.title||row.i)}</td><td>${row.r??''}</td></tr>`).join('');

  // candidates from retriever (exclude seen)
  const seen=new Set((user2items.get(u)||[]).map(x=>x.i));
  const uIdx=tf.tensor1d([userIndex.get(u)],'int32');
  let cand = idx2item.map((iid,ii)=>({iid,ii})).filter(x=>!seen.has(x.iid));

  let twScores = [];
  if(tw){
    const s = await tw.scoreUserAgainstAll(uIdx); twScores = Array.from(s.dataSync()); s.dispose();
    // top 200 candidates
    cand = cand.map(o=>({ii:o.ii, iid:o.iid, s:twScores[o.ii]})).sort((a,b)=>b.s-a.s).slice(0,200);
  }else{
    // if no retriever, start with all unseen
    cand = cand.slice(0,200).map(o=>({ii:o.ii, iid:o.iid, s:0}));
  }

  // render retriever top-10
  $('twTbl').innerHTML = cand.slice(0,10).map((o,ix)=>`<tr><td>${ix+1}</td><td>${escapeHtml(items.get(o.iid)?.title||o.iid)}</td><td>${o.s.toFixed(3)}</td></tr>`).join('');

  // SASRec rerank by true sequence
  let final = cand;
  if(sa){
    const seqL=parseInt($('saLen').value,10);
    const histSorted=(user2items.get(u)||[]).slice().sort((a,b)=>a.ts-b.ts);
    const idxs = histSorted.map(x=> itemIndex.get(x.i)+1 );
    const last = idxs.slice(-seqL);
    const seq = Array(Math.max(0, seqL-last.length)).fill(0).concat(last);
    const S = tf.tensor2d([seq],[1,seqL],'int32');
    const scores = await sa.scoreItems(S, cand.map(x=>x.ii+1)); // +1 because SASRec uses 1..N
    const arr = Array.from(scores.dataSync());
    scores.dispose(); S.dispose();
    final = cand.map((o,ix)=>({ ...o, sSa: arr[ix]}));
  }else{
    final = cand.map(o=>({ ...o, sSa: 0 }));
  }

  // Optional PPR re-rank
  if($('usePPR').checked){
    const pr = personalizedPageRankForUser(u, user2items, item2users, {alpha:0.15,iters:20});
    final = final.map(o=>({ ...o, sPpr: (pr.get(o.iid)||0) }));
  }else{
    final = final.map(o=>({ ...o, sPpr: 0 }));
  }

  // blend: 0.3 SASRec, 0.1 PPR, 0.6 retriever (normalize z-score like)
  function norm(arr){ const m=arr.reduce((s,v)=>s+v,0)/Math.max(1,arr.length); const sd=Math.sqrt(arr.reduce((s,v)=>s+(v-m)*(v-m),0)/Math.max(1,arr.length)); return arr.map(v=> sd>1e-6 ? (v-m)/sd : 0); }
  const nzTw = norm(final.map(x=>x.s));
  const nzSa = norm(final.map(x=>x.sSa));
  const nzPr = norm(final.map(x=>x.sPpr));
  final = final.map((o,ix)=>({ ...o, score: 0.6*nzTw[ix] + 0.3*nzSa[ix] + 0.1*nzPr[ix] }))
               .sort((a,b)=>b.score-a.score)
               .slice(0,10);

  $('saTbl').innerHTML = final.map((o,ix)=>`<tr><td>${ix+1}</td><td>${escapeHtml(items.get(o.iid)?.title||o.iid)}</td><td>${o.score.toFixed(3)}</td></tr>`).join('');
  uIdx.dispose();
  $('demoLine').textContent += ' — recommendations generated!';
}

// ---------- Metrics (Recall@10 & nDCG@10 on valid users) ----------
async function computeMetrics(){
  const body=$('metricsBody'); if(!valid.length){ body.textContent='No validation split available.'; return; }

  const sampleNeg = 500, K=10;
  const usersV = Array.from(new Set(valid.map(x=>x.u)));
  const pick = usersV.slice(0, Math.min(120, usersV.length));

  async function evalModel(){
    let hits=0, dcg=0, ideal=0;
    for(const u of pick){
      const seen = new Set((user2items.get(u)||[]).map(x=>x.i));
      const pos = valid.filter(x=>x.u===u).map(x=>x.i); if(!pos.length) continue;

      const neg=[]; while(neg.length<sampleNeg){ const iid=idx2item[(Math.random()*idx2item.length)|0]; if(!seen.has(iid)&&!pos.includes(iid)) neg.push(iid); }
      const cand = pos.slice(0,Math.min(5,pos.length)).concat(neg);

      // scores: retriever → SASRec → (no PPR in offline)
      const uIdx = tf.tensor1d([userIndex.get(u)], 'int32');
      let base = [];
      if(tw){ const s = await tw.scoreItems(uIdx, cand.map(i=>itemIndex.get(i))); base = Array.from(s.dataSync()); s.dispose(); }
      else base = new Array(cand.length).fill(0);

      if(sa){
        const seqL=parseInt($('saLen').value,10);
        const hist=(user2items.get(u)||[]).slice().sort((a,b)=>a.ts-b.ts).map(x=> itemIndex.get(x.i)+1);
        const last = hist.slice(-seqL); const seq = Array(Math.max(0,seqL-last.length)).fill(0).concat(last);
        const S=tf.tensor2d([seq],[1,seqL],'int32');
        const s = await sa.scoreItems(S, cand.map(i=>itemIndex.get(i)+1));
        const arr=Array.from(s.dataSync()); s.dispose(); S.dispose();

        // z-score blend (0.6, 0.4)
        const n1 = (arr)=>{ const m=arr.reduce((a,b)=>a+b,0)/arr.length; const sd=Math.sqrt(arr.reduce((s,v)=>s+(v-m)*(v-m),0)/arr.length)||1; return arr.map(v=>(v-m)/sd); };
        const b=n1(base), a=n1(arr);
        base = base.map((_,ix)=> 0.6*b[ix]+0.4*a[ix] );
      }
      uIdx.dispose();

      const zipped=cand.map((iid,ix)=>({iid, s:base[ix], rel: pos.includes(iid)?1:0})).sort((a,b)=>b.s-a.s).slice(0,K);
      hits += zipped.some(z=>z.rel>0) ? 1 : 0;
      dcg  += zipped.reduce((s,z,i)=> s + (z.rel>0 ? 1/Math.log2(i+2) : 0), 0);
      ideal += 1;
    }
    return { recallAt10: hits/Math.max(1,pick.length), ndcgAt10: dcg/Math.max(1,ideal) };
  }

  const M = await evalModel();
  body.innerHTML = `Validation users sampled: ${fmt(Math.min(120, new Set(valid.map(x=>x.u)).size))}
  <br/>Cook-next stack — Recall@10 <b>${M.recallAt10.toFixed(3)}</b> · nDCG@10 <b>${M.ndcgAt10.toFixed(3)}</b>`;
}

// ---------- events ----------
$('btnLoad').addEventListener('click', handleLoad);
$('btnTrainTw').addEventListener('click', ()=>trainTwoTower().catch(console.error));
$('btnTrainSa').addEventListener('click', ()=>trainSASRec().catch(console.error));
$('btnDemo').addEventListener('click', ()=>demoOnce().catch(console.error));
