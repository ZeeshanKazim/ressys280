/* app.js (ONE-TIME PATCHED VERSION)
   - Robust data loader that works with ./data/u.item and ./data/u.data OR ./u.item and ./u.data
   - Everything else unchanged: training loop, charts, test + table.
   - Requires two-tower.js (TwoTowerModel) and the DOM from index.html provided earlier.
*/

/* ==============================
   Tiny DOM helpers & state
================================ */
const $ = (id) => document.getElementById(id);
const S = {
  // raw parsed
  interactions: [],                 // [{userId,itemId,rating,ts}]
  items: new Map(),                 // itemId -> {title, year, genres:Int8Array(18)}
  users: new Set(),
  itemIds: new Set(),

  // indexers
  userId2idx: new Map(),
  itemId2idx: new Map(),
  idx2userId: [],
  idx2itemId: [],

  // quick lookups
  userToRated: new Map(),           // userId -> Set(itemId)
  userTopRated: new Map(),          // userId -> [{itemId,title,year,rating,ts}]

  // models
  baseline: null,                   // TwoTowerModel (no hidden)
  deep: null,                       // TwoTowerModel (1 hidden)

  // training
  lossSeriesBaseline: [],
  lossSeriesDeep: []
};

/* ==============================
   Utils
================================ */
function setStatus(msg){ $('status').textContent = `Status: ${msg}`; }

function randChoice(arr){ return arr[(Math.random()*arr.length)|0]; }

function shuffleInPlace(a){
  for(let i=a.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [a[i],a[j]]=[a[j],a[i]]; }
  return a;
}

/* Simple Canvas Line Plot (no external libs) */
function clearCanvas(ctx){ ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height); }
function plotLoss(ctx, seriesA, seriesB){
  clearCanvas(ctx);
  const w=ctx.canvas.width, h=ctx.canvas.height;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,w,h);

  const pad=32;
  const maxLen = Math.max(seriesA.length, seriesB.length, 1);
  const maxVal = Math.max(
    ...seriesA.map(x=>x.y),
    ...seriesB.map(x=>x.y),
    1.0
  );
  // axes
  ctx.strokeStyle='#374151'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,h-pad); ctx.lineTo(w-pad,h-pad); ctx.stroke();

  const draw = (series, color) => {
    if(!series.length) return;
    ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.beginPath();
    for(let i=0;i<series.length;i++){
      const x = pad + (i/(Math.max(series.length-1,1))) * (w-2*pad);
      const y = h-pad - (series[i].y/maxVal) * (h-2*pad);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  };
  // baseline (green) vs deep (red)
  draw(seriesA, '#22c55e');
  draw(seriesB, '#ef4444');

  // legend
  ctx.fillStyle='#9ca3af'; ctx.font='12px ui-sans-serif,system-ui';
  ctx.fillText('Baseline (no hidden)', pad, pad-8);
  ctx.fillStyle='#ef4444'; ctx.fillRect(pad+130-8, pad-16, 10, 10);
  ctx.fillStyle='#22c55e'; ctx.fillRect(pad-8, pad-16, 10, 10);
  ctx.fillStyle='#9ca3af';
  ctx.fillText('Deep (1 hidden)', pad+130, pad-8);
}

/* PCA (very small, numeric) — power method on covariance for top-2 vectors */
function pca2D(matrix /* Float32Array rows concatenated */, rows, cols){
  // center
  const mean = new Float32Array(cols);
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++) mean[c]+=matrix[r*cols+c];
  }
  for(let c=0;c<cols;c++) mean[c]/=rows;
  const X = new Float32Array(rows*cols);
  for(let r=0;r<rows;r++) for(let c=0;c<cols;c++){
    X[r*cols+c]=matrix[r*cols+c]-mean[c];
  }

  function powerVec(iter=50){
    let v=new Float32Array(cols);
    for(let c=0;c<cols;c++) v[c]=Math.random()-0.5;
    for(let it=0;it<iter;it++){
      // w = X^T(X v)
      const Xv=new Float32Array(rows);
      for(let r=0;r<rows;r++){
        let s=0; for(let c=0;c<cols;c++) s+=X[r*cols+c]*v[c];
        Xv[r]=s;
      }
      const w=new Float32Array(cols);
      for(let c=0;c<cols;c++){
        let s=0; for(let r=0;r<rows;r++) s+=X[r*cols+c]*Xv[r];
        w[c]=s;
      }
      // normalize
      let n=0; for(let c=0;c<cols;c++) n+=w[c]*w[c];
      n=Math.sqrt(n)||1;
      for(let c=0;c<cols;c++) v[c]=w[c]/n;
    }
    return v;
  }
  const pc1=powerVec(30);
  // deflate roughly
  const Xproj1 = new Float32Array(rows);
  for(let r=0;r<rows;r++){
    let s=0; for(let c=0;c<cols;c++) s+=X[r*cols+c]*pc1[c];
    Xproj1[r]=s;
    for(let c=0;c<cols;c++) X[r*cols+c]-=s*pc1[c];
  }
  const pc2=powerVec(30);

  // project to 2D
  const out=new Float32Array(rows*2);
  for(let r=0;r<rows;r++){
    let a=0,b=0;
    for(let c=0;c<cols;c++){ const val=matrix[r*cols+c]-mean[c]; a+=val*pc1[c]; b+=val*pc2[c]; }
    out[r*2]=a; out[r*2+1]=b;
  }
  return out;
}

function drawProjection(ctx, points /* Float32Array [x,y]* */, titles){
  clearCanvas(ctx);
  const w=ctx.canvas.width,h=ctx.canvas.height;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,w,h);
  let minX=+Infinity,minY=+Infinity,maxX=-Infinity,maxY=-Infinity;
  for(let i=0;i<points.length;i+=2){
    const x=points[i], y=points[i+1];
    if(x<minX)minX=x; if(x>maxX)maxX=x; if(y<minY)minY=y; if(y>maxY)maxY=y;
  }
  const pad=20, sx=(w-2*pad)/(maxX-minX||1), sy=(h-2*pad)/(maxY-minY||1);
  ctx.fillStyle='#60a5fa';
  for(let i=0;i<points.length;i+=2){
    const x=pad+(points[i]-minX)*sx, y=h-pad-(points[i+1]-minY)*sy;
    ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill();
  }
  // (Note: hover labels can be added; omitted for simplicity)
}

/* ==============================
   Robust fetch helper + Loader
================================ */
// <--- The only new helper for the "one-time" fix
async function fetchTextTry(paths) {
  for (const p of paths) {
    try {
      const r = await fetch(p + (p.includes('?') ? '' : `?v=${Date.now()}`)); // cache-bust
      if (r.ok) return await r.text();
    } catch (_) { /* try next */ }
  }
  throw new Error(`Could not fetch any of: ${paths.join(', ')}`);
}

// <--- Patched loadData() that tries ./data/* first, then ./*
async function loadData(){
  await tf.ready();
  try{
    const itemTxt = await fetchTextTry(['./data/u.item', './u.item', 'u.item', 'data/u.item']);
    const dataTxt = await fetchTextTry(['./data/u.data', './u.data', 'u.data', 'data/u.data']);

    // reset
    S.items.clear(); S.itemIds.clear(); S.users.clear();
    S.interactions.length=0; S.userToRated.clear(); S.userTopRated.clear();

    // parse u.item
    const linesItem = itemTxt.split('\n').filter(Boolean);
    for(const line of linesItem){
      const parts=line.split('|'); if(parts.length<2) continue;
      const id=+parts[0]; let title=parts[1]; let year=null;
      const m=title.match(/\((\d{4})\)\s*$/); if(m){ year=+m[1]; title=title.replace(/\s*\(\d{4}\)\s*$/,''); }
      const flags=parts.slice(5).map(x=>x==='1'?1:0);
      const use=(flags.length>=19)? flags.slice(1) : flags; // drop 'Unknown'
      const g=new Int8Array(18);
      for(let k=0;k<Math.min(18,use.length);k++) g[k]=use[k];
      S.items.set(id,{title,year,genres:g});
      S.itemIds.add(id);
    }

    // parse u.data
    const linesData = dataTxt.split('\n').filter(Boolean);
    for(const line of linesData){
      const [u,i,r,t]=line.split('\t');
      const userId=+u,itemId=+i,rating=+r,ts=+t;
      if(Number.isFinite(userId)&&Number.isFinite(itemId)){
        S.interactions.push({userId,itemId,rating,ts});
        S.users.add(userId);
      }
    }

    // indexers
    const userIds=[...S.users].sort((a,b)=>a-b), itemIds=[...S.itemIds].sort((a,b)=>a-b);
    S.idx2userId=userIds; S.idx2itemId=itemIds;
    S.userId2idx=new Map(userIds.map((u,idx)=>[u,idx]));
    S.itemId2idx=new Map(itemIds.map((i,idx)=>[i,idx]));

    // quick maps
    for(const {userId,itemId} of S.interactions){
      if(!S.userToRated.has(userId)) S.userToRated.set(userId,new Set());
      S.userToRated.get(userId).add(itemId);
    }
    // top rated per user (for the Test table)
    const byUser=new Map();
    for(const r of S.interactions){ if(!byUser.has(r.userId)) byUser.set(r.userId,[]); byUser.get(r.userId).push(r); }
    for(const [uid,arr] of byUser){
      arr.sort((a,b)=> (b.rating-a.rating)||(b.ts-a.ts));
      const top = arr.slice(0,60).map(x=>({ itemId:x.itemId, rating:x.rating, ts:x.ts,
        title:S.items.get(x.itemId)?.title ?? String(x.itemId), year:S.items.get(x.itemId)?.year ?? '' }));
      S.userTopRated.set(uid, top);
    }

    $('btn-train').disabled=false;
    $('btn-test').disabled=false;
    setStatus(`data loaded — users: ${userIds.length}, items: ${itemIds.length}, interactions: ${S.interactions.length}`);
  }catch(e){
    console.error(e);
    setStatus('fetch failed. Ensure ./data/u.item and ./data/u.data exist (or next to index.html).');
  }
}

/* ==============================
   Training + Batching
================================ */
function readCfg(){
  return {
    maxInt: parseInt($('cfg-max-int').value,10) || 80000,
    embDim: parseInt($('cfg-emb-dim').value,10) || 32,
    hidDim: parseInt($('cfg-hid-dim').value,10) || 64,
    batch:  parseInt($('cfg-batch').value,10)   || 256,
    epochs: parseInt($('cfg-epochs').value,10)  || 5,
    lr:     parseFloat($('cfg-lr').value)       || 0.003,
    lossType: $('cfg-loss').value,              // 'softmax' or 'bpr'
    trainBaseline: $('cfg-train-baseline').value !== 'No'
  };
}

function buildPairs(maxN){
  const xs = S.interactions.slice(0, Math.min(maxN, S.interactions.length));
  shuffleInPlace(xs);
  const users = new Int32Array(xs.length);
  const items = new Int32Array(xs.length);
  for(let k=0;k<xs.length;k++){
    users[k]=S.userId2idx.get(xs[k].userId);
    items[k]=S.itemId2idx.get(xs[k].itemId);
  }
  return {users, items, size: xs.length};
}

function* batcher(users, items, batch){
  let i=0;
  while(i<users.length){
    const end=Math.min(i+batch, users.length);
    yield {
      u: tf.tensor1d(users.subarray(i,end),'int32'),
      p: tf.tensor1d(items.subarray(i,end),'int32')
    };
    i=end;
  }
}

async function train(){
  if(!S.idx2userId.length){ setStatus('load data first.'); return; }
  const cfg=readCfg();
  setStatus('initializing models…');

  // build training arrays
  const {users, items} = buildPairs(cfg.maxInt);

  // Item side features (genres) for the deep tower
  const numItems=S.idx2itemId.length;
  const genres = new Float32Array(numItems*18);
  for(let i=0;i<numItems;i++){
    const itemId=S.idx2itemId[i], g=S.items.get(itemId)?.genres;
    if(g) for(let k=0;k<18;k++) genres[i*18+k]=g[k];
  }

  // (re)create models
  S.deep?.dispose?.();
  S.baseline?.dispose?.();
  S.deep = new TwoTowerModel(S.idx2userId.length, S.idx2itemId.length, cfg.embDim, {
    deep:true, hiddenDim: cfg.hidDim, lossType: cfg.lossType, learningRate: cfg.lr, itemFeatures:{data:genres, dim:18}
  });
  if(cfg.trainBaseline){
    S.baseline = new TwoTowerModel(S.idx2userId.length, S.idx2itemId.length, cfg.embDim, {
      deep:false, hiddenDim: 0, lossType: cfg.lossType, learningRate: cfg.lr
    });
  } else {
    S.baseline = null;
  }

  const ctx = $('loss-canvas').getContext('2d');
  S.lossSeriesBaseline.length=0; S.lossSeriesDeep.length=0;
  setStatus('training…');

  const iterAll = cfg.epochs * Math.ceil(users.length / cfg.batch);
  let it=0;

  for(let e=0;e<cfg.epochs;e++){
    for(const {u,p} of batcher(users, items, cfg.batch)){
      // deep step
      const lossDeep = await S.deep.trainStep(u,p);
      S.lossSeriesDeep.push({x:it, y:lossDeep});

      // baseline step (optional)
      if(S.baseline){
        const lossBase = await S.baseline.trainStep(u,p);
        S.lossSeriesBaseline.push({x:it, y:lossBase});
      }

      plotLoss(ctx, S.lossSeriesBaseline, S.lossSeriesDeep);
      setStatus(`training… ${++it}/${iterAll} (epoch ${e+1}/${cfg.epochs})`);
      await tf.nextFrame();
      u.dispose(); p.dispose();
    }
  }

  // Embedding projection (sample ~1000)
  const itemEmb = S.deep.getAllItemEmbeddings(); // Float32Array [numItems, embDim]
  const sampleN = Math.min(1000, numItems);
  const idxs = Array.from({length:numItems},(_,i)=>i); shuffleInPlace(idxs);
  const take = idxs.slice(0,sampleN);
  const mat = new Float32Array(sampleN*cfg.embDim);
  const titles=[];
  for(let r=0;r<take.length;r++){
    const i=take[r];
    mat.set(itemEmb.subarray(i*cfg.embDim,(i+1)*cfg.embDim), r*cfg.embDim);
    titles.push(S.items.get(S.idx2itemId[i])?.title || String(S.idx2itemId[i]));
  }
  const proj = pca2D(mat, sampleN, cfg.embDim);
  drawProjection($('proj-canvas').getContext('2d'), proj, titles);

  setStatus('training done. Click Test to view recommendations.');
}

/* ==============================
   Inference / Test table
================================ */
function buildScoresAndTopK(model, userId, k, excludeSet){
  const uIdx=S.userId2idx.get(userId);
  if(uIdx==null) return [];
  const scores=model.getScoresForAllItems(uIdx); // Float32Array length=numItems
  // mask rated
  for(const itemId of excludeSet){
    const ii=S.itemId2idx.get(itemId);
    if(ii!=null) scores[ii] = -1e9;
  }
  // top-k indices
  const N=scores.length;
  const idxs=Array.from({length:N},(_,i)=>i);
  idxs.sort((a,b)=>scores[b]-scores[a]);
  const out=[];
  for(let i=0;i<Math.min(k,N);i++){
    const ii=idxs[i], id=S.idx2itemId[ii], s=scores[ii];
    const it=S.items.get(id);
    out.push({rank:i+1, itemId:id, title:it?.title||String(id), year:it?.year||'', score:+s.toFixed(4)});
  }
  return out.slice(0,k);
}

function renderTable(userId, topRated, recDeep, recBase){
  const box=$('table-box');
  const th = (t)=>`<th>${t}</th>`;
  const row = (a,b,c)=>`<tr>
    <td class="num">${a?.rank??''}</td><td>${a?.title??''}</td><td class="num">${a?.rating??''}</td><td class="num">${a?.year??''}</td>
    <td class="sep"></td>
    <td class="num">${b?.rank??''}</td><td>${b?.title??''}</td><td class="num">${b?.score??''}</td><td class="num">${b?.year??''}</td>
    <td class="sep"></td>
    <td class="num">${c?.rank??''}</td><td>${c?.title??''}</td><td class="num">${c?.score??''}</td><td class="num">${c?.year??''}</td>
  </tr>`;
  const rows=[];
  for(let i=0;i<10;i++) rows.push(row(topRated[i], recDeep[i], recBase?.[i]));
  box.innerHTML = `
    <h3>Top-10 Rated vs Recommended (Baseline vs Deep) — User ${userId}</h3>
    <table class="cmp">
      <thead>
        <tr>
          ${th('#')+th('Top-rated title')+th('★')+th('Year')}
          <th class="sep"></th>
          ${th('#')+th('Recommended (Deep)')+th('Score')+th('Year')}
          <th class="sep"></th>
          ${th('#')+th('Recommended (Baseline)')+th('Score')+th('Year')}
        </tr>
      </thead>
      <tbody>${rows.join('')}</tbody>
    </table>`;
}

function testOne(){
  if(!S.deep){ setStatus('train first.'); return; }
  // pick a user with >=20 ratings
  const candidates=[...S.userTopRated.entries()].filter(([,arr])=>arr.length>=20).map(([u])=>u);
  const userId = randChoice(candidates);
  const ratedList=S.userTopRated.get(userId)||[];
  const exclude = new Set(ratedList.map(x=>x.itemId));
  const topRated = ratedList.slice(0,10).map((x,i)=>({rank:i+1, ...x}));

  const recDeep = buildScoresAndTopK(S.deep, userId, 10, exclude);
  const recBase = S.baseline ? buildScoresAndTopK(S.baseline, userId, 10, exclude) : null;

  renderTable(userId, topRated, recDeep, recBase);
  setStatus('recommendations generated successfully!');
}

/* ==============================
   Wire up
================================ */
window.addEventListener('load', ()=>{
  $('btn-load').addEventListener('click', async()=>{ $('btn-load').disabled=true; setStatus('loading…'); await loadData(); $('btn-load').disabled=false; });
  $('btn-train').addEventListener('click', async()=>{ $('btn-train').disabled=true; await train(); $('btn-train').disabled=false; });
  $('btn-test').addEventListener('click', ()=> testOne());
});
