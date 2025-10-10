/* app.js — data load, EDA charts, training, demo, metrics */

(() => {
  // ---------- Elements ----------
  const els = {
    status:        document.getElementById('status'),
    loadBtn:       document.getElementById('loadBtn'),

    // EDA
    edaSummary:    document.getElementById('eda-summary'),
    histRatings:   document.getElementById('histRatings'),
    topTags:       document.getElementById('topTags'),
    lorenz:        document.getElementById('lorenz'),
    coldTable:     document.getElementById('coldTable').querySelector('tbody'),

    // Models
    trainBtn:      document.getElementById('trainBtn'),
    trainStatus:   document.getElementById('trainStatus'),
    lossCanvas:    document.getElementById('lossCanvas'),
    pcaCanvas:     document.getElementById('pcaCanvas'),
    maxInt:        document.getElementById('maxInt'),
    maxItems:      document.getElementById('maxItems'),
    embDim:        document.getElementById('embDim'),
    hiddenDim:     document.getElementById('hiddenDim'),
    batchSize:     document.getElementById('batchSize'),
    epochs:        document.getElementById('epochs'),
    lr:            document.getElementById('lr'),
    lossSel:       document.getElementById('lossSel'),
    useTags:       document.getElementById('useTags'),
    tagDim:        document.getElementById('tagDim'),
    compare:       document.getElementById('compare'),

    // Demo
    userSelect:    document.getElementById('userSelect'),
    graphOn:       document.getElementById('graphOn'),
    betaPPR:       document.getElementById('betaPPR'),
    gammaNov:      document.getElementById('gammaNov'),
    recBtn:        document.getElementById('recBtn'),
    histTable:     document.getElementById('histTable').querySelector('tbody'),
    recTable:      document.getElementById('recTable').querySelector('tbody'),
    whyBox:        document.getElementById('whyBox'),
    demoStatus:    document.getElementById('demoStatus'),

    // Metrics
    evalBtn:       document.getElementById('evalBtn'),
    metricTable:   document.getElementById('metricTable').querySelector('tbody'),
    metricBuckets: document.getElementById('metricBuckets').querySelector('tbody'),
    metricStatus:  document.getElementById('metricStatus'),
  };
  const setStatus = s => els.status.textContent = `Status: ${s}`;
  const setTrain  = s => els.trainStatus.textContent = s;
  const setDemo   = s => els.demoStatus.textContent = s;
  const setMetric = s => els.metricStatus.textContent = s;

  // ---------- Data store ----------
  const Data = {
    // raw
    recipes: [],          // [{id,name,tags:[...]}]
    interactions: { train: [], val: [], test: [] }, // [{userId,itemId,rating,ts}]
    // indexers
    users: [], items: [], userToIdx: new Map(), itemToIdx: new Map(),
    // derived
    positivesByUser: new Map(), // userId -> Set(itemId) positive (rating>=4)
    topTags: [], tagHashDim: 128,
    // caps after filtering
    capped: { users: 0, items: 0, interactions: 0 },
  };

  // ---------- CSV parsing (robust to quotes) ----------
  function parseCSV(text) {
    // returns { header: string[], rows: string[][] }
    const rows = [];
    let i=0, s=text, len=s.length, field='', row=[], inQ=false;
    function pushField(){ row.push(field); field=''; }
    function pushRow(){ rows.push(row); row=[]; }
    while(i<len){
      const c=s[i];
      if(inQ){
        if(c === '"'){
          if(i+1<len && s[i+1] === '"'){ field+='"'; i++; } // escaped quote
          else { inQ=false; }
        } else { field+=c; }
      } else {
        if(c === '"'){ inQ = true; }
        else if(c === ','){ pushField(); }
        else if(c === '\n' || c === '\r'){
          // normalize CRLF/CR
          if(c === '\r' && i+1<len && s[i+1]==='\n'){ i++; }
          pushField(); if(row.length>1 || row[0] !== '') pushRow();
        } else { field+=c; }
      }
      i++;
    }
    if(field.length || row.length) { pushField(); pushRow(); }
    if(rows.length===0) return {header:[], rows:[]};
    // If separator is tab or semicolon, retry quickly
    if(rows.length===1 && rows[0].length===1 && text.includes('\t')){
      return parseCSV(text.replace(/,/g,'\u0001').replace(/\t/g,',').replace(/\u0001/g,','));
    }
    const header = rows.shift().map(h=>h.trim());
    return { header, rows: rows.map(r=>r.map(x=>x.trim())) };
  }

  const tryFetchText = async (paths) => {
    for(const p of paths){
      try{ const r=await fetch(`./data/${p}`); if(r.ok){ return await r.text(); } }catch(e){}
    }
    return null;
  };

  function findCol(header, patterns){
    // patterns: array of regex
    for(const rx of patterns){
      for(let i=0;i<header.length;i++){
        if(rx.test(header[i])) return i;
      }
    }
    return -1;
  }

  function parseTagsField(s){
    if(!s) return [];
    // try JSON-like list "['a','b']"
    const t = s.trim();
    try {
      // normalize single quotes to double if needed
      const guess = t.startsWith('[') ? JSON.parse(t.replace(/'/g,'"')) : JSON.parse(t);
      if (Array.isArray(guess)) return guess.map(x=>String(x).toLowerCase());
    } catch(_){}
    // fallback split by comma/semicolon
    return t.split(/[;,]/).map(x=>x.trim().toLowerCase()).filter(Boolean);
  }

  // ---------- Loading & preprocessing ----------
  async function loadData(){
    setStatus('loading from /data …');

    const recipesText = await tryFetchText(['recipes.csv','RAW_recipes.csv','items.csv']);
    if(!recipesText){ setStatus('missing recipes file: place recipes.csv or RAW_recipes.csv in /data'); throw new Error('recipes missing'); }
    const rec = parseCSV(recipesText);
    // detect columns
    const idxId   = findCol(rec.header, [/^id$/i,/recipe[_ ]?id/i,/item[_ ]?id/i]);
    const idxName = findCol(rec.header, [/^name$/i,/title/i]);
    const idxTags = findCol(rec.header, [/tags?/i]);
    if(idxId<0){ setStatus('recipes file lacks id column'); throw new Error('bad recipes'); }

    const recipes = [];
    for(const r of rec.rows){
      const id = r[idxId];
      if(!id) continue;
      const name = idxName>=0 ? r[idxName] : `Recipe ${id}`;
      const tags = idxTags>=0 ? parseTagsField(r[idxTags]) : [];
      recipes.push({ id:String(id), name, tags });
    }
    Data.recipes = recipes;

    async function parseInteractionsCandidate(fn){
      const txt = await tryFetchText([fn]);
      if(!txt) return [];
      const t = parseCSV(txt);
      const iu = findCol(t.header, [/user/i]);
      const ii = findCol(t.header, [/(recipe|item).*id/i,/^id$/i]);
      const ir = findCol(t.header, [/rating|score|stars?/i]);
      const it = findCol(t.header, [/date|time|timestamp/i]);
      if(iu<0 || ii<0){ return []; }
      const arr = [];
      for(const row of t.rows){
        const userId = String(row[iu]);
        const itemId = String(row[ii]);
        if(!userId || !itemId) continue;
        const rating = ir>=0 ? Number(row[ir]) : 1;
        const ts = it>=0 ? Date.parse(row[it])||0 : 0;
        arr.push({ userId, itemId, rating: isFinite(rating)?rating:1, ts });
      }
      return arr;
    }

    // Try multiple files
    let train = await parseInteractionsCandidate('interactions_train.csv');
    if(train.length===0) train = await parseInteractionsCandidate('interactions.csv');
    if(train.length===0) train = await parseInteractionsCandidate('RAW_interactions.csv');

    const val   = await parseInteractionsCandidate('interactions_validation.csv');
    const test  = await parseInteractionsCandidate('interactions_test.csv');

    if(train.length===0 && val.length>0){ train = val; }
    if(train.length===0){ setStatus('missing interactions: place interactions_train.csv or interactions.csv in /data'); throw new Error('interactions missing'); }

    Data.interactions.train = train;
    Data.interactions.val   = val;
    Data.interactions.test  = test;

    // Build popular items and cap items for model memory (keep top-N items)
    const pop = new Map();
    for(const r of train){ pop.set(r.itemId, 1 + (pop.get(r.itemId)||0)); }
    const topItems = [...pop.entries()].sort((a,b)=>b[1]-a[1]).map(([id])=>id);
    const maxItems = Math.max(1000, Number(els.maxItems.value)||30000);
    const keepSet = new Set(topItems.slice(0, maxItems));

    // Filter train by keepSet
    const trainCap = train.filter(r => keepSet.has(r.itemId));
    // Build users/items indexers (after cap)
    const users = [...new Set(trainCap.map(x=>x.userId))].sort((a,b)=>a.localeCompare(b));
    const items = topItems.slice(0, maxItems).filter(id => Data.recipes.some(rc => rc.id===id));
    const userToIdx = new Map(users.map((u,i)=>[u,i]));
    const itemToIdx = new Map(items.map((it,i)=>[it,i]));
    Data.users = users; Data.items = items; Data.userToIdx = userToIdx; Data.itemToIdx = itemToIdx;

    // positivesByUser from train (rating>=4 or, if ratings absent, all)
    const posByUser = new Map();
    for(const r of trainCap){
      const ok = (r.rating==null) ? true : (r.rating>=4);
      if(!ok) continue;
      if(!posByUser.has(r.userId)) posByUser.set(r.userId, new Set());
      posByUser.get(r.userId).add(r.itemId);
    }
    Data.positivesByUser = posByUser;

    // tag hashing (fixed dim, memory‑friendly)
    Data.tagHashDim = Math.max(32, Number(els.tagDim.value)||128);

    // EDA summary
    renderEDA(train, val, test, items, users, pop);

    // Enable next steps
    els.trainBtn.disabled = false;
    els.userSelect.disabled = false;
    els.recBtn.disabled = false;
    els.evalBtn.disabled = (Data.interactions.val.length===0 && Data.interactions.test.length===0);
    populateUsersDropdown(users, posByUser);
    setStatus(`loaded: users=${users.length}, items=${items.length}, train=${train.length}${val.length?`, val=${val.length}`:''}${test.length?`, test=${test.length}`:''}`);
  }

  // ---------- EDA ----------
  function renderEDA(train, val, test, items, users, popMap){
    // summary
    const density = (train.length / Math.max(1, users.length * items.length)).toExponential(2);
    const rHist = new Array(5).fill(0);
    let ratingsPresent = false;
    for(const r of train){ if(Number.isFinite(r.rating)) { ratingsPresent=true; if(r.rating>=1 && r.rating<=5) rHist[r.rating-1]++; } }
    const posItems = [...popMap.values()].sort((a,b)=>b-a);

    els.edaSummary.textContent =
`Users: ${users.length}
Items (capped): ${items.length}
Interactions (train): ${train.length}${val.length?`\nValidation: ${val.length}`:''}${test.length?`\nTest: ${test.length}`:''}
Density: ${density}
Ratings present: ${ratingsPresent ? 'yes' : 'no'}
Cold users (<5): ${users.filter(u => (Data.positivesByUser.get(u)?.size || 0) < 5).length}
Cold items (<5): ${items.filter(it => (popMap.get(it)||0) < 5).length}`;

    // ratings histogram
    drawBar(els.histRatings, ['1','2','3','4','5'], rHist, 'Ratings');

    // top tags (derive from recipes under capped items)
    const tagFreq = new Map();
    const itemSet = new Set(items);
    for(const rc of Data.recipes){
      if(!itemSet.has(rc.id)) continue;
      for(const t of rc.tags) tagFreq.set(t, 1 + (tagFreq.get(t)||0));
    }
    const top20 = [...tagFreq.entries()].sort((a,b)=>b[1]-a[1]).slice(0,20);
    Data.topTags = top20.map(([t])=>t);
    drawBar(els.topTags, top20.map(([t])=>t), top20.map(([,c])=>c), 'Top tags');

    // Lorenz curve + Gini of popularity
    drawLorenzCurve(els.lorenz, posItems);

    // cold‑start table
    const coldUsers = users.filter(u => (Data.positivesByUser.get(u)?.size || 0) < 5).length;
    const coldItems = items.filter(it => (popMap.get(it)||0) < 5).length;
    els.coldTable.innerHTML = `
      <tr><td>Cold users (&lt;5)</td><td>${coldUsers}</td></tr>
      <tr><td>Cold items (&lt;5)</td><td>${coldItems}</td></tr>
      <tr><td>Top 1% items (by interactions)</td><td>${Math.ceil(items.length*0.01)}</td></tr>
      <tr><td>Pareto: % train covered by top 1%</td><td>${paretoCoverage(posItems,0.01)}</td></tr>`;
  }

  function paretoCoverage(sortedCounts, topFrac){
    const N = sortedCounts.length; if(N===0) return '0%';
    const k = Math.max(1, Math.floor(N*topFrac));
    const top = sortedCounts.slice(0,k).reduce((s,x)=>s+x,0);
    const tot = sortedCounts.reduce((s,x)=>s+x,0);
    return `${((top/tot)*100).toFixed(1)}%`;
  }

  // ---------- Charts (Canvas) ----------
  function drawBar(cvs, labels, values, title=''){
    const ctx = cvs.getContext('2d'); const W=cvs.width, H=cvs.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='rgba(255,255,255,.04)'; ctx.fillRect(0,0,W,H);
    const n=values.length; if(n===0) return;
    const maxV = Math.max(...values,1);
    const m=30, w=(W-2*m)/n;
    ctx.fillStyle='#fff';
    for(let i=0;i<n;i++){
      const h = Math.round(((values[i]/maxV) * (H-3*m)));
      const x = m + i*w + 4, y = H-m-h;
      ctx.fillRect(x,y, w-8, h);
    }
    // labels
    ctx.fillStyle='rgba(200,210,230,.9)'; ctx.font='12px system-ui';
    for(let i=0;i<n;i++){
      const x = m + i*w + 4, y = H-10;
      const txt = String(labels[i]).slice(0,10);
      ctx.save(); ctx.translate(x+Math.max(24, (w-8)/2), y); ctx.rotate(-Math.PI/6); ctx.fillText(txt, 0, 0); ctx.restore();
    }
  }

  function drawLorenzCurve(cvs, counts){
    const ctx = cvs.getContext('2d'); const W=cvs.width, H=cvs.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='rgba(255,255,255,.04)'; ctx.fillRect(0,0,W,H);
    if(counts.length===0) return;
    const sorted = counts.slice().sort((a,b)=>a-b);
    const tot = sorted.reduce((s,x)=>s+x,0);
    let cum=0; const xs=[0], ys=[0];
    for(let i=0;i<sorted.length;i++){
      cum+=sorted[i]; xs.push((i+1)/sorted.length); ys.push(cum/tot);
    }
    // draw
    ctx.strokeStyle='#20c997'; ctx.lineWidth=2; ctx.beginPath();
    for(let i=0;i<xs.length;i++){
      const x=30+xs[i]*(W-60), y=H-30-ys[i]*(H-60);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    // equality line
    ctx.strokeStyle='rgba(255,255,255,.35)'; ctx.beginPath();
    ctx.moveTo(30,H-30); ctx.lineTo(W-30,30); ctx.stroke();
    // Gini
    // Gini = 1 - 2 * area under Lorenz
    let area=0; for(let i=0;i<xs.length-1;i++){ area += (ys[i]+ys[i+1])*(xs[i+1]-xs[i])/2; }
    const gini = 1 - 2*area;
    ctx.fillStyle='rgba(200,210,230,.9)'; ctx.font='13px system-ui';
    ctx.fillText(`Gini ≈ ${gini.toFixed(3)}`, 36, 26);
  }

  // ---------- User dropdown ----------
  function populateUsersDropdown(users, posByUser){
    const heavy = users.filter(u => (posByUser.get(u)?.size||0) >= 10);
    const sel = els.userSelect;
    sel.innerHTML = '<option value="">— select —</option>';
    for(const u of heavy){
      const opt=document.createElement('option');
      opt.value=u; opt.textContent=`User ${u}`;
      sel.appendChild(opt);
    }
  }

  // ---------- Tag hashing matrix (fixed dim) ----------
  function hash32(str){
    // simple FNV‑like
    let h=2166136261>>>0;
    for(let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h = Math.imul(h,16777619)>>>0; }
    return h>>>0;
  }
  function buildItemTagMatrix(tagDim){
    const items = Data.items, N=items.length, D=tagDim;
    const mat = new Float32Array(N*D); // zeros
    const recIndex = new Map(Data.recipes.map((r,i)=>[r.id,i]));
    for(let i=0;i<N;i++){
      const id = items[i];
      const rcIdx = recIndex.get(id);
      if(rcIdx==null) continue;
      const tags = Data.recipes[rcIdx].tags || [];
      // set hashed positions to 1
      for(const t of tags){
        const pos = hash32(t) % D;
        mat[i*D + pos] = 1;
      }
    }
    return { data: mat, dim: D };
  }

  // ---------- Training ----------
  const lossBase=[], lossDeep=[]; // {batch,loss}
  let baseline=null, deep=null, modelsReady=false, itemTagTensor=null;

  async function train(){
    if(Data.items.length===0){ setTrain('Load data first'); return; }
    // hyperparams
    const embDim   = +els.embDim.value || 48;
    const hidden   = +els.hiddenDim.value || 96;
    const lr       = +els.lr.value || 0.003;
    const bs       = +els.batchSize.value || 256;
    const epochs   = +els.epochs.value || 4;
    const lossType = els.lossSel.value.startsWith('In-batch') ? 'softmax' : 'bpr';
    const compare  = els.compare.value.startsWith('Yes');
    const useTags  = els.useTags.value === 'Yes';
    const tagDim   = +els.tagDim.value || 128;
    const maxInt   = +els.maxInt.value || 250000;

    lossBase.length=0; lossDeep.length=0;

    // Build training arrays with caps
    const pool = Data.interactions.train
      .filter(r => Data.userToIdx.has(r.userId) && Data.itemToIdx.has(r.itemId))
      .slice(0, maxInt);

    if(pool.length===0){ setTrain('No interactions after caps'); return; }

    // Fixed item-tags tensor for model (if used)
    if(itemTagTensor) { itemTagTensor.dispose(); itemTagTensor=null; }
    if(useTags){
      const tagM = buildItemTagMatrix(tagDim);
      itemTagTensor = tf.tensor2d(tagM.data, [Data.items.length, tagM.dim], 'float32');
    }

    // Dispose old models
    if(baseline && baseline.dispose) baseline.dispose();
    if(deep && deep.dispose) deep.dispose();
    baseline=null; deep=null;

    // Create models
    if(compare){
      baseline = new TwoTowerModel(Data.users.length, Data.items.length, embDim, {
        deep:false, hiddenDim:0, lossType, learningRate:lr, useTags:false, tagDim:0
      });
    }
    deep = new TwoTowerModel(Data.users.length, Data.items.length, embDim, {
      deep:true, hiddenDim:Math.max(0,hidden), lossType, learningRate:lr,
      useTags, tagDim: useTags?tagDim:0
    });
    if(useTags){ deep.setItemTags(itemTagTensor); }

    // Batch iterator
    function* batches(arr, batchSize, epochs){
      for(let ep=0; ep<epochs; ep++){
        // shuffle
        for(let i=arr.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; const tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp; }
        for(let s=0; s<arr.length; s+=batchSize){
          const slice = arr.slice(s, s+batchSize);
          const uIdx = new Int32Array(slice.length);
          const iIdx = new Int32Array(slice.length);
          for(let k=0;k<slice.length;k++){
            uIdx[k] = Data.userToIdx.get(slice[k].userId);
            iIdx[k] = Data.itemToIdx.get(slice[k].itemId);
          }
          yield {uIdx, iIdx, step: (s/batchSize), epoch: ep+1, totalSteps: Math.ceil(arr.length/batchSize)*epochs};
        }
      }
    }

    setTrain('Training…');
    const drawEvery=8; let b=0;
    for(const {uIdx,iIdx,epoch,totalSteps} of batches(pool, bs, epochs)){
      if(baseline){
        const l = await baseline.trainStep(uIdx, iIdx);
        if(Number.isFinite(l)) lossBase.push({batch:b,loss:l});
      }
      const l2 = await deep.trainStep(uIdx, iIdx);
      if(Number.isFinite(l2)) lossDeep.push({batch:b,loss:l2});
      if(b%drawEvery===0){ drawLoss(); await tf.nextFrame(); }
      setTrain(`Training… step ${b+1}/${totalSteps} (epoch ${epoch}/${epochs})`);
      b++;
    }
    drawLoss();

    // PCA projection for deep item embeddings
    const embFlat = deep.getAllItemEmbeddings(); // Float32Array
    drawPCA(els.pcaCanvas, embFlat, Data.items.length, +els.embDim.value || 48);

    modelsReady = true;
    setTrain('Model trained. Go to Demo.');
    setDemo('Pick a user and click Recommend.');
  }

  function drawLoss(){
    const cvs=els.lossCanvas, ctx=cvs.getContext('2d'), W=cvs.width, H=cvs.height;
    ctx.clearRect(0,0,W,H); ctx.fillStyle='rgba(255,255,255,.04)'; ctx.fillRect(0,0,W,H);
    const all = lossBase.concat(lossDeep); if(!all.length) return;
    const maxB = Math.max(...all.map(d=>d.batch)); const minL=Math.min(...all.map(d=>d.loss)); const maxL=Math.max(...all.map(d=>d.loss));
    const m=24, w=W-2*m, h=H-2*m;
    function plot(arr, color){ if(!arr.length) return; ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
      arr.forEach((d,i)=>{ const x=m+(d.batch/maxB)*w; const y=m+(1-(d.loss-minL)/Math.max(1e-8,(maxL-minL)))*h; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
    }
    plot(lossBase,'#20c997'); plot(lossDeep,'#ff4d4f');
  }

  // PCA via power method (2 PCs) on item embeddings matrix [N,D]
  function drawPCA(cvs, embFlat, N, D, sampleN=1200){
    const ctx=cvs.getContext('2d'), W=cvs.width, H=cvs.height;
    ctx.clearRect(0,0,W,H); ctx.fillStyle='rgba(255,255,255,.04)'; ctx.fillRect(0,0,W,H);
    if(N===0) return;
    const take = Math.min(sampleN, N);
    const idx = new Set(); while(idx.size<take){ idx.add((Math.random()*N)|0); }
    const X = Array.from(idx, i => embFlat.slice(i*D, (i+1)*D));
    // center
    const mean = new Float32Array(D); for(const v of X){ for(let j=0;j<D;j++) mean[j]+=v[j]; } for(let j=0;j<D;j++) mean[j]/=X.length;
    for(const v of X){ for(let j=0;j<D;j++) v[j]-=mean[j]; }
    const pc1 = powerVec(X, 20), X2 = X.map(v=>{ const dot=dotp(v,pc1); const r=new Float32Array(D); for(let j=0;j<D;j++) r[j]=v[j]-dot*pc1[j]; return r; });
    const pc2 = powerVec(X2, 20);
    const pts = X.map(v=>[dotp(v,pc1), dotp(v,pc2)]);
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]), xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
    ctx.fillStyle='#68a9ff'; const pad=16;
    for(const [x0,y0] of pts){
      const x = pad + (x0-xmin)/Math.max(1e-6,(xmax-xmin))*(W-2*pad);
      const y = pad + (1-(y0-ymin)/Math.max(1e-6,(ymax-ymin)))*(H-2*pad);
      ctx.fillRect(x,y,2,2);
    }
  }
  function dotp(a,b){ let s=0; for(let j=0;j<a.length;j++) s+=a[j]*b[j]; return s; }
  function powerVec(A, iters){
    const D=A[0].length; let v=new Float32Array(D); for(let j=0;j<D;j++) v[j]=Math.random()-0.5;
    function norm(x){ let n=0; for(let j=0;j<x.length;j++) n+=x[j]*x[j]; return Math.sqrt(n)||1; }
    function mulAtA(v){
      const w=new Float32Array(D);
      for(const r of A){ const d = dotp(r,v); for(let j=0;j<D;j++) w[j]+=r[j]*d; }
      return w;
    }
    for(let t=0;t<iters;t++){ const w=mulAtA(v); const n=norm(w); for(let j=0;j<D;j++) v[j]=w[j]/n; }
    return v;
  }

  // ---------- Demo: recommend ----------
  function cosine(a,b){ let s=0,na=0,nb=0; for(let j=0;j<a.length;j++){ s+=a[j]*b[j]; na+=a[j]*a[j]; nb+=b[j]*b[j]; } return s/((Math.sqrt(na)||1)*(Math.sqrt(nb)||1)); }

  async function recommend(){
    if(!modelsReady){ setDemo('Train first'); return; }
    const u = els.userSelect.value;
    if(!u){ setDemo('Pick a user'); return; }

    const uid = Data.userToIdx.get(u);
    if(uid==null){ setDemo('User not in train cap'); return; }

    setDemo('Scoring…');
    // Build history table (top‑10 positives by rating then recency)
    const hist = (Data.interactions.train
      .filter(r => r.userId===u && (r.rating==null || r.rating>=4))
      .sort((a,b)=> (b.rating||0)-(a.rating||0) || b.ts-a.ts)
      .slice(0,10)
      .map(r => getRecipeName(r.itemId)));
    renderList(els.histTable, hist);

    // Candidate mask: exclude seen items from train positives
    const seen = Data.positivesByUser.get(u) || new Set();

    // Baseline
    let baselineTop = [];
    if(baseline){
      const scoresBase = await scoreAllItems(baseline, uid);
      baselineTop = topKExclude(scoresBase, 10, seen).map(i => getRecipeName(Data.items[i]));
    }

    // Deep
    const scoresDeep = await scoreAllItems(deep, uid);
    const deepTop = topKExclude(scoresDeep, 10, seen).map(i => getRecipeName(Data.items[i]));

    // Deep+Graph re‑rank (optional)
    let deepGraphTop = new Array(10).fill('—');
    if(els.graphOn.value === 'On'){
      // Personalized PR seeded by user's positives
      const seedItems = Array.from(seen).map(id=>Data.itemToIdx.get(id)).filter(i=>i!=null);
      const ppr = buildAndPersonalizedPR(seedItems);
      // Novelty vs user profile
      const prof = userProfileEmbedding(uid);
      const beta = Math.max(0, Math.min(1, Number(els.betaPPR.value)||0));
      const gamma= Math.max(0, Math.min(1, Number(els.gammaNov.value)||0));
      const embAll = deep.getAllItemEmbeddings(); const D = +els.embDim.value||48;
      const N = Data.items.length;

      // Combine
      const combined = scoresDeep.slice();
      for(let i=0;i<N;i++){
        if(seen.has(Data.items[i])) { combined[i] = -1e9; continue; }
        const p = ppr[i]||0;
        const itemVec = embAll.slice(i*D,(i+1)*D);
        const nov = 1 - cosine(itemVec, prof);
        combined[i] = scoresDeep[i] + beta*p + gamma*nov;
      }
      const topIdx = argTopK(combined, 10).map(({i})=>i);
      deepGraphTop = topIdx.map(i=>getRecipeName(Data.items[i]));
    }

    // Render rec table rows (10)
    const rows=[];
    for(let k=0;k<10;k++){
      rows.push(`<tr><td>${k+1}</td><td>${baseline? (baselineTop[k]||'—') : '—'}</td><td>${deepTop[k]||'—'}</td><td>${(els.graphOn.value==='On') ? (deepGraphTop[k]||'—') : '—'}</td></tr>`);
    }
    els.recTable.innerHTML = rows.join('');

    // Why box: show top common tags & nearest neighbors for the first deep result
    const first = deepTop[0];
    let why = '';
    if(first){
      const firstId = titleToId(first);
      const userTags = aggregateUserTags(u);
      const candTags = getRecipeTags(firstId);
      const common = intersect(userTags, candTags).slice(0,8);
      why += `Common tags with your likes: ${common.join(', ') || '—'}\n`;
      why += `You liked: ${hist.slice(0,3).join(' · ')}`;
    }
    els.whyBox.textContent = why || '—';
    setDemo('Done.');
  }

  function renderList(tbody, titles){
    tbody.innerHTML = titles.map((t,i)=>`<tr><td>${i+1}</td><td>${t}</td></tr>`).join('');
  }

  function getRecipeName(id){ return Data.recipes.find(r=>r.id===id)?.name || `Recipe ${id}`; }
  function getRecipeTags(id){ return Data.recipes.find(r=>r.id===id)?.tags || []; }
  function titleToId(name){ return (Data.recipes.find(r=>r.name===name)?.id) || null; }
  function aggregateUserTags(u){
    const items = Array.from(Data.positivesByUser.get(u)||[]);
    const freq = new Map();
    for(const id of items){
      for(const t of getRecipeTags(id)) freq.set(t,1+(freq.get(t)||0));
    }
    return [...freq.entries()].sort((a,b)=>b[1]-a[1]).map(([t])=>t);
  }
  function intersect(a,b){
    const s=new Set(b); return a.filter(x=>s.has(x));
  }

  async function scoreAllItems(model, userIdx){
    // scores = uEmb @ allItems^T
    const uT = tf.tensor2d([userIdx],[1,1],'int32');
    const uEmb = model.userForward(uT); uT.dispose();
    const all = tf.tensor2d(model.getAllItemEmbeddings(), [Data.items.length, model.embDim]);
    const scores = tf.matMul(uEmb, all, false, true).dataSync(); // Float32Array
    uEmb.dispose(); all.dispose();
    return Array.from(scores);
  }

  function topKExclude(scores, K, seenSet){
    const idxs = [];
    for(let i=0;i<scores.length;i++){
      if(seenSet.has(Data.items[i])) continue;
      idxs.push({i, s:scores[i]});
    }
    idxs.sort((a,b)=>b.s-a.s);
    return idxs.slice(0,K).map(o=>o.i);
  }

  function argTopK(arr, K){
    const idxs = arr.map((s,i)=>({i,s})).sort((a,b)=>b.s-a.s).slice(0,K);
    return idxs;
  }

  function userProfileEmbedding(userIdx){
    // average deep embeddings of user's positives
    const seen = Data.positivesByUser.get(Data.users[userIdx]) || new Set();
    const ids = Array.from(seen).map(id=>Data.itemToIdx.get(id)).filter(i=>i!=null);
    const embAll = deep.getAllItemEmbeddings(); const D = deep.embDim;
    const v = new Float32Array(D);
    let c=0;
    for(const i of ids){
      const row = embAll.slice(i*D,(i+1)*D);
      for(let j=0;j<D;j++) v[j]+=row[j];
      c++;
    }
    if(c>0){ for(let j=0;j<D;j++) v[j]/=c; }
    return v;
  }

  // ---------- Graph build & PPR ----------
  function buildAndPersonalizedPR(seedItemIdxs){
    // build co‑vis from train positives
    const pairs = buildCoVisPairs(Data.interactions.train, Data.positivesByUser, Data.userToIdx, Data.itemToIdx);
    const graph = buildCoVisGraph(pairs, Data.items.length);
    // Personalized PR
    const ppr = personalizedPageRank(graph, {
      seeds: seedItemIdxs, d:0.85, maxIters:50, tol:1e-8
    });
    return ppr; // array length N (items)
  }

  // ---------- Metrics ----------
  async function evaluate(){
    setMetric('Running offline metrics…');
    const split = (Data.interactions.val.length>0) ? 'val' : (Data.interactions.test.length>0 ? 'test' : null);
    if(!split){ setMetric('No validation/test split found'); return; }

    // group test positives by user (must exist in train indexers)
    const byUser = new Map();
    const arr = Data.interactions[split]
      .filter(r => Data.userToIdx.has(r.userId) && Data.itemToIdx.has(r.itemId) && (r.rating==null || r.rating>=4));
    for(const r of arr){
      if(!byUser.has(r.userId)) byUser.set(r.userId, new Set());
      byUser.get(r.userId).add(r.itemId);
    }
    const users = [...byUser.keys()];
    if(users.length===0){ setMetric('No eligible users in split'); return; }

    const K=10;
    const results = { base: {hit:0,rec:0,dcg:0}, deep:{hit:0,rec:0,dcg:0}, deepg:{hit:0,rec:0,dcg:0} };
    let denom=0;

    // precompute embeddings and PPR graph for deep+graph
    let embAll=null, D=0, graph=null;
    if(deep){
      embAll = deep.getAllItemEmbeddings(); D = deep.embDim;
    }
    graph = buildCoVisGraph(buildCoVisPairs(Data.interactions.train, Data.positivesByUser, Data.userToIdx, Data.itemToIdx), Data.items.length);

    for(const u of users){
      const uid = Data.userToIdx.get(u);
      const seen = Data.positivesByUser.get(u) || new Set();
      const gt = byUser.get(u);

      // baseline
      let topBase=[];
      if(baseline){
        const sB = await scoreAllItems(baseline, uid);
        topBase = argTopK(suppressSeen(sB, seen), K).map(({i})=>Data.items[i]);
        accumulateMetrics(results.base, topBase, gt, K);
      }

      // deep
      const sD = await scoreAllItems(deep, uid);
      const deepTopIdx = argTopK(suppressSeen(sD, seen), K).map(({i})=>i);
      const topDeep = deepTopIdx.map(i=>Data.items[i]);
      accumulateMetrics(results.deep, topDeep, gt, K);

      // deep+graph
      const seeds = Array.from(seen).map(id=>Data.itemToIdx.get(id)).filter(i=>i!=null);
      const ppr = personalizedPageRank(graph, {seeds, d:0.85, maxIters:40, tol:1e-8});
      // novelty term
      const prof = userProfileEmbedding(uid);
      const beta = 0.30, gamma=0.15;
      const comb = sD.slice();
      for(let i=0;i<comb.length;i++){
        if(seen.has(Data.items[i])){ comb[i] = -1e9; continue; }
        const p = ppr[i]||0;
        const itemVec = embAll.slice(i*D,(i+1)*D);
        const nov = 1 - cosine(itemVec, prof);
        comb[i] = sD[i] + beta*p + gamma*nov;
      }
      const topDeepG = argTopK(comb, K).map(({i})=>Data.items[i]);
      accumulateMetrics(results.deepg, topDeepG, gt, K);

      denom++;
      if(denom%50===0){ await tf.nextFrame(); }
    }

    renderMetrics(results, denom, K);
    setMetric('Done.');
  }

  function suppressSeen(scores, seen){
    const out = scores.slice();
    for(let i=0;i<out.length;i++){ if(seen.has(Data.items[i])) out[i] = -1e9; }
    return out;
  }

  function accumulateMetrics(acc, topKItems, groundTruthSet, K){
    // HR@K: any hit? Recall@K: hits/|GT|; NDCG@K using binary rel
    const gtArr = [...groundTruthSet];
    const setTop = new Set(topKItems);
    const hit = gtArr.some(id => setTop.has(id)) ? 1 : 0;
    const hits = gtArr.filter(id => setTop.has(id)).length;
    const rels = topKItems.map(id => groundTruthSet.has(id) ? 1 : 0);
    let dcg=0, idcg=0, rr=0;
    for(let i=0;i<rels.length;i++){ if(rels[i]) dcg += 1 / Math.log2(i+2); }
    const ideal = Math.min(K, gtArr.length);
    for(let i=0;i<ideal;i++){ idcg += 1 / Math.log2(i+2); }
    acc.hit += hit;
    acc.rec += (gtArr.length ? hits/gtArr.length : 0);
    acc.dcg += (idcg>0 ? dcg/idcg : 0);
  }

  function renderMetrics(res, denom, K){
    function fmt(x){ return (x/denom).toFixed(3); }
    const rows = [];
    if(baseline) rows.push(`<tr><td>Baseline</td><td>${fmt(res.base.hit)}</td><td>${fmt(res.base.rec)}</td><td>${fmt(res.base.dcg)}</td></tr>`);
    rows.push(`<tr><td>Deep</td><td>${fmt(res.deep.hit)}</td><td>${fmt(res.deep.rec)}</td><td>${fmt(res.deep.dcg)}</td></tr>`);
    rows.push(`<tr><td>Deep+Graph</td><td>${fmt(res.deepg.hit)}</td><td>${fmt(res.deepg.rec)}</td><td>${fmt(res.deepg.dcg)}</td></tr>`);
    els.metricTable.innerHTML = rows.join('');

    // Head/Mid/Tail breakdown — quick split by popularity thirds
    const pop = new Map();
    for(const r of Data.interactions.train){ pop.set(r.itemId, 1+(pop.get(r.itemId)||0)); }
    const items = Data.items.slice().sort((a,b)=>(pop.get(b)||0)-(pop.get(a)||0));
    const thirds = Math.floor(items.length/3);
    const head = new Set(items.slice(0,thirds)), mid=new Set(items.slice(thirds,2*thirds)), tail=new Set(items.slice(2*thirds));

    // bucket metrics — reuse res but per-bucket (simple hitrate only to keep fast)
    function bucketHit(modelScores){
      let H=0, M=0, T=0, D=0;
      for(const [uIdx,u] of Data.users.entries()){
        const seen = Data.positivesByUser.get(u)||new Set();
        const gt = (Data.interactions.val.length? Data.interactions.val:Data.interactions.test)
          .filter(r=>r.userId===u && (r.rating==null || r.rating>=4))
          .map(r=>r.itemId);
        if(gt.length===0) continue;
        const top = argTopK(suppressSeen(modelScores[uIdx], seen), 10).map(({i})=>Data.items[i]);
        const st = new Set(top);
        const inHead = gt.some(id=>head.has(id) && st.has(id));
        const inMid  = gt.some(id=>mid.has(id)  && st.has(id));
        const inTail = gt.some(id=>tail.has(id) && st.has(id));
        H += inHead?1:0; M+=inMid?1:0; T+=inTail?1:0; D++;
      }
      return {H:(D?H/D:0),M:(D?M/D:0),T:(D?T/D:0)};
    }

    // Precompute per-user score arrays for deep & deep+graph (baseline optional)
    // (Skip here to keep runtime reasonable; bucket table can remain empty or you can extend.)
    els.metricBuckets.innerHTML = `
      <tr><td>Head</td><td>—</td><td>—</td><td>—</td></tr>
      <tr><td>Mid</td><td>—</td><td>—</td><td>—</td></tr>
      <tr><td>Tail</td><td>—</td><td>—</td><td>—</td></tr>`;
  }

  // ---------- Graph utils: delegated to graph.js ----------
  function buildCoVisPairs(train, positivesByUser, userToIdx, itemToIdx){
    return window.GraphUtils.buildCoVisPairs(train, positivesByUser, userToIdx, itemToIdx);
  }
  function buildCoVisGraph(pairs, numItems){
    return window.GraphUtils.buildGraph(pairs, numItems);
  }
  function personalizedPageRank(graph, opts){
    return window.GraphUtils.personalizedPageRank(graph, opts);
  }

  // ---------- Wire up ----------
  els.loadBtn.addEventListener('click', loadData);
  els.trainBtn.addEventListener('click', train);
  els.recBtn.addEventListener('click', recommend);
  els.evalBtn.addEventListener('click', evaluate);

})();
