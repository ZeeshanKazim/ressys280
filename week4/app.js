// app.js
// Two-Tower retrieval demo with a Deep variant (MLP towers + genres).
// Pure client-side; reads /data/u.data and /data/u.item.

(() => {
  // ---------- Global State ----------
  const state = {
    interactions: [],                 // {userId, itemId, rating, ts}
    items: new Map(),                 // itemId -> {title, year, genres:Int8Array(18)}
    users: new Set(), itemIds: new Set(),
    userToRated: new Map(),           // userId -> Set(itemId)
    userTopRated: new Map(),          // userId -> [{itemId, rating, ts, title, year}]
    // indexers
    userId2idx: new Map(), itemId2idx: new Map(),
    idx2userId: [], idx2itemId: [],
    // models
    baseline: null, deep: null,
    trained: { baseline:false, deep:false },
    // config
    cfg: { max:80000, dim:32, hid:64, batch:256, epochs:5, lr:0.003, loss:'softmax', both:'yes' },
    // chart
    loss: { baseline:[], deep:[] },
    // projection points
    projPoints: []
  };

  // ---------- DOM ----------
  const $ = id => document.getElementById(id);
  const setStatus = t => { $('status').textContent = `Status: ${t}`; };

  // ---------- Tiny multi-series line plot ----------
  class MultiChart {
    constructor(canvas){
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
    }
    draw(seriesObj){ // {name:[y1,y2,...]}
      const W=this.canvas.width, H=this.canvas.height, ctx=this.ctx;
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle='#0c1424'; ctx.fillRect(0,0,W,H);
      ctx.strokeStyle='#21324a'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(40,10); ctx.lineTo(40,H-20); ctx.lineTo(W-10,H-20); ctx.stroke();

      const arrays = Object.values(seriesObj).filter(a=>a && a.length);
      if (!arrays.length) return;
      const all = arrays.flat();
      const ymin = Math.min(...all), ymax = Math.max(...all);
      const maxLen = Math.max(...arrays.map(a=>a.length));
      const sx = x => 50 + (x/(Math.max(1,maxLen-1))) * (W-64);
      const sy = y => (H-24) - ((y - ymin)/Math.max(1e-9, ymax-ymin)) * (H-34);

      const palette = { baseline:'#3fb950', deep:'#e50914' };

      for (const [name, arr] of Object.entries(seriesObj)){
        if (!arr || !arr.length) continue;
        ctx.strokeStyle = palette[name] || '#6ea8fe';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i=0;i<arr.length;i++){
          const x=sx(i), y=sy(arr[i]);
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.stroke();
      }

      ctx.fillStyle='#9fb0c5'; ctx.font='12px ui-monospace, monospace';
      ctx.fillText(`${ymax.toFixed(3)}`, 6, sy(ymax));
      ctx.fillText(`${ymin.toFixed(3)}`, 6, sy(ymin));
    }
  }
  const lossChart = new MultiChart($('lossCanvas'));

  // ---------- Config ----------
  function readConfig() {
    state.cfg.max   = Math.max(1000, parseInt($('cfg-max').value,10) || 80000);
    state.cfg.dim   = Math.max(8, parseInt($('cfg-dim').value,10) || 32);
    state.cfg.hid   = Math.max(16, parseInt($('cfg-hid').value,10) || 64);
    state.cfg.batch = Math.max(32, parseInt($('cfg-batch').value,10) || 256);
    state.cfg.epochs= Math.max(1, parseInt($('cfg-epochs').value,10) || 5);
    state.cfg.lr    = Math.max(1e-5, parseFloat($('cfg-lr').value) || 0.003);
    state.cfg.loss  = $('cfg-loss').value;
    state.cfg.both  = $('cfg-both').value;
  }

  // ---------- Data Loading ----------
  async function loadData() {
    setStatus('loading data…');
    const [uItemResp, uDataResp] = await Promise.all([ fetch('data/u.item'), fetch('data/u.data') ]);
    if (!uItemResp.ok || !uDataResp.ok) {
      setStatus('failed to fetch /data files — ensure /data/u.item and /data/u.data exist');
      throw new Error('fetch error');
    }

    // genre names (MovieLens-100K order) — 19 with "Unknown" in u.item; we ignore that one
    const genre19 = [
      'Unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy',
      'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
    ];

    // Parse u.item (pipe-delimited)
    const tItem = await uItemResp.text();
    const linesItem = tItem.split('\n').filter(l => l.trim().length);
    for (const line of linesItem) {
      const parts = line.split('|');
      const id = parseInt(parts[0],10);
      const titleRaw = parts[1] || String(id);
      let title = titleRaw, year = null;
      const m = titleRaw.match(/\((\d{4})\)\s*$/);
      if (m) { year = +m[1]; title = titleRaw.replace(/\s*\(\d{4}\)\s*$/,''); }
      // trailing flags: 19 or 18
      const flags = parts.slice(5).map(x => x==='1'?1:0);
      const use = (flags.length>=19) ? flags.slice(1) : flags; // drop "Unknown"
      const g = new Int8Array(18);
      for (let k=0;k<Math.min(18,use.length);k++) g[k]=use[k];

      state.items.set(id, {title, year, genres:g});
      state.itemIds.add(id);
    }

    // Parse u.data (tab-delimited)
    const tData = await uDataResp.text();
    const linesData = tData.split('\n').filter(l => l.trim().length);
    for (const line of linesData) {
      const [u,i,r,ts] = line.split('\t');
      const userId=+u, itemId=+i, rating=+r, t=+ts;
      if (!Number.isFinite(userId) || !Number.isFinite(itemId)) continue;
      state.interactions.push({userId,itemId,rating,ts:t});
      state.users.add(userId);
    }

    // Build user->rated and top-rated lists
    for (const {userId,itemId} of state.interactions) {
      if (!state.userToRated.has(userId)) state.userToRated.set(userId, new Set());
      state.userToRated.get(userId).add(itemId);
    }
    const byUser = new Map();
    for (const row of state.interactions) {
      if (!byUser.has(row.userId)) byUser.set(row.userId, []);
      byUser.get(row.userId).push(row);
    }
    for (const [uid, arr] of byUser.entries()) {
      arr.sort((a,b)=> (b.rating-a.rating) || (b.ts-a.ts));
      const top = arr.slice(0,60).map(x=> ({
        itemId:x.itemId, rating:x.rating, ts:x.ts,
        title: state.items.get(x.itemId)?.title ?? String(x.itemId),
        year:  state.items.get(x.itemId)?.year ?? ''
      }));
      state.userTopRated.set(uid, top);
    }

    // Indexers
    const userIds = Array.from(state.users).sort((a,b)=>a-b);
    const itemIds = Array.from(state.itemIds).sort((a,b)=>a-b);
    state.idx2userId = userIds; state.idx2itemId = itemIds;
    state.userId2idx = new Map(userIds.map((u,idx)=>[u,idx]));
    state.itemId2idx = new Map(itemIds.map((i,idx)=>[i,idx]));

    setStatus(`data loaded — users: ${userIds.length}, items: ${itemIds.length}, interactions: ${state.interactions.length}`);
    $('btn-train').classList.remove('secondary');
    $('btn-test').classList.remove('secondary');
  }

  // ---------- Training data ----------
  function buildPairs(maxN){
    const N = Math.min(maxN, state.interactions.length);
    const idxs = [...Array(state.interactions.length)].map((_,i)=>i);
    for (let i=idxs.length-1;i>0;i--){const j=(Math.random()*(i+1))|0; [idxs[i],idxs[j]]=[idxs[j],idxs[i]];}
    const out=[];
    for (let k=0;k<N;k++){
      const row=state.interactions[idxs[k]];
      const u=state.userId2idx.get(row.userId), it=state.itemId2idx.get(row.itemId);
      if (u!=null && it!=null) out.push([u,it]);
    }
    return out;
  }

  function buildItemGenreTensor(){ // [numItems, 18] float32
    const N = state.idx2itemId.length;
    const G = 18;
    const buf = new Float32Array(N*G);
    for (let i=0;i<N;i++){
      const id = state.idx2itemId[i];
      const g = state.items.get(id)?.genres || new Int8Array(G);
      for (let k=0;k<G;k++) buf[i*G+k] = g[k];
    }
    return tf.tensor2d(buf, [N,G], 'float32');
  }

  // ---------- Training ----------
  async function train() {
    if (!state.interactions.length) { setStatus('load data first'); return; }
    readConfig();
    state.loss = {baseline:[], deep:[]};
    lossChart.draw(state.loss);
    setStatus('initializing models…');

    const pairs = buildPairs(state.cfg.max);
    const U = state.idx2userId.length, I = state.idx2itemId.length;

    // Baseline (optional)
    if (state.cfg.both === 'yes') {
      if (!state.baseline) state.baseline = new TwoTowerModel(U, I, state.cfg.dim, state.cfg.loss);
      state.baseline.setOptimizer(tf.train.adam(state.cfg.lr));
      await trainOne(state.baseline, pairs, 'baseline');
      state.trained.baseline = true;
    }

    // Deep (uses genres + MLP)
    const itemGenre = buildItemGenreTensor(); // constant features
    if (!state.deep) state.deep = new TwoTowerDeepModel(U, I, state.cfg.dim, 18, state.cfg.hid, state.cfg.loss, itemGenre);
    state.deep.setOptimizer(tf.train.adam(state.cfg.lr));
    await trainOne(state.deep, pairs, 'deep');
    state.trained.deep = true;

    // Projection of deep item outputs
    await drawProjectionSample(state.deep);
    setStatus('training done — ready to test');
  }

  async function trainOne(model, pairs, key){
    const B=state.cfg.batch, epochs=state.cfg.epochs;
    const steps=Math.ceil(pairs.length/B);
    const lossArr = state.loss[key];

    for (let ep=0; ep<epochs; ep++){
      for (let s=0; s<steps; s++){
        const start=s*B, end=Math.min((s+1)*B, pairs.length);
        const batch=pairs.slice(start,end);
        const uIdx=new Int32Array(batch.length), iIdx=new Int32Array(batch.length);
        for (let t=0;t<batch.length;t++){ uIdx[t]=batch[t][0]; iIdx[t]=batch[t][1]; }
        const L = await model.trainStep(uIdx, iIdx);
        lossArr.push(L);
        if ((s&1)===0) lossChart.draw(state.loss);
        setStatus(`${key}: epoch ${ep+1}/${epochs} — step ${s+1}/${steps} — loss ${L.toFixed(4)}`);
        await tf.nextFrame();
      }
    }
    lossChart.draw(state.loss);
  }

  // ---------- Projection (PCA on deep item outputs) ----------
  async function drawProjectionSample(model){
    const canvas=$('projCanvas'), ctx=canvas.getContext('2d'), W=canvas.width, H=canvas.height;
    ctx.clearRect(0,0,W,H); ctx.fillStyle='#0c1424'; ctx.fillRect(0,0,W,H);

    const N = state.idx2itemId.length;
    const maxSamp = Math.min(1000, N);
    const idxs=[...Array(N).keys()];
    for (let i=idxs.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [idxs[i],idxs[j]]=[idxs[j],idxs[i]]; }
    const sample = idxs.slice(0,maxSamp);

    // get deep item outputs in batches
    const chunk=2048;
    const outChunks=[];
    for (let s=0;s<sample.length;s+=chunk){
      const block = sample.slice(s, Math.min(sample.length, s+chunk));
      const iT = tf.tensor1d(block,'int32');
      const eT = model.itemForward(iT); // [B,D]
      outChunks.push(eT);
    }
    const emb = tf.concat(outChunks, 0); outChunks.forEach(t=>t.dispose()); // [S,D]

    let XY;
    try {
      XY = tf.tidy(()=>{
        const mean = emb.mean(0), Xc=emb.sub(mean);
        const svd = tf.svd ? tf.svd(Xc, true) : (tf.linalg && tf.linalg.svd ? tf.linalg.svd(Xc,true) : null);
        const V2 = svd.v.slice([0,0],[-1,2]);
        return Xc.matMul(V2);
      });
    } catch(e) {
      XY = emb.slice([0,0],[-1,2]);
    }
    const xy = await XY.array();
    emb.dispose(); XY.dispose();

    const xs=xy.map(p=>p[0]), ys=xy.map(p=>p[1]);
    const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
    const pad=16;
    state.projPoints = [];
    ctx.fillStyle='#e6eef788';
    for (let k=0;k<xy.length;k++){
      const id = state.idx2itemId[sample[k]];
      const title = state.items.get(id)?.title ?? String(id);
      const x = pad + ((xs[k]-xmin)/Math.max(1e-6,xmax-xmin))*(W-2*pad);
      const y = H-pad - ((ys[k]-ymin)/Math.max(1e-6,ymax-ymin))*(H-2*pad);
      ctx.beginPath(); ctx.arc(x,y,2,0,6.28); ctx.fill();
      state.projPoints.push({x,y,title});
    }

    canvas.onmousemove = (ev)=>{
      const r=canvas.getBoundingClientRect();
      const mx=(ev.clientX-r.left)*(canvas.width/r.width), my=(ev.clientY-r.top)*(canvas.height/r.height);
      ctx.clearRect(0,0,W,H); ctx.fillStyle='#0c1424'; ctx.fillRect(0,0,W,H); ctx.fillStyle='#e6eef788';
      for (const p of state.projPoints){ ctx.beginPath(); ctx.arc(p.x,p.y,2,0,6.28); ctx.fill(); }
      let best=null, bestD=9;
      for (const p of state.projPoints){ const d=(p.x-mx)**2+(p.y-my)**2; if (d<bestD){bestD=d; best=p;} }
      if (best){ ctx.fillStyle='#fff'; ctx.font='12px system-ui, sans-serif';
        const t=best.title.length>42?best.title.slice(0,39)+'…':best.title;
        ctx.fillText(t, Math.min(W-6-ctx.measureText(t).width, Math.max(6,best.x+8)), Math.max(14,best.y-8));
      }
    };
  }

  // ---------- Test ----------
  async function testOnce(){
    if (!state.trained.deep && !state.trained.baseline) { setStatus('train first'); return; }

    // pick user with >=20 ratings
    const candidates=[];
    for (const [u, arr] of state.userTopRated.entries()) if (arr.length>=20) candidates.push(u);
    if (!candidates.length){ setStatus('no users with ≥20 ratings'); return; }
    const userId = candidates[(Math.random()*candidates.length)|0];
    const left = state.userTopRated.get(userId).slice(0,10);

    const middle = state.trained.baseline ? await recommendWithBaseline(userId, 10) : [];
    const right  = state.trained.deep     ? await recommendWithDeep(userId, 10)     : [];

    renderThree(userId, left, middle, right);
    setStatus(`tested user ${userId}`);
  }

  async function recommendWithBaseline(userId, K){
    const rated = state.userToRated.get(userId) || new Set();
    const uIdx = state.userId2idx.get(userId);
    const uEmb = state.baseline.getUserEmbeddingForIndex(uIdx); // [1,D]
    const itemEmb = state.baseline.getItemEmbedding();          // [N,D]
    const scores = await batchedDot(uEmb, itemEmb);
    uEmb.dispose();

    const arr=[];
    for (let i=0;i<scores.length;i++){
      const iid = state.idx2itemId[i];
      if (rated.has(iid)) continue;
      arr.push({ idx:i, score:scores[i] });
    }
    arr.sort((a,b)=> b.score-a.score);
    const top = arr.slice(0,K).map(o=>{
      const iid=state.idx2itemId[o.idx];
      return { itemId:iid, score:o.score, title:state.items.get(iid)?.title ?? String(iid) };
    });
    return top;
  }

  async function recommendWithDeep(userId, K){
    const rated = state.userToRated.get(userId) || new Set();
    const uIdx = state.userId2idx.get(userId);
    const uOut = state.deep.getUserEmbeddingForIndex(uIdx); // [1,D]
    const N = state.idx2itemId.length, chunk=2048;
    const scores = new Float32Array(N);

    for (let s=0;s<N;s+=chunk){
      const end=Math.min(N,s+chunk);
      const idx = tf.tensor1d([...Array(end-s).keys()].map(x=>x+s),'int32');
      const iOut = state.deep.itemForward(idx);            // [c,D]
      const block = tf.tidy(()=> iOut.matMul(uOut.transpose()).squeeze()); // [c]
      const vals = await block.data();
      scores.set(vals,s);
      block.dispose(); iOut.dispose(); idx.dispose();
      await tf.nextFrame();
    }
    uOut.dispose();

    const arr=[];
    for (let i=0;i<scores.length;i++){
      const iid = state.idx2itemId[i];
      if (rated.has(iid)) continue;
      arr.push({ idx:i, score:scores[i] });
    }
    arr.sort((a,b)=> b.score-a.score);
    const top = arr.slice(0,K).map(o=>{
      const iid=state.idx2itemId[o.idx];
      return { itemI
