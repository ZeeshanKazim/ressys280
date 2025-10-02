/* app.js
   - Loads MovieLens 100K (u.item, u.data) from ./data/
   - Builds user/item indexers and interactions
   - Trains Two-Tower baseline + deep models
   - Plots loss, draws PCA, and renders comparison table
*/

(() => {
  const els = {
    status: document.getElementById('status'),
    loadBtn: document.getElementById('loadBtn'),
    trainBtn: document.getElementById('trainBtn'),
    testBtn:  document.getElementById('testBtn'),
    loss:     document.getElementById('lossCanvas'),
    pca:      document.getElementById('pcaCanvas'),
    table:    document.getElementById('comparison'),
    // hyperparams
    maxInt:   document.getElementById('maxInt'),
    embDim:   document.getElementById('embDim'),
    hidden:   document.getElementById('hiddenDim'),
    batch:    document.getElementById('batchSize'),
    epochs:   document.getElementById('epochs'),
    lr:       document.getElementById('lr'),
    lossSel:  document.getElementById('lossSel'),
    compare:  document.getElementById('compare'),
  };

  function setStatus(msg){ els.status.textContent = `Status: ${msg}`; }
  function enTrain(on){ els.trainBtn.disabled = !on; }
  function enTest(on){ els.testBtn.disabled = !on; }

  // ---------- Data loading ----------
  const MovieDataLoader = {
    async load(itemPath, dataPath){
      const [itemText, dataText] = await Promise.all([
        fetch(itemPath).then(r=>{ if(!r.ok) throw new Error('u.item fetch'); return r.text(); }),
        fetch(dataPath).then(r=>{ if(!r.ok) throw new Error('u.data fetch'); return r.text(); }),
      ]);

      // parse u.item
      // id|title|release_date|...|19 genre flags  OR sometimes 18 flags (without "Unknown")
      const linesI = itemText.split('\n').filter(Boolean);
      const items = new Map(); // id -> {title, year}
      let maxItemId = 0;
      for(const ln of linesI){
        const p = ln.split('|');
        const id = parseInt(p[0],10);
        if (!Number.isFinite(id)) continue;
        const title = p[1] || `Movie ${id}`;
        const year = (title.match(/\((\d{4})\)/)||[])[1] || '';
        items.set(id, {title, year});
        if (id>maxItemId) maxItemId=id;
      }

      // parse u.data
      const linesD = dataText.split('\n').filter(Boolean);
      const interactions = [];
      const userSet = new Set();
      const itemSet = new Set();

      for(const ln of linesD){
        const [u,i,r,t] = ln.split('\t');
        const userId = parseInt(u,10);
        const itemId = parseInt(i,10);
        const rating = parseInt(r,10);
        const ts     = parseInt(t,10);
        if(!Number.isFinite(userId) || !Number.isFinite(itemId)) continue;
        interactions.push({userId, itemId, rating, ts});
        userSet.add(userId); itemSet.add(itemId);
      }

      // indexers
      const users = Array.from(userSet).sort((a,b)=>a-b);
      const userToIdx = new Map(users.map((u,ix)=>[u,ix]));
      const itemsSorted = Array.from(itemSet).sort((a,b)=>a-b);
      const itemToIdx = new Map(itemsSorted.map((it,ix)=>[it,ix]));

      // build ratingsByUser for test/eval
      const byUser = new Map(); // userId -> array of {itemId, rating, ts}
      for(const r of interactions){
        if(!byUser.has(r.userId)) byUser.set(r.userId, []);
        byUser.get(r.userId).push({itemId:r.itemId, rating:r.rating, ts:r.ts});
      }
      // keep only users with at least 20 ratings (for sampling)
      const heavyUsers = users.filter(u => byUser.get(u)?.length >= 20);

      return {
        interactions, users, items: itemsSorted, userToIdx, itemToIdx, itemsMeta: items,
        ratingsByUser: byUser, heavyUsers,
        numUsers: users.length, numItems: itemsSorted.length
      };
    },

    // build shuffled minibatches (userIdx, posItemIdx)
    *makeBatches(DATA, batchSize, epochs, maxInteractions){
      const pool = DATA.interactions.slice(0, maxInteractions ?? DATA.interactions.length);
      for(let ep=0; ep<epochs; ep++){
        // shuffle each epoch
        for(let i=pool.length-1;i>0;i--){
          const j = (Math.random()* (i+1))|0; const tmp = pool[i]; pool[i] = pool[j]; pool[j]=tmp;
        }
        for(let s=0; s<pool.length; s+=batchSize){
          const slice = pool.slice(s, s+batchSize);
          const uIdx = new Int32Array(slice.length);
          const iIdx = new Int32Array(slice.length);
          for(let k=0;k<slice.length;k++){
            uIdx[k] = DATA.userToIdx.get(slice[k].userId);
            iIdx[k] = DATA.itemToIdx.get(slice[k].itemId);
          }
          yield {uIdx, iIdx, step: s/batchSize, epoch: ep+1, stepsPerEpoch: Math.ceil(pool.length/batchSize), totalSteps: Math.ceil(pool.length/batchSize)*epochs};
        }
      }
    },

    // recommendation helper
    recommendForRandomUser: async ({DATA, baseline, deepModel, topK=10})=>{
      // pick user with >= 20 ratings
      const userId = DATA.heavyUsers[(Math.random()*DATA.heavyUsers.length)|0];
      const rated = DATA.ratingsByUser.get(userId).slice().sort((a,b)=>{
        if(b.rating!==a.rating) return b.rating-a.rating;
        return b.ts - a.ts;
      });
      const topRatedTitles = rated.slice(0, topK).map(r => DATA.itemsMeta.get(r.itemId)?.title || `Movie ${r.itemId}`);

      const uIdx = DATA.userToIdx.get(userId);
      const ratedSet = new Set(rated.map(x=>x.itemId));

      // compute recs
      const run = async (model) => {
        if(!model) return [];
        const uT = tf.tensor2d([uIdx],[1,1],'int32');
        const uEmb = model.userForward(uT); // [1,dim]
        uT.dispose();
        // all item embeddings
        const all = tf.tensor2d(model.getAllItemEmbeddings(), [DATA.numItems, model.embDim]);
        // scores = uEmb @ all^T
        const scores = tf.matMul(uEmb, all, false, true).dataSync(); // Float32Array length numItems
        uEmb.dispose(); all.dispose();

        // topK excluding rated
        const idxs = [];
        for(let i=0;i<DATA.numItems;i++){
          const itId = DATA.items[i];
          if(!ratedSet.has(itId)) idxs.push({i, s: scores[i]});
        }
        idxs.sort((a,b)=>b.s-a.s);
        const top = idxs.slice(0, topK).map(o=>DATA.itemsMeta.get(DATA.items[o.i])?.title || `Movie ${DATA.items[o.i]}`);
        return top;
      };

      const baselineRec = await run(baseline);
      const deepRec     = await run(deepModel);

      return { topRated: topRatedTitles, recBase: baselineRec, recDeep: deepRec };
    }
  };

  // expose for dev (optional)
  window.MovieDataLoader = MovieDataLoader;

  // ---------- Plotting ----------
  const lossHistoryBase = [];
  const lossHistoryDeep = [];

  function drawLoss(){
    const cvs = els.loss, ctx = cvs.getContext('2d');
    const W=cvs.width, H=cvs.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='rgba(255,255,255,.04)'; ctx.fillRect(0,0,W,H);

    const all = lossHistoryBase.concat(lossHistoryDeep);
    if (!all.length) return;

    const maxBatch = Math.max(...all.map(d=>d.batch));
    const minLoss  = Math.min(...all.map(d=>d.loss));
    const maxLoss  = Math.max(...all.map(d=>d.loss));
    const margin = 24, plotW=W-margin*2, plotH=H-margin*2;
    const xy = (d) => {
      const x = margin + (d.batch / Math.max(1,maxBatch)) * plotW;
      const y = margin + (1 - (d.loss - minLoss)/Math.max(1e-8,(maxLoss-minLoss))) * plotH;
      return [x,y];
    };

    // axes
    ctx.strokeStyle='rgba(255,255,255,.15)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(margin, margin); ctx.lineTo(margin, H-margin); ctx.lineTo(W-margin, H-margin); ctx.stroke();

    // baseline green
    if(lossHistoryBase.length){
      ctx.strokeStyle='#20c997'; ctx.lineWidth=2; ctx.beginPath();
      lossHistoryBase.forEach((d,i)=>{ const [x,y]=xy(d); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
    }
    // deep red
    if(lossHistoryDeep.length){
      ctx.strokeStyle='#ff4d4f'; ctx.lineWidth=2; ctx.beginPath();
      lossHistoryDeep.forEach((d,i)=>{ const [x,y]=xy(d); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
    }
  }

  // quick PCA (power method for 2 PCs)
  function drawPCA(matrix, sampleN=1200){
    const cvs = els.pca, ctx = cvs.getContext('2d');
    const W=cvs.width, H=cvs.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='rgba(255,255,255,.04)'; ctx.fillRect(0,0,W,H);
    if(!matrix || !matrix.length) return;

    const D = matrix[0].length, M = matrix.length;
    const idx = [];
    const take = Math.min(sampleN, M);
    for(let i=0;i<take;i++) idx.push((Math.random()*M)|0);
    const X = idx.map(i=>matrix[i]);

    // center
    const mean = new Float32Array(D);
    X.forEach(v=>{ for(let j=0;j<D;j++) mean[j]+=v[j]; });
    for(let j=0;j<D;j++) mean[j] /= X.length;
    const C = X.map(v=>{
      const r = new Float32Array(D);
      for(let j=0;j<D;j++) r[j]=v[j]-mean[j];
      return r;
    });

    const power = (A,iters=20)=>{
      let v = new Float32Array(D);
      for(let j=0;j<D;j++) v[j]=Math.random()-0.5;
      let n = Math.hypot(...v); for(let j=0;j<D;j++) v[j]/=(n||1);
      for(let t=0;t<iters;t++){
        const w = new Float32Array(D);
        for(const row of A){
          let dot=0; for(let k=0;k<D;k++) dot += row[k]*v[k];
          for(let k=0;k<D;k++) w[k]+= row[k]*dot;
        }
        n = Math.hypot(...w); for(let j=0;j<D;j++) v[j]=w[j]/(n||1);
      }
      return v;
    };
    const pc1 = power(C,20);
    const C2 = C.map(r=>{
      let dot=0; for(let k=0;k<D;k++) dot+= r[k]*pc1[k];
      const rr = new Float32Array(D);
      for(let k=0;k<D;k++) rr[k]= r[k]-dot*pc1[k];
      return rr;
    });
    const pc2 = power(C2,20);

    const pts = C.map(r=>{
      let x=0,y=0; for(let k=0;k<D;k++){ x+= r[k]*pc1[k]; y+= r[k]*pc2[k]; }
      return [x,y];
    });
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    const xMin=Math.min(...xs), xMax=Math.max(...xs);
    const yMin=Math.min(...ys), yMax=Math.max(...ys);

    ctx.fillStyle = '#68a9ff';
    const pad=16;
    pts.forEach(([x,y])=>{
      const px = (x-xMin)/Math.max(1e-6,(xMax-xMin));
      const py = (y-yMin)/Math.max(1e-6,(yMax-yMin));
      const cx = pad + px*(W-pad*2);
      const cy = pad + (1-py)*(H-pad*2);
      ctx.fillRect(cx,cy,2,2);
    });
  }

  // ---------- Training/Test orchestration ----------
  let DATA = null, baseline = null, deepModel = null, trained=false;

  async function onLoad(){
    try{
      setStatus('loading data…');
      DATA = await MovieDataLoader.load('./data/u.item','./data/u.data');
      setStatus(`data loaded — users: ${DATA.numUsers}, items: ${DATA.numItems}, interactions: ${DATA.interactions.length}`);
      enTrain(true); enTest(false); trained=false;
      lossHistoryBase.length=0; lossHistoryDeep.length=0; drawLoss(); drawPCA([]);
    }catch(e){
      console.error(e);
      setStatus('fetch failed. Ensure ./data/u.item and ./data/u.data exist (case-sensitive).');
    }
  }

  async function onTrain(){
    if(!DATA){ setStatus('load data first'); return; }
    // hyperparams
    const embDim = +els.embDim.value || 40;
    const hidden = +els.hidden.value || 64;
    const lr     = +els.lr.value     || 0.003;
    const batch  = +els.batch.value  || 256;
    const epochs = +els.epochs.value || 5;
    const maxInt = +els.maxInt.value || DATA.interactions.length;
    const lossType = (els.lossSel.value.startsWith('In-batch')) ? 'softmax' : 'bpr';
    const compare = els.compare.value.startsWith('Yes');

    // dispose any prior models
    if(baseline && baseline.dispose) baseline.dispose();
    if(deepModel && deepModel.dispose) deepModel.dispose();
    baseline=null; deepModel=null; trained=false;

    setStatus('building models…');
    if(compare){
      baseline = new TwoTowerModel(DATA.numUsers, DATA.numItems, embDim, {
        deep:false, hiddenDim:0, lossType, learningRate:lr
      });
    }
    deepModel = new TwoTowerModel(DATA.numUsers, DATA.numItems, embDim, {
      deep:true, hiddenDim:hidden, lossType, learningRate:lr
    });

    // train
    setStatus('training…');
    enTrain(false); enTest(false);
    lossHistoryBase.length=0; lossHistoryDeep.length=0;

    const drawEvery = 10;
    let batchId = 0, stepSeen = 0;
    for (const {uIdx, iIdx, step, epoch, stepsPerEpoch, totalSteps} of MovieDataLoader.makeBatches(DATA, batch, epochs, maxInt)){
      if (baseline) {
        const l = await baseline.trainStep(uIdx, iIdx);
        if(Number.isFinite(l)) lossHistoryBase.push({batch:batchId, loss:l});
      }
      {
        const l = await deepModel.trainStep(uIdx, iIdx);
        if(Number.isFinite(l)) lossHistoryDeep.push({batch:batchId, loss:l});
      }
      batchId++; stepSeen++;
      if (batchId % drawEvery === 0){ drawLoss(); await tf.nextFrame(); }
      setStatus(`training… step ${stepSeen}/${stepsPerEpoch*epochs} (epoch ${epoch}/${epochs})`);
    }

    drawLoss();

    // PCA from deep item embeddings
    const flat = deepModel.getAllItemEmbeddings(); // Float32Array
    const embMat = [];
    for (let i=0;i<DATA.numItems;i++){
      embMat.push(Array.from(flat.slice(i*embDim, (i+1)*embDim)));
    }
    drawPCA(embMat);

    setStatus('training finished.');
    trained=true; enTest(true);
  }

  async function onTest(){
    if(!trained){ setStatus('train first'); return; }
    setStatus('generating recommendations…');
    const {topRated, recBase, recDeep} = await MovieDataLoader.recommendForRandomUser({
      DATA, baseline, deepModel, topK:10
    });
    renderTable(topRated, recBase, recDeep);
    setStatus('recommendations generated successfully!');
  }

  function renderTable(topRated, recBase, recDeep){
    const tr = a => a.map((t,i)=>`<tr><td>${i+1}</td><td>${t}</td></tr>`).join('');
    els.table.innerHTML = `
      <div class="grid-3">
        <div>
          <h4>Top-10 Rated (historical)</h4>
          <table class="t"><thead><tr><th>#</th><th>Movie</th></tr></thead><tbody>${tr(topRated)}</tbody></table>
        </div>
        <div>
          <h4>Top-10 Recommended (Baseline)</h4>
          <table class="t"><thead><tr><th>#</th><th>Movie</th></tr></thead><tbody>${tr(recBase)}</tbody></table>
        </div>
        <div>
          <h4>Top-10 Recommended (Deep)</h4>
          <table class="t"><thead><tr><th>#</th><th>Movie</th></tr></thead><tbody>${tr(recDeep)}</tbody></table>
        </div>
      </div>`;
  }

  // wire buttons once
  els.loadBtn.addEventListener('click', onLoad);
  els.trainBtn.addEventListener('click', onTrain);
  els.testBtn.addEventListener('click', onTest);

  // expose for dev
  window.__state = () => ({DATA, baseline, deepModel, lossHistoryBase, lossHistoryDeep});
})();
