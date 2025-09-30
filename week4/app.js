// app.js — wires UI, loads data, trains baseline+deep, tests, plots.
// Defensive wiring + clear status logs so buttons always respond.

(() => {
  // ----- small utils -----
  const $ = id => document.getElementById(id);
  const setStatus = t => { const el=$('status'); if (el) el.textContent = `Status: ${t}`; console.log('[status]', t); };

  // ----- global runtime state -----
  const S = {
    interactions: [],
    items: new Map(),              // itemId -> {title, year, genres:Int8Array(18)}
    users: new Set(), itemIds: new Set(),
    userToRated: new Map(),        // userId -> Set(itemId)
    userTopRated: new Map(),       // userId -> [{itemId,rating,ts,title,year}]
    userId2idx: new Map(), itemId2idx: new Map(), idx2userId: [], idx2itemId: [],
    baseline: null, deep: null, trained: {baseline:false, deep:false},
    cfg: { max:80000, dim:32, hid:64, batch:256, epochs:5, lr:0.003, loss:'softmax', both:'yes' },
    loss: { baseline:[], deep:[] },
    projPoints: []
  };

  // ----- tiny chart for losses -----
  class MultiChart {
    constructor(canvas){ this.c=canvas; this.ctx=canvas.getContext('2d'); }
    draw(series){ const W=this.c.width,H=this.c.height,ctx=this.ctx;
      ctx.clearRect(0,0,W,H); ctx.fillStyle='#0c1424'; ctx.fillRect(0,0,W,H);
      ctx.strokeStyle='#21324a'; ctx.beginPath(); ctx.moveTo(40,10); ctx.lineTo(40,H-20); ctx.lineTo(W-10,H-20); ctx.stroke();
      const vals=Object.values(series).flat(); if(!vals.length)return;
      const ymin=Math.min(...vals), ymax=Math.max(...vals); const maxLen=Math.max(...Object.values(series).map(a=>a.length),1);
      const sx=x=>50+(x/(maxLen-1))* (W-64), sy=y=>(H-24)-((y-ymin)/Math.max(1e-9,ymax-ymin))*(H-34);
      const col={baseline:'#3fb950', deep:'#e50914'};
      for(const [k,arr] of Object.entries(series)){ if(!arr.length)continue; ctx.strokeStyle=col[k]||'#6ea8fe'; ctx.lineWidth=2; ctx.beginPath();
        arr.forEach((y,i)=>{const X=sx(i),Y=sy(y); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y)}); ctx.stroke(); }
      ctx.fillStyle='#9fb0c5'; ctx.font='12px ui-monospace,monospace'; ctx.fillText(ymax.toFixed(3),6,sy(ymax)); ctx.fillText(ymin.toFixed(3),6,sy(ymin));
    }
  }
  const lossChart = new MultiChart($('lossCanvas'));

  // ----- read config from UI -----
  function readCfg(){
    S.cfg.max   = Math.max(1000, +$('cfg-max').value || 80000);
    S.cfg.dim   = Math.max(8,    +$('cfg-dim').value || 32);
    S.cfg.hid   = Math.max(16,   +$('cfg-hid').value || 64);
    S.cfg.batch = Math.max(32,   +$('cfg-batch').value || 256);
    S.cfg.epochs= Math.max(1,    +$('cfg-epochs').value || 5);
    S.cfg.lr    = Math.max(1e-5, +$('cfg-lr').value || 0.003);
    S.cfg.loss  = $('cfg-loss').value === 'bpr' ? 'bpr' : 'softmax';
    S.cfg.both  = $('cfg-both').value === 'no' ? 'no' : 'yes';
  }

  // ----- data loading -----
  async function loadData(){
    setStatus('loading data… (check Console if it stalls)');
    if (typeof tf === 'undefined') { setStatus('TensorFlow.js failed to load'); return; }

    let itemTxt, dataTxt;
    try{
      const [r1,r2] = await Promise.all([fetch('data/u.item'), fetch('data/u.data')]);
      if(!r1.ok) throw new Error('missing data/u.item');
      if(!r2.ok) throw new Error('missing data/u.data');
      itemTxt = await r1.text(); dataTxt = await r2.text();
    }catch(e){
      console.error(e);
      setStatus('fetch failed. Ensure /data/u.item and /data/u.data exist (case-sensitive).');
      return;
    }

    // Parse u.item
    S.items.clear(); S.itemIds.clear();
    const linesI = itemTxt.split('\n').filter(Boolean);
    for (const line of linesI){
      const parts = line.split('|'); if (parts.length<2) continue;
      const id = +parts[0]; let titleRaw = parts[1] || String(id); let title=titleRaw, year=null;
      const m = titleRaw.match(/\((\d{4})\)\s*$/); if(m){ year=+m[1]; title = titleRaw.replace(/\s*\(\d{4}\)\s*$/,''); }
      const flags = parts.slice(5).map(x=>x==='1'?1:0);
      const use = (flags.length>=19)? flags.slice(1) : flags; // drop "Unknown"
      const g = new Int8Array(18); for(let k=0;k<Math.min(18,use.length);k++) g[k]=use[k];
      S.items.set(id,{title,year,genres:g}); S.itemIds.add(id);
    }

    // Parse u.data
    S.interactions.length=0; S.users.clear();
    const linesD = dataTxt.split('\n').filter(Boolean);
    for (const line of linesD){
      const [u,i,r,t] = line.split('\t'); const userId=+u,itemId=+i,rating=+r,ts=+t;
      if(!Number.isFinite(userId)||!Number.isFinite(itemId)) continue;
      S.interactions.push({userId,itemId,rating,ts}); S.users.add(userId);
    }

    // user->rated, top rated
    S.userToRated.clear(); S.userTopRated.clear();
    for (const {userId,itemId} of S.interactions){
      if(!S.userToRated.has(userId)) S.userToRated.set(userId,new Set());
      S.userToRated.get(userId).add(itemId);
    }
    const byUser = new Map();
    for (const row of S.interactions){ if(!byUser.has(row.userId)) byUser.set(row.userId,[]); byUser.get(row.userId).push(row); }
    for (const [uid, arr] of byUser){
      arr.sort((a,b)=> (b.rating-a.rating)||(b.ts-a.ts));
      const top = arr.slice(0,60).map(x=>({ itemId:x.itemId, rating:x.rating, ts:x.ts,
        title:S.items.get(x.itemId)?.title ?? String(x.itemId), year:S.items.get(x.itemId)?.year ?? '' }));
      S.userTopRated.set(uid, top);
    }

    // indexers
    const userIds=[...S.users].sort((a,b)=>a-b);
    const itemIds=[...S.itemIds].sort((a,b)=>a-b);
    S.idx2userId=userIds; S.idx2itemId=itemIds;
    S.userId2idx=new Map(userIds.map((u,idx)=>[u,idx]));
    S.itemId2idx=new Map(itemIds.map((i,idx)=>[i,idx]));

    $('btn-train')?.classList.remove('secondary');
    $('btn-test')?.classList.remove('secondary');
    setStatus(`data loaded — users: ${userIds.length}, items: ${itemIds.length}, interactions: ${S.interactions.length}`);
  }

  // helpers
  function buildPairs(maxN){
    const idxs = S.interactions.map((_,i)=>i);
    for(let i=idxs.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [idxs[i],idxs[j]]=[idxs[j],idxs[i]]; }
    const N=Math.min(maxN, idxs.length), out=[];
    for(let k=0;k<N;k++){ const r=S.interactions[idxs[k]];
      const u=S.userId2idx.get(r.userId), it=S.itemId2idx.get(r.itemId);
      if(u!=null && it!=null) out.push([u,it]);
    }
    return out;
  }
  function buildItemGenreTensor(){
    const N=S.idx2itemId.length, G=18, buf=new Float32Array(N*G);
    for(let i=0;i<N;i++){ const id=S.idx2itemId[i]; const g=S.items.get(id)?.genres || new Int8Array(G);
      for(let k=0;k<G;k++) buf[i*G+k]=g[k]; }
    return tf.tensor2d(buf,[N,G],'float32');
  }

  // ----- training -----
  async function train(){
    if (!S.interactions.length){ setStatus('load data first'); return; }
    readCfg(); S.loss={baseline:[],deep:[]}; lossChart.draw(S.loss);
    const pairs=buildPairs(S.cfg.max);
    const U=S.idx2userId.length, I=S.idx2itemId.length;

    // baseline
    if (S.cfg.both==='yes'){
      if(!S.baseline) S.baseline = new TwoTowerModel(U,I,S.cfg.dim, S.cfg.loss);
      S.baseline.setOptimizer(tf.train.adam(S.cfg.lr));
      await trainOne(S.baseline, pairs, 'baseline');
      S.trained.baseline=true;
    }

    // deep
    const itemGenre = buildItemGenreTensor();
    if(!S.deep) S.deep = new TwoTowerDeepModel(U,I,S.cfg.dim,18,S.cfg.hid,S.cfg.loss,itemGenre);
    S.deep.setOptimizer(tf.train.adam(S.cfg.lr));
    await trainOne(S.deep, pairs, 'deep');
    S.trained.deep=true;

    await drawProjectionSample(S.deep);
    setStatus('training done — ready to test');
  }

  async function trainOne(model, pairs, key){
    const B=S.cfg.batch, epochs=S.cfg.epochs, steps=Math.ceil(pairs.length/B), series=S.loss[key];
    for (let ep=0; ep<epochs; ep++){
      for (let s=0;s<steps;s++){
        const start=s*B, end=Math.min((s+1)*B,pairs.length), batch=pairs.slice(start,end);
        const u=new Int32Array(batch.length), it=new Int32Array(batch.length);
        for(let t=0;t<batch.length;t++){ u[t]=batch[t][0]; it[t]=batch[t][1]; }
        const L = await model.trainStep(u,it);
        series.push(L); if((s&1)===0) lossChart.draw(S.loss);
        setStatus(`${key}: epoch ${ep+1}/${epochs} — step ${s+1}/${steps} — loss ${L.toFixed(4)}`);
        await tf.nextFrame();
      }
    }
    lossChart.draw(S.loss);
  }

  // ----- projection -----
  async function drawProjectionSample(model){
    const cvs=$('projCanvas'), ctx=cvs.getContext('2d'), W=cvs.width,H=cvs.height;
    ctx.clearRect(0,0,W,H); ctx.fillStyle='#0c1424'; ctx.fillRect(0,0,W,H);

    const N=S.idx2itemId.length, sampleN=Math.min(1000,N);
    const idxs=[...Array(N).keys()]; for(let i=idxs.length-1;i>0;i--){const j=(Math.random()*(i+1))|0; [idxs[i],idxs[j]]=[idxs[j],idxs[i]];}
    const sample=idxs.slice(0,sampleN);

    const chunk=2048, outs=[];
    for(let s=0;s<sample.length;s+=chunk){
      const block=sample.slice(s,Math.min(sample.length,s+chunk));
      const iT=tf.tensor1d(block,'int32');
      const eT=model.itemForward(iT); outs.push(eT); iT.dispose();
    }
    const E=tf.concat(outs,0); outs.forEach(t=>t.dispose()); // [S,D]

    let XY;
    try{
      XY=tf.tidy(()=>{ const X=E.sub(E.mean(0)), svd=(tf.svd?tf.svd:tf.linalg.svd)(X,true), V2=svd.v.slice([0,0],[-1,2]); return X.matMul(V2); });
    }catch{ XY=E.slice([0,0],[-1,2]); }
    const xy=await XY.array(); E.dispose(); XY.dispose();

    const xs=xy.map(p=>p[0]), ys=xy.map(p=>p[1]);
    const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys), pad=16;
    S.projPoints=[];
    ctx.fillStyle='#e6eef788';
    for(let k=0;k<xy.length;k++){
      const id=S.idx2itemId[sample[k]];
      const x=pad+((xs[k]-xmin)/Math.max(1e-6,xmax-xmin))*(W-2*pad);
      const y=H-pad-((ys[k]-ymin)/Math.max(1e-6,ymax-ymin))*(H-2*pad);
      ctx.beginPath(); ctx.arc(x,y,2,0,6.28); ctx.fill();
      S.projPoints.push({x,y,title:S.items.get(id)?.title ?? String(id)});
    }
    cvs.onmousemove = ev => {
      const r=cvs.getBoundingClientRect();
      const mx=(ev.clientX-r.left)*(cvs.width/r.width), my=(ev.clientY-r.top)*(cvs.height/r.height);
      ctx.clearRect(0,0,W,H); ctx.fillStyle='#0c1424'; ctx.fillRect(0,0,W,H); ctx.fillStyle='#e6eef788';
      for(const p of S.projPoints){ ctx.beginPath(); ctx.arc(p.x,p.y,2,0,6.28); ctx.fill(); }
      let best=null,bestD=9; for(const p of S.projPoints){ const d=(p.x-mx)**2+(p.y-my)**2; if(d<bestD){bestD=d; best=p;} }
      if(best){ ctx.fillStyle='#fff'; ctx.font='12px system-ui'; const t=best.title.length>42?best.title.slice(0,39)+'…':best.title;
        ctx.fillText(t, Math.min(W-6-ctx.measureText(t).width, Math.max(6,best.x+8)), Math.max(14,best.y-8)); }
    };
  }

  // ----- test -----
  async function testOnce(){
    if(!S.trained.deep && !S.trained.baseline){ setStatus('train first'); return; }
    const candidates=[]; for(const [u,arr] of S.userTopRated) if(arr.length>=20) candidates.push(u);
    if(!candidates.length){ setStatus('no users with ≥20 ratings'); return; }
    const userId=candidates[(Math.random()*candidates.length)|0];
    const left=S.userTopRated.get(userId).slice(0,10);
    const mid = S.trained.baseline? await recommendWithBaseline(userId,10) : [];
    const right = S.trained.deep? await recommendWithDeep(userId,10) : [];
    renderThree(userId,left,mid,right);
    setStatus(`tested user ${userId}`);
  }

  async function recommendWithBaseline(userId,K){
    const rated=S.userToRated.get(userId)||new Set(); const uIdx=S.userId2idx.get(userId);
    const u=S.baseline.getUserEmbeddingForIndex(uIdx); const M=S.baseline.getItemEmbedding();
    const scores=await batchedDot(u,M); u.dispose();
    const arr=[]; for(let i=0;i<scores.length;i++){ const iid=S.idx2itemId[i]; if(rated.has(iid)) continue; arr.push({i,score:scores[i]}); }
    arr.sort((a,b)=>b.score-a.score);
    return arr.slice(0,K).map(o=>({ itemId:S.idx2itemId[o.i], score:o.score, title:S.items.get(S.idx2itemId[o.i])?.title ?? String(S.idx2itemId[o.i]) }));
  }
  async function recommendWithDeep(userId,K){
    const rated=S.userToRated.get(userId)||new Set(); const uIdx=S.userId2idx.get(userId);
    const u=S.deep.getUserEmbeddingForIndex(uIdx); const N=S.idx2itemId.length, chunk=2048, scores=new Float32Array(N);
    for(let s=0;s<N;s+=chunk){
      const end=Math.min(N,s+chunk), idx=tf.tensor1d([...Array(end-s).keys()].map(x=>x+s),'int32'), e=S.deep.itemForward(idx);
      const block=tf.tidy(()=> e.matMul(u.transpose()).squeeze()); const vals=await block.data(); scores.set(vals,s);
      block.dispose(); e.dispose(); idx.dispose(); await tf.nextFrame();
    }
    u.dispose();
    const arr=[]; for(let i=0;i<scores.length;i++){ const iid=S.idx2itemId[i]; if(rated.has(iid)) continue; arr.push({i,score:scores[i]}); }
    arr.sort((a,b)=>b.score-a.score);
    return arr.slice(0,K).map(o=>({ itemId:S.idx2itemId[o.i], score:o.score, title:S.items.get(S.idx2itemId[o.i])?.title ?? String(S.idx2itemId[o.i]) }));
  }
  async function batchedDot(uEmb, itemEmb){
    const N=itemEmb.shape[0], D=itemEmb.shape[1], chunk=2048, out=new Float32Array(N);
    for(let s=0;s<N;s+=chunk){
      const e=Math.min(N,s+chunk), slice=itemEmb.slice([s,0],[e-s,D]), block=tf.tidy(()=> slice.matMul(uEmb.transpose()).squeeze());
      const vals=await block.data(); out.set(vals,s); block.dispose(); slice.dispose(); await tf.nextFrame();
    } return out;
  }

  // ----- render three tables -----
  function renderThree(userId, left, mid, right){
    const root=$('results'); root.innerHTML='';
    const mk=(title,rows,cols)=>{ const box=document.createElement('div');
      const h=document.createElement('h3'); h.textContent=title; h.style.margin='4px 0 6px'; h.style.fontSize='16px'; box.appendChild(h);
      const table=document.createElement('table'), thead=document.createElement('thead'), trh=document.createElement('tr');
      cols.forEach(c=>{const th=document.createElement('th'); th.textContent=c; trh.appendChild(th)}); thead.appendChild(trh); table.appendChild(thead);
      const tb=document.createElement('tbody'); rows.forEach((r,i)=>{ const tr=document.createElement('tr');
        const c1=document.createElement('td'); c1.textContent=String(i+1); tr.appendChild(c1);
        const c2=document.createElement('td'); c2.textContent=r.title; tr.appendChild(c2);
        const c3=document.createElement('td'); c3.textContent=(r.rating!=null)?r.rating:(r.score!=null? r.score.toFixed(3) : ''); tr.appendChild(c3);
        if(r.year!=null){ const c4=document.createElement('td'); c4.textContent=r.year||''; tr.appendChild(c4); } tb.appendChild(tr);
      }); table.appendChild(tb); box.appendChild(table); return box; };
    root.appendChild(mk('Top-10 Rated (history)', left, ['#','Title','Rating','Year']));
    if (mid.length) root.appendChild(mk('Top-10 Recommended — Baseline', mid, ['#','Title','Score']));
    if (right.length) root.appendChild(mk('Top-10 Recommended — Deep', right, ['#','Title','Score']));
  }

  // ----- wire buttons safely -----
  window.addEventListener('DOMContentLoaded', () => {
    console.log('%cApp booted','color:#3fb950'); setStatus('idle (click “Load Data”)');
    $('btn-load')?.addEventListener('click', () => { try{ loadData(); }catch(e){ console.error(e); setStatus('load error (see Console)'); }});
    $('btn-train')?.addEventListener('click', () => { try{ train(); }catch(e){ console.error(e); setStatus('train error (see Console)'); }});
    $('btn-test')?.addEventListener('click', () => { try{ testOnce(); }catch(e){ console.error(e); setStatus('test error (see Console)'); }});
  });
})();
