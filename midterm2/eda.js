/* eda.js — fully client-side EDA for GitHub Pages
   Works if index.html + eda.js are in /midterm2/ OR /midterm2/data/
   Looks for CSVs in the same folder and in /data/.
*/
(() => {
  // ------------------- tiny DOM helpers -------------------
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => (typeof n === 'number' ? n.toLocaleString() : n);

  // ------------------- global state -------------------
  let recipes = new Map(); // id -> {name, minutes, n_ingredients, tags[], date?}
  let interactions = [];   // {u, i, r, ts, y, m, d, hr, dow}
  let users = new Set();
  let items = new Set();
  let tagFreq = new Map();
  let user2 = new Map();   // u -> {cnt,sum}
  let item2 = new Map();   // i -> {cnt,sum}
  let loaded = false;

  // ------------------- safe onload -------------------
  window.addEventListener('DOMContentLoaded', () => {
    bindTabs();
    $('loadBtn').addEventListener('click', handleLoad);
    $('redraw').addEventListener('click', drawAll);
    $('topK').addEventListener('change', drawAll);
    $('edgeMin').addEventListener('change', drawAll);
    $('nodeMax').addEventListener('change', drawAll);
    $('timeGroup').addEventListener('change', drawAll);
  });

  // ------------------- tabs -------------------
  function bindTabs(){
    document.querySelectorAll('.tab').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
        document.querySelectorAll('.tabpane').forEach(p=>p.classList.add('hidden'));
        btn.classList.add('active');
        $(btn.dataset.tab).classList.remove('hidden');
      });
    });
  }

  // ------------------- robust loaders -------------------
  async function fetchFirstExisting(paths){
    for (const p of paths){
      try{
        const r = await fetch(p, {cache:'no-store'});
        if (r.ok){
          const t = await r.text();
          return { path:p, text:t };
        }
      }catch(e){ /* try next */ }
    }
    return null;
  }

  function csvSplit(line){
    // split by commas but respect quotes
    return line.match(/("([^"]|"")*"|[^,]*)/g).filter(s=>s!==',').map(s=>s.replace(/^,?/,''));
  }
  function splitLines(text){ return text.split(/\r?\n/).filter(Boolean); }

  function detectPaths(){
    // index may live at /midterm2/ or /midterm2/data/
    const here = window.location.pathname;
    const base = here.replace(/[^/]+$/, ''); // folder of index.html
    // try same folder first, then /data/ relative variations
    const prefixes = [
      '', './', 'data/', './data/', '../data/'
    ];
    return {
      recipes: prefixes.map(p=>p+'RAW_recipes.csv')
        .concat(prefixes.map(p=>p+'PP_recipes.csv')),
      inter: prefixes.map(p=>p+'interactions_train.csv')
    };
  }

  // ------------------- parsers -------------------
  function parseRecipes(text){
    const lines = splitLines(text);
    const head = csvSplit(lines.shift().trim()).map(h=>h.toLowerCase());
    const idx = {
      id: head.findIndex(h=>/^id$|(^|_)id$/.test(h)),
      name: head.findIndex(h=>/(^|_)name$|title/.test(h)),
      minutes: head.findIndex(h=>/(minutes|cook_time)/.test(h)),
      n_ing: head.findIndex(h=>/(n_ingredients|num_ingredients)/.test(h)),
      tags: head.findIndex(h=>/tags/.test(h)),
      date: head.findIndex(h=>/(submitted|date)/.test(h))
    };
    for (const ln of lines){
      const cols = csvSplit(ln);
      const id = parseInt(cols[idx.id],10);
      if (!Number.isInteger(id)) continue;
      const obj = {
        name: (idx.name>=0? cols[idx.name] : `Recipe ${id}`).replace(/^"|"$/g,''),
        minutes: idx.minutes>=0 ? parseFloat(cols[idx.minutes])||null : null,
        n_ingredients: idx.n_ing>=0 ? parseInt(cols[idx.n_ing],10)||null : null,
        tags: [],
        date: idx.date>=0 ? Date.parse(cols[idx.date])||null : null
      };
      if (idx.tags>=0 && cols[idx.tags]){
        let raw = cols[idx.tags].trim();
        // handle "['a', 'b']" or JSON-ish
        raw = raw.replace(/^\s*\[|\]\s*$/g,'');
        const parts = raw.split(/['"]\s*,\s*['"]|,\s*/g)
          .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
          .filter(Boolean);
        obj.tags = parts.slice(0, 40);
      }
      recipes.set(id, obj);
      items.add(id);
      for (const t of obj.tags){
        tagFreq.set(t, (tagFreq.get(t)||0)+1);
      }
    }
  }

  function parseInteractions(text){
    const lines = splitLines(text);
    const head = csvSplit(lines.shift().trim()).map(h=>h.toLowerCase());
    const idx = {
      u: head.findIndex(h=>/(^|_)user/.test(h)),
      i: head.findIndex(h=>/(^|_)(item|recipe)(_?id)?$/.test(h) || /(recipe_id)/.test(h)),
      r: head.findIndex(h=>/(^|_)rating$|stars/.test(h)),
      t: head.findIndex(h=>/(^|_)(time|date|timestamp)/.test(h))
    };
    for (const ln of lines){
      const cols = csvSplit(ln);
      const u = parseInt(cols[idx.u],10);
      const i = parseInt(cols[idx.i],10);
      if (!Number.isInteger(u) || !Number.isInteger(i)) continue;
      const r = idx.r>=0 && cols[idx.r]!=='' ? parseFloat(cols[idx.r]) : null;
      const ts = idx.t>=0 ? (Date.parse(cols[idx.t]) || parseInt(cols[idx.t],10) || null) : null;

      const d = ts ? new Date(ts) : null;
      const row = {
        u, i, r,
        ts: ts,
        y: d? d.getUTCFullYear(): null,
        m: d? (d.getUTCMonth()+1): null,
        d: d? d.getUTCDate(): null,
        hr: d? d.getUTCHours(): null,
        dow: d? d.getUTCDay(): null
      };
      interactions.push(row);
      users.add(u);

      // aggregates for user/item
      const U = user2.get(u) || {cnt:0,sum:0};
      U.cnt++; U.sum += (r??0); user2.set(u,U);
      const I = item2.get(i) || {cnt:0,sum:0};
      I.cnt++; I.sum += (r??0); item2.set(i,I);
    }
  }

  // ------------------- draw helpers (canvas) -------------------
  function clearCanvas(ctx){
    const c = ctx.canvas;
    const dpr = window.devicePixelRatio||1;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = Math.max(1, w*dpr);
    c.height = Math.max(1, h*dpr);
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr,dpr);
    ctx.clearRect(0,0,w,h);
    return {W:w,H:h};
  }
  function drawBars(id, arr, opts={}){
    const ctx = $(id).getContext('2d');
    const {W,H} = clearCanvas(ctx);
    const pad = 28, base = H-pad;
    const maxV = Math.max(1, ...arr);
    const bw = (W-pad*2)/arr.length - 6;
    ctx.strokeStyle='#213047';
    ctx.beginPath(); ctx.moveTo(pad,base+0.5); ctx.lineTo(W-pad,base+0.5); ctx.stroke();
    ctx.fillStyle='#cbd5e1';
    arr.forEach((v,k)=>{
      const x = pad + k*(bw+6);
      const h = (v/maxV)*(H-pad*2);
      ctx.fillRect(x, base-h, Math.max(1,bw), h);
    });
  }
  function drawLine(id, pts){
    const ctx = $(id).getContext('2d'); const {W,H} = clearCanvas(ctx);
    if (!pts.length) return;
    const pad = 28, base = H-pad;
    const maxX = Math.max(...pts.map(p=>p.x)) || 1;
    const maxY = Math.max(...pts.map(p=>p.y)) || 1;
    const sx = (x)=> pad + (x/maxX)*(W-pad*2);
    const sy = (y)=> base - (y/maxY)*(H-pad*2);
    ctx.strokeStyle='#7dd3fc';
    ctx.beginPath();
    ctx.moveTo(sx(pts[0].x), sy(pts[0].y));
    for (let i=1;i<pts.length;i++) ctx.lineTo(sx(pts[i].x), sy(pts[i].y));
    ctx.stroke();
  }
  function drawScatter(id, pts){
    const ctx = $(id).getContext('2d'); const {W,H} = clearCanvas(ctx);
    if (!pts.length) return;
    const pad=28, base=H-pad;
    const maxX = Math.max(...pts.map(p=>p.x))||1;
    const maxY = Math.max(...pts.map(p=>p.y))||1;
    const sx = (x)=> pad + (x/maxX)*(W-pad*2);
    const sy = (y)=> base - (y/maxY)*(H-pad*2);
    ctx.fillStyle='#cbd5e1';
    for (const p of pts){
      const x=sx(p.x), y=sy(p.y);
      ctx.fillRect(x-1,y-1,2,2);
    }
  }
  function drawHeatmap(id, grid, maxV){
    const ctx = $(id).getContext('2d'); const {W,H} = clearCanvas(ctx);
    const rows = grid.length, cols = grid[0]?.length || 0;
    if (!rows || !cols) return;
    const pad = 30, gw = (W-pad*2)/cols, gh = (H-pad*2)/rows;
    for (let r=0;r<rows;r++){
      for (let c=0;c<cols;c++){
        const v = grid[r][c];
        const t = maxV? v/maxV : 0;
        // simple viridis-ish ramp
        const col = `hsl(${240-240*t}, 70%, ${20+50*t}%)`;
        ctx.fillStyle = col;
        ctx.fillRect(pad + c*gw, pad + r*gh, gw-1, gh-1);
      }
    }
  }

  // ------------------- draw all panels -------------------
  function drawAll(){
    if (!loaded) return;

    // badges
    const dens = (interactions.length / Math.max(1, users.size*items.size)).toExponential(3);
    $('badges').innerHTML =
      badge(`Recipes: ${fmt(recipes.size)}`) +
      badge(`Users: ${fmt(users.size)}`) +
      badge(`Items: ${fmt(items.size)}`) +
      badge(`Interactions: ${fmt(interactions.length)}`) +
      badge(`Density: ${dens}`) +
      badge(`Ratings: ${hasRatings()? 'yes':'no'}`) +
      badge(`Time: ${hasDates()? 'yes':'no'}`) +
      badge(`Minutes: ${hasMinutes()? 'yes':'no'}`);

    // counters
    fillKPI('counters', [
      ['Users', users.size],
      ['Items', items.size],
      ['Interactions', interactions.length],
      ['Density', dens],
      ['Cold users <5', coldUserCount()],
      ['Cold items <5', coldItemCount()],
    ]);

    // KPIs
    fillKPI('kpis', [
      ['Ratings present', hasRatings() ? 'yes':'no'],
      ['Avg rating', avgRating().toFixed(2)],
      ['Top 1% items', Math.max(1,Math.floor(0.01*items.size))],
      ['% interactions by top 1%', paretoCoverage(0.01).toFixed(1)+'%'],
      ['Median minutes', medianMinutes() ?? '—'],
      ['Median n_ingredients', medianIngredients() ?? '—']
    ]);

    // histograms
    drawBars('histRatings', ratingsHist(), {});
    drawBars('histRatings2', ratingsHist(), {});

    // user activity
    drawBars('histUser', bucketCounts(Array.from(user2.values()).map(v=>v.cnt),
      [1,2,3,5,10,20,50,100]));
    // scatter users
    const uscat = [];
    user2.forEach(v=>{ uscat.push({x:v.cnt, y:v.cnt? v.sum/v.cnt : 0}); });
    drawScatter('userScatter', uscat);

    // item popularity
    drawBars('histItem', bucketCounts(Array.from(item2.values()).map(v=>v.cnt),
      [1,2,3,5,10,20,100,500,1000]));
    const iscat = [];
    item2.forEach(v=>{ iscat.push({x:v.cnt, y:v.cnt? v.sum/v.cnt : 0}); });
    drawScatter('itemScatter', iscat);

    // recipe features
    drawBars('histMinutes', numericHist(Array.from(recipes.values()).map(r=>r.minutes).filter(v=>v!=null), 20, 0, 600));
    drawBars('histIngred', numericHist(Array.from(recipes.values()).map(r=>r.n_ingredients).filter(v=>v!=null), 16, 0, 40));

    // time charts
    drawLine('lineTime', timeSeries($('timeGroup').value));
    drawLine('lineTime2', timeSeries('month'));

    // heatmap dow × hour (collapse if hour missing)
    const hm = heatmapDowHour();
    if (hm.rows && hm.cols) {
      drawHeatmap('dowHour', hm.grid, hm.max);
    } else {
      const ctx = $('dowHour').getContext('2d'); clearCanvas(ctx);
    }

    // tags
    drawBars('topTags', topTags(+$('topK').value).slice(0,20).map(x=>x[1]));
    drawBars('topTags30', topTags(30).map(x=>x[1]));

    // simple tag co-occurrence graph (force-like layout light)
    drawTagGraph();

    // lorenz curve + pareto
    drawLorenzAndPareto();
  }

  function badge(text){ return `<span class="pill">${text}</span>`; }
  function fillKPI(id, pairs){
    $(id).innerHTML = pairs.map(([t,v])=>(
      `<div class="cell"><div class="t">${t}</div><div class="v">${v}</div></div>`
    )).join('');
  }

  // ------------------- metrics helpers -------------------
  function hasRatings(){ return interactions.some(x=>x.r!=null); }
  function hasDates(){ return interactions.some(x=>x.ts!=null); }
  function hasMinutes(){ return Array.from(recipes.values()).some(r=>r.minutes!=null); }
  function avgRating(){
    let s=0,c=0; for (const x of interactions){ if (x.r!=null){s+=x.r; c++;} }
    return c? s/c : 0;
  }
  function bucketCounts(vals, edges){
    // edges are right-closed buckets values
    const B = new Array(edges.length+1).fill(0);
    for (const v of vals){
      let idx = edges.findIndex(e=>v<=e);
      if (idx<0) idx = edges.length;
      B[idx]++;
    }
    return B;
  }
  function numericHist(vals, bins, minX, maxX){
    if (!vals.length) return new Array(bins).fill(0);
    const lo = minX!=null? minX : Math.min(...vals);
    const hi = maxX!=null? maxX : Math.max(...vals);
    const B = new Array(bins).fill(0), w = (hi-lo)/bins;
    for (const v of vals){
      if (v==null) continue;
      const ix = Math.max(0, Math.min(bins-1, Math.floor((v-lo)/w)));
      B[ix]++;
    }
    return B;
  }
  function ratingsHist(){
    const h=[0,0,0,0,0];
    for (const x of interactions){
      const r = x.r!=null ? Math.round(Math.max(1,Math.min(5,x.r))) : null;
      if (r) h[r-1]++;
    }
    return h;
  }
  function timeSeries(group){
    if (!hasDates()) return [];
    const key = (x)=>{
      const d = new Date(x.ts);
      if (group==='year') return d.getUTCFullYear();
      if (group==='week'){
        const dt = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
        const day = (dt.getUTCDay()+6)%7;
        dt.setUTCDate(dt.getUTCDate()-day);
        return +dt; // week start
      }
      // month
      return d.getUTCFullYear()*12 + d.getUTCMonth()+1;
    };
    const cnt = new Map();
    for (const x of interactions){ if (x.ts) cnt.set(key(x),(cnt.get(key(x))||0)+1); }
    const arr = Array.from(cnt.entries()).sort((a,b)=>a[0]-b[0]).map(([k,v],i)=>({x:i+1,y:v}));
    return arr;
  }
  function heatmapDowHour(){
    const grid = Array.from({length:7},()=>Array.from({length:24},()=>0));
    let any=false, max=0;
    for (const x of interactions){
      if (x.dow!=null && x.hr!=null){
        grid[x.dow][x.hr]++; any=true; if (grid[x.dow][x.hr]>max) max=grid[x.dow][x.hr];
      }
    }
    return any? {rows:7,cols:24,grid,max} : {rows:0,cols:0,grid:[],max:0};
    }

  function medianMinutes(){
    const arr = Array.from(recipes.values()).map(r=>r.minutes).filter(v=>v!=null).sort((a,b)=>a-b);
    if (!arr.length) return null;
    const m = arr.length>>1;
    return arr.length%2? arr[m] : Math.round((arr[m-1]+arr[m])/2);
  }
  function medianIngredients(){
    const arr = Array.from(recipes.values()).map(r=>r.n_ingredients).filter(v=>v!=null).sort((a,b)=>a-b);
    if (!arr.length) return null;
    const m = arr.length>>1;
    return arr.length%2? arr[m] : Math.round((arr[m-1]+arr[m])/2);
  }
  function coldUserCount(){
    let c=0; user2.forEach(v=>{ if (v.cnt<5) c++; }); return c;
  }
  function coldItemCount(){
    let c=0; item2.forEach(v=>{ if (v.cnt<5) c++; }); return c;
  }
  function paretoCoverage(frac){
    const arr = Array.from(item2.values()).map(v=>v.cnt).sort((a,b)=>b-a);
    const topN = Math.max(1, Math.floor(frac*arr.length));
    const topSum = arr.slice(0,topN).reduce((s,v)=>s+v,0);
    return 100*topSum/Math.max(1,interactions.length);
  }

  function drawLorenzAndPareto(){
    // Lorenz
    const counts = Array.from(item2.values()).map(v=>v.cnt).sort((a,b)=>a-b);
    const total = counts.reduce((s,v)=>s+v,0) || 1;
    let acc=0; const lor = counts.map(v=>{ acc+=v; return acc/total; });
    const pts = lor.map((y,i)=>({x:i, y}));
    drawLine('lorenz', pts);
    const gini = 1 - 2 * lor.reduce((s,y,i)=> s + y/(counts.length||1), 0) / (counts.length||1);
    $('giniLine').textContent = `Gini ≈ ${gini.toFixed(3)}`;

    // Pareto bar (top 1% vs others)
    const topPct = paretoCoverage(0.01);
    const ctx = $('pareto').getContext('2d'); const {W,H} = clearCanvas(ctx);
    const pad=28, base=H-pad, data=[topPct, 100-topPct];
    const bw=(W-pad*2)/2 - 10; const maxV=100;
    ctx.fillStyle='#cbd5e1';
    for (let k=0;k<2;k++){
      const h = (data[k]/maxV)*(H-pad*2);
      const x = pad + k*(bw+20);
      ctx.fillRect(x, base-h, bw, h);
    }
    $('paretoLine').textContent = `Top 1% items (~${Math.max(1,Math.floor(items.size*0.01))}) cover ${topPct.toFixed(1)}% of all interactions.`;
  }

  function topTags(K){
    const freq = Array.from(tagFreq.entries()).sort((a,b)=>b[1]-a[1]).slice(0,K);
    return freq;
  }

  function drawTagGraph(){
    // very lightweight force-ish plot based on tag co-occurrence
    const edgeMin = +$('edgeMin').value;
    const nodeMax = +$('nodeMax').value;
    const top = new Set(topTags(nodeMax).map(x=>x[0]));
    // build co-occur counts
    const co = new Map(); // "a|b" -> w
    recipes.forEach(r=>{
      const tags = (r.tags||[]).filter(t=>top.has(t));
      for (let i=0;i<tags.length;i++){
        for (let j=i+1;j<tags.length;j++){
          const a = tags[i], b = tags[j];
          const key = a<b? `${a}|${b}`:`${b}|${a}`;
          co.set(key, (co.get(key)||0)+1);
        }
      }
    });
    const edges = Array.from(co.entries()).filter(([,w])=>w>=edgeMin).map(([k,w])=>{
      const [a,b] = k.split('|'); return {a,b,w};
    });
    const nodes = Array.from(top.values()).map(t=>({id:t, x:Math.random(), y:Math.random(), vx:0, vy:0}));

    const id2 = new Map(nodes.map((n,i)=>[n.id,i]));
    // basic relax iterations
    for (let it=0;it<120;it++){
      // spring on edges
      for (const e of edges){
        const na = nodes[id2.get(e.a)], nb = nodes[id2.get(e.b)];
        const dx = nb.x - na.x, dy = nb.y - na.y;
        const dist = Math.hypot(dx,dy)+1e-6;
        const k = 0.02 * Math.log(1+e.w);
        const fx = k*dx/dist, fy=k*dy/dist;
        na.vx += fx; na.vy += fy; nb.vx -= fx; nb.vy -= fy;
      }
      // repulsion
      for (let i=0;i<nodes.length;i++){
        for (let j=i+1;j<nodes.length;j++){
          const a=nodes[i], b=nodes[j];
          const dx=b.x-a.x, dy=b.y-a.y, d2=dx*dx+dy*dy+1e-6;
          const k=0.0005/d2;
          a.vx -= k*dx; a.vy -= k*dy; b.vx += k*dx; b.vy += k*dy;
        }
      }
      // integrate + clamp
      for (const n of nodes){
        n.x = Math.min(1, Math.max(0, n.x + n.vx));
        n.y = Math.min(1, Math.max(0, n.y + n.vy));
        n.vx*=0.85; n.vy*=0.85;
      }
    }
    // draw
    const ctx = $('tagGraph').getContext('2d'); const {W,H} = clearCanvas(ctx); const pad=10;
    // edges
    ctx.strokeStyle='#374151';
    for (const e of edges){
      const a=nodes[id2.get(e.a)], b=nodes[id2.get(e.b)];
      ctx.globalAlpha = Math.min(1, 0.2 + e.w/100);
      ctx.beginPath();
      ctx.moveTo(pad + a.x*(W-pad*2), pad + a.y*(H-pad*2));
      ctx.lineTo(pad + b.x*(W-pad*2), pad + b.y*(H-pad*2));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // nodes
    ctx.fillStyle='#e5e7eb';
    ctx.font='12px Inter, sans-serif';
    ctx.textAlign='center';
    nodes.forEach(n=>{
      const x = pad + n.x*(W-pad*2), y = pad + n.y*(H-pad*2);
      ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
      ctx.fillText(n.id.slice(0,18), x, y-8);
    });
  }

  // ------------------- load handler -------------------
  async function handleLoad(){
    try{
      $('status').textContent = 'Status: loading…';
      $('badNews').classList.add('hidden');
      recipes.clear(); interactions.length=0;
      users.clear(); items.clear(); tagFreq.clear(); user2.clear(); item2.clear();

      const paths = detectPaths();
      // prefer RAW_recipes, fallback to PP_recipes
      const rec = await fetchFirstExisting(paths.recipes);
      const itx = await fetchFirstExisting(paths.inter);

      if (!rec || !itx){
        throw new Error('CSV files not found. Place files next to index.html or in a /data/ folder.');
      }

      parseRecipes(rec.text);
      parseInteractions(itx.text);

      loaded = true;
      $('status').textContent = 'Status: loaded.';
      // badges line
      drawAll();
    }catch(err){
      console.error(err);
      $('status').textContent = 'Status: error.';
      $('badNews').classList.remove('hidden');
      $('badNews').innerHTML = `
        <b>Couldn’t load CSVs.</b><br>
        Place files as <code>RAW_recipes.csv</code> or <code>PP_recipes.csv</code> and <code>interactions_train.csv</code>
        either in the <b>same folder</b> as this page or in a child folder named <b>/data/</b>.<br><br>
        Tried multiple locations relative to <code>${window.location.pathname}</code>.
      `;
    }
  }
})();
