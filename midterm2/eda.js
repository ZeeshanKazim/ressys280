/* EDA — GitHub Pages friendly (no build, no libs)
   Works with:
     /data/RAW_recipes.csv or /data/PP_recipes.csv (any one)
     /data/interactions_train.csv
   Also tries repo root (RAW_recipes.csv, PP_recipes.csv, interactions_train.csv)
*/

/* ---------- tiny DOM helpers ---------- */
const $ = (id) => document.getElementById(id);
const fmt = (n) => (typeof n === 'number' ? n.toLocaleString() : n);
function onReady(fn){ document.readyState !== 'loading' ? fn() : document.addEventListener('DOMContentLoaded', fn); }

/* ---------- global state ---------- */
let Recipes = new Map();   // id -> {title, minutes?, n_ingr?, tags[]}
let Train = [];            // [{u,i,r,ts}]
let Users = new Set();
let Items = new Set();
let user2items = new Map(); // u -> [{i,r,ts}]
let item2users = new Map(); // i -> Set(u)

let tag2cnt = new Map();
let tag2idx = new Map(), idx2tag = [];
let topK = 200;

let filesUsed = { recipes: '—', interactions: '—' };

/* ---------- loader utilities ---------- */
async function fetchFirst(paths){
  for (const p of paths){
    try{
      const res = await fetch(p, {cache:'no-store'});
      if (res.ok){
        const text = await res.text();
        return {path: p, text};
      }
    }catch(_){/* try next */}
  }
  return null;
}

function splitCSVLine(line){
  // split on commas not inside quotes
  return line.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
}
function splitLines(t){ return t.split(/\r?\n/).filter(Boolean); }

function parseRecipes(csv){
  const lines = splitLines(csv);
  if (!lines.length) return;
  const header = splitCSVLine(lines.shift()).map(h=>h.trim());
  const idIdx = header.findIndex(h=>/^id$|(^|_)id$/i.test(h));
  const nameIdx = header.findIndex(h=>/(^|_)(name|title)$/i.test(h));
  const minsIdx = header.findIndex(h=>/minute/i.test(h));
  const nIngIdx = header.findIndex(h=>/n_?ingredients/i.test(h));
  const tagsIdx = header.findIndex(h=>/tags/i.test(h));
  for (const ln of lines){
    const cols = splitCSVLine(ln);
    const id = parseInt(cols[idIdx],10);
    if (!Number.isInteger(id)) continue;
    const title = (cols[nameIdx]??`Recipe ${id}`).replace(/^"|"$/g,'');
    const minutes = minsIdx>=0 ? parseFloat(cols[minsIdx]) : null;
    const n_ing = nIngIdx>=0 ? parseInt(cols[nIngIdx],10) : null;

    let tags=[];
    if (tagsIdx>=0 && cols[tagsIdx]){
      let raw = cols[tagsIdx].trim();
      // handle "['tag','tag']" or "tag1, tag2"
      raw = raw.replace(/^\s*\[|\]\s*$/g,'');
      tags = raw.split(/['"]\s*,\s*['"]|,\s*/g)
                .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
                .filter(Boolean)
                .slice(0,24);
    }
    Recipes.set(id, {title, minutes, n_ing, tags});
  }
}

function parseInteractions(csv){
  const lines = splitLines(csv);
  if (!lines.length) return;
  const header = splitCSVLine(lines.shift()).map(h=>h.trim().toLowerCase());
  const uIdx = header.findIndex(h=>/user/.test(h));
  const iIdx = header.findIndex(h=>/(item|recipe)_?id/.test(h));
  const rIdx = header.findIndex(h=>/rating|score/.test(h));
  const tIdx = header.findIndex(h=>/time|date/.test(h));

  for (const ln of lines){
    const cols = splitCSVLine(ln);
    const u = parseInt(cols[uIdx],10);
    const i = parseInt(cols[iIdx],10);
    if (!Number.isInteger(u) || !Number.isInteger(i)) continue;
    const r = rIdx>=0 && cols[rIdx]!=='' ? parseFloat(cols[rIdx]) : 1;
    let ts = 0;
    if (tIdx>=0 && cols[tIdx]!==''){
      const d = Date.parse(cols[tIdx]);
      ts = Number.isFinite(d) ? d : 0;
    }
    Train.push({u,i,r,ts});
    Users.add(u); Items.add(i);
    if (!user2items.has(u)) user2items.set(u,[]);
    user2items.get(u).push({i,r,ts});
    if (!item2users.has(i)) item2users.set(i,new Set());
    item2users.get(i).add(u);
  }
}

function buildTagStats(){
  tag2cnt.clear();
  for (const rec of Recipes.values()){
    for (const t of (rec.tags||[])){
      tag2cnt.set(t, (tag2cnt.get(t)||0)+1);
    }
  }
  const sorted = Array.from(tag2cnt.entries()).sort((a,b)=>b[1]-a[1]).slice(0, topK);
  tag2idx = new Map(sorted.map(([t],i)=>[t,i]));
  idx2tag = sorted.map(([t])=>t);
}

/* ---------- draw helpers ---------- */
function clearCanvas(ctx){
  const c = ctx.canvas;
  const dpr = Math.max(1, window.devicePixelRatio||1);
  const w = c.clientWidth, h = c.clientHeight;
  if (w===0 || h===0){ return; }
  c.width = Math.floor(w*dpr);
  c.height = Math.floor(h*dpr);
  ctx.setTransform(1,0,0,1,0,0);
  ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
}
function drawBars(id, values, labels=null){
  const ctx = $(id).getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight;
  const pad = 28, gap = 6;
  const n = values.length||1;
  const maxV = Math.max(1, ...values);
  const bw = (W - pad*2 - gap*(n-1)) / n;
  ctx.strokeStyle="#233046";
  ctx.beginPath(); ctx.moveTo(pad,H-pad+0.5); ctx.lineTo(W-pad,H-pad+0.5); ctx.stroke();
  ctx.fillStyle="#e5e7eb";
  for (let i=0;i<n;i++){
    const v = values[i];
    const h = (v/maxV) * (H - pad*2);
    const x = pad + i*(bw+gap);
    ctx.fillRect(x, H-pad-h, bw, h);
  }
  if (labels && n<=30){
    ctx.fillStyle="#94a3b8"; ctx.font="11px Inter, sans-serif"; ctx.textAlign="center";
    for (let i=0;i<n;i++){
      const x = pad + i*(bw+gap) + bw/2;
      ctx.save(); ctx.translate(x, H-pad+12); ctx.rotate(-Math.PI/2.7);
      ctx.fillText(String(labels[i]).slice(0,20), 0, 0);
      ctx.restore();
    }
  }
}
function drawLine(id, pairs){ // [{x,y}]
  const ctx = $(id).getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight;
  const pad = 28;
  if (!pairs.length) return;
  const maxX = Math.max(...pairs.map(p=>p.x)), minX = Math.min(...pairs.map(p=>p.x));
  const maxY = Math.max(...pairs.map(p=>p.y),1), minY = 0;
  const sx = (x)=> pad + ( (x-minX)/(maxX-minX||1) ) * (W - pad*2);
  const sy = (y)=> (H - pad) - ( (y-minY)/(maxY-minY||1) ) * (H - pad*2);
  ctx.strokeStyle="#60a5fa";
  ctx.beginPath(); ctx.moveTo(sx(pairs[0].x), sy(pairs[0].y));
  for (let i=1;i<pairs.length;i++) ctx.lineTo(sx(pairs[i].x), sy(pairs[i].y));
  ctx.stroke();
}
function drawHeat(id, mat){ // 7 x 24
  const ctx = $(id).getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight;
  const pad = 24, rows = mat.length, cols = mat[0]?.length||0;
  if (!rows || !cols) return;
  const max = Math.max(1, ...mat.flat());
  const cw = (W - pad*2)/cols, ch = (H - pad*2)/rows;
  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      const v = mat[r][c]/max;
      const col = `hsl(${200 - 200*v}, 70%, ${18 + 60*v}%)`;
      ctx.fillStyle = col;
      ctx.fillRect(pad + c*cw, pad + r*ch, cw-1, ch-1);
    }
  }
}

function scatter(id, pts){ // [{x,y}], light
  const ctx = $(id).getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight, pad = 28;
  if (!pts.length) return;
  const maxX = Math.max(...pts.map(p=>p.x),1), minX = Math.min(...pts.map(p=>p.x));
  const maxY = Math.max(...pts.map(p=>p.y),1), minY = Math.min(...pts.map(p=>p.y));
  const sx = (x)=> pad + ((x-minX)/(maxX-minX||1))*(W-pad*2);
  const sy = (y)=> (H-pad) - ((y-minY)/(maxY-minY||1))*(H-pad*2);
  ctx.fillStyle="#cbd5e1";
  for (const p of pts){
    ctx.fillRect(sx(p.x)-1, sy(p.y)-1, 2, 2);
  }
}

/* ---------- metrics / charts ---------- */
function computeAndRenderAll(){
  topK = parseInt($('topK').value,10) || 200;
  buildTagStats();

  // badges
  const density = (Train.length / (Users.size * Math.max(1, Items.size)));
  const ratingsPresent = Train.some(x=>x.r!=null && !Number.isNaN(x.r));
  const timePresent = Train.some(x=>x.ts>0);
  const minutesPresent = Array.from(Recipes.values()).some(r=>Number.isFinite(r.minutes));
  const coldUsers = Array.from(user2items.values()).filter(a=>a.length<5).length;
  const coldItems = Array.from(item2users.values()).filter(s=>s.size<5).length;

  $('bRecipes').textContent = `Recipes: ${fmt(Recipes.size)}`;
  $('bUsers').textContent = `Users: ${fmt(Users.size)}`;
  $('bItems').textContent = `Items: ${fmt(Items.size)}`;
  $('bInter').textContent = `Interactions: ${fmt(Train.length)}`;
  $('bDensity').textContent = `Density: ${density.toExponential(3)}`;
  $('bRatings').textContent = `Ratings: ${ratingsPresent?'yes':'no'}`;
  $('bTime').textContent = `Time: ${timePresent?'yes':'no'}`;
  $('bMinutes').textContent = `Minutes: ${minutesPresent?'yes':'no'}`;
  $('bColdUsers').textContent = `Cold users<5: ${fmt(coldUsers)}`;
  $('bColdItems').textContent = `Cold items<5: ${fmt(coldItems)}`;
  $('bFiles').textContent = `files: ${filesUsed.recipes}, ${filesUsed.interactions}`;

  // KPI tables
  const top1pct = Math.max(1, Math.floor(Items.size*0.01));
  const icnt = new Map();
  for (const r of Train){ icnt.set(r.i,(icnt.get(r.i)||0)+1); }
  const topItems = Array.from(icnt.values()).sort((a,b)=>b-a).slice(0, top1pct);
  const coveredPct = (topItems.reduce((s,v)=>s+v,0) / Math.max(1,Train.length))*100;

  $('kpisTbl').innerHTML =
    `<tr><td>Users</td><td>${fmt(Users.size)}</td></tr>
     <tr><td>Items</td><td>${fmt(Items.size)}</td></tr>
     <tr><td>Interactions</td><td>${fmt(Train.length)}</td></tr>
     <tr><td>Ratings present</td><td>${ratingsPresent?'yes':'no'}</td></tr>`;
  $('moreKpisTbl').innerHTML =
    `<tr><td>Avg rating</td><td>${avg(Train.map(x=>x.r)).toFixed(2)}</td></tr>
     <tr><td>Top 1% items</td><td>${fmt(top1pct)}</td></tr>
     <tr><td>% interactions by top 1%</td><td>${coveredPct.toFixed(1)}%</td></tr>`;

  // Ratings histogram (1..5)
  const buckets=[0,0,0,0,0];
  for (const r of Train){ const v = Math.round(Math.max(1, Math.min(5, r.r||1))); buckets[v-1]++; }
  drawBars('histRatings', buckets, ['1','2','3','4','5']);

  // Interactions time series by selected grouping
  drawTimeSeries();

  // DOW x hour heatmap
  drawDowHour();

  // Users — activity and scatter rating vs activity
  const uCnt = new Map();
  for (const r of Train){ uCnt.set(r.u,(uCnt.get(r.u)||0)+1); }
  const uVals = Array.from(uCnt.values());
  drawBars('userActivity', makeHistogram(uVals, [1,2,3,5,10,20,50,100]), ['1','2','3','5','10','20','50','100+']);
  const userScatter = Array.from(user2items.entries()).slice(0, 4000).map(([u,arr])=>{
    const avgR = avg(arr.map(a=>a.r));
    return {x: arr.length, y: avgR};
  });
  scatter('userScatter', userScatter);

  // Items — popularity and scatter rating vs popularity
  const iVals = Array.from(icnt.values());
  drawBars('itemPopularity', makeHistogram(iVals, [1,2,3,5,10,20,100,500,1000]),
           ['1','2','3','5','10','20','100','500','1000+']);

  const itemScatterPts = Array.from(item2users.entries()).slice(0, 4000).map(([i,setU])=>{
    const arr = (user2items.get([...setU][0])||[]); // just to have any sample rating if needed
    const ratings = (Train.filter(x=>x.i===i).map(x=>x.r));
    return {x: setU.size, y: ratings.length? avg(ratings): 0};
  });
  scatter('itemScatter', itemScatterPts);

  // Minutes & ingredients hist
  const minutes = Array.from(Recipes.values()).map(r=>r.minutes).filter(x=>Number.isFinite(x) && x>=0 && x<1000);
  drawBars('minutesHist', makeContHist(minutes, 20));
  const nIngr = Array.from(Recipes.values()).map(r=>r.n_ing).filter(x=>Number.isFinite(x) && x>0 && x<60);
  drawBars('ingrHist', makeContHist(nIngr, 20));

  // Tags — top 30 and simple co-occurrence graph
  const top30 = Array.from(tag2cnt.entries()).sort((a,b)=>b[1]-a[1]).slice(0,30);
  drawBars('topTags', top30.map(([,c])=>c), top30.map(([t])=>t));
  drawTagGraph();

  // Long-tail: Lorenz & Pareto bar
  drawLorenzAndPareto(icnt, coveredPct);
}

function drawTimeSeries(){
  const group = $('timeGroup').value; // month|week|day
  const fmtKey = (ts)=>{
    const d = new Date(ts || 0);
    if (!Number.isFinite(d.getTime())) return 'unknown';
    const y = d.getUTCFullYear();
    const m = d.getUTCMonth()+1;
    const day = d.getUTCDate();
    if (group==='month') return `${y}-${String(m).padStart(2,'0')}`;
    if (group==='week'){
      // ISO week number
      const tmp = new Date(Date.UTC(y, d.getUTCMonth(), day));
      const dayNum = (tmp.getUTCDay()+6)%7;
      tmp.setUTCDate(tmp.getUTCDate() - dayNum + 3);
      const week1 = new Date(Date.UTC(tmp.getUTCFullYear(),0,4));
      const weekNo = 1 + Math.round(((tmp - week1)/86400000 - 3 + ((week1.getUTCDay()+6)%7))/7);
      return `${tmp.getUTCFullYear()}-W${String(weekNo).padStart(2,'0')}`;
    }
    return `${y}-${String(m).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
  };
  const counts = new Map();
  for (const r of Train){ const k = fmtKey(r.ts); counts.set(k,(counts.get(k)||0)+1); }
  const keys = Array.from(counts.keys()).sort();
  const pairs = keys.map((k,idx)=>({x: idx, y: counts.get(k)}));
  drawLine('interTime', pairs);
  drawLine('byDate', pairs);
}
function drawDowHour(){
  // 7x24 matrix
  const M = Array.from({length:7},()=>Array(24).fill(0));
  for (const r of Train){
    if (!r.ts) continue;
    const d = new Date(r.ts);
    if (!Number.isFinite(d.getTime())) continue;
    const dow = (d.getUTCDay()+6)%7; // 0..6 (Mon..Sun) but we just keep 0..6
    const hr = d.getUTCHours();
    M[dow][hr]++;
  }
  drawHeat('dowHeat', M);
  // hour marginal
  const byHr = Array(24).fill(0); for (let h=0;h<24;h++) for (let d=0;d<7;d++) byHr[h]+=M[d][h];
  drawBars('byHour', byHr);
}
function drawLorenzAndPareto(icnt, coveredPct){
  const counts = Array.from(icnt.values()).sort((a,b)=>a-b);
  const total = counts.reduce((s,v)=>s+v,0) || 1;
  let acc=0; const lor = counts.map(v=>{ acc+=v; return acc/total; });
  const pairs = lor.map((y,i)=>({x:i, y}));
  // overlay equality line by drawing twice (line() already draws blue; draw equality with grey)
  const ctx = $('lorenz').getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight, pad=28;
  // equality
  ctx.strokeStyle='#334155';
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,pad); ctx.stroke();
  // observed
  if (pairs.length){
    const maxX = pairs[pairs.length-1].x||1;
    ctx.strokeStyle='#60a5fa'; ctx.beginPath();
    const sx = (x)=> pad + (x/maxX)*(W-pad*2);
    const sy = (y)=> (H-pad) - y*(H-pad*2);
    ctx.moveTo(sx(0), sy(pairs[0].y||0));
    for (let i=1;i<pairs.length;i++) ctx.lineTo(sx(pairs[i].x), sy(pairs[i].y));
    ctx.stroke();
  }
  const gini = 1 - 2 * lor.reduce((s,y)=>s+y/(counts.length||1),0) / (counts.length||1);
  $('gini').textContent = `Gini ≈ ${gini.toFixed(3)}`;

  // Pareto bar
  const ctx2 = $('pareto').getContext('2d'); clearCanvas(ctx2);
  const W2 = ctx2.canvas.clientWidth, H2 = ctx2.canvas.clientHeight, pad2=28;
  const pct = Math.max(0, Math.min(100, coveredPct));
  const w = (W2 - pad2*2);
  ctx2.fillStyle = '#60a5fa'; ctx2.fillRect(pad2, H2-60, w*(pct/100), 40);
  ctx2.fillStyle = '#1f2937'; ctx2.fillRect(pad2 + w*(pct/100), H2-60, w*(1 - pct/100), 40);
  ctx2.fillStyle = '#cbd5e1'; ctx2.font = '12px Inter,sans-serif';
  ctx2.fillText('Top 1% items', pad2, H2-70);
  ctx2.fillText('Others', pad2 + w*(pct/100) + 8, H2-70);
  $('paretoTxt').textContent = `Top 1% items (~${Math.max(1, Math.floor(Items.size*0.01))}) cover ${pct.toFixed(1)}% of all interactions.`;
}

function drawTagGraph(){
  const minCo = parseInt($('minCo').value,10)||40;
  const maxNodes = parseInt($('maxNodes').value,10)||80;
  // Build co-occurrence on top K tags only
  const counts = new Map(); // "a|b" -> weight
  const tagSeen = new Set(idx2tag);
  for (const rec of Recipes.values()){
    const tags = (rec.tags||[]).filter(t=>tagSeen.has(t));
    for (let i=0;i<tags.length;i++){
      for (let j=i+1;j<tags.length;j++){
        const a = tags[i], b = tags[j];
        const key = a<b ? `${a}|${b}` : `${b}|${a}`;
        counts.set(key,(counts.get(key)||0)+1);
      }
    }
  }
  // nodes: top by degree until maxNodes
  const deg = new Map(); // tag -> degree weight
  for (const [key,w] of counts){
    if (w<minCo) continue;
    const [a,b] = key.split('|');
    deg.set(a,(deg.get(a)||0)+w);
    deg.set(b,(deg.get(b)||0)+w);
  }
  const nodes = Array.from(deg.entries()).sort((a,b)=>b[1]-a[1]).slice(0, maxNodes).map(([t])=>t);
  const nodeSet = new Set(nodes);
  const edges = [];
  for (const [key,w] of counts){
    if (w<minCo) continue;
    const [a,b] = key.split('|');
    if (nodeSet.has(a) && nodeSet.has(b)) edges.push([a,b,w]);
  }

  // simple circular layout
  const ctx = $('tagGraph').getContext('2d'); clearCanvas(ctx);
  const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight;
  const cx=W/2, cy=H/2, R = Math.min(W,H)/2 - 40;
  const pos = new Map();
  nodes.forEach((t,idx)=>{
    const ang = (idx/nodes.length)*Math.PI*2;
    pos.set(t, {x: cx + R*Math.cos(ang), y: cy + R*Math.sin(ang)});
  });
  // edges
  ctx.lineWidth = 1;
  for (const [a,b,w] of edges){
    const A = pos.get(a), B = pos.get(b);
    ctx.strokeStyle = `rgba(96,165,250, ${Math.min(0.15 + w/(minCo*8), 0.9)})`;
    ctx.beginPath(); ctx.moveTo(A.x,A.y); ctx.lineTo(B.x,B.y); ctx.stroke();
  }
  // nodes + labels
  ctx.fillStyle = '#cbd5e1';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  for (const t of nodes){
    const p = pos.get(t);
    ctx.beginPath(); ctx.arc(p.x,p.y,3,0,Math.PI*2); ctx.fill();
    ctx.fillText(t, p.x, p.y-8);
  }
}

/* ---------- utils ---------- */
function avg(arr){ if (!arr.length) return 0; return arr.reduce((s,v)=>s+(Number.isFinite(v)?v:0),0)/arr.length; }
function makeHistogram(vals, edges){ // edges ascending; final bin is ">= last"
  const out = Array(edges.length).fill(0);
  for (const v of vals){
    let idx=edges.length-1;
    for (let e=0;e<edges.length-1;e++){ if (v<=edges[e]){ idx=e; break; } }
    out[idx]++;
  }
  return out;
}
function makeContHist(vals, bins){
  if (!vals.length) return [];
  const min = Math.min(...vals), max = Math.max(...vals);
  const step = (max-min)/bins || 1;
  const out = Array(bins).fill(0);
  for (const v of vals){
    const idx = Math.min(bins-1, Math.max(0, Math.floor((v-min)/step)));
    out[idx]++;
  }
  return out;
}

/* ---------- tabs ---------- */
function initTabs(){
  document.querySelectorAll('.tab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.tabpane').forEach(p=>p.classList.add('hidden'));
      $(btn.dataset.tab).classList.remove('hidden');
    });
  });
}

/* ---------- Load + Redraw ---------- */
async function handleLoad(){
  try{
    $('status').textContent = 'Status: loading…';
    // Reset
    Recipes.clear(); Train.length=0; Users.clear(); Items.clear();
    user2items.clear(); item2users.clear(); tag2cnt.clear();

    const recipes = await fetchFirst(['data/RAW_recipes.csv','data/PP_recipes.csv','RAW_recipes.csv','PP_recipes.csv']);
    if (!recipes) throw new Error('Could not find RAW_recipes.csv or PP_recipes.csv in / or /data/');
    parseRecipes(recipes.text); filesUsed.recipes = recipes.path;

    const interactions = await fetchFirst(['data/interactions_train.csv','interactions_train.csv']);
    if (!interactions) throw new Error('Could not find interactions_train.csv in / or /data/');
    parseInteractions(interactions.text); filesUsed.interactions = interactions.path;

    computeAndRenderAll();
    $('status').textContent = 'Status: loaded';
  }catch(err){
    console.error(err);
    $('status').textContent = 'Status: ' + err.message;
  }
}
function handleRedraw(){
  if (!Train.length){ $('status').textContent = 'Status: load the data first.'; return; }
  $('status').textContent = 'Status: redrawing…';
  computeAndRenderAll();
  $('status').textContent = 'Status: ready';
}

/* ---------- boot ---------- */
onReady(()=>{
  initTabs();
  $('btnLoad').addEventListener('click', handleLoad);
  $('btnRedraw').addEventListener('click', handleRedraw);
  $('topK').addEventListener('change', handleRedraw);
  $('minCo').addEventListener('change', handleRedraw);
  $('maxNodes').addEventListener('change', handleRedraw);
  $('timeGroup').addEventListener('change', handleRedraw);
});
