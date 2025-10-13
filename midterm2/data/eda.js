/* data/eda.js — zero-dep, GitHub-Pages-friendly EDA
   Loads /data/RAW_recipes.csv or /data/PP_recipes.csv and /data/interactions_train.csv.
   Works even if files move to repo root (tries multiple paths).
*/

// ---------- tiny DOM helpers ----------
const $ = (id)=>document.getElementById(id);
const setText = (id, s)=> { const el=$(id); if(el) el.textContent=s; };
const fmt = (n)=> typeof n==="number" ? n.toLocaleString() : n;

// ---------- global state ----------
let recipes = new Map(); // id -> {name, minutes, n_ing, tags[]}
let users = new Set();
let items = new Set();   // recipe ids present in interactions
let inter = [];          // [{u,i,r,ts}]
let user2rows = new Map();
let item2rows = new Map();
let tagFreq = new Map();

let topK=200, minEdge=40, maxNodes=80, timeGroup="month";

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tabpane").forEach(p=>p.classList.add("hidden"));
    $(btn.dataset.tab).classList.remove("hidden");
  });
});

// ---------- controls ----------
$('topK').addEventListener('change', ()=>{ topK = clamp(parseInt($('topK').value,10)||200,20,1000); redrawAll();});
$('minEdge').addEventListener('change', ()=>{ minEdge = clamp(parseInt($('minEdge').value,10)||40,2,200); redrawTags(); });
$('maxNodes').addEventListener('change', ()=>{ maxNodes = clamp(parseInt($('maxNodes').value,10)||80,30,300); redrawTags(); });
$('timeGrp').addEventListener('change', ()=>{ timeGroup = $('timeGrp').value; redrawTime(); });

// ---------- load ----------
$('btnLoad').addEventListener('click', ()=>loadAll().catch(err=>{
  console.error(err);
  setText('status','Status: failed to load. Make sure CSVs exist in /data or root. (See console)');
}));
$('btnRedraw').addEventListener('click', ()=>redrawAll());

// ---------- utils ----------
function clamp(x,a,b){ return Math.max(a,Math.min(b,x)); }
function splitCSVLine(line){
  // split on commas not inside quotes
  const out=[]; let cur=''; let q=false;
  for(let i=0;i<line.length;i++){
    const c=line[i];
    if(c==='"'){
      if(q && line[i+1]==='"'){ cur+='"'; i++; } else { q=!q; }
    }else if(c===',' && !q){ out.push(cur); cur=''; }
    else cur+=c;
  }
  out.push(cur);
  return out;
}
function cleanQuotes(s){ return s ? s.replace(/^"|"$/g,'') : s; }
function toInt(x){ const v=parseInt(x,10); return Number.isFinite(v)?v:null; }
function toFloat(x){ const v=parseFloat(x); return Number.isFinite(v)?v:null; }
function tryPaths(paths){ return Promise.any(paths.map(p=>fetch(p, {cache:'no-store'}))); }
function parsePyList(str){
  if(!str) return [];
  let s=String(str).trim();
  // remove brackets
  s=s.replace(/^\s*\[|\]\s*$/g,'');
  if(!s) return [];
  const parts = s.split(/['"]\s*,\s*['"]|,\s*/g)
    .map(t=>t.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
    .filter(Boolean);
  return Array.from(new Set(parts));
}

// ---------- parsing ----------
function parseRecipesCSV(text){
  recipes.clear(); tagFreq.clear();
  const rows=text.replace(/^\uFEFF/,'').split(/\r?\n/).filter(Boolean);
  const header=splitCSVLine(rows.shift()).map(h=>h.toLowerCase());
  const idIx = idxOf(header, ['id','recipe_id']);
  const nameIx = idxOf(header, ['name','title']);
  const minutesIx = idxOf(header, ['minutes','cook_time','time']);
  const ingrIx = idxOf(header, ['n_ingredients','ingredients_count','n_ing']);
  const tagsIx = idxOf(header, ['tags','tag_list','pp_tags']);
  for(const line of rows){
    const cols=splitCSVLine(line);
    const id = toInt(cols[idIx]); if(id===null) continue;
    const name = cleanQuotes(cols[nameIx]||`Recipe ${id}`);
    const minutes = toInt(cols[minutesIx]);
    const n_ing = toInt(cols[ingrIx]);
    const tags = tagsIx>=0 ? parsePyList(cols[tagsIx]) : [];
    recipes.set(id,{name,minutes,n_ing,tags});
    for(const t of tags){ tagFreq.set(t,(tagFreq.get(t)||0)+1); }
  }
}
function idxOf(arr, keys){
  for(const k of keys){
    const ix = arr.findIndex(h=>h===k || h.endsWith('_'+k));
    if(ix>=0) return ix;
  }
  return -1;
}
function parseInterCSV(text){
  inter.length=0; users.clear(); items.clear();
  user2rows.clear(); item2rows.clear();
  const rows=text.replace(/^\uFEFF/,'').split(/\r?\n/).filter(Boolean);
  const header=splitCSVLine(rows.shift()).map(h=>h.toLowerCase());
  const uIx = idxOf(header,['user','user_id','profile','uid']);
  const iIx = idxOf(header,['item','recipe_id','rid','iid']);
  const rIx = idxOf(header,['rating','score','stars']);
  const tIx = idxOf(header,['date','time','timestamp','ts','review_date']);
  for(const line of rows){
    const c=splitCSVLine(line);
    const u = toInt(c[uIx]), i = toInt(c[iIx]);
    if(u===null || i===null) continue;
    const r = (rIx>=0 && c[rIx]!=='' && c[rIx]!=null) ? toFloat(c[rIx]) : null;
    const tsRaw = (tIx>=0 ? Date.parse(c[tIx]) : NaN);
    const ts = Number.isFinite(tsRaw) ? tsRaw : 0;
    inter.push({u,i,r,ts});
    users.add(u); items.add(i);
    if(!user2rows.has(u)) user2rows.set(u,[]);
    if(!item2rows.has(i)) item2rows.set(i,[]);
    user2rows.get(u).push({i,r,ts});
    item2rows.get(i).push({u,r,ts});
  }
}

// ---------- loading orchestrator ----------
async function loadAll(){
  setText('status','Status: loading…');

  let recResp;
  try{
    recResp = await tryPaths([
      './data/PP_recipes.csv', './data/RAW_recipes.csv',
      'data/PP_recipes.csv','data/RAW_recipes.csv',
      './PP_recipes.csv','./RAW_recipes.csv'
    ]);
  }catch{
    throw new Error('recipes CSV not found in /data or root');
  }
  const recText = await recResp.text();
  parseRecipesCSV(recText);

  let itResp;
  try{
    itResp = await tryPaths([
      './data/interactions_train.csv','data/interactions_train.csv','./interactions_train.csv'
    ]);
  }catch{
    throw new Error('interactions_train.csv not found in /data or root');
  }
  const itText = await itResp.text();
  parseInterCSV(itText);

  setText('status','Status: loaded');
  fillCounters();
  redrawAll();
}

// ---------- counters / KPIs ----------
function fillCounters(){
  // presence
  const haveRatings = inter.some(x=>x.r!=null);
  const haveTime = inter.some(x=>x.ts>0);
  const haveMinutes = Array.from(recipes.values()).some(r=>Number.isFinite(r.minutes));
  const haveIng = Array.from(recipes.values()).some(r=>Number.isFinite(r.n_ing));

  const density = inter.length / (users.size * Math.max(1, items.size));
  const coldUsers = Array.from(user2rows.values()).filter(a=>a.length<5).length;
  const coldItems = Array.from(item2rows.values()).filter(a=>a.length<5).length;

  setText('c_recipes',`Recipes: ${fmt(recipes.size)}`);
  setText('c_users',`Users: ${fmt(users.size)}`);
  setText('c_items',`Items: ${fmt(items.size)}`);
  setText('c_inter',`Interactions: ${fmt(inter.length)}`);
  setText('c_density',`Density: ${density.toExponential(3)}`);
  setText('c_ratings',`Ratings: ${haveRatings?'yes':'no'}`);
  setText('c_time',`Time: ${haveTime?'yes':'no'}`);
  setText('c_minutes',`Minutes: ${haveMinutes?'yes':'no'}`);
  setText('c_ing',`n_ingredients: ${haveIng?'yes':'no'}`);
  setText('c_coldU',`Cold users<5: ${fmt(coldUsers)}`);
  setText('c_coldI',`Cold items<5: ${fmt(coldItems)}`);

  $('kpisA').innerHTML =
    `<span class="stat pill">Users<br><b>${fmt(users.size)}</b></span>
     <span class="stat pill">Items<br><b>${fmt(items.size)}</b></span>
     <span class="stat pill">Interactions<br><b>${fmt(inter.length)}</b></span>`;

  const top1pct = Math.max(1, Math.floor(0.01*items.size));
  const counts = Array.from(item2rows.values()).map(a=>a.length).sort((a,b)=>b-a);
  const covered = (counts.slice(0,top1pct).reduce((s,v)=>s+v,0) / Math.max(1,inter.length))*100;

  $('kpisB').innerHTML =
    `<span class="stat pill">Density<br><b>${density.toExponential(3)}</b></span>
     <span class="stat pill">Cold users&lt;5<br><b>${fmt(coldUsers)}</b></span>
     <span class="stat pill">Cold items&lt;5<br><b>${fmt(coldItems)}</b></span>
     <span class="stat pill">Top 1% items<br><b>${fmt(top1pct)}</b></span>
     <span class="stat pill">% interactions by top 1%<br><b>${covered.toFixed(1)}%</b></span>`;
}

// ---------- drawing primitives ----------
function ctxPrep(id){
  const c=$(id); const dpr=window.devicePixelRatio||1;
  c.width = Math.max(1, c.clientWidth*dpr);
  c.height = Math.max(1, c.clientHeight*dpr);
  const g=c.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,c.clientWidth,c.clientHeight);
  return [g,c.clientWidth,c.clientHeight];
}
function drawBars(id, arr, opts={}){
  const [g,W,H]=ctxPrep(id);
  const pad=28, base=H-pad, bw = (W-pad*2)/arr.length - (opts.gap??6);
  const max = opts.max ?? Math.max(1,...arr);
  g.strokeStyle="#213047"; g.beginPath(); g.moveTo(pad,base+0.5); g.lineTo(W-pad,base+0.5); g.stroke();
  g.fillStyle="#dbeafe";
  arr.forEach((v,i)=>{
    const h = (v/max)*(H-pad*2);
    const x = pad + i*(bw+(opts.gap??6));
    g.fillRect(x, base-h, bw, h);
  });
  if(opts.labels){
    g.fillStyle="#93a3b5"; g.font="11px Inter";
    const step = Math.ceil(opts.labels.length/Math.max(8, Math.floor(W/90)));
    opts.labels.forEach((t,i)=>{
      if(i%step) return;
      const x = pad + i*(bw+(opts.gap??6)) + bw/2;
      g.save(); g.translate(x, H-6); g.rotate(-Math.PI/3); g.fillText(t,0,0); g.restore();
    });
  }
}
function drawLine(id, pts){
  const [g,W,H]=ctxPrep(id);
  const pad=28, base=H-pad;
  const maxX = Math.max(1,...pts.map(p=>p.x));
  const maxY = Math.max(1,...pts.map(p=>p.y));
  const sx = (x)=> pad + (x/maxX)*(W-pad*2);
  const sy = (y)=> base - (y/maxY)*(H-pad*2);
  g.strokeStyle="#7dd3fc"; g.beginPath();
  pts.forEach((p,i)=> i? g.lineTo(sx(p.x),sy(p.y)) : g.moveTo(sx(p.x),sy(p.y)));
  g.stroke();
}
function drawScatter(id, pts){
  const [g,W,H]=ctxPrep(id);
  const pad=28, base=H-pad;
  const maxX = Math.max(1,...pts.map(p=>p.x)), maxY=Math.max(1,...pts.map(p=>p.y));
  const sx=(x)=> pad + (x/maxX)*(W-pad*2), sy=(y)=> base - (y/maxY)*(H-pad*2);
  g.fillStyle="#cbd5e1";
  pts.slice(0,8000).forEach(p=>{ g.fillRect(sx(p.x)-1.5, sy(p.y)-1.5, 3,3); });
}
function drawHeat(id, mat){
  const [g,W,H]=ctxPrep(id);
  const rows=mat.length, cols=mat[0]?.length||0;
  if(!rows||!cols) return;
  const pad=40; const cw=(W-pad*2)/cols; const ch=(H-pad*2)/rows;
  const max = Math.max(1, ...mat.flat());
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const v = mat[r][c]/max; // 0..1
      const col = heatColor(v);
      g.fillStyle = col;
      g.fillRect(pad+c*cw, pad+r*ch, cw-1, ch-1);
    }
  }
  // axes labels
  g.fillStyle="#93a3b5"; g.font="11px Inter";
  const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  days.forEach((d,i)=> g.fillText(d, 8, pad + (i+0.7)*ch) );
  for(let c=0;c<cols;c+=Math.max(1,Math.floor(cols/8))){
    g.fillText(String(c), pad + c*cw, H-8);
  }
}
function heatColor(v){
  // simple yellow-green scale
  const a = Math.min(1, Math.max(0, v));
  const r = 255, g = Math.floor(255*Math.min(1,a*1.2)), b = 60;
  return `rgb(${r},${g},${b},${0.12 + 0.88*a})`;
}

// simple pan / zoom on tag graph canvas
function enablePanZoom(canvas, state){
  let dragging=false, lx=0, ly=0;
  canvas.addEventListener('mousedown', e=>{dragging=true; lx=e.clientX; ly=e.clientY;});
  window.addEventListener('mouseup', ()=>dragging=false);
  window.addEventListener('mousemove', e=>{
    if(!dragging) return;
    const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
    state.tx += dx; state.ty += dy; drawTagGraph();
  });
  canvas.addEventListener('wheel', e=>{
    e.preventDefault();
    const s = e.deltaY<0 ? 1.1 : 0.9;
    state.scale = clamp(state.scale*s, 0.3, 4);
    drawTagGraph();
  }, {passive:false});
}

// ---------- redrawers ----------
function redrawAll(){
  redrawOverview();
  redrawUsers();
  redrawItems();
  redrawTags();
  redrawTime();
  redrawLongTail();
}

function redrawOverview(){
  // ratings hist
  const hist=[0,0,0,0,0];
  for(const r of inter){ if(r.r!=null){ const v=Math.max(1,Math.min(5,Math.round(r.r))); hist[v-1]++; } }
  drawBars('histRatings', hist);

  // interactions over time (month)
  const byMonth = new Map();
  inter.forEach(x=>{
    const d = x.ts? new Date(x.ts): null;
    if(!d) return;
    const key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;
    byMonth.set(key, (byMonth.get(key)||0)+1);
  });
  const keys = Array.from(byMonth.keys()).sort();
  const pts = keys.map((k,ix)=>({x:ix+1, y:byMonth.get(k)}));
  drawLine('interOver', pts);

  // heatmap dow x hour
  const mat = Array.from({length:7}, ()=>Array(24).fill(0));
  let anyHour=false;
  for(const x of inter){
    if(!x.ts) continue;
    const d=new Date(x.ts);
    const h=d.getHours();
    if(Number.isFinite(h)){ anyHour = true; mat[(d.getDay()+6)%7][h]++; }
  }
  if(anyHour) drawHeat('heat', mat);
  else { const [g,W,H]=ctxPrep('heat'); g.fillStyle="#93a3b5"; g.fillText("No hours in timestamps; heatmap collapsed.", 16, 24); }
}

function redrawUsers(){
  // user activity
  const cnt = new Map(); inter.forEach(x=>cnt.set(x.u,(cnt.get(x.u)||0)+1));
  const arr = Array.from(cnt.values());
  const b = bucketize(arr,[1,2,3,5,10,20,50,100,200,500],'≤');
  drawBars('histUser', b.buckets, {labels:b.labels, gap:4});
  // scatter: avg rating vs interactions
  const pts=[];
  cnt.forEach((c,u)=>{
    const rows = user2rows.get(u)||[];
    const rvals = rows.map(r=>r.r).filter(v=>v!=null);
    const avg = rvals.length ? (rvals.reduce((s,v)=>s+v,0)/rvals.length) : 0;
    pts.push({x:c, y:avg});
  });
  drawScatter('scatterUser', pts);
}

function redrawItems(){
  const cnt = new Map(); inter.forEach(x=>cnt.set(x.i,(cnt.get(x.i)||0)+1));
  const arr = Array.from(cnt.values());
  const b = bucketize(arr,[1,2,3,5,10,20,50,100,500,1000],'≤');
  drawBars('histItem', b.buckets, {labels:b.labels, gap:4});
  // item scatter
  const pts=[];
  cnt.forEach((c,i)=>{
    const rows = item2rows.get(i)||[];
    const rvals = rows.map(r=>r.r).filter(v=>v!=null);
    const avg = rvals.length ? (rvals.reduce((s,v)=>s+v,0)/rvals.length) : 0;
    pts.push({x:c, y:avg});
  });
  drawScatter('scatterItem', pts);

  // minutes
  const mins = Array.from(recipes.values()).map(r=>r.minutes).filter(v=>Number.isFinite(v) && v>0 && v<10000);
  drawBars('histMinutes', histogram(mins, 40));

  // ingredients
  const ings = Array.from(recipes.values()).map(r=>r.n_ing).filter(v=>Number.isFinite(v) && v>=0 && v<100);
  drawBars('histIng', histogram(ings, 40));
}

function redrawTags(){
  // topK tags
  const top = Array.from(tagFreq.entries()).sort((a,b)=>b[1]-a[1]).slice(0,topK);
  const top30 = top.slice(0,30);
  drawBars('topTags', top30.map(([,c])=>c), {labels: top30.map(([t])=>t)});

  // build co-occurrence edges among top tags
  const topSet = new Set(top.map(([t])=>t));
  const idx = new Map(Array.from(topSet).map((t,i)=>[t,i]));
  const deg = new Array(topSet.size).fill(0);
  const edges = new Map(); // "a|b" -> weight

  // sweep recipes
  for(const r of recipes.values()){
    const ts = (r.tags||[]).filter(t=>topSet.has(t));
    if(ts.length<2) continue;
    for(let i=0;i<ts.length;i++){
      for(let j=i+1;j<ts.length;j++){
        const a=ts[i], b=ts[j];
        const key= a<b? `${a}|${b}` : `${b}|${a}`;
        const w = (edges.get(key)||0)+1; edges.set(key,w);
      }
    }
  }
  const E = Array.from(edges.entries()).filter(([,w])=>w>=minEdge)
               .sort((a,b)=>b[1]-a[1]).slice(0, 5000);

  // degree
  for(const [k,w] of E){
    const [a,b]=k.split('|'); deg[idx.get(a)]++; deg[idx.get(b)]++;
  }
  // pick nodes with highest degree up to maxNodes
  const nodes = Array.from(topSet).map(t=>({t, d:deg[idx.get(t)]||0}))
                    .sort((a,b)=>b.d-a.d).slice(0, maxNodes);
  const keep = new Set(nodes.map(n=>n.t));
  const edges2 = E.filter(([k])=>{ const [a,b]=k.split('|'); return keep.has(a)&&keep.has(b); });

  // layout (simple circular, then small relaxation)
  const N = nodes.length;
  const pts = nodes.map((n,i)=>({
    t:n.t, x: Math.cos(2*Math.PI*i/N), y: Math.sin(2*Math.PI*i/N), vx:0, vy:0
  }));
  const pos = new Map(pts.map(p=>[p.t,p]));

  for(let step=0; step<80; step++){
    // repulsion
    for(let i=0;i<N;i++){
      for(let j=i+1;j<N;j++){
        const a=pts[i], b=pts[j];
        let dx=a.x-b.x, dy=a.y-b.y, d=Math.hypot(dx,dy)+1e-3;
        const f=0.001/(d*d);
        dx/=d; dy/=d; a.vx+=dx*f; a.vy+=dy*f; b.vx-=dx*f; b.vy-=dy*f;
      }
    }
    // attraction
    for(const [k,w] of edges2){
      const [ta,tb]=k.split('|'); const a=pos.get(ta), b=pos.get(tb);
      let dx=b.x-a.x, dy=b.y-a.y, d=Math.hypot(dx,dy)+1e-3;
      const f=0.0015*Math.log(d+1)*Math.min(3,w/100); dx/=d; dy/=d;
      a.vx+=dx*f; a.vy+=dy*f; b.vx-=dx*f; b.vy-=dy*f;
    }
    // integrate & damping
    for(const p of pts){ p.x+=p.vx; p.y+=p.vy; p.vx*=0.9; p.vy*=0.9; }
  }

  // store into state & render
  tagGraphState.nodes = pts;
  tagGraphState.edges = edges2.map(([k,w])=>{
    const [ta,tb]=k.split('|'); return {a:pos.get(ta), b:pos.get(tb), w};
  });
  drawTagGraph();
}

// state for pan/zoom
const tagGraphState = { nodes:[], edges:[], scale: 160, tx: 0, ty: 0 };
enablePanZoom($('tagGraph'), tagGraphState);

function drawTagGraph(){
  const [g,W,H]=ctxPrep('tagGraph');
  const cx = W/2 + tagGraphState.tx, cy = H/2 + tagGraphState.ty, S = tagGraphState.scale;
  // edges
  g.strokeStyle="#2b3a55";
  for(const e of tagGraphState.edges){
    g.lineWidth = Math.min(4, 0.5 + e.w/80);
    g.beginPath();
    g.moveTo(cx + e.a.x*S, cy + e.a.y*S);
    g.lineTo(cx + e.b.x*S, cy + e.b.y*S);
    g.stroke();
  }
  // nodes + labels
  g.fillStyle="#e5e7eb"; g.font="12px Inter"; g.textAlign="center";
  for(const n of tagGraphState.nodes){
    const x=cx+n.x*S, y=cy+n.y*S;
    g.beginPath(); g.arc(x,y,4,0,Math.PI*2); g.fill();
    g.fillText(n.t, x, y-8);
  }
}

function redrawTime(){
  // calendar by month
  const counts=new Map();
  for(const x of inter){
    if(!x.ts) continue;
    const d=new Date(x.ts);
    const key=`${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;
    counts.set(key,(counts.get(key)||0)+1);
  }
  const keys=Array.from(counts.keys()).sort();
  const line=keys.map((k,i)=>({x:i+1, y:counts.get(k)}));
  drawLine('interCalendar', line);

  // grouped by selected granularity
  const c2=new Map();
  for(const x of inter){
    if(!x.ts) continue;
    const d=new Date(x.ts);
    let key;
    if(timeGroup==='week'){
      // ISO week
      const t=new Date(Date.UTC(d.getFullYear(),d.getMonth(),d.getDate()));
      const day=(t.getUTCDay()+6)%7; t.setUTCDate(t.getUTCDate()-day+3);
      const week1=new Date(Date.UTC(t.getUTCFullYear(),0,4));
      const week=Math.round(((t-week1)/86400000-3+(week1.getUTCDay()+6)%7)/7)+1;
      key=`${t.getUTCFullYear()}-W${String(week).padStart(2,'0')}`;
    }else if(timeGroup==='day'){
      key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
    }else{
      key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;
    }
    c2.set(key,(c2.get(key)||0)+1);
  }
  const k2=Array.from(c2.keys()).sort();
  const line2=k2.map((k,i)=>({x:i+1, y:c2.get(k)}));
  drawLine('interGrouped', line2);
}

function redrawLongTail(){
  const counts = Array.from(item2rows.values()).map(a=>a.length).sort((a,b)=>a-b);
  const total = counts.reduce((s,v)=>s+v,0)||1;
  let acc=0; const lor=counts.map(v=>{acc+=v; return acc/total;});
  // lorenz
  const pts = lor.map((y,i)=>({x:i+1,y}));
  drawLine('lorenz', pts);
  const gini = 1 - 2 * lor.reduce((s,y,i)=> s + y/(counts.length||1), 0) / (counts.length||1);
  setText('giniLine',`Gini ≈ ${gini.toFixed(3)}`);

  // Pareto bar
  const top1 = Math.max(1, Math.floor(0.01*counts.length));
  const covered = (counts.slice(-top1).reduce((s,v)=>s+v,0) / total)*100;
  drawBars('pareto', [covered, 100-covered], {gap:40});
  setText('paretoLine', `Top ${top1} items (~1%) cover ${covered.toFixed(1)}% of all interactions.`);
}

// ---------- helpers ----------
function bucketize(values, edges, mode='≤'){
  const buckets=new Array(edges.length).fill(0);
  const labels=[];
  for(let i=0;i<edges.length;i++){
    const prev = i? edges[i-1]: null;
    const cur = edges[i];
    labels.push( prev==null ? `${cur}` : `${prev+1}-${cur}` );
  }
  values.forEach(v=>{
    for(let i=0;i<edges.length;i++){
      if(v<=edges[i]){ buckets[i]++; return; }
    }
  });
  return {buckets, labels};
}
function histogram(values, bins=30){
  if(!values.length) return [];
  const min=Math.min(...values), max=Math.max(...values);
  const w=(max-min)/bins || 1;
  const arr=new Array(bins).fill(0);
  for(const v of values){
    const ix=Math.max(0, Math.min(bins-1, Math.floor((v-min)/w)));
    arr[ix]++;
  }
  return arr;
}
