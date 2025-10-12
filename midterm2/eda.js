/* eda-app.js — static, business-first EDA for food recommender (GitHub Pages ready) */
/* Loads RAW_recipes.csv or PP_recipes.csv and interactions_train.csv from ./ or ./data/ */

const $ = (id)=>document.getElementById(id);
const fmt = (n)=> typeof n==='number' ? n.toLocaleString() : n;

// ---------------- state ----------------
let items = new Map();           // id -> {name, minutes, n_ingredients, tags[]}
let users = new Set();
let interactions = [];           // [{u,i,r,ts}]
let user2rows = new Map();       // u -> rows
let item2rows = new Map();       // i -> rows
let tag2count = new Map();

let loadedFiles = {recipes:'—', inter:'—'};
let topK = 200, edgeMin = 40, nodeCap = 80;

// ---------------- tabs ----------------
document.querySelectorAll('.tab').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tabpane').forEach(p=>p.classList.add('hidden'));
    btn.classList.add('active');
    $(btn.dataset.tab).classList.remove('hidden');
  });
});

// ---------------- IO helpers ----------------
async function fetchFirst(paths){
  for (const p of paths){
    try{
      const r = await fetch(p, {cache:'no-store'});
      if (r.ok) return {path:p, text: await r.text()};
    }catch{}
  }
  return null;
}
function splitCSVLine(line){
  return line.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g);
}

// recipes: RAW_recipes.csv or PP_recipes.csv
function parseRecipes(text){
  const lines = text.split(/\r?\n/).filter(Boolean);
  const head = splitCSVLine(lines.shift()).map(s=>s.toLowerCase());
  const idxId  = head.findIndex(h=>/^id$|(^|_)id$/.test(h));
  const idxName = head.findIndex(h=>/(name|title)/.test(h));
  const idxTags = head.findIndex(h=>/tags/.test(h));
  const idxMin  = head.findIndex(h=>/minute/.test(h));
  const idxIng  = head.findIndex(h=>/ingredient/.test(h));
  for (const ln of lines){
    const cols = splitCSVLine(ln);
    const id = parseInt(cols[idxId],10); if (!Number.isInteger(id)) continue;
    const name = (cols[idxName]||`Recipe ${id}`).replace(/^"|"$/g,'');
    const minutes = idxMin>=0 ? parseInt(cols[idxMin]||'0',10)||0 : 0;
    const n_ing = idxIng>=0 ? parseInt(cols[idxIng]||'0',10)||0 : 0;
    let tags=[];
    if (idxTags>=0 && cols[idxTags]){
      let raw = cols[idxTags].trim();
      raw = raw.replace(/^\s*\[|\]\s*$/g,''); // strip [ ... ]
      tags = raw.split(/['"]\s*,\s*['"]|,\s*/g)
                .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
                .filter(Boolean);
    }
    items.set(id,{name, minutes, n_ingredients:n_ing, tags});
    for (const t of tags){ tag2count.set(t,(tag2count.get(t)||0)+1); }
  }
}

// interactions: interactions_train.csv
function parseInteractions(text){
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return;
  const head = splitCSVLine(lines.shift()).map(s=>s.toLowerCase());
  const uIdx = head.findIndex(h=>/user/.test(h));
  const iIdx = head.findIndex(h=>/(item|recipe)_?id/.test(h));
  const rIdx = head.findIndex(h=>/rating/.test(h));
  const tIdx = head.findIndex(h=>/(time|date|timestamp)/.test(h));
  for (const ln of lines){
    const cols = splitCSVLine(ln);
    const u = parseInt(cols[uIdx],10), i = parseInt(cols[iIdx],10);
    if (!Number.isInteger(u) || !Number.isInteger(i)) continue;
    const r = rIdx>=0 && cols[rIdx]!=='' ? parseFloat(cols[rIdx]) : 0;
    const ts = tIdx>=0 ? (Date.parse(cols[tIdx])||0) : 0;
    const row = {u,i,r,ts};
    interactions.push(row);
    users.add(u);
    if (!user2rows.has(u)) user2rows.set(u,[]);
    user2rows.get(u).push(row);
    if (!item2rows.has(i)) item2rows.set(i,[]);
    item2rows.get(i).push(row);
  }
}

// ---------------- draw utils ----------------
function prepCanvas(id){
  const ctx = $(id).getContext('2d');
  const c = ctx.canvas, d = window.devicePixelRatio||1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = Math.max(1,w*d); c.height = Math.max(1,h*d);
  ctx.setTransform(d,0,0,d,0,0);
  ctx.clearRect(0,0,w,h);
  return {ctx,w,h};
}
function drawBars(id, vals, opts={}){
  const {ctx,w,h} = prepCanvas(id);
  const pad=28, base=h-pad, bw=(w-pad*2)/Math.max(1,vals.length)-6;
  const maxV = opts.max ?? Math.max(1,...vals);
  ctx.strokeStyle="#243244"; ctx.beginPath(); ctx.moveTo(pad,base+0.5); ctx.lineTo(w-pad,base+0.5); ctx.stroke();
  ctx.fillStyle="#e5e7eb";
  vals.forEach((v,i)=>{
    const x = pad + i*(bw+6);
    const hh = (v/maxV)*(h-pad*2);
    ctx.fillRect(x, base-hh, bw, hh);
  });
}
function drawLine(id, pts){
  const {ctx,w,h} = prepCanvas(id);
  if (!pts.length) return;
  const pad=24, base=h-pad;
  const maxX = Math.max(...pts.map(p=>p.x)), maxY = Math.max(1,...pts.map(p=>p.y));
  const sx = (x)=> pad + (x/maxX)*(w-pad*2);
  const sy = (y)=> base - (y/maxY)*(h-pad*2);
  ctx.strokeStyle="#7dd3fc"; ctx.beginPath();
  ctx.moveTo(sx(pts[0].x), sy(pts[0].y));
  for (let k=1;k<pts.length;k++) ctx.lineTo(sx(pts[k].x), sy(pts[k].y));
  ctx.stroke();
}
function drawScatter(id, pts){
  const {ctx,w,h} = prepCanvas(id);
  if (!pts.length) return;
  const pad=28;
  const maxX = Math.max(1,...pts.map(p=>p.x)), maxY = Math.max(1,...pts.map(p=>p.y));
  const sx = (x)=> pad + (x/maxX)*(w-pad*2);
  const sy = (y)=> (h-pad) - (y/maxY)*(h-pad*2);
  ctx.fillStyle="#cbd5e1";
  for (const p of pts){ ctx.fillRect(sx(p.x)-1, sy(p.y)-1, 2, 2); }
}
function drawHeatmap(id, matrix, opts={}){
  const {ctx,w,h} = prepCanvas(id);
  const rows = matrix.length, cols = matrix[0]?.length||0;
  if (!rows || !cols) return;
  const pad=30; const cw=(w-pad*2)/cols, ch=(h-pad*2)/rows;
  const maxV = Math.max(1, ...matrix.flat());
  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      const v = matrix[r][c];
      const t = v/maxV; // 0..1
      // simple viridis-ish
      const col = `hsl(${200 - 200*t}, 80%, ${20+50*t}%)`;
      ctx.fillStyle = col;
      ctx.fillRect(pad+c*cw, pad+r*ch, cw-1, ch-1);
    }
  }
}

// ---------------- business metrics ----------------
function median(arr){ const a=arr.slice().sort((x,y)=>x-y); const n=a.length; return n? (n%2? a[(n-1)/2] : 0.5*(a[n/2-1]+a[n/2])) : 0; }
function avg(arr){ return arr.length? arr.reduce((s,v)=>s+v,0)/arr.length : 0; }

// personas by tag substrings
const PERSONA_RULES = [
  {key:'Fast cook',      match:['15-minutes','30-minutes','60-minutes','quick']},
  {key:'Healthy eater',  match:['low-fat','low-calorie','low-cholesterol','low-sodium','low-carb','healthy']},
  {key:'Dessert lover',  match:['dessert','cake','cookies','sweet','pudding','pie']},
  {key:'Breakfast/Brunch', match:['breakfast','brunch']},
  {key:'Vegetarian/Vegan', match:['vegetarian','vegan']},
  {key:'Gluten-free',    match:['gluten-free']}
];
function personaOf(tags){
  const tLower = (tags||[]).map(t=>t.toLowerCase());
  let best=null, bestScore=0;
  for (const p of PERSONA_RULES){
    let s=0;
    for (const k of p.match) s += tLower.some(t=>t.includes(k)) ? 1 : 0;
    if (s>bestScore){ bestScore=s; best=p.key; }
  }
  return best || 'General';
}

// ---------------- compute & render ----------------
function renderBadges(){
  const hasRatings = interactions.some(x=>x.r && !Number.isNaN(x.r));
  const hasTime = interactions.some(x=>x.ts>0);
  const hasMinutes = Array.from(items.values()).some(x=>x.minutes>0);
  const chips = [
    `Recipes: ${fmt(items.size)}`,
    `Users: ${fmt(users.size)}`,
    `Items: ${fmt(new Set(interactions.map(x=>x.i)).size)}`,
    `Interactions: ${fmt(interactions.length)}`,
    `Density: ${(interactions.length/(Math.max(1,users.size)*Math.max(1,items.size))).toExponential(3)}`,
    `Ratings: ${hasRatings?'yes':'no'}`,
    `Time: ${hasTime?'yes':'no'}`,
    `Minutes: ${hasMinutes?'yes':'no'}`
  ];
  $('badges').innerHTML = chips.map(c=>`<span class="chip">${c}</span>`).join('');
}

function renderCounters(){
  const coldUsers = Array.from(user2rows.values()).filter(a=>a.length<5).length;
  const coldItems = Array.from(item2rows.values()).filter(a=>a.length<5).length;
  const div = $('counters');
  div.innerHTML = [
    `Users <b>${fmt(users.size)}</b>`,
    `Items <b>${fmt(item2rows.size)}</b>`,
    `Interactions <b>${fmt(interactions.length)}</b>`,
    `Density <b>${(interactions.length/(Math.max(1,users.size)*Math.max(1,items.size))).toExponential(3)}</b>`,
    `Cold users &lt;5 <b>${fmt(coldUsers)}</b>`,
    `Cold items &lt;5 <b>${fmt(coldItems)}</b>`,
    `Ratings present <b>${interactions.some(x=>x.r && !Number.isNaN(x.r))?'yes':'no'}</b>`
  ].map(s=>`<span class="chip">${s}</span>`).join('');
}

function renderKPIs(){
  const iu = users.size? interactions.length/users.size : 0;
  const ii = item2rows.size? interactions.length/item2rows.size : 0;
  const itemCoverage = items.size? (item2rows.size/items.size*100) : 0;
  const mins = Array.from(items.values()).map(x=>x.minutes||0).filter(x=>x>0);
  const medMin = median(mins);
  $('kpibox').innerHTML = [
    `avg interactions/user <b>${iu.toFixed(2)}</b>`,
    `avg interactions/item <b>${ii.toFixed(2)}</b>`,
    `items w/ ≥1 interaction <b>${(itemCoverage).toFixed(1)}%</b>`,
    `median minutes (all recipes) <b>${medMin?medMin.toFixed(0):'—'}</b>`
  ].map(s=>`<span class="chip">${s}</span>`).join('');
}

function drawOverview(){
  // interactions over time
  const grp = {};
  for (const r of interactions){
    const d = r.ts ? new Date(r.ts) : null;
    let key;
    const g = $('timeGroup').value.trim().toLowerCase();
    if (d){
      if (g==='week'){
        const y=d.getUTCFullYear();
        const w=Math.ceil((((d - new Date(Date.UTC(y,0,1)))/86400000)+4)/7);
        key=`${y}-W${w.toString().padStart(2,'0')}`;
      }else{ // month default
        key = `${d.getUTCFullYear()}-${(d.getUTCMonth()+1).toString().padStart(2,'0')}`;
      }
    }else key='unknown';
    grp[key]=(grp[key]||0)+1;
  }
  const xs = Object.keys(grp).filter(k=>k!=='unknown').sort();
  const pts = xs.map((k,idx)=>({x:idx+1, y:grp[k]}));
  drawLine('lineInteractions', pts);

  // ratings histogram (1..5)
  const hist=[0,0,0,0,0];
  interactions.forEach(r=>{ const v = Math.round(Math.max(1,Math.min(5, r.r||0))); if (v>=1&&v<=5) hist[v-1]++; });
  drawBars('histRatings', hist, {max:Math.max(...hist,1)});

  // dow × hour heatmap
  const mat = Array.from({length:7},()=>Array(24).fill(0));
  for (const r of interactions){
    if (!r.ts) continue;
    const d = new Date(r.ts);
    const dow = (d.getUTCDay()+6)%7; // Mon=0
    const hr = d.getUTCHours();
    mat[dow][hr]++;
  }
  drawHeatmap('dowHour', mat);
}

function drawUsers(){
  // activity hist
  const counts = Array.from(user2rows.values()).map(a=>a.length);
  const buckets = [0,0,0,0,0,0,0,0];
  counts.forEach(v=>{
    const idx = v===1?0 : v<=2?1 : v<=3?2 : v<=5?3 : v<=10?4 : v<=20?5 : v<=50?6 : 7;
    buckets[idx]++;
  });
  drawBars('histUser', buckets);

  // avg rating vs activity
  const map = new Map();
  for (const [u,rows] of user2rows){
    const cnt = rows.length;
    const r = rows.filter(x=>x.r>0).map(x=>x.r);
    const avgR = r.length? avg(r) : 0;
    map.set(u, {x:cnt, y:avgR});
  }
  drawScatter('scatterUser', Array.from(map.values()));
}

function drawItems(){
  // popularity hist
  const counts = Array.from(item2rows.values()).map(a=>a.length);
  const buckets = [0,0,0,0,0,0,0,0,0];
  counts.forEach(v=>{
    const idx = v===1?0 : v<=2?1 : v<=3?2 : v<=5?3 : v<=10?4 : v<=20?5 : v<=100?6 : v<=500?7 : 8;
    buckets[idx]++;
  });
  drawBars('histItem', buckets);

  // item avg rating vs popularity
  const pts = [];
  for (const [i,rows] of item2rows){
    const cnt = rows.length;
    const r = rows.filter(x=>x.r>0).map(x=>x.r);
    const avgR = r.length? avg(r) : 0;
    pts.push({x:cnt, y:avgR});
  }
  drawScatter('scatterItem', pts);

  // minutes
  drawBars('histMinutes', histogram(Array.from(items.values()).map(x=>x.minutes||0), [0,15,30,45,60,90,120,180,240]));

  // ingredients
  drawBars('histIngr', histogram(Array.from(items.values()).map(x=>x.n_ingredients||0), [0,3,5,7,9,12,15,20,30]));
}
function histogram(values, edges){
  const b = Array(edges.length).fill(0);
  for (const v of values){
    for (let i=0;i<edges.length;i++){
      if (v<=edges[i]){ b[i]++; break; }
    }
  }
  return b;
}

function drawTags(){
  // top-K
  const top = Array.from(tag2count.entries()).sort((a,b)=>b[1]-a[1]).slice(0,30);
  drawBars('barTags', top.map(([,c])=>c), {max:Math.max(1,...top.map(([,c])=>c))});

  // co-occurrence network (projected as grid heatmap-like on canvas)
  // Build lightweight adjacency for topK tags:
  const topAll = Array.from(tag2count.entries()).sort((a,b)=>b[1]-a[1]).slice(0,topK).map(([t])=>t);
  const index = new Map(topAll.map((t,i)=>[t,i]));
  const adj = Array.from({length:topAll.length},()=>Array(topAll.length).fill(0));
  for (const it of items.values()){
    const list = (it.tags||[]).filter(t=>index.has(t));
    for (let a=0;a<list.length;a++){
      for (let b=a+1;b<list.length;b++){
        const i=index.get(list[a]), j=index.get(list[b]);
        adj[i][j]++; adj[j][i]++;
      }
    }
  }
  // keep strongest edges ≥ edgeMin and at most nodeCap nodes
  const deg = adj.map(row=>row.reduce((s,v)=>s+(v>=edgeMin?1:0),0));
  const nodes = Array.from(deg.map((d,i)=>({i,d}))).sort((a,b)=>b.d-a.d).slice(0,Math.min(nodeCap,deg.length)).map(n=>n.i);
  const {ctx,w,h} = prepCanvas('tagGraph');
  const pad=30, R=Math.min(w,h)/2 - 40, cx=w/2, cy=h/2;
  ctx.fillStyle="#cbd5e1"; ctx.strokeStyle="#3b556f";
  // positions on circle
  const pos = new Map();
  nodes.forEach((ii,k)=>{ const ang=2*Math.PI*k/nodes.length; pos.set(ii,{x:cx+R*Math.cos(ang), y:cy+R*Math.sin(ang)}); });
  // edges
  ctx.globalAlpha = 0.5;
  for (const i of nodes){
    for (const j of nodes){
      if (j<=i) continue;
      if (adj[i][j] >= edgeMin){
        const a=pos.get(i), b=pos.get(j);
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;
  // nodes
  ctx.font = "12px Inter";
  nodes.forEach(i=>{
    const p = pos.get(i);
    ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill();
    const tag = topAll[i];
    ctx.fillText(tag, p.x+6, p.y+2);
  });
}

function drawTime(){
  // ratings hist duplicate
  const hist=[0,0,0,0,0]; interactions.forEach(r=>{ const v = Math.round(Math.max(1,Math.min(5, r.r||0))); if (v>=1&&v<=5) hist[v-1]++; });
  drawBars('histRatings2', hist, {max:Math.max(...hist,1)});

  // interactions by month
  const grp={}; for (const r of interactions){ if (!r.ts) continue; const d=new Date(r.ts); const k=`${d.getUTCFullYear()}-${(d.getUTCMonth()+1).toString().padStart(2,'0')}`; grp[k]=(grp[k]||0)+1; }
  const xs = Object.keys(grp).sort(); const pts = xs.map((k,i)=>({x:i+1,y:grp[k]}));
  drawLine('lineInteractions2', pts);
}

function drawLongtail(){
  const counts = Array.from(item2rows.values()).map(a=>a.length).sort((a,b)=>a-b);
  const total = counts.reduce((s,v)=>s+v,0)||1;
  let acc=0; const lor = counts.map(v=>{acc+=v; return acc/total;});
  // Lorenz
  const pts = lor.map((y,i)=>({x:i+1, y}));
  drawLine('lorenz', pts);
  // Gini (approx via Lorenz area)
  const gini = 1 - 2 * lor.reduce((s,y,i)=> s + y/(counts.length||1), 0) / (counts.length||1);
  $('giniLine').textContent = `Gini ≈ ${gini.toFixed(3)}`;
  // Pareto bar: top 1% items
  const topCut = Math.max(1, Math.floor(0.01*counts.length));
  const covered = counts.slice(-topCut).reduce((s,v)=>s+v,0)/(total||1)*100;
  $('paretoLine').textContent = `Top ${topCut} items (~1%) cover ${covered.toFixed(1)}% of interactions.`;
  drawBars('pareto', [covered, 100-covered], {max:100});
}

function drawPersonas(){
  // Build user -> dominant persona via their tagged items
  const personaCount = new Map(); const personaStats = new Map(); // key -> {users:Set, ratings:[], minutes:[]}
  const userPersona = new Map();
  for (const [u,rows] of user2rows){
    const allTags = [];
    rows.forEach(r=>{ const it=items.get(r.i); if (it && it.tags) allTags.push(...it.tags); });
    const key = personaOf(allTags);
    userPersona.set(u,key);
    if (!personaCount.has(key)) personaCount.set(key,0);
    personaCount.set(key, personaCount.get(key)+1);
  }
  // stats per persona
  for (const [u,rows] of user2rows){
    const k = userPersona.get(u)||'General';
    if (!personaStats.has(k)) personaStats.set(k,{users:new Set(), ratings:[], minutes:[], tags:new Map()});
    const st = personaStats.get(k);
    st.users.add(u);
    for (const r of rows){
      if (r.r>0) st.ratings.push(r.r);
      const it=items.get(r.i);
      if (it){
        if (it.minutes>0) st.minutes.push(it.minutes);
        for (const t of (it.tags||[])){ st.tags.set(t,(st.tags.get(t)||0)+1); }
      }
    }
  }
  // bar
  const order = Array.from(personaCount.entries()).sort((a,b)=>b[1]-a[1]);
  drawBars('barPersonas', order.map(([,c])=>c), {max:Math.max(1,...order.map(([,c])=>c))});
  // table
  $('personaTbl').innerHTML = order.map(([k,c])=>{
    const st = personaStats.get(k)||{ratings:[], minutes:[], tags:new Map(), users:new Set()};
    const topTags = Array.from(st.tags.entries()).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([t])=>t).join(', ');
    const avgR = st.ratings.length? avg(st.ratings).toFixed(2) : '—';
    const medM = st.minutes.length? median(st.minutes).toFixed(0) : '—';
    return `<tr><td>${k}</td><td>${fmt(st.users.size||0)}</td><td>${topTags||'—'}</td><td>${avgR}</td><td>${medM}</td></tr>`;
  }).join('');
}

// ---------------- event wiring ----------------
$('btnLoad').addEventListener('click', loadAll);
$('btnRedraw').addEventListener('click', ()=>{
  topK = parseInt($('kTags').value,10)||200;
  edgeMin = parseInt($('edgeMin').value,10)||40;
  nodeCap = parseInt($('nodeCap').value,10)||80;
  drawEverything();
});

async function loadAll(){
  try{
    $('status').textContent = 'Status: loading…';
    items.clear(); users.clear(); interactions.length=0; user2rows.clear(); item2rows.clear(); tag2count.clear();

    const r = await fetchFirst(['./PP_recipes.csv','./RAW_recipes.csv','data/PP_recipes.csv','data/RAW_recipes.csv']);
    const t = await fetchFirst(['./interactions_train.csv','data/interactions_train.csv']);
    if (!r || !t) throw new Error('CSV files not found near index.html or ./data/');

    parseRecipes(r.text); loadedFiles.recipes = r.path;
    parseInteractions(t.text); loadedFiles.inter = t.path;

    $('status').textContent = 'Status: loaded';
    const badges = [
      `Recipes: <b>${fmt(items.size)}</b>`,
      `Users: <b>${fmt(users.size)}</b>`,
      `Items: <b>${fmt(item2rows.size)}</b>`,
      `Interactions: <b>${fmt(interactions.length)}</b>`,
      `Density: <b>${(interactions.length/(Math.max(1,users.size)*Math.max(1,items.size))).toExponential(3)}</b>`,
      `Ratings: <b>${interactions.some(x=>x.r && !Number.isNaN(x.r))?'yes':'no'}</b>`,
      `Time: <b>${interactions.some(x=>x.ts>0)?'yes':'no'}</b>`,
      `Minutes: <b>${Array.from(items.values()).some(x=>x.minutes>0)?'yes':'no'}</b>`,
      `(files: ${loadedFiles.recipes}, ${loadedFiles.inter})`
    ];
    $('badges').innerHTML = badges.map(s=>`<span class="chip">${s}</span>`).join('');

    drawEverything();
  }catch(err){
    console.error(err);
    $('status').textContent = 'Status: failed to load. Ensure CSVs are in repo root or ./data/.';
  }
}

function drawEverything(){
  renderBadges();
  renderCounters();
  renderKPIs();
  drawOverview();
  drawUsers();
  drawItems();
  drawTags();
  drawTime();
  drawLongtail();
  drawPersonas();
}
