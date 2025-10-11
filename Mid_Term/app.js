/* app.js — data load, EDA, training loops, testing UI
   Works with /data/recipes.csv (or RAW_recipes.csv) and /data/interactions_train.csv (or interactions.csv)
   No build step; pure client-side.
*/

/* ----------------------- Globals & helpers ----------------------- */
const qs = (s) => document.querySelector(s);
const byId = (s) => document.getElementById(s);

let RECIPES = new Map();  // itemId -> {title, tags:Set<string>}
let ITEMS = [];           // [{id, title}]
let INTERACTIONS = [];    // [{userId, itemId, rating, ts}]
let user2idx = new Map(), item2idx = new Map();
let idx2user = [], idx2item = [];

let perUser = new Map();  // userId -> [{itemId, rating, ts}]
let perItem = new Map();  // itemId -> count
let tagCounts = new Map();

let baselineModel = null;
let deepModel = null;

let itemTagVocab = [];    // top-K tags used by Deep tower
let itemTagIndex = new Map(); // tag -> idx
let itemTagMatrix = null; // Float32Array length = ITEMS * K

// Drawing util (simple 2D canvas)
function clearCanvas(c){ const g=c.getContext('2d'); g.clearRect(0,0,c.width,c.height); }

/* drawBars: bars = [{x,label,val}]  */
function drawBars(canvas, arr, {yMax=null, yTicks=4, rotate=false}={}){
  const g = canvas.getContext('2d');
  g.clearRect(0,0,canvas.width,canvas.height);
  const W = canvas.width, H = canvas.height, P=36;
  const n = arr.length;
  const maxV = yMax ?? Math.max(1, ...arr.map(a=>a.val));
  const bw = (W - P*2) / n;
  g.fillStyle = '#ffffff';
  g.strokeStyle = '#3a5285';
  g.lineWidth = 1;

  // grid
  g.globalAlpha = 0.2;
  for(let t=0;t<=yTicks;t++){
    const y = H-P - (H-P*2) * (t/yTicks);
    g.beginPath(); g.moveTo(P,y); g.lineTo(W-P,y); g.stroke();
  }
  g.globalAlpha = 1;

  // bars
  for(let i=0;i<n;i++){
    const h = ((H-P*2) * (arr[i].val / maxV));
    g.fillRect(P + i*bw + 2, H-P-h, Math.max(1,bw-4), h);
  }

  // x labels
  g.fillStyle = '#9fb0d9';
  g.font = '12px system-ui';
  g.textAlign='center';
  if (n<=60){ // avoid clutter with big N
    for(let i=0;i<n;i++){
      const x = P + i*bw + bw/2;
      if(rotate){
        g.save(); g.translate(x,H-P+12); g.rotate(-Math.PI/3);
        g.fillText(arr[i].label, 0, 0); g.restore();
      } else {
        g.fillText(arr[i].label, x, H-P+14);
      }
    }
  }
}

/* drawLine: y array in [0..1] normalized; will scale automatically */
function drawLine(canvas, y){
  const g = canvas.getContext('2d');
  g.clearRect(0,0,canvas.width,canvas.height);
  const W = canvas.width, H = canvas.height, P = 20;
  const n = y.length;
  if(!n){ return; }
  const maxV = Math.max(...y), minV=Math.min(...y);
  const scale = (v)=> H-P - (H-P*2)*( (v-minV) / Math.max(1e-9, (maxV-minV)) );

  g.strokeStyle = '#60a5fa';
  g.lineWidth = 2;
  g.beginPath();
  for(let i=0;i<n;i++){
    const x = P + (W-P*2)*(i/Math.max(1,n-1));
    const yy = scale(y[i]);
    if(i===0) g.moveTo(x,yy); else g.lineTo(x,yy);
  }
  g.stroke();
}

/* ----------------------- CSV parsing ----------------------- */
async function fetchCSV(path){
  const resp = await fetch(path);
  if(!resp.ok) throw new Error(`fetch failed: ${path}`);
  const text = await resp.text();
  return new Promise((resolve)=>{
    Papa.parse(text, { header:true, skipEmptyLines:true, dynamicTyping:true, complete:(r)=>resolve(r.data) });
  });
}

/* Parse recipes: accept columns:
   - id or recipe_id
   - name or title
   - tags (JSON-ish: "['easy','vegan']" or '[]')
*/
function parseRecipes(rows){
  RECIPES.clear(); ITEMS.length=0; tagCounts.clear();
  for(const r of rows){
    const id = r.id ?? r.recipe_id ?? r.RecipeId ?? r.item_id;
    const title = (r.name ?? r.title ?? `Recipe ${id}`) + '';
    if(id == null) continue;

    // tags
    let raw = r.tags ?? r.Tags ?? '';
    let tags = [];
    if (typeof raw === 'string'){
      let s = raw.trim();
      if (s.startsWith('[') && s.endsWith(']')){
        try{
          // fix single quotes
          s = s.replace(/'/g, '"');
          const list = JSON.parse(s);
          if (Array.isArray(list)) tags = list.map(x=>String(x).toLowerCase());
        }catch(_){}
      } else if (s){
        tags = s.split(',').map(x=>x.trim().toLowerCase()).filter(Boolean);
      }
    }

    const tagSet = new Set(tags);
    for(const t of tagSet){ tagCounts.set(t, (tagCounts.get(t)||0)+1); }

    RECIPES.set(Number(id), { title, tags: tagSet });
  }
  // ITEMS array for stable ordering
  ITEMS = Array.from(RECIPES.entries()).map(([id,obj])=>({id, title:obj.title}));
}

/* Parse interactions: accept columns:
   - user_id, recipe_id/item_id, rating, date/timestamp
*/
function parseInteractions(rows){
  INTERACTIONS.length=0; perUser.clear(); perItem.clear();
  for(const r of rows){
    const u = r.user_id ?? r.user ?? r.uid ?? r.UserId ?? r.profile_id;
    const it = r.recipe_id ?? r.item_id ?? r.RecipeId ?? r.id;
    const rating = Number(r.rating ?? r.score ?? r.stars ?? 1);
    let ts = r.timestamp ?? r.date ?? r.time;
    if (typeof ts === 'string') ts = Date.parse(ts) || 0;
    ts = Number(ts)||0;
    if (u==null || it==null) continue;
    const userId = Number(u), itemId = Number(it);
    INTERACTIONS.push({ userId, itemId, rating, ts });

    if(!perUser.has(userId)) perUser.set(userId, []);
    perUser.get(userId).push({ itemId, rating, ts });

    perItem.set(itemId, (perItem.get(itemId)||0)+1);
  }
}

/* Build indexers */
function buildIndexers(){
  const users = Array.from(new Set(INTERACTIONS.map(x=>x.userId))).sort((a,b)=>a-b);
  const items = Array.from(new Set(INTERACTIONS.map(x=>x.itemId))).sort((a,b)=>a-b);
  user2idx = new Map(users.map((u,i)=>[u,i])); idx2user = users;
  item2idx = new Map(items.map((m,i)=>[m,i]));  idx2item = items;
}

/* Tag vocab for Deep tower (top-K by df) */
function buildTagVocab(k=200){
  const arr = Array.from(tagCounts.entries()).sort((a,b)=>b[1]-a[1]).slice(0,k);
  itemTagVocab = arr.map(([t])=>t);
  itemTagIndex = new Map(itemTagVocab.map((t,i)=>[t,i]));

  const K = itemTagVocab.length;
  const M = idx2item.length;
  itemTagMatrix = new Float32Array(M*K); // dense 0/1
  for(let m=0;m<M;m++){
    const itemId = idx2item[m];
    const info = RECIPES.get(itemId);
    if(!info) continue;
    for(const t of info.tags){
      const j = itemTagIndex.get(t);
      if(j != null) itemTagMatrix[m*K + j] = 1;
    }
  }
}

/* ----------------------- EDA visuals & metrics ----------------------- */
function computeEDA(){
  // counters
  const nUsers = user2idx.size;
  const nItems = item2idx.size;
  const nInt   = INTERACTIONS.length;
  const density = (nInt / Math.max(1, nUsers*nItems)).toExponential(2);
  const ratingsPresent = INTERACTIONS.some(x=>x.rating!=null) ? 'yes' : 'no';

  // cold start (users/items with <5)
  let coldUsers=0, coldItems=0;
  for(const [u,arr] of perUser) if(arr.length<5) coldUsers++;
  for(const [it,c] of perItem) if(c<5) coldItems++;

  byId('eda-counters').textContent =
    `Users: ${nUsers}  Items: ${nItems}  Interactions: ${nInt}  ` +
    `Density: ${density}  Ratings present: ${ratingsPresent}  ` +
    `Cold users (<5): ${coldUsers}  Cold items (<5): ${coldItems}`;

  // ratings histogram 1..5
  const hist = [0,0,0,0,0];
  for(const x of INTERACTIONS){
    const r = Math.max(1, Math.min(5, Math.round(Number(x.rating)||1)));
    hist[r-1]++;
  }
  drawBars(byId('chart-ratings'),
    hist.map((v,i)=>({label:String(i+1), val:v})));

  // top 20 tags
  const topTags = Array.from(tagCounts.entries()).sort((a,b)=>b[1]-a[1]).slice(0,20);
  drawBars(byId('chart-tags'),
    topTags.map(([t,c])=>({label:t, val:c})), {rotate:true});

  // user activity (bucketed)
  const userLens = Array.from(perUser.values()).map(a=>a.length).sort((a,b)=>a-b);
  const ua = bucketCounts(userLens, [1,2,3,5,10,20,50,100,200]);
  drawBars(byId('chart-user-activity'), ua.map(([lab,c])=>({label:lab,val:c})));

  // item popularity (bucketed)
  const itemLens = Array.from(perItem.values()).map(x=>x).sort((a,b)=>a-b);
  const ip = bucketCounts(itemLens, [1,2,3,5,10,20,50,100,200,500]);
  drawBars(byId('chart-item-pop'), ip.map(([lab,c])=>({label:lab,val:c})));

  // Lorenz curve + gini
  const sorted = itemLens.slice().sort((a,b)=>a-b);
  const total = sorted.reduce((a,b)=>a+b,0) || 1;
  const cum = []; let s=0;
  for(let i=0;i<sorted.length;i++){ s+=sorted[i]; cum.push(s/total); }
  const pts = [0, ...cum.map((v,i)=>({x:(i+1)/sorted.length, y:v}))];
  drawLorenz(byId('chart-lorenz'), pts);
  const gini = 1 - (2 / sorted.length) * cum.reduce((acc,v,i)=>acc + v, 0);
  byId('gini').textContent = `Gini ≈ ${gini.toFixed(3)}`;

  // cold table
  const tbody = byId('cold-table').querySelector('tbody'); tbody.innerHTML='';
  addRow(tbody, 'Cold users (&lt;5)', coldUsers);
  addRow(tbody, 'Cold items (&lt;5)', coldItems);
  const top1pctN = Math.max(1, Math.round(nItems*0.01));
  const topItems = Array.from(perItem.entries()).sort((a,b)=>b[1]-a[1]).slice(0, top1pctN);
  const topCover = (topItems.reduce((a,[,c])=>a+c,0)/Math.max(1,INTERACTIONS.length))*100;
  addRow(tbody, 'Top 1% items (by interactions)', top1pctN);
  addRow(tbody, 'Pareto: % train covered by top 1%', `${topCover.toFixed(1)}%`);

  function addRow(tb, k, v){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
    tb.appendChild(tr);
  }
}

function bucketCounts(vals, cuts){
  const labs = [], counts = Array(cuts.length+1).fill(0);
  for(let i=0;i<cuts.length;i++){
    labs.push(i===0? `${1}` : `${cuts[i-1]+1}–${cuts[i]}`);
  }
  labs.push(`>${cuts[cuts.length-1]}`);
  for(const v of vals){
    let idx = cuts.findIndex(c=>v<=c);
    if(idx===-1) idx = cuts.length;
    counts[idx]++;
  }
  return labs.map((lab,i)=>[lab,counts[i]]);
}

function drawLorenz(canvas, pts){
  const g = canvas.getContext('2d');
  g.clearRect(0,0,canvas.width,canvas.height);
  const W=canvas.width, H=canvas.height, P=28;

  const X = (t)=> P + (W-P*2)*t;
  const Y = (t)=> H-P - (H-P*2)*t;

  // diagonal
  g.strokeStyle = '#3a5285'; g.lineWidth=1; g.globalAlpha=0.5;
  g.beginPath(); g.moveTo(P,H-P); g.lineTo(W-P,P); g.stroke(); g.globalAlpha=1;

  // curve
  g.strokeStyle = '#34d399'; g.lineWidth=2;
  g.beginPath(); g.moveTo(P, H-P);
  for(const p of pts){ g.lineTo(X(p.x), Y(p.y)); }
  g.stroke();
}

/* ----------------------- Training data prep ----------------------- */
function buildIndexArrays(limit=80000){
  // Select positives (keep rating >= 4 strongly; include some 3s)
  const positives = [];
  for(const x of INTERACTIONS){
    if (x.rating >= 4) positives.push(x);
    else if (x.rating === 3 && Math.random()<0.3) positives.push(x);
  }
  const shuffled = positives.sort(()=>Math.random()-0.5).slice(0, limit);
  const u = new Int32Array(shuffled.length);
  const i = new Int32Array(shuffled.length);
  for(let k=0;k<shuffled.length;k++){
    u[k] = user2idx.get(shuffled[k].userId);
    i[k] = item2idx.get(shuffled[k].itemId);
  }
  return {u,i,count:shuffled.length};
}

/* ----------------------- UI: Load ----------------------- */
async function handleLoad(){
  const s = byId('status-load');
  s.textContent = 'Status: loading…';

  // Recipes first (prefer recipes.csv, then RAW_recipes.csv)
  let recipesRows = null;
  try { recipesRows = await fetchCSV('data/recipes.csv'); }
  catch(_){ try{ recipesRows = await fetchCSV('data/RAW_recipes.csv'); } catch(e){ s.textContent = 'Status: failed to load recipes CSV'; return; } }
  parseRecipes(recipesRows);

  // Interactions (prefer interactions_train.csv, then interactions.csv)
  let ixRows = null;
  try { ixRows = await fetchCSV('data/interactions_train.csv'); }
  catch(_){ try{ ixRows = await fetchCSV('data/interactions.csv'); } catch(e){ s.textContent = 'Status: failed to load interactions CSV'; return; } }
  parseInteractions(ixRows);

  buildIndexers();

  // (Optional) cap items used to those present in interactions
  ITEMS = idx2item.map(id => ({id, title: RECIPES.get(id)?.title || `Recipe ${id}`}));

  computeEDA();

  s.textContent = `Status: loaded. users=${user2idx.size}, items=${item2idx.size}, interactions=${INTERACTIONS.length}`;
}

/* ----------------------- UI: Train Baseline ----------------------- */
async function handleTrainBaseline(){
  if (user2idx.size===0 || item2idx.size===0){ byId('status-b').textContent='Load data first.'; return; }
  const emb = parseInt(byId('emb-b').value,10);
  const ep  = parseInt(byId('ep-b').value,10);
  const bs  = parseInt(byId('bs-b').value,10);
  const lr  = Number(byId('lr-b').value);
  const maxint = parseInt(byId('maxint').value,10);

  const {u,i,count} = buildIndexArrays(maxint);
  const chart = byId('loss-b'); drawLine(chart, []);
  byId('status-b').textContent = `Training baseline… (samples=${count})`;

  // (re)create model
  baselineModel?.dispose();
  baselineModel = new TwoTowerModel(user2idx.size, item2idx.size, emb, lr);

  const losses = [];
  await baselineModel.train(u,i,{epochs:ep,batchSize:bs, onBatch:(k,loss)=>{
    losses.push(loss); drawLine(chart,losses);
    if(k%20===0) byId('status-b').textContent = `Training baseline… step ${k}, loss ${loss.toFixed(4)}`;
  }});
  byId('status-b').textContent = `Baseline done. Final loss ${losses.at(-1)?.toFixed(4)}`;

  await drawProjection(baselineModel.getItemEmbeddingMatrix());
}

/* ----------------------- UI: Train Deep ----------------------- */
async function handleTrainDeep(){
  if (user2idx.size===0 || item2idx.size===0){ byId('status-d').textContent='Load data first.'; return; }
  const emb = parseInt(byId('emb-d').value,10);
  const ep  = parseInt(byId('ep-d').value,10);
  const bs  = parseInt(byId('bs-d').value,10);
  const lr  = Number(byId('lr-d').value);
  const K   = parseInt(byId('tags-k').value,10);

  buildTagVocab(K);

  const {u,i,count} = buildIndexArrays(parseInt(byId('maxint').value,10));
  const chart = byId('loss-d'); drawLine(chart, []);
  byId('status-d').textContent = `Training deep… (K=${K}, samples=${count})`;

  deepModel?.dispose();
  deepModel = new TwoTowerDeepModel({
    numUsers:user2idx.size, numItems:item2idx.size, embDim:emb, lr,
    tagVocabSize:itemTagVocab.length, itemTagMatrix, // dense [M*K]
  });

  const losses=[];
  await deepModel.train(u,i,{epochs:ep,batchSize:bs, onBatch:(k,loss)=>{
    losses.push(loss); drawLine(chart,losses);
    if(k%20===0) byId('status-d').textContent = `Training deep… step ${k}, loss ${loss.toFixed(4)}`;
  }});
  byId('status-d').textContent = `Deep model done. Final loss ${losses.at(-1)?.toFixed(4)}`;

  await drawProjection(deepModel.getItemEmbeddingMatrix());
}

/* ----------------------- PCA / projection ----------------------- */
async function drawProjection(itemEmbMatrix){
  // itemEmbMatrix: tf.Variable [numItems, embDim]
  const c = byId('proj'); const g = c.getContext('2d');
  clearCanvas(c);

  await tf.nextFrame();
  const coords = tf.tidy(()=>{
    const E = itemEmbMatrix;                // [M,D]
    const { v } = tf.linalg.svd(E, true);   // V: [D,D]
    const V2 = v.slice([0,0],[v.shape[0],2]); // take first 2 PCs
    const XY = E.matMul(V2);                // [M,2]
    return XY;
  });
  const xy = await coords.array();
  coords.dispose();

  // normalize to canvas
  const xs = xy.map(p=>p[0]); const ys = xy.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs),
        minY=Math.min(...ys), maxY=Math.max(...ys);
  const P=24, W=c.width, H=c.height;
  const SX=(x)=> P + (W-P*2) * ((x-minX)/Math.max(1e-9,maxX-minX));
  const SY=(y)=> H-P - (H-P*2) * ((y-minY)/Math.max(1e-9,maxY-minY));

  g.fillStyle='#a6b7e8';
  for(let m=0;m<xy.length;m++){
    const x = SX(xy[m][0]), y = SY(xy[m][1]);
    g.fillRect(x-1,y-1,2,2);
  }
}

/* ----------------------- Demo / Test ----------------------- */
function pickRandomQualifiedUser(minRatings=20){
  const pool = Array.from(perUser.entries()).filter(([u,arr])=>arr.length>=minRatings);
  if(!pool.length) return null;
  const [userId, arr] = pool[Math.floor(Math.random()*pool.length)];
  // top-10 history by rating desc, then recent ts
  const hist = arr.slice().sort((a,b)=> (b.rating-a.rating) || (b.ts-a.ts)).slice(0,10);
  return { userId, hist };
}

async function handleTest(){
  const s = byId('status-test');
  if (!baselineModel && !deepModel){ s.textContent='Train at least one model first.'; return; }

  const pick = pickRandomQualifiedUser(20);
  if (!pick){ s.textContent='No user with ≥20 ratings in this split.'; return; }
  const userId = pick.userId; const seen = new Set(perUser.get(userId).map(x=>x.itemId));

  // history table
  fillHistTable(pick.hist);

  // scores: baseline / deep
  const b = baselineModel ? await scoreUser(userId, baselineModel, seen) : [];
  const d = deepModel ? await scoreUser(userId, deepModel, seen) : [];

  const useGraph = byId('use-graph').checked;
  let b2=b, d2=d;
  if (useGraph){
    // build graph once (cached in graph.js global)
    ensureGraph(INTERACTIONS);
    const seeds = pick.hist.map(x=>x.itemId);
    const ppr = personalizedPageRank(seeds, 0.15, 30);
    b2 = rerankWithGraph(b, ppr, 0.3);
    d2 = rerankWithGraph(d, ppr, 0.3);
  }

  fillRecTable('rec-b', b2);
  fillRecTable('rec-d', d2);
  s.textContent = 'Status: recommendations generated successfully!';
}

function fillHistTable(arr){
  const tb = byId('hist-table').querySelector('tbody'); tb.innerHTML='';
  arr.forEach((x,idx)=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${idx+1}</td><td>${(RECIPES.get(x.itemId)?.title)||x.itemId}</td><td>${x.rating}</td>`;
    tb.appendChild(tr);
  });
}

function fillRecTable(id, arr){
  const tb = byId(id).querySelector('tbody'); tb.innerHTML='';
  arr.slice(0,10).forEach((x,idx)=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${idx+1}</td><td>${RECIPES.get(idx2item[x.idx])?.title || idx2item[x.idx]}</td><td>${x.score.toFixed(4)}</td>`;
    tb.appendChild(tr);
  });
}

async function scoreUser(userId, model, seen){
  const uIdx = user2idx.get(userId);
  if (uIdx==null) return [];
  const uEmb = model.getUserEmbedding(uIdx); // tf.Tensor [1,D] or [D]
  const S = tf.tidy(()=>{
    const all = model.getItemEmbeddingMatrix(); // [M,D]
    const U = uEmb.reshape([1,-1]);             // [1,D]
    const logits = U.matMul(all.transpose());   // [1,M]
    return logits.squeeze();
  });
  const sc = Array.from(await S.data());
  S.dispose(); uEmb.dispose && uEmb.dispose();

  // produce sorted indices excluding seen
  const res = [];
  for(let m=0;m<sc.length;m++){
    const itemId = idx2item[m];
    if (seen.has(itemId)) continue;
    res.push({idx:m, score: sc[m]});
  }
  res.sort((a,b)=>b.score-a.score);
  return res.slice(0,50);
}

/* ----------------------- Wire up ----------------------- */
function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.tab===name));
  ['eda','models','demo','metrics'].forEach(k=>{
    byId(`tab-${k}`).style.display = (k===name? 'block':'none');
  });
}
document.querySelectorAll('.tab').forEach(btn=>{
  btn.addEventListener('click', ()=> switchTab(btn.dataset.tab));
});

byId('btn-load').addEventListener('click', handleLoad);
byId('btn-train-b').addEventListener('click', handleTrainBaseline);
byId('btn-train-d').addEventListener('click', handleTrainDeep);
byId('btn-test').addEventListener('click', handleTest);

// initial
switchTab('eda');
