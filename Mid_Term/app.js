/* app.js
   Wires the UI, loads CSVs, renders EDA, and calls the models and graph modules.
   No ES modules so it works on GitHub Pages. */

(() => {
  "use strict";

  // ---------- DOM helpers ----------
  const $ = (id) => document.getElementById(id);
  const on = (el, ev, fn) => el.addEventListener(ev, fn);

  // Small canvas bar chart (no external lib)
  function drawBars(canvas, series, opts = {}) {
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext("2d");
    const W = canvas.clientWidth * dpr, H = canvas.clientHeight * dpr;
    canvas.width = W; canvas.height = H;
    ctx.clearRect(0,0,W,H);
    ctx.font = `${12*dpr}px system-ui`;
    ctx.fillStyle = "#9fb0c0";
    ctx.strokeStyle = "#243447";
    ctx.lineWidth = 1*dpr;

    const pad = 24*dpr, gap = 8*dpr;
    const n = series.length || 1;
    const barW = Math.max(1, (W - pad*2 - gap*(n-1)) / n);
    const maxY = Math.max(1, Math.max(...series.map(s => s.v)));
    for (let i=0;i<n;i++){
      const x = pad + i*(barW+gap);
      const h = Math.round((H - pad*2) * (series[i].v / maxY));
      const y = H - pad - h;
      // bar
      ctx.fillStyle = "#4aa3ff";
      ctx.fillRect(x, y, barW, h);
      // label
      ctx.fillStyle = "#9fb0c0";
      const label = String(series[i].k).slice(0,12);
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, x + (barW - tw)/2, H - pad/2);
    }
    // axes
    ctx.strokeStyle = "#243447";
    ctx.beginPath(); ctx.moveTo(pad, H-pad); ctx.lineTo(W-pad, H-pad); ctx.stroke();
  }

  // ---------- global app state ----------
  const state = {
    items: new Map(),      // itemId -> { title, tags:Array<string> }
    users: new Set(),      // user ids present (train)
    interactions: [],      // {u,i,r,ts}
    userToItems: new Map(),// user -> Map(item->rating)
    itemToUsers: new Map(),// item -> Set(user)
    tagIndex: new Map(),   // tag -> column index
    tagMatrix: null,       // Float32Array [numItems x K] (for deep tower)
    numItems: 0,
    numUsers: 0,
    loaded: false,

    // models
    baseModel: null,
    deepModel: null,

    // metrics
    baseLosses: [],
    deepLosses: [],
    proj2d: null
  };

  // ---------- error surfacing ----------
  function showError(e){
    const box = $("errBox");
    box.style.display = "block";
    box.textContent = (e && e.stack) ? e.stack : String(e);
    console.error(e);
  }
  window.addEventListener("error", (ev) => showError(ev.error || ev.message));

  // ---------- CSV parsing (robust enough for Kaggle RAW_recipes & interactions) ----------
  function parseCSV(text){
    const out = [];
    const lines = text.replace(/\r/g,"").split("\n");
    if (!lines.length) return out;
    const header = splitCSVLine(lines[0]);
    for (let i=1;i<lines.length;i++){
      const line = lines[i];
      if (!line || !line.trim()) continue;
      const cols = splitCSVLine(line);
      const row = {};
      for (let j=0;j<header.length;j++){
        row[header[j]] = cols[j] ?? "";
      }
      out.push(row);
    }
    return out;
  }
  function splitCSVLine(line){
    const res = [];
    let cur = "", inQ = false;
    for (let i=0;i<line.length;i++){
      const c = line[i];
      if (c === '"'){
        if (inQ && line[i+1] === '"'){ cur+='"'; i++; }
        else inQ = !inQ;
      } else if (c === ',' && !inQ) {
        res.push(cur); cur = "";
      } else {
        cur += c;
      }
    }
    res.push(cur);
    return res;
  }

  // ---------- data loading ----------
  async function loadData() {
    try {
      $("status").textContent = "Status: loading…";
      $("btnLoad").disabled = true;

      const [recipesTxt, interTxt] = await Promise.all([
        fetch("./data/RAW_recipes.csv").then(r => {
          if (!r.ok) throw new Error("RAW_recipes.csv not found at ./data/");
          return r.text();
        }),
        fetch("./data/interactions_train.csv").then(r => {
          if (!r.ok) throw new Error("interactions_train.csv not found at ./data/");
          return r.text();
        })
      ]);

      const recipes = parseCSV(recipesTxt);
      const inters  = parseCSV(interTxt);

      // items
      state.items.clear();
      for (const row of recipes){
        const id = Number(row.id ?? row.recipe_id);
        if (!Number.isFinite(id)) continue;
        const title = (row.name || row.title || `Recipe ${id}`).trim();
        // tags column like: "['60-minutes-or-less','time-to-make',...]"
        const tags = [];
        const raw = (row.tags || "").trim();
        if (raw) {
          const m = raw.match(/\[([^\]]*)\]/);
          const inside = m ? m[1] : raw;
          // split on quotes/comma
          inside.split(/'([^']*)'/g).forEach((tok, idx) => {
            if (idx % 2 === 1 && tok.trim()) tags.push(tok.trim());
          });
        }
        state.items.set(id, { title, tags });
      }

      // interactions (user_id,recipe_id,rating,date)
      state.interactions.length = 0;
      state.users.clear();
      state.userToItems.clear();
      state.itemToUsers.clear();

      for (const row of inters){
        const u = Number(row.user_id ?? row.userId);
        const i = Number(row.recipe_id ?? row.itemId);
        const r = Number(row.rating ?? row.score ?? 0);
        if (!Number.isFinite(u) || !Number.isFinite(i) || !state.items.has(i)) continue;
        const ts = row.date ? Date.parse(row.date) : (row.timestamp ? Number(row.timestamp) * 1000 : 0);
        state.interactions.push({u,i,r,ts});
        state.users.add(u);
        if (!state.userToItems.has(u)) state.userToItems.set(u, new Map());
        state.userToItems.get(u).set(i,r);
        if (!state.itemToUsers.has(i)) state.itemToUsers.set(i, new Set());
        state.itemToUsers.get(i).add(u);
      }

      state.numUsers = state.users.size;
      state.numItems = state.items.size;
      state.loaded = true;

      $("datasetLine").textContent =
        `Users: ${state.numUsers}  Items: ${state.numItems}  Interactions: ${state.interactions.length}`;

      // enable training now
      $("btnTrainBase").disabled = false;
      $("btnTrainDeep").disabled = false;

      // build tag index for deep tower
      buildTagIndex( Number($("dTags").value) );

      // EDA charts
      renderEDA();

      // Build co-vis graph (lightweight) for optional re-rank
      Graph.buildCoVis(state.interactions, { maxItems: state.numItems });

      $("status").textContent =
        `Status: loaded. users=${state.numUsers}, items=${state.numItems}, interactions=${state.interactions.length}`;
    } catch (e) {
      $("status").textContent = "Status: fetch failed. Ensure /data/*.csv exist (case-sensitive).";
      showError(e);
    } finally {
      $("btnLoad").disabled = false;
    }
  }

  function buildTagIndex(maxFeatures){
    // collect tag frequencies
    const freq = new Map();
    for (const {tags} of state.items.values()){
      for (const t of tags) freq.set(t, 1 + (freq.get(t) || 0));
    }
    // top K by frequency
    const top = [...freq.entries()].sort((a,b)=> b[1]-a[1]).slice(0, Math.max(10, maxFeatures|0));
    state.tagIndex.clear();
    top.forEach(([t], idx) => state.tagIndex.set(t, idx));
    const K = state.tagIndex.size;

    // sparse → dense one-hot (Float32Array numItems x K)
    state.tagMatrix = new Float32Array(state.numItems * K);
    const itemIds = [...state.items.keys()];
    const id2row = new Map(itemIds.map((id,row)=>[id,row]));
    for (const [id, info] of state.items.entries()){
      const row = id2row.get(id);
      if (row == null) continue;
      for (const t of info.tags){
        const j = state.tagIndex.get(t);
        if (j != null) state.tagMatrix[row*K + j] = 1;
      }
    }
  }

  // ---------- EDA ----------
  function renderEDA(){
    if (!state.loaded) return;

    // Ratings histogram (1..5)
    const hist = new Array(5).fill(0);
    for (const x of state.interactions){
      const r = Math.max(1, Math.min(5, Math.round(x.r || 0)));
      hist[r-1]++;
    }
    drawBars($("histRatings"), hist.map((v,i)=>({k:i+1, v})));

    // User activity (#ratings per user, binned)
    const perUser = [];
    for (const [u, m] of state.userToItems) perUser.push(m.size);
    const binsU = [1,2,3,5,10,20,50,100,200,500, 99999];
    const hu = new Array(binsU.length).fill(0);
    for (const c of perUser){
      const idx = binsU.findIndex(b => c<=b);
      hu[idx]++;
    }
    drawBars($("userActivity"), binsU.map((b,i)=>({k: i? `${binsU[i-1]+1}–${b}`: "1", v:hu[i]})));

    // Item popularity (#ratings per item)
    const perItem = [];
    for (const [i, s] of state.itemToUsers) perItem.push(s.size);
    const binsI = [1,2,3,5,10,20,50,100,200,500, 99999];
    const hi = new Array(binsI.length).fill(0);
    for (const c of perItem){
      const idx = binsI.findIndex(b => c<=b);
      hi[idx]++;
    }
    drawBars($("itemPop"), binsI.map((b,i)=>({k: i? `${binsI[i-1]+1}–${b}`: "1", v:hi[i]})));

    // Top tags
    const tf = new Map();
    for (const it of state.items.values()){
      for (const t of it.tags) tf.set(t, 1+(tf.get(t)||0));
    }
    const top = [...tf.entries()].sort((a,b)=> b[1]-a[1]).slice(0,20).map(([k,v])=>({k,v}));
    drawBars($("topTags"), top.length? top : [{k:"(no tags)", v:1}]);

    // Lorenz/Gini (approx with item popularity)
    perItem.sort((a,b)=>a-b);
    const n = perItem.length || 1;
    let cum=0, area=0;
    for (let i=0;i<n;i++){
      cum += perItem[i];
      area += cum;
    }
    const lorenzArea = area / (cum*n);
    const gini = Math.max(0, 1 - 2*lorenzArea);
    $("gini").textContent = `Gini ≈ ${gini.toFixed(3)}`;
    // simple polyline
    const lor = $("lorenz"); const ctx = lor.getContext("2d");
    const dpr = window.devicePixelRatio||1, W=lor.clientWidth*dpr, H=lor.clientHeight*dpr;
    lor.width=W; lor.height=H; ctx.clearRect(0,0,W,H);
    ctx.strokeStyle="#4aa3ff"; ctx.lineWidth=2*dpr;
    ctx.beginPath();
    ctx.moveTo(8*dpr,H-8*dpr);
    cum=0;
    for (let i=0;i<n;i++){
      cum += perItem[i];
      const x = 8*dpr + (W-16*dpr)*(i+1)/n;
      const y = (H-8*dpr) - (H-16*dpr)*(cum/perItem.reduce((a,b)=>a+b,0));
      ctx.lineTo(x,y);
    }
    ctx.stroke();

    // Cold-start counts
    const coldUsers = [...state.userToItems].filter(([u,m]) => (m.size<5)).length;
    const coldItems = [...state.itemToUsers].filter(([i,s]) => (s.size<5)).length;
    const top1pctN = Math.max(1, Math.floor(state.numItems*0.01));
    const top1Items = [...state.itemToUsers.entries()].sort((a,b)=> b[1].size - a[1].size).slice(0, top1pctN);
    const covered = new Set();
    for (const [i,setU] of top1Items) for (const u of setU) covered.add(u);
    const pareto = (100 * covered.size / state.users.size).toFixed(1) + "%";
    $("coldTable").innerHTML = `
      <tr><td>Cold users (&lt;5)</td><td>${coldUsers}</td></tr>
      <tr><td>Cold items (&lt;5)</td><td>${coldItems}</td></tr>
      <tr><td>Top 1% items (by interactions)</td><td>${top1pctN}</td></tr>
      <tr><td>Pareto: % train covered by top 1%</td><td>${pareto}</td></tr>
    `;
  }

  // ---------- training ----------
  async function trainBaseline(){
    try{
      $("btnTrainBase").disabled = true;
      const embDim = +$("bEmbDim").value|0;
      const epochs = +$("bEpochs").value|0;
      const batch  = +$("bBatch").value|0;
      const lr     = +$("bLR").value;
      const maxI   = +$("bMaxI").value|0;

      const {numUsers, numItems, batches} = buildBatches(maxI);
      state.baseModel?.dispose?.();
      state.baseModel = new TwoTowerModel(numUsers, numItems, embDim);

      const chart = $("bLoss");
      state.baseLosses.length = 0;
      $("bLine").textContent = "Training baseline…";
      await TwoTower.trainInBatchSoftmax(state.baseModel, batches, {
        epochs, batchSize: batch, lr,
        onBatchEnd: (step, loss) => {
          state.baseLosses.push(loss);
          drawLoss(chart, state.baseLosses);
          $("bLine").textContent = `Training… step ${step+1}, loss ${loss.toFixed(4)}`;
        }
      });
      $("bLine").textContent = `Baseline done. Final loss ${state.baseLosses.at(-1).toFixed(4)}`;
      // Enable test if at least baseline exists
      $("btnTest").disabled = false;

      // Project embeddings to 2D
      state.proj2d = await TwoTower.projectItems2D(state.baseModel);
      drawProjection($("proj"), state.proj2d);

    }catch(e){ showError(e); }
    finally{ $("btnTrainBase").disabled = false; }
  }

  async function trainDeep(){
    try{
      $("btnTrainDeep").disabled = true;
      const embDim = +$("dEmbDim").value|0;
      const epochs = +$("dEpochs").value|0;
      const batch  = +$("dBatch").value|0;
      const lr     = +$("dLR").value;
      const K      = +$("dTags").value|0;
      buildTagIndex(K);

      const {numUsers, numItems, batches} = buildBatches(80000); // deep usually benefits from more
      state.deepModel?.dispose?.();
      state.deepModel = new TwoTowerDeep(numUsers, numItems, embDim, state.tagMatrix, state.tagIndex.size);

      const chart = $("dLoss");
      state.deepLosses.length = 0;
      $("dLine").textContent = "Training deep…";
      await TwoTower.trainInBatchSoftmax(state.deepModel, batches, {
        epochs, batchSize: batch, lr,
        onBatchEnd: (step, loss) => {
          state.deepLosses.push(loss);
          drawLoss(chart, state.deepLosses);
          $("dLine").textContent = `Training deep… step ${step+1}, loss ${loss.toFixed(4)}`;
        }
      });
      $("dLine").textContent = `Deep done. Final loss ${state.deepLosses.at(-1).toFixed(4)}`;
      $("btnTest").disabled = false;
    }catch(e){ showError(e); }
    finally{ $("btnTrainDeep").disabled = false; }
  }

  function buildBatches(maxInteractions){
    const users = [...state.users].sort((a,b)=>a-b);
    const items = [...state.items.keys()].sort((a,b)=>a-b);
    const u2idx = new Map(users.map((u,i)=>[u,i]));
    const i2idx = new Map(items.map((it,i)=>[it,i]));
    const inter = maxInteractions>0 ? state.interactions.slice(0, maxInteractions) : state.interactions;
    const batches = inter.map(({u,i}) => ({u: u2idx.get(u), i: i2idx.get(i)}));
    return { numUsers: users.length, numItems: items.length, batches };
  }

  function drawLoss(canvas, arr){
    const series = arr.map((v,i)=>({k:i+1, v}));
    drawBars(canvas, series);
  }

  // ---------- demo / infer ----------
  async function runDemo(){
    try{
      $("btnTest").disabled = true;
      $("testLine").textContent = "Sampling a user with ≥20 ratings…";

      // pick a user with >=20 ratings
      const goodUsers = [...state.userToItems.entries()].filter(([u,m]) => m.size >= 20);
      if (!goodUsers.length){
        $("testLine").textContent = "No user with ≥20 ratings in this split.";
        return;
      }
      const [user, m] = goodUsers[ Math.floor(Math.random()*goodUsers.length) ];
      $("testLine").textContent = `User ${user} selected (${m.size} ratings). Scoring…`;

      const users = [...state.users].sort((a,b)=>a-b);
      const items = [...state.items.keys()].sort((a,b)=>a-b);
      const u2idx = new Map(users.map((u,i)=>[u,i]));
      const i2idx = new Map(items.map((it,i)=>[it,i]));
      const seen = new Set(state.userToItems.get(user)?.keys() || []);

      // history top-10
      const hist = [...(state.userToItems.get(user) || new Map()).entries()]
                   .sort((a,b)=> b[1]-a[1]).slice(0,10);
      $("tHistory").innerHTML = hist.map(([iid,r],k)=>
        `<tr><td>${k+1}</td><td>${escapeHTML(state.items.get(iid)?.title || iid)}</td><td>${r}</td></tr>`
      ).join("");

      // Baseline scores
      if (state.baseModel){
        const uIdx = u2idx.get(user);
        const scores = await TwoTower.scoreAllItems(state.baseModel, uIdx);
        const baseTop = topKExclude(scores, items, seen, 10);
        $("tBase").innerHTML = baseTop.map(([iid,sc],k)=>
          `<tr><td>${k+1}</td><td>${escapeHTML(state.items.get(iid)?.title || iid)}</td><td>${sc.toFixed(3)}</td></tr>`
        ).join("");
      } else {
        $("tBase").innerHTML = `<tr><td colspan="3" class="muted">Train baseline first.</td></tr>`;
      }

      // Deep scores
      if (state.deepModel){
        const uIdx = u2idx.get(user);
        const scores = await TwoTower.scoreAllItems(state.deepModel, uIdx);
        let deepTop = topKExclude(scores, items, seen, 10);

        // Optional graph re-rank
        if ($("useGraph").checked){
          deepTop = Graph.rerankWithPPR(user, deepTop, state.userToItems, 0.15);
        }

        $("tDeep").innerHTML = deepTop.map(([iid,sc],k)=>
          `<tr><td>${k+1}</td><td>${escapeHTML(state.items.get(iid)?.title || iid)}</td><td>${sc.toFixed(3)}</td></tr>`
        ).join("");
      } else {
        $("tDeep").innerHTML = `<tr><td colspan="3" class="muted">Train deep model to see this.</td></tr>`;
      }

      $("testLine").textContent = "Recommendations generated successfully!";
    }catch(e){ showError(e); }
    finally{ $("btnTest").disabled = false; }
  }

  function topKExclude(scores, itemIds, seenSet, K){
    const arr = itemIds.map((iid, idx)=> [iid, scores[idx]]);
    arr.sort((a,b)=> b[1]-a[1]);
    const out=[];
    for (const [iid, sc] of arr){
      if (seenSet.has(iid)) continue;
      out.push([iid, sc]);
      if (out.length>=K) break;
    }
    return out;
  }

  function drawProjection(canvas, pts){
    if (!pts) return;
    const dpr = window.devicePixelRatio||1, ctx = canvas.getContext("2d");
    const W = canvas.clientWidth*dpr, H=canvas.clientHeight*dpr;
    canvas.width=W; canvas.height=H; ctx.clearRect(0,0,W,H);
    // normalize to 5% padding
    let minX=+Infinity, maxX=-Infinity, minY=+Infinity, maxY=-Infinity;
    for (const [x,y] of pts){ if(x<minX)minX=x; if(x>maxX)maxX=x; if(y<minY)minY=y; if(y>maxY)maxY=y; }
    const pad=20*dpr;
    for (const [x,y] of pts){
      const X = pad + (W-2*pad) * (x - minX) / (maxX - minX + 1e-8);
      const Y = pad + (H-2*pad) * (y - minY) / (maxY - minY + 1e-8);
      ctx.fillStyle="#4aa3ff";
      ctx.fillRect(X, H-Y, 2*dpr, 2*dpr);
    }
  }

  // ---------- utils ----------
  function escapeHTML(s){ return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

  // tabs
  function setupTabs(){
    const tabs = document.querySelectorAll(".tab");
    tabs.forEach(btn => on(btn, "click", () => {
      tabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      for (const sec of ["eda","models","demo","metrics"]){
        $(sec).classList.toggle("hide", sec !== btn.dataset.tab);
      }
    })));
  }

  // ---------- boot ----------
  on(document, "DOMContentLoaded", () => {
    setupTabs();
    on($("btnLoad"), "click", () => loadData().catch(showError));
    on($("btnTrainBase"), "click", () => trainBaseline().catch(showError));
    on($("btnTrainDeep"), "click", () => trainDeep().catch(showError));
    on($("btnTest"), "click", () => runDemo().catch(showError));
    $("status").textContent = "Status: ready. Click “Load Data”.";
  });

})();
