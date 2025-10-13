/* eda.js — 2025-grade, static EDA for Food Recommenders
   Works on GitHub Pages. No servers. Put CSVs in repo root or /data/.
   Data expected:
     - RAW_recipes.csv (preferred) or PP_recipes.csv
       id, name/title, tags (python-list string), minutes, n_ingredients
     - interactions_train.csv
       user_id, item_id, rating, date (timestamp or parseable string)
*/

(() => {
  // ---------- DOM helpers ----------
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // ---------- state ----------
  let recipes = new Map(); // id -> {title,tags[],minutes,n_ingredients}
  let interactions = [];   // {u,i,r,ts,date}
  let user2items = new Map(); // u -> array {i,r,ts}
  let item2users = new Map(); // i -> Set(u)
  let tagFreq = new Map();
  let minutesAvail = false, nIngAvail = false, ratingsAvail = false, datesAvail = false;

  // ---------- file loading ----------
  async function fetchTextFirst(paths) {
    for (const p of paths) {
      try {
        const res = await fetch(p, { cache: "no-store" });
        if (res.ok) {
          const text = await res.text();
          return { path: p, text };
        }
      } catch (_) {}
    }
    return null;
  }

  function parseCSV(text) {
    return new Promise((resolve) => {
      Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        fastMode: false,
        complete: (out) => resolve(out.data),
      });
    });
  }

  function detectCol(cols, wanted) {
    // wanted: array of regexps to try in order
    const lower = cols.map((c) => c.toLowerCase());
    for (const rx of wanted) {
      const ix = lower.findIndex((c) => rx.test(c));
      if (ix >= 0) return cols[ix];
    }
    return null;
  }

  function safeListCell(v) {
    if (v == null) return [];
    if (Array.isArray(v)) return v;
    let s = String(v).trim();
    // remove outer quotes if any
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) s = s.slice(1, -1);
    // strip brackets
    if (s.startsWith("[") && s.endsWith("]")) s = s.slice(1, -1);
    if (!s) return [];
    return s
      .split(/['"]\s*,\s*['"]|,\s*/g)
      .map((x) => x.replace(/^['"]|['"]$/g, "").trim())
      .filter(Boolean);
  }

  function toTs(v) {
    if (v == null || v === "") return 0;
    if (typeof v === "number" && isFinite(v)) {
      // numeric epoch? accept seconds or ms range
      if (v > 1e12) return Math.floor(v / 1000);
      if (v > 1e10) return Math.floor(v / 1000);
      return Math.floor(v);
    }
    const t = Date.parse(String(v));
    return isNaN(t) ? 0 : Math.floor(t / 1000);
  }

  async function loadAll() {
    $('status').textContent = 'Status: searching files…';

    const recTry = [
      './RAW_recipes.csv','RAW_recipes.csv','data/RAW_recipes.csv',
      './PP_recipes.csv','PP_recipes.csv','data/PP_recipes.csv'
    ];
    const intTry = [
      './interactions_train.csv','interactions_train.csv','data/interactions_train.csv'
    ];

    const rec = await fetchTextFirst(recTry);
    const inte = await fetchTextFirst(intTry);

    if (!rec || !inte) {
      $('status').textContent = 'Status: missing CSVs. Expect RAW_recipes/PP_recipes + interactions_train.';
      throw new Error('Missing files');
    }
    $('status').textContent = 'Status: parsing…';

    const recRows = await parseCSV(rec.text);
    const intRows = await parseCSV(inte.text);

    // detect recipe columns
    const recCols = recRows.length ? Object.keys(recRows[0]) : [];
    const idCol = detectCol(recCols, [/^id$|(^|_)id$/i, /(recipe|item)_?id/i]) || recCols[0];
    const nameCol = detectCol(recCols, [/name|title/i]) || idCol;
    const tagsCol = detectCol(recCols, [/tags/i]);
    const minutesCol = detectCol(recCols, [/minute/i]);
    const ningCol = detectCol(recCols, [/n_ingredients/i, /ingredients?(_count)?/i]);

    recipes.clear();
    tagFreq.clear();
    minutesAvail = !!minutesCol;
    nIngAvail = !!ningCol;

    for (const row of recRows) {
      const id = parseInt(row[idCol], 10);
      if (!Number.isInteger(id)) continue;
      const title = (row[nameCol] ?? `Recipe ${id}`) + '';
      const tags = tagsCol ? safeListCell(row[tagsCol]).slice(0, 48) : [];
      const minutes = minutesCol ? Number(row[minutesCol]) : null;
      const ning = ningCol ? Number(row[ningCol]) : null;
      recipes.set(id, { title, tags, minutes, n_ingredients: ning });
      for (const t of tags) tagFreq.set(t, (tagFreq.get(t) || 0) + 1);
    }

    // detect interaction columns
    const intCols = intRows.length ? Object.keys(intRows[0]) : [];
    const uCol = detectCol(intCols, [/user/i]);
    const iCol = detectCol(intCols, [/(item|recipe)_?id/i]) || detectCol(intCols, [/id/i]);
    const rCol = detectCol(intCols, [/rating|stars|score/i]);
    const dCol = detectCol(intCols, [/time|date/i]);

    if (!uCol || !iCol) throw new Error('user_id or item_id missing in interactions');

    interactions.length = 0;
    user2items.clear();
    item2users.clear();
    ratingsAvail = !!rCol;
    datesAvail = !!dCol;

    // keep only interactions for recipes we have
    for (const row of intRows) {
      const u = Number(row[uCol]);
      const i = Number(row[iCol]);
      if (!Number.isFinite(u) || !Number.isFinite(i)) continue;
      if (!recipes.has(i)) continue;
      const r = rCol ? Number(row[rCol]) : null;
      const ts = dCol ? toTs(row[dCol]) : 0;
      const obj = { u, i, r, ts, date: dCol ? row[dCol] : null };
      interactions.push(obj);

      if (!user2items.has(u)) user2items.set(u, []);
      user2items.get(u).push({ i, r, ts });
      if (!item2users.has(i)) item2users.set(i, new Set());
      item2users.get(i).add(u);
    }

    // summary KPIs
    const nUsers = user2items.size;
    const nItems = item2users.size;
    const nInter = interactions.length;
    const density = nInter / Math.max(1, nUsers * nItems);
    const userCounts = Array.from(user2items.values(), arr => arr.length);
    const itemCounts = Array.from(item2users.values(), s => s.size);
    const coldUsers = userCounts.filter(c => c < 5).length;
    const coldItems = itemCounts.filter(c => c < 5).length;

    $('fileLine').innerHTML =
      `<span class="chip">Recipes: <b>${fmt(recipes.size)}</b></span>
       <span class="chip">Users: <b>${fmt(nUsers)}</b></span>
       <span class="chip">Items: <b>${fmt(nItems)}</b></span>
       <span class="chip">Interactions: <b>${fmt(nInter)}</b></span>
       <span class="chip">Density: <b>${density.toExponential(2)}</b></span>
       <span class="chip">Ratings: <b>${ratingsAvail?'yes':'no'}</b></span>
       <span class="chip">Time: <b>${datesAvail?'yes':'no'}</b></span>
       <span class="chip">Minutes: <b>${minutesAvail?'yes':'no'}</b></span>
       <span class="chip">n_ingredients: <b>${nIngAvail?'yes':'no'}</b></span>
       <span class="chip">Cold users&lt;5: <b>${fmt(coldUsers)}</b></span>
       <span class="chip">Cold items&lt;5: <b>${fmt(coldItems)}</b></span>`;

    $('status').textContent = 'Status: loaded';
    drawAll();
  }

  // ---------- math utils ----------
  function giniFromCounts(countsAsc) {
    const n = countsAsc.length;
    if (!n) return 0;
    const total = countsAsc.reduce((s, v) => s + v, 0);
    if (total === 0) return 0;
    // trapezoid area between (0,0) and (1,1)
    let cum = 0, area = 0;
    let prevX = 0, prevY = 0;
    for (let i = 0; i < n; i++) {
      cum += countsAsc[i];
      const x = (i + 1) / n;
      const y = cum / total;
      area += 0.5 * (y + prevY) * (x - prevX);
      prevX = x; prevY = y;
    }
    const g = 1 - 2 * area;
    return Math.max(0, Math.min(1, g));
  }

  function groupTs(ts, mode='month') {
    const d = new Date(ts * 1000);
    if (mode === 'hour') return d.toISOString().slice(0, 13) + ':00';
    if (mode === 'day')  return d.toISOString().slice(0, 10);
    if (mode === 'week') {
      // ISO year-week
      const dt = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
      const dayNum = (dt.getUTCDay() + 6) % 7;
      dt.setUTCDate(dt.getUTCDate() - dayNum + 3);
      const firstThursday = new Date(Date.UTC(dt.getUTCFullYear(),0,4));
      const week = 1 + Math.round(((dt - firstThursday)/86400000 - 3)/7);
      return `${dt.getUTCFullYear()}-W${String(week).padStart(2,'0')}`;
    }
    // month
    return d.toISOString().slice(0, 7);
  }

  // ---------- charts ----------
  function plotBar(id, x, y, title, xTitle='') {
    Plotly.newPlot(id, [{x, y, type:'bar'}], {
      title, margin:{t:28,l:40,r:10,b:40}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      xaxis:{title:xTitle, gridcolor:'#1c2541'}, yaxis:{gridcolor:'#1c2541'}
    }, {displaylogo:false, responsive:true});
  }
  function plotHist(id, values, nbins, title, xTitle='') {
    Plotly.newPlot(id, [{x: values, type:'histogram', nbinsx: nbins}], {
      title, margin:{t:28,l:40,r:10,b:40}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      xaxis:{title:xTitle, gridcolor:'#1c2541'}, yaxis:{gridcolor:'#1c2541'}
    }, {displaylogo:false, responsive:true});
  }
  function plotScatter(id, x, y, title, xTitle='', yTitle='') {
    Plotly.newPlot(id, [{x, y, mode:'markers', type:'scattergl', marker:{size:6, opacity:0.7}}], {
      title, margin:{t:28,l:50,r:10,b:50}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      xaxis:{title:xTitle, gridcolor:'#1c2541'}, yaxis:{title:yTitle, gridcolor:'#1c2541'}
    }, {displaylogo:false, responsive:true});
  }
  function plotLine(id, x, y, title, xTitle='') {
    Plotly.newPlot(id, [{x, y, mode:'lines+markers'}], {
      title, margin:{t:28,l:40,r:10,b:40}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      xaxis:{title:xTitle, gridcolor:'#1c2541'}, yaxis:{gridcolor:'#1c2541'}
    }, {displaylogo:false, responsive:true});
  }
  function plotHeat(id, z, x, y, title) {
    Plotly.newPlot(id, [{z, x, y, type:'heatmap', colorscale:'Viridis'}], {
      title, margin:{t:28,l:60,r:10,b:40}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'
    }, {displaylogo:false, responsive:true});
  }

  function drawCards() {
    const nUsers = user2items.size;
    const nItems = item2users.size;
    const nInter = interactions.length;
    const density = nInter / Math.max(1, nUsers * nItems);
    const userCounts = Array.from(user2items.values(), arr => arr.length);
    const itemCounts = Array.from(item2users.values(), s => s.size);
    const coldUsers = userCounts.filter(c => c < 5).length;
    const coldItems = itemCounts.filter(c => c < 5).length;
    const ratingVals = ratingsAvail ? interactions.map(x => clamp(Number(x.r)||0, 1, 5)).filter(Number.isFinite) : [];
    const ratingMean = ratingVals.length ? (ratingVals.reduce((a,b)=>a+b,0)/ratingVals.length) : 0;

    const top1pctCount = Math.max(1, Math.floor(0.01*nItems));
    const sortedItems = itemCounts.slice().sort((a,b)=>b-a);
    const top1Covered = sortedItems.slice(0, top1pctCount).reduce((s,v)=>s+v,0) / Math.max(1,nInter) * 100;

    const html = `
      <div class="stat"><div class="h">Users</div><div class="v">${fmt(nUsers)}</div></div>
      <div class="stat"><div class="h">Items</div><div class="v">${fmt(nItems)}</div></div>
      <div class="stat"><div class="h">Interactions</div><div class="v">${fmt(nInter)}</div></div>
      <div class="stat"><div class="h">Density</div><div class="v">${density.toExponential(2)}</div></div>
      <div class="stat"><div class="h">Cold users &lt;5</div><div class="v">${fmt(coldUsers)}</div></div>
      <div class="stat"><div class="h">Cold items &lt;5</div><div class="v">${fmt(coldItems)}</div></div>
      <div class="stat"><div class="h">Ratings present</div><div class="v">${ratingsAvail ? 'yes' : 'no'}</div></div>
      <div class="stat"><div class="h">Avg rating</div><div class="v">${ratingsAvail ? ratingMean.toFixed(2) : '—'}</div></div>
      <div class="stat"><div class="h">Top 1% items</div><div class="v">${fmt(top1pctCount)}</div></div>
      <div class="stat"><div class="h">% interactions by top 1%</div><div class="v">${top1Covered.toFixed(1)}%</div></div>
    `;
    $('cards').innerHTML = html;
  }

  function drawOverview() {
    // time series
    const mode = ($('timeGroup').value || 'month').toLowerCase();
    const arr = interactions.filter(x => x.ts && Number.isFinite(x.ts));
    const counts = _.countBy(arr, x => groupTs(x.ts, mode));
    const keys = Object.keys(counts).sort();
    const vals = keys.map(k => counts[k]);
    plotLine('tsSeries', keys, vals, `Interactions over time (${mode})`, mode);
    $('tsNote').textContent = `${fmt(vals.reduce((a,b)=>a+b,0))} events with date`;

    // DOW/Hour heatmap
    const hasHour = arr.some(x => {
      const d = new Date(x.ts*1000);
      return d.getUTCHours() !== 0;
    });
    const hours = [...Array(24).keys()];
    const dows = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    const grid = dows.map(()=> hours.map(()=>0));
    for (const x of arr) {
      const d = new Date(x.ts*1000);
      const dow = (d.getUTCDay()+6)%7;
      const hr = hasHour ? d.getUTCHours() : 0;
      grid[dow][hr] += 1;
    }
    plotHeat('dowHeat', grid, hours, dows, 'DOW × Hour');
  }

  function drawUsers() {
    const userCounts = Array.from(user2items.values(), arr => arr.length);
    plotHist('userHist', userCounts, 40, 'User activity (interactions per user)', 'count');

    if (!ratingsAvail) {
      $('userScatter').innerHTML = '<div class="muted">Avg rating requires rating column</div>';
      return;
    }
    // avg rating vs activity
    const avgByUser = [];
    for (const [u, arr] of user2items.entries()) {
      const ratings = arr.map(x=>x.r).filter(v=>v!=null && isFinite(v));
      if (!ratings.length) continue;
      const mean = ratings.reduce((a,b)=>a+b,0)/ratings.length;
      avgByUser.push({c: arr.length, m: mean});
    }
    plotScatter('userScatter', avgByUser.map(x=>x.c), avgByUser.map(x=>x.m), 'User avg rating vs activity', 'interactions', 'avg rating');
  }

  function drawItems() {
    const itemCounts = Array.from(item2users.values(), s => s.size);
    plotHist('itemHist', itemCounts, 50, 'Item popularity (interactions per item)', 'count');

    if (ratingsAvail) {
      // per-item avg rating vs popularity
      const byItem = new Map(); // i -> {sum, n}
      for (const row of interactions) {
        if (row.r == null || !isFinite(row.r)) continue;
        const o = byItem.get(row.i) || {sum:0,n:0};
        o.sum += row.r; o.n += 1;
        byItem.set(row.i, o);
      }
      const xs=[], ys=[];
      for (const [i,o] of byItem.entries()) {
        xs.push(o.n);
        ys.push(o.sum/o.n);
      }
      plotScatter('itemScatter', xs, ys, 'Item avg rating vs popularity', 'interactions', 'avg rating');
    } else {
      $('itemScatter').innerHTML = '<div class="muted">Avg rating requires rating column</div>';
    }

    const recList = Array.from(recipes.values());
    if (minutesAvail) {
      const mins = recList.map(r=> Number(r.minutes)).filter(v=>isFinite(v) && v>=0 && v<600);
      plotHist('minutesHist', mins, 40, 'Recipe time (minutes)', 'minutes');
    } else {
      $('minutesHist').innerHTML = '<div class="muted">minutes column not found</div>';
    }

    if (nIngAvail) {
      const ings = recList.map(r=> Number(r.n_ingredients)).filter(v=>isFinite(v) && v>=0 && v<=80);
      plotHist('ingCountHist', ings, 40, 'Ingredients count', 'count');
    } else {
      $('ingCountHist').innerHTML = '<div class="muted">n_ingredients/ingredients_count not found</div>';
    }
  }

  function drawTags() {
    const K = parseInt(($('topK').value||'200'),10);
    const top = Array.from(tagFreq.entries()).sort((a,b)=>b[1]-a[1]).slice(0, Math.max(30, Math.min(1000, K)));
    const top30 = top.slice(0,30);
    plotBar('topTags', top30.map(([t])=>t), top30.map(([,c])=>c), 'Top tags (top 30)', '');

    // co-occurrence graph among top-N nodes
    const nodeMax = parseInt(($('nodeMax').value||'60'),10);
    const nodeTags = new Set(top.slice(0, nodeMax).map(([t])=>t));
    const edgeMin = parseInt(($('edgeMin').value||'40'),10);

    // build index of tags per recipe for only recipes that appear in interactions
    const usedItems = new Set(interactions.map(x=>x.i));
    const co = new Map(); // "a||b" -> count
    for (const iid of usedItems) {
      const rec = recipes.get(iid); if (!rec) continue;
      const ts = (rec.tags || []).filter(t=>nodeTags.has(t));
      if (ts.length>1) {
        for (let i=0;i<ts.length;i++){
          for (let j=i+1;j<ts.length;j++){
            const a = ts[i], b = ts[j];
            const key = a < b ? `${a}||${b}` : `${b}||${a}`;
            co.set(key, (co.get(key)||0)+1);
          }
        }
      }
    }
    const nodes = Array.from(nodeTags).map((t,idx)=>({id:t, idx}));
    const nodeIndex = new Map(nodes.map((n,i)=>[n.id,i]));
    const edges = [];
    for (const [k,c] of co.entries()){
      if (c < edgeMin) continue;
      const [a,b] = k.split('||');
      const s = nodeIndex.get(a), t = nodeIndex.get(b);
      if (s==null || t==null) continue;
      edges.push({source:s, target:t, weight:c});
    }
    drawForceGraph('tagGraph', nodes, edges);
  }

  function drawTime() {
    if (ratingsAvail) {
      const ratings = interactions.map(x=> clamp(Number(x.r)||0, 1, 5)).filter(Number.isFinite);
      plotHist('ratingHist', ratings, 20, 'Ratings histogram', 'rating (1–5)');
    } else {
      $('ratingHist').innerHTML = '<div class="muted">No ratings column found</div>';
    }

    // calendar distribution (by month or week)
    const mode = ($('timeGroup').value || 'month').toLowerCase();
    const arr = interactions.filter(x => x.ts && Number.isFinite(x.ts));
    const counts = _.countBy(arr, x => groupTs(x.ts, mode));
    const keys = Object.keys(counts).sort();
    const vals = keys.map(k => counts[k]);
    plotBar('calendar', keys, vals, `Interactions by ${mode}`, mode);
  }

  function drawCold() {
    // Lorenz & Gini
    const itemCounts = Array.from(item2users.values(), s => s.size);
    const asc = itemCounts.slice().sort((a,b)=>a-b);
    const total = asc.reduce((s,v)=>s+v,0) || 1;
    const lorX = [], lorY = [];
    let cum=0;
    for (let i=0;i<asc.length;i++){
      cum += asc[i];
      lorX.push((i+1)/asc.length);
      lorY.push(cum/total);
    }
    Plotly.newPlot('lorenz', [
      {x:[0,1], y:[0,1], mode:'lines', line:{dash:'dot'}, name:'equality'},
      {x:lorX, y:lorY, mode:'lines', name:'observed'}
    ], {
      title:'Lorenz curve', margin:{t:28,l:40,r:10,b:40}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      xaxis:{title:'fraction of items', gridcolor:'#1c2541'}, yaxis:{title:'fraction of interactions', gridcolor:'#1c2541'},
      legend:{orientation:'h'}
    }, {displaylogo:false, responsive:true});

    const g = giniFromCounts(asc);
    $('giniLine').textContent = `Gini ≈ ${g.toFixed(3)}`;

    // Pareto coverage
    const nItems = item2users.size;
    const topK = Math.max(1, Math.floor(0.01*nItems));
    const desc = itemCounts.slice().sort((a,b)=>b-a);
    const covered = desc.slice(0, topK).reduce((s,v)=>s+v,0) / Math.max(1, interactions.length) * 100;
    plotBar('pareto', ['Top 1% items','Others'], [covered, 100-covered], 'Pareto coverage', '%');
    $('paretoNote').textContent = `Top ${fmt(topK)} items (~1%) cover ${covered.toFixed(1)}% of all interactions.`;
  }

  // ---------- force graph (D3) ----------
  function drawForceGraph(svgId, nodes, links){
    const svg = d3.select(`#${svgId}`);
    svg.selectAll('*').remove();

    const width = svg.node().clientWidth || 800;
    const height = svg.node().clientHeight || 500;

    const zoom = d3.zoom().scaleExtent([0.3, 4]).on("zoom", (e) => g.attr("transform", e.transform));
    const g = svg.append("g");
    svg.call(zoom);

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d=>d.idx).distance(d=>100 + (300/Math.sqrt(d.weight||1))))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width/2, height/2));

    const link = g.append("g").attr("stroke","#6b7fa8").attr("stroke-opacity",0.6)
      .selectAll("line").data(links).enter().append("line")
      .attr("stroke-width", d => Math.max(1, Math.log2(1+d.weight)));

    const node = g.append("g").selectAll("circle").data(nodes).enter().append("circle")
      .attr("r", 6)
      .attr("fill", "#9fccff")
      .attr("stroke", "#274b7a")
      .attr("stroke-width", 1.2)
      .call(d3.drag()
        .on("start", (event,d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (event,d) => { d.fx = event.x; d.fy = event.y; })
        .on("end",  (event,d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    const labels = g.append("g").selectAll("text").data(nodes).enter().append("text")
      .text(d=>d.id).attr("font-size", 10).attr("fill", "#e6efff");

    sim.on("tick", () => {
      link.attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
      labels.attr("x", d => d.x + 8).attr("y", d => d.y + 3);
    });
  }

  // ---------- draw all ----------
  function drawAll(){
    drawCards();
    drawOverview();
    drawUsers();
    drawItems();
    drawTags();
    drawTime();
    drawCold();
  }

  // ---------- tabs ----------
  function bindTabs(){
    const nav = $('tabs');
    nav.addEventListener('click', (e)=>{
      const btn = e.target.closest('.tab'); if (!btn) return;
      document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tabpane').forEach(p=>p.classList.remove('active'));
      $(tab).classList.add('active');
      // on demand redraw sizing-sensitive charts
      if (tab === 'tags') drawTags();
      if (tab === 'overview') drawOverview();
      if (tab === 'items') drawItems();
      if (tab === 'users') drawUsers();
      if (tab === 'time') drawTime();
      if (tab === 'cold') drawCold();
    });
  }

  // ---------- events ----------
  window.addEventListener('resize', _.throttle(()=>{
    const active = document.querySelector('.tabpane.active')?.id;
    if (active) {
      if (active==='overview') drawOverview();
      else if (active==='users') drawUsers();
      else if (active==='items') drawItems();
      else if (active==='tags') drawTags();
      else if (active==='time') drawTime();
      else if (active==='cold') drawCold();
    }
  }, 400));

  $('btnLoad').addEventListener('click', ()=>{
    loadAll().catch(err => {
      console.error(err);
      $('status').textContent = 'Status: failed to load/parse CSVs (see console)';
    });
  });
  $('btnRedraw').addEventListener('click', ()=> drawAll());
  $('topK').addEventListener('change', ()=> drawTags());
  $('edgeMin').addEventListener('change', ()=> drawTags());
  $('nodeMax').addEventListener('change', ()=> drawTags());
  $('timeGroup').addEventListener('change', ()=> { drawOverview(); drawTime(); });

  bindTabs();
})();
