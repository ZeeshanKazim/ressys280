// graph.js — Tab A: build route graph, Personalized PageRank-ish score + constraints
import {$, loadJSON, loadCSV, haversineKm, fmtMoney, fmtDuration, addMinutes, parseConstraints, circle, line} from './utils.js';

export const GraphState = {
  airports: {},     // IATA -> {name, city, lat, lon}
  airlines: {},     // IATA -> {name, alliance}
  routes: [],       // {origin,dest,count,carriers:[...]}
  adj: new Map(),   // origin -> [{to, count, carriers}]
  loaded:false
};

async function tryLoad(fn){ try{ return await fn(); }catch(_){ return null; } }

export async function loadGraphData(){
  // airports: prefer JSON; else fallback to your CSV "Full_Merge_of_All_Unique Airports.csv"
  const airportsJSON = await tryLoad(()=>loadJSON('data/airports.json'));
  let airports = airportsJSON;
  if(!airports){
    const rows = await loadCSV('data/Full_Merge_of_All_Unique%20Airports.csv');
    // tolerant columns: IATA/Code, City, Latitude, Longitude, Name
    const pick = (r, keys)=> keys.find(k=>k in r && r[k]);
    airports = {};
    for(const r of rows){
      const code = (r.IATA || r.Code || r.code || r.AirportCode || "").trim();
      if(!code) continue;
      const name = (r.Name || r.AirportName || code).trim();
      const city = (r.City || r.Municipality || r.Town || "").trim();
      const lat = Number(r.Latitude || r.lat || r.Lat || r.latitude);
      const lon = Number(r.Longitude || r.lon || r.Lon || r.longitude);
      if(Number.isFinite(lat) && Number.isFinite(lon)){
        airports[code] = {name, city, lat, lon};
      }
    }
  }

  // routes: prefer JSON; else fallback to your CSV "Full_Merge_of_All_Unique_Routes.csv"
  const routesJSON = await tryLoad(()=>loadJSON('data/routes.json'));
  let routes = routesJSON;
  if(!routes){
    const rows = await loadCSV('data/Full_Merge_of_All_Unique_Routes.csv');
    // tolerant columns: Origin, Destination, Airline, Count optional
    const group = new Map();
    for(const r of rows){
      const o = (r.Origin || r.origin || r.From || "").trim();
      const d = (r.Destination || r.dest || r.To || "").trim();
      const c = (r.Airline || r.airline || r.Carrier || "").trim().toUpperCase();
      if(!o || !d) continue;
      const key = `${o}>${d}`;
      if(!group.has(key)) group.set(key, {origin:o, dest:d, count:0, carriers:new Set()});
      const g = group.get(key);
      g.count += 1;
      if(c) g.carriers.add(c);
    }
    routes = Array.from(group.values()).map(g=>({origin:g.origin,dest:g.dest,count:g.count,carriers:Array.from(g.carriers)}));
  }

  // airlines with alliances (optional); if missing, create bare map
  const airlines = await tryLoad(()=>loadJSON('data/airlines.json')) || deduceAirlines(routes);

  GraphState.airports = airports;
  GraphState.airlines = airlines;
  GraphState.routes = routes;
  GraphState.adj.clear();
  for(const r of routes){
    if(!GraphState.adj.has(r.origin)) GraphState.adj.set(r.origin,[]);
    GraphState.adj.get(r.origin).push({to:r.dest, count:r.count, carriers:r.carriers});
  }
  GraphState.loaded = true;
}

function deduceAirlines(routes){
  const m = {};
  for(const r of routes){
    for(const c of (r.carriers||[])){ if(!m[c]) m[c] = {name:c, alliance:""}; }
  }
  return m;
}

// ---------- path search ≤ 1 stop ----------
function findPaths(origin, dest, maxStops){
  const paths = [];
  if(maxStops===0){
    const outs = GraphState.adj.get(origin)||[];
    for(const e of outs){ if(e.to===dest) paths.push([origin,dest]); }
    return paths;
  }
  const outs = GraphState.adj.get(origin)||[];
  for(const e of outs){
    if(e.to===dest) paths.push([origin,dest]);
    const outs2 = GraphState.adj.get(e.to)||[];
    for(const e2 of outs2){
      if(e2.to===dest && e.to!==origin) paths.push([origin,e.to,dest]);
    }
  }
  const seen = new Set();
  return paths.filter(p=> (seen.has(p.join(">"))?false:(seen.add(p.join(">")),true)));
}

// ---------- popularity-ish PPR ----------
function pprScore(path){
  let pop = 0;
  for(let i=0;i<path.length-1;i++){
    const a=path[i], b=path[i+1];
    const e = (GraphState.adj.get(a)||[]).find(x=>x.to===b);
    pop += e? e.count : 0;
  }
  return Math.log1p(pop);
}

function pickCarriers(path, preferredAlliance){
  const carriers = [];
  for(let i=0;i<path.length-1;i++){
    const a=path[i], b=path[i+1];
    const e = (GraphState.adj.get(a)||[]).find(x=>x.to===b);
    if(!e){ carriers.push("??"); continue; }
    const opts = preferredAlliance
      ? e.carriers.filter(c=> (GraphState.airlines[c]?.alliance===preferredAlliance))
      : e.carriers;
    carriers.push(opts[0] || e.carriers[0] || "??");
  }
  return carriers;
}

function priceDuration(path, dateStr, timeStr){
  let km=0;
  for(let i=0;i<path.length-1;i++){
    const A = GraphState.airports[path[i]];
    const B = GraphState.airports[path[i+1]];
    if(!A || !B) return {price:9999, minutes:9999, depart:timeStr||"09:00", arrive:timeStr||"09:00"};
    km += haversineKm(A,B);
  }
  const price = 40 + km*0.12 + (path.length-2)*70;
  const minutes = (km/800)*60 + (path.length-2)*70;
  const depart = timeStr || "09:00";
  const arrive = addMinutes(dateStr, depart, minutes);
  return {price, minutes, depart, arrive};
}

function violatesRedEye(departHHmm){ const [hh]=departHHmm.split(":").map(Number); return (hh>=22 || hh<2); }
function violatesTimeBound(arriveBefore, departAfter, departHHmm, arriveHHmm){
  let bad=false;
  if(arriveBefore && arriveHHmm>arriveBefore) bad=true;
  if(departAfter && departHHmm<departAfter) bad=true;
  return bad;
}

function scoreItin(ppr, minutes, price, tradeoff){
  const zDur = 1/(1+minutes/300);
  const zPri = 1/(1+price/600);
  return 0.45*ppr + 0.25*((1-tradeoff)*zDur + tradeoff*zPri) + 0.30*(zDur*zPri);
}

export function runGraphTab(params){
  const {origin,destination,maxStops,alliance,tradeoff,date,time,freeText} = params;
  const constraints = parseConstraints(freeText);
  const paths = findPaths(origin, destination, maxStops);
  const rows = [];

  for(const path of paths){
    const ppr = pprScore(path);
    const carriers = pickCarriers(path, alliance || constraints.preferAlliance);
    const {price, minutes, depart, arrive} = priceDuration(path, date, time);
    const stops = path.length-2;

    if(constraints.noRedEye && violatesRedEye(depart)) continue;
    if(violatesTimeBound(constraints.arriveBefore, constraints.departAfter, depart, arrive)) continue;
    if(constraints.maxPrice && price>constraints.maxPrice) continue;

    const s = scoreItin(ppr, minutes, price, tradeoff);
    const why = [
      `popular route (+${ppr.toFixed(2)})`,
      stops===0? "nonstop":"1-stop",
      (alliance || constraints.preferAlliance) ? `alliance: ${alliance || constraints.preferAlliance}`:""
    ].filter(Boolean).join(", ");

    rows.push({ score:s, path, carriers, stops, depart, arrive, minutes, price, why });
  }

  rows.sort((a,b)=>b.score-a.score);
  renderGraphTable(rows.slice(0,10));
  drawRouteGraph();
  return rows.slice(0,10);
}

function renderGraphTable(rows){
  const tb = $('#graphTable').querySelector('tbody');
  tb.innerHTML = "";
  rows.forEach((r,i)=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i+1}</td>
      <td>${r.path.join(" → ")}</td>
      <td><span class="badge">${r.carriers.join("/")}</span></td>
      <td>${r.stops}</td>
      <td>${r.depart} → ${r.arrive}</td>
      <td>${fmtDuration(r.minutes)}</td>
      <td class="price">${fmtMoney(r.price)}</td>
      <td>${r.score.toFixed(3)}</td>
      <td>${r.why}</td>`;
    tb.appendChild(tr);
  });
}

function drawRouteGraph(){
  const canvas = $('#graphCanvas'); const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = "#2b476f"; ctx.fillStyle="#b9d3fb";

  const nodes = Object.keys(GraphState.airports).slice(0,50);
  const pos = new Map();
  nodes.forEach((iata,idx)=>{
    const seed = hash(iata);
    const x = 60 + (seed%1000)/1000*(canvas.width-120);
    const y = 60 + (seed%777)/777*(canvas.height-120);
    pos.set(iata,{x,y});
  });

  for(const r of GraphState.routes){
    if(!pos.has(r.origin) || !pos.has(r.dest)) continue;
    const a = pos.get(r.origin), b = pos.get(r.dest);
    const alpha = Math.min(0.8, 0.2 + Math.log1p(r.count)/6);
    line(ctx, a.x, a.y, b.x, b.y, alpha);
  }
  for(const iata of nodes){
    const deg = (GraphState.adj.get(iata)||[]).length;
    const r = Math.max(3, Math.min(10, 3+deg*0.7));
    const p = pos.get(iata);
    circle(ctx, p.x, p.y, r);
    ctx.fillStyle="#9fb0c3"; ctx.font="10px Inter";
    ctx.fillText(iata, p.x+6, p.y-6);
    ctx.fillStyle="#b9d3fb";
  }
}

function hash(s){ let h=0; for(let i=0;i<s.length;i++) h=(h*31 + s.charCodeAt(i))>>>0; return h; }
