// text.js — Tab B: reviews → vectors; MMR; constraints (no LLM)
import {$, loadJSON, loadCSV, cosine, normalize, mmr, fmtMoney, fmtDuration, addMinutes, parseConstraints} from './utils.js';
import { GraphState } from './graph.js';

export const TextState = {
  reviewEmb: {},    // airline -> [v1..vd]  normalized
  reviews: [],      // {airline, route, rating, text}
  dim: 0,
  loaded:false
};

async function tryLoad(fn){ try{ return await fn(); }catch(_){ return null; } }

// derive simple 10-dim vectors from reviews if JSON embeddings absent
const VOCAB = ["seat","legroom","food","staff","ontime","lounge","wifi","service","comfort","price"];

export async function loadTextData(){
  const embJSON = await tryLoad(()=>loadJSON('data/review_embeddings.json'));
  let reviewEmb = embJSON, dim = embJSON ? (Object.values(embJSON)[0]||[]).length : 0;

  // reviews CSV: sample "reviews.csv" OR your "Airline_Reviews.csv"
  const reviewsCSV = await tryLoad(()=>loadCSV('data/reviews.csv')) ||
                     await tryLoad(()=>loadCSV('data/Airline_Reviews.csv')) || [];
  const normalizedRows = reviewsCSV.map(r=>{
    const airline = (r.airline || r.Airline || r.carrier || r.Carrier || "").toUpperCase().trim();
    const route = (r.route || r.Route || r.origin_dest || "").trim();
    const rating = Number(r.rating || r.Rating || r.score || 0);
    const text = (r.text || r.Review || r.review || "").trim();
    return {airline, route, rating, text};
  }).filter(x=>x.airline);

  // if no JSON vectors, compute per-airline TF (VOCAB) from reviews
  if(!reviewEmb){
    const sums = {}; // airline -> [counts]
    for(const row of normalizedRows){
      const v = new Array(VOCAB.length).fill(0);
      const t = (row.text||"").toLowerCase();
      VOCAB.forEach((w,i)=>{ const m = (t.match(new RegExp(`\\b${w}\\b`, 'g'))||[]).length; v[i]+=m; });
      (sums[row.airline] ||= new Array(VOCAB.length).fill(0)).forEach((_,i)=> sums[row.airline][i]+=v[i]);
    }
    reviewEmb = {};
    for(const k of Object.keys(sums)) reviewEmb[k] = normalize(sums[k].map(x=>x||0.001));
    dim = VOCAB.length;
  }

  TextState.reviewEmb = reviewEmb;
  TextState.dim = dim;
  TextState.reviews = normalizedRows;
  TextState.loaded = true;
}

function intentVector(freeText){
  const v = new Array(VOCAB.length).fill(0);
  const t = (freeText||"").toLowerCase();
  VOCAB.forEach((w,i)=>{ const m = (t.match(new RegExp(`\\b${w}\\b`, 'g'))||[]).length; v[i]=m; });
  if(v.every(x=>x===0)){ v[VOCAB.indexOf("comfort")] = 1; v[VOCAB.indexOf("service")] = 1; }
  return normalize(v);
}

function candidatesFromGraph(origin, destination, date, time, maxStops, alliancePref){
  const outs0 = GraphState.adj.get(origin)||[];
  const list = [];
  if(maxStops===0){
    for(const e of outs0) if(e.to===destination) list.push([origin,destination]);
  }else{
    for(const e of outs0){
      if(e.to===destination) list.push([origin,destination]);
      const outs1 = GraphState.adj.get(e.to)||[];
      for(const e2 of outs1){ if(e2.to===destination && e.to!==origin) list.push([origin,e.to,destination]); }
    }
  }
  const seen = new Set();
  const paths = list.filter(p=> (seen.has(p.join(">"))?false:(seen.add(p.join(">")),true)));

  function pickCarrier(a,b){
    const e = (GraphState.adj.get(a)||[]).find(x=>x.to===b);
    if(!e) return "??";
    const opts = alliancePref ? e.carriers.filter(c=> GraphState.airlines[c]?.alliance===alliancePref) : e.carriers;
    return opts[0] || e.carriers[0] || "??";
  }
  function priceDuration(path){
    let km=0;
    for(let i=0;i<path.length-1;i++){
      const A = GraphState.airports[path[i]], B = GraphState.airports[path[i+1]];
      if(!A || !B) return {price:9999, minutes:9999, depart:time||"09:00", arrive:time||"09:00"};
      const R=6371, dLat=(B.lat-A.lat)*Math.PI/180, dLon=(B.lon-A.lon)*Math.PI/180;
      const s = Math.sin(dLat/2)**2 + Math.cos(A.lat*Math.PI/180)*Math.cos(B.lat*Math.PI/180)*Math.sin(dLon/2)**2;
      km += 2*R*Math.asin(Math.sqrt(s));
    }
    const price = 40 + km*0.12 + (path.length-2)*70;
    const minutes = (km/800)*60 + (path.length-2)*70;
    const depart = time || "09:00";
    const arrive = addMinutes(date, depart, minutes);
    return {price, minutes, depart, arrive};
  }

  return paths.map(p=>{
    const carriers = []; for(let i=0;i<p.length-1;i++) carriers.push(pickCarrier(p[i], p[i+1]));
    const airline = carriers[0];
    const meta = priceDuration(p);
    return {path:p, airline, ...meta, stops:p.length-2, carriers};
  });
}

export function runTextTab(params){
  const {origin,destination,maxStops,alliance,tradeoff,date,time,freeText} = params;
  const constraints = parseConstraints(freeText);
  const intent = intentVector(freeText);
  const items = candidatesFromGraph(origin,destination,date,time,maxStops, (alliance || constraints.preferAlliance));

  const cands = [];
  for(const it of items){
    const vec = TextState.reviewEmb[it.airline] || new Array(TextState.dim||VOCAB.length).fill(0);
    const sim = cosine(intent, vec);
    const zDur = 1/(1+it.minutes/300), zPri = 1/(1+it.price/600);
    const base = 0.55*sim + 0.45*((1-tradeoff)*zDur + tradeoff*zPri);

    if(constraints.noRedEye){ const [hh]=it.depart.split(":").map(Number); if(hh>=22 || hh<2) continue; }
    if(constraints.arriveBefore && it.arrive>constraints.arriveBefore) continue;
    if(constraints.departAfter && it.depart<constraints.departAfter) continue;
    if(constraints.maxPrice && it.price>constraints.maxPrice) continue;

    cands.push({ id: it.path.join(">")+"|"+it.airline, score: base, vec, meta: it });
  }

  const top = mmr(cands, 10, 0.7);
  renderTextTable(top);
  drawEmbeddings();
  return top;
}

function renderTextTable(selected){
  const tb = $('#textTable').querySelector('tbody');
  tb.innerHTML = "";
  selected.forEach((s,i)=>{
    const m = s.meta;
    const why = `matches reviews on comfort/service; ${(m.stops===0?"nonstop":"1-stop")}${GraphState.airlines[m.airline]?.alliance ? ", alliance: "+GraphState.airlines[m.airline].alliance : ""}`;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i+1}</td>
      <td>${m.path.join(" → ")}</td>
      <td><span class="badge">${m.airline}</span></td>
      <td>${m.stops}</td>
      <td>${m.depart} → ${m.arrive}</td>
      <td>${fmtDuration(m.minutes)}</td>
      <td class="price">${fmtMoney(m.price)}</td>
      <td>${s.score.toFixed(3)}</td>
      <td>${why}</td>`;
    tb.appendChild(tr);
  });
}

function drawEmbeddings(){
  const canvas = $('#embCanvas'); const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#b9d3fb"; ctx.strokeStyle="#2b476f";
  const labels = Object.keys(TextState.reviewEmb).slice(0,80);
  const points = labels.map((k,i)=>{
    const v = TextState.reviewEmb[k];
    let x,y;
    if(v.length>=2){ x = v[0]; y = v[1]; }
    else { const ang = (i/labels.length)*Math.PI*2; x = Math.cos(ang); y = Math.sin(ang); }
    const px = 40 + (x+1)/2 * (canvas.width-80);
    const py = 40 + (y+1)/2 * (canvas.height-80);
    return {k, x:px, y:py};
  });
  for(const p of points){ ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill(); }
  ctx.fillStyle="#9fb0c3"; ctx.font="10px Inter";
  for(const p of points){ ctx.fillText(p.k, p.x+6, p.y-6); }
  ctx.fillStyle="#b9d3fb";
}
