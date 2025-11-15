// utils.js — shared helpers (I/O, math, parsing, MMR, formatting)
export const $ = (id) => document.getElementById(id);

// --------------- I/O ---------------
export async function loadJSON(path){
  const r = await fetch(path);
  if(!r.ok) throw new Error(`fetch failed: ${path}`);
  return r.json();
}
export async function loadCSV(path){
  const r = await fetch(path);
  if(!r.ok) throw new Error(`fetch failed: ${path}`);
  const text = await r.text();
  const lines = text.split(/\r?\n/).filter(l=>l.trim().length>0);
  const header = splitCSV(lines.shift());
  return lines.map(line=>{
    const cells = splitCSV(line);
    const obj = {};
    header.forEach((h,i)=> obj[h.trim()] = cells[i]);
    return obj;
  });
}
function splitCSV(line){
  const out=[]; let cur=""; let inQ=false;
  for(let i=0;i<line.length;i++){
    const ch=line[i];
    if(ch==='"'){ if(line[i+1]==='"'){cur+='"'; i++;} else inQ=!inQ; }
    else if(ch===',' && !inQ){ out.push(cur); cur=""; }
    else cur+=ch;
  }
  out.push(cur);
  return out;
}

// --------------- math ---------------
export const deg2rad = d => d*Math.PI/180;
export function haversineKm(a, b){
  const R=6371;
  const dLat=deg2rad(b.lat-a.lat), dLon=deg2rad(b.lon-a.lon);
  const s = Math.sin(dLat/2)**2 + Math.cos(deg2rad(a.lat))*Math.cos(deg2rad(b.lat))*Math.sin(dLon/2)**2;
  return 2*R*Math.asin(Math.sqrt(s));
}
export function normalize(v){
  const n = Math.sqrt(v.reduce((s,x)=>s+x*x,0)) || 1;
  return v.map(x=>x/n);
}
export function dot(a,b){ let s=0; for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
export function cosine(a,b){
  const na = Math.sqrt(dot(a,a)) || 1;
  const nb = Math.sqrt(dot(b,b)) || 1;
  return dot(a,b)/(na*nb);
}

// --------------- MMR ---------------
export function mmr(cands, k=10, lambda=0.7){
  const selected = [];
  const rest = cands.slice().sort((a,b)=>b.score-a.score);
  while(selected.length<k && rest.length){
    let best=null, bestVal=-1e9, bestIdx=-1;
    for(let i=0;i<rest.length;i++){
      const c = rest[i];
      const rel = c.score;
      let div = 0;
      if(selected.length){
        const sims = selected.map(s=> cosine(s.vec, c.vec));
        div = Math.max(...sims);
      }
      const val = lambda*rel - (1-lambda)*div;
      if(val>bestVal){ bestVal=val; best=c; bestIdx=i; }
    }
    selected.push(best);
    rest.splice(bestIdx,1);
  }
  return selected;
}

// --------------- formatting ---------------
export const fmtMoney = (n)=> `$${Math.round(n).toLocaleString()}`;
export function fmtDuration(min){
  const h=Math.floor(min/60), m=Math.round(min%60);
  return `${h}h ${m}m`;
}
export function addMinutes(dateStr, timeStr, minutes){
  const d = new Date(`${dateStr}T${timeStr || "00:00"}:00`);
  d.setMinutes(d.getMinutes()+minutes);
  return d.toTimeString().slice(0,5);
}

// --------------- light query parsing ---------------
export function parseConstraints(freeText){
  const q = (freeText||"").toLowerCase();
  const constraints = {
    noRedEye: /no\s*red[-\s]?eye/.test(q),
    arriveBefore: (q.match(/arrive (before|by) (\d{1,2}):?(\d{2})?/ )||[]).slice(2,4).join(":") || null,
    departAfter: (q.match(/depart (after|post) (\d{1,2}):?(\d{2})?/ )||[]).slice(2,4).join(":") || null,
    preferAlliance: /sky\s?team|skyteam/.test(q) ? "SkyTeam" :
                    /one\s?world|oneworld/.test(q) ? "Oneworld" :
                    /star alliance|star\s?alliance/.test(q) ? "Star Alliance" : null,
    maxPrice: (q.match(/under\s*\$?(\d{2,5})|<\s*\$?(\d{2,5})/)||[]).slice(1).find(Boolean) || null
  };
  if(constraints.maxPrice) constraints.maxPrice = Number(constraints.maxPrice);
  return constraints;
}

// --------------- canvas helpers ---------------
export function circle(ctx,x,y,r){ ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); }
export function line(ctx,x1,y1,x2,y2,alpha){
  ctx.save(); ctx.globalAlpha = alpha; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); ctx.restore();
}
