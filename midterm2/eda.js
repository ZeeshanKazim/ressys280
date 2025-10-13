/* eda.js — fully client-side EDA for GitHub Pages
   Looks for CSVs next to this file OR under ./data/.
   Robust status + error messages, no external libs.
*/

/* ============= tiny DOM helpers ============= */
const $ = (id) => document.getElementById(id);
const fmt = (x) => (typeof x === "number" ? x.toLocaleString() : x);

/* ============= state ============= */
let items = new Map();   // itemId -> {title, tags[], minutes, n_ing}
let users = new Set();
let interactions = [];   // {u,i,r,ts}

let user2rows = new Map(); // u -> [{i,r,ts}]
let item2rows = new Map(); // i -> [{u,r,ts}]

let idx2item = [];       // stable iteration
let tag2freq = new Map();
let colsDetected = {
  hasRatings: false,
  hasTime: false,
  hasMinutes: false,
  hasNing: false
};

/* ============= bootstrapping ============= */
document.addEventListener("DOMContentLoaded", () => {
  // tabs
  document.querySelectorAll(".tab").forEach(b=>{
    b.addEventListener("click", ()=>{
      document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
      document.querySelectorAll(".pane").forEach(p=>p.classList.remove("active"));
      b.classList.add("active");
      $(b.dataset.pane).classList.add("active");
    });
  });

  $("btnLoad").addEventListener("click", () => safeRun(handleLoad));
  $("btnRedraw").addEventListener("click", () => safeRun(redrawAll));

  // initial status
  setStatus("Status: ready. Click “Load data”.");
});

/* ============= error guard ============= */
function safeRun(fn){
  try{ fn(); }catch(err){
    console.error(err);
    setStatus("Status: error — " + (err.message||err));
  }
}
function setStatus(s){ $("status").textContent = s; }

/* ============= loader ============= */
async function fetchFirstExisting(paths){
  for (const p of paths){
    try{
      const res = await fetch(p, {cache:"no-store"});
      if (res.ok){
        const text = await res.text();
        return {path:p, text};
      }
    }catch(_){/* try next */}
  }
  return null;
}

function splitCSVLines(text){
  return text.split(/\r?\n/).filter(Boolean);
}
function splitCSVRow(row){
  // split by commas not inside quotes
  return row.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/g).map(s => s.replace(/^"|"$/g,""));
}

function parseRecipes(csvText){
  const lines = splitCSVLines(csvText);
  if (!lines.length) return;

  const header = splitCSVRow(lines.shift()).map(h=>h.trim().toLowerCase());
  const idIdx = header.findIndex(h=>/^id$|(^|_)id$/.test(h));
  const nameIdx = header.findIndex(h=>/(name|title)/.test(h));
  const tagsIdx = header.findIndex(h=>/tags/.test(h));
  const minIdx = header.findIndex(h=>/minute/.test(h));
  const ningIdx = header.findIndex(h=>/n_?ingredients/.test(h));

  for (const ln of lines){
    const cols = splitCSVRow(ln);
    const id = parseInt(cols[idIdx],10);
    if (!Number.isInteger(id)) continue;

    const title = (cols[nameIdx] || `Recipe ${id}`).trim();
    let tags = [];
    if (tagsIdx >= 0 && cols[tagsIdx]){
      // handle "['a','b']" formats
      let raw = cols[tagsIdx].trim();
      raw = raw.replace(/^\s*\[|\]\s*$/g,"");
      tags = raw.split(/['"]\s*,\s*['"]|,\s*/g)
                .map(s=>s.replace(/^\s*['"]?|['"]?\s*$/g,'').trim())
                .filter(Boolean)
                .slice(0, 32);
    }

    const minutes = (minIdx>=0 && cols[minIdx] !== "") ? parseFloat(cols[minIdx]) : null;
    const n_ing = (ningIdx>=0 && cols[ningIdx] !== "") ? parseInt(cols[ningIdx],10) : null;

    items.set(id, {title, tags, minutes, n_ing});
  }

  colsDetected.hasMinutes = Array.from(items.values()).some(x=>typeof x.minutes === "number");
  colsDetected.hasNing = Array.from(items.values()).some(x=>Number.isInteger(x.n_ing));

  // tag frequencies
  tag2freq.clear();
  for (const it of items.values()){
    for (const t of it.tags||[]){
      tag2freq.set(t, (tag2freq.get(t)||0)+1);
    }
  }
}

function parseInteractions(csvText){
  const lines = splitCSVLines(csvText);
  if (!lines.length) return;
  const header = splitCSVRow(lines.shift()).map(h=>h.trim().toLowerCase());

  const uIdx = header.findIndex(h=>/user/.test(h));
  const iIdx = header.findIndex(h=>/(item|recipe)_?id/.test(h));
  const rIdx = header.findIndex(h=>/rating/.test(h));
  const tIdx = header.findIndex(h=>/(time|date)/.test(h));

  for (const ln of lines){
    const cols = splitCSVRow(ln);
    const u = parseInt(cols[uIdx],10);
    const i = parseInt(cols[iIdx],10);
    if (!Number.isInteger(u) || !Number.isInteger(i)) continue;
    const r = rIdx>=0 && cols[rIdx] !== "" ? parseFloat(cols[rIdx]) : null;
    const ts = tIdx>=0 ? (Date.parse(cols[tIdx]) || 0) : 0;

    interactions.push({u,i,r,ts});
    users.add(u);
    if (!user2rows.has(u)) user2rows.set(u, []);
    user2rows.get(u).push({i,r,ts});
    if (!item2rows.has(i)) item2rows.set(i, []);
    item2rows.get(i).push({u,r,ts});
  }

  colsDetected.hasRatings = interactions.some(x=>x.r!=null);
  colsDetected.hasTime = interactions.some(x=>x.ts>0);

  idx2item = Array.from(new Set(interactions.map(x=>x.i))).sort
