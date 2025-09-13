/* ---------- Cosine similarity over genre multi-hot vectors ---------- */
let genreVocab = [];
const movieVec = new Map();
const itemStats = new Map(); // {count, sum, avg}

/* Build genre vocabulary from movies */
function buildGenreVocab(){
  const s = new Set();
  movies.forEach(m => m.genres.forEach(g => s.add(g)));
  genreVocab = Array.from(s).sort();
}

/* Convert genres[] -> binary vector aligned with vocab */
function toVec(genres){
  const set = new Set(genres);
  return genreVocab.map(g => (set.has(g) ? 1 : 0));
}

/* Precompute vectors for all movies */
function precomputeVectors(){
  buildGenreVocab();
  movieVec.clear();
  movies.forEach(m => movieVec.set(m.id, toVec(m.genres)));
}

/* Cosine similarity */
function cosine(a,b){
  let dot=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){ const ai=a[i], bi=b[i]; dot+=ai*bi; na+=ai*ai; nb+=bi*bi; }
  return (na && nb) ? dot / (Math.sqrt(na)*Math.sqrt(nb)) : 0;
}

/* Basic popularity stats from ratings to show "Top Picks" row */
function computeStats(){
  itemStats.clear();
  for (const r of ratings){
    const s = itemStats.get(r.itemId) || {count:0,sum:0,avg:0};
    s.count++; s.sum += r.rating; itemStats.set(r.itemId, s);
  }
  for (const [id,s] of itemStats){
    s.avg = s.sum / s.count;
  }
}
function getTopPicks(n=18){
  const withStats = movies.map(m => ({ m, s: itemStats.get(m.id) || {count:0, avg:0} }));
  withStats.sort((a,b) => (b.s.count - a.s.count) || (b.s.avg - a.s.avg));
  return withStats.slice(0,n).map(x => x.m);
}

/* ----------------------- UI Helpers ----------------------- */
function $(id){ return document.getElementById(id); }

function populateMoviesDropdown(){
  const sel = $('movie-select');
  while (sel.options.length > 1) sel.remove(1);
  const sorted = [...movies].sort((a,b)=>a.title.localeCompare(b.title));
  for (const m of sorted){
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.textContent = m.title;
    sel.appendChild(opt);
  }
}

function titleInitials(t){
  return t.split(/[\s:–-]+/).filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase();
}
function gradFromId(id){
  const h = (id*37) % 360;
  const h2 = (h+40)%360;
  return `linear-gradient(160deg, hsla(${h},70%,45%,.9), hsla(${h2},70%,35%,.9))`;
}
function makeCard(m){
  const card = document.createElement('div'); card.className = 'card'; card.title = m.title;

  const poster = document.createElement('div'); poster.className='poster';
  poster.style.setProperty('--grad', gradFromId(m.id));
  poster.textContent = titleInitials(m.title);

  const meta = document.createElement('div'); meta.className='meta';
  const h = document.createElement('p'); h.className='title'; h.textContent = m.title;
  const chips = document.createElement('div'); chips.className='chips';
  (m.genres.slice(0,3)).forEach(g=>{
    const c=document.createElement('span'); c.className='chip'; c.textContent=g; chips.appendChild(c);
  });

  meta.appendChild(h); meta.appendChild(chips);
  card.appendChild(poster); card.appendChild(meta);

  // click a card to simulate selecting it
  card.addEventListener('click', () => {
    $('movie-select').value = String(m.id);
    getRecommendations();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  return card;
}
function renderRow(containerId, list){
  const row = $(containerId);
  row.innerHTML = '';
  list.forEach(m => row.appendChild(makeCard(m)));
}

/* -------------------- Main Recommendation -------------------- */
function getRecommendations(){
  const res = $('result');
  const sel = $('movie-select');
  const idStr = sel.value;
  if (!idStr){ res.textContent = 'Please select a movie first.'; return; }

  const likedId = parseInt(idStr,10);
  const liked = movies.find(m => m.id === likedId);
  if (!liked){ res.textContent = 'Selected movie not found.'; return; }

  const likedVec = movieVec.get(likedId) || toVec(liked.genres);
  const candidates = movies.filter(m => m.id !== likedId);

  const scored = candidates.map(c => {
    const s = cosine(likedVec, movieVec.get(c.id) || toVec(c.genres));
    return { ...c, score: s };
  }).sort((a,b)=>b.score - a.score);

  const top = scored.slice(0, 18); // fill a nice row
  renderRow('row-recs', top);

  $('row-recs').parentElement.querySelector('.row-title').textContent =
    `Because you liked: ${liked.title}`;
  res.textContent = `Showing cosine-similar titles to "${liked.title}"`;
}

/* -------------------- Search (simple filter) -------------------- */
function setupSearch(){
  const input = $('search');
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q){ renderRow('row-popular', getTopPicks()); return; }
    const matches = movies.filter(m => m.title.toLowerCase().includes(q)).slice(0, 30);
    renderRow('row-popular', matches);
  });
}

/* -------------------- Init -------------------- */
window.onload = async () => {
  try{
    await loadData();          // from data.js
    precomputeVectors();       // build cosine vectors
    computeStats();            // for "Top Picks"

    populateMoviesDropdown();
    renderRow('row-popular', getTopPicks());
    setupSearch();

    const r = $('result'); r.textContent = 'Data loaded. Using Cosine Similarity.';
    r.classList.remove('muted');
  }catch(e){
    /* data.js already shows a friendly message */
  }
};
