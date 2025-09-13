/* ---------------- Cosine similarity on genre multi-hot ---------------- */
let genreVocab = [];
const movieVec = new Map();
const itemStats = new Map(); // popularity for "Top Picks"

/* Build vocab and vectors */
function buildGenreVocab(){
  const s = new Set();
  movies.forEach(m => m.genres.forEach(g => s.add(g)));
  genreVocab = Array.from(s).sort();
}
function toVec(genres){
  const set = new Set(genres);
  return genreVocab.map(g => (set.has(g) ? 1 : 0));
}
function precomputeVectors(){
  buildGenreVocab();
  movieVec.clear();
  movies.forEach(m => movieVec.set(m.id, toVec(m.genres)));
}
function cosine(a,b){
  let dot=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){ const ai=a[i], bi=b[i]; dot+=ai*bi; na+=ai*ai; nb+=bi*bi; }
  return (na && nb) ? dot / (Math.sqrt(na)*Math.sqrt(nb)) : 0;
}

/* Popularity stats (for a nice "Top Picks" rail) */
function computeStats(){
  itemStats.clear();
  for (const r of ratings){
    const s = itemStats.get(r.itemId) || {count:0,sum:0,avg:0};
    s.count++; s.sum += r.rating; itemStats.set(r.itemId, s);
  }
  for (const [id,s] of itemStats){ s.avg = s.sum / s.count; }
}
function getTopPicks(n=18){
  const withStats = movies.map(m => ({ m, s: itemStats.get(m.id) || {count:0, avg:0} }));
  withStats.sort((a,b) => (b.s.count - a.s.count) || (b.s.avg - a.s.avg));
  return withStats.slice(0,n).map(x => x.m);
}

/* ---------------- UI helpers ---------------- */
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
  const h = (id*37) % 360, h2 = (h+40)%360;
  return `linear-gradient(160deg, hsla(${h},70%,45%,.9), hsla(${h2},70%,35%,.9))`;
}

/* makeCard can show an optional similarity badge if item.score exists */
function makeCard(m){
  const card = document.createElement('div'); card.className='card'; card.title = m.title;

  const poster = document.createElement('div'); poster.className='poster';
  poster.style.setProperty('--grad', gradFromId(m.id));
  poster.textContent = titleInitials(m.title);

  /* similarity badge */
  if (typeof m.score === 'number'){
    const badge = document.createElement('div');
    badge.className = 'badge';
    badge.textContent = `${Math.round(m.score*100)}% match`;
    poster.appendChild(badge);
  }

  const meta = document.createElement('div'); meta.className='meta';
  const h = document.createElement('p'); h.className='title'; h.textContent = m.title;
  const chips = document.createElement('div'); chips.className='chips';
  m.genres.slice(0,3).forEach(g => {
    const s = document.createElement('span'); s.className='chip-mini'; s.textContent = g;
    chips.appendChild(s);
  });

  meta.appendChild(h); meta.appendChild(chips);
  card.appendChild(poster); card.appendChild(meta);

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

/* ---------------- Search ---------------- */
function setupSearch(){
  const input = $('search');
  const section = $('search-section');
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q){
      section.classList.add('hidden');
      renderRow('row-popular', getTopPicks());
      return;
    }
    const matches = movies.filter(m => m.title.toLowerCase().includes(q)).slice(0, 30);
    section.classList.toggle('hidden', matches.length === 0);
    renderRow('row-search', matches);
  });
}

/* ---------------- Theme toggle ---------------- */
function setupTheme(){
  const p = document.getElementById('btn-prime');
  const n = document.getElementById('btn-netflix');
  if (p) p.addEventListener('click', ()=> document.body.setAttribute('data-theme','prime'));
  if (n) n.addEventListener('click', ()=> document.body.setAttribute('data-theme','netflix'));
}

/* ---------------- Main recommendation (COSINE) ---------------- */
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

  const top = scored.slice(0, 18);
  renderRow('row-recs', top);

  $('row-recs').parentElement.querySelector('.row-title').textContent =
    `Because you liked: ${liked.title}`;
  res.textContent = `Using Cosine Similarity • showing ${top.length} similar titles`;
}

/* ---------------- Init ---------------- */
window.onload = async () => {
  try{
    await loadData();          // from data.js
    precomputeVectors();       // cosine vectors
    computeStats();            // popularity for "Top Picks"

    setupTheme();
    setupSearch();
    populateMoviesDropdown();
    renderRow('row-popular', getTopPicks());

    const r = $('result');
    r.textContent = `Data loaded: ${movies.length} movies, ${ratings.length} ratings. Select a movie.`;
    r.classList.remove('muted');

    // helpful if something silently fails later
    if (movies.length === 0) {
      r.textContent = 'No movies parsed. Check u.item format/location.';
    }
  }catch(e){
    // data.js already prints a friendly message
  }
};
