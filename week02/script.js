/* ---------- Cosine similarity on genre multi-hot vectors ---------- */
let genreVocab = [];
const movieVec = new Map();
const itemStats = new Map();

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

/* Popularity for the Top Picks row */
function computeStats(){
  itemStats.clear();
  for (const r of ratings){
    const s = itemStats.get(r.itemId) || {count:0,sum:0,avg:0};
    s.count++; s.sum += r.rating; itemStats.set(r.itemId, s);
  }
  for (const s of itemStats.values()){ s.avg = s.sum / s.count; }
}
function topPicks(n=18){
  const arr = movies.map(m => ({ m, s: itemStats.get(m.id) || {count:0,avg:0} }));
  arr.sort((a,b)=> (b.s.count - a.s.count) || (b.s.avg - a.s.avg));
  return arr.slice(0,n).map(x=>x.m);
}

/* ---------- UI helpers ---------- */
const $ = id => document.getElementById(id);

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

function initials(t){
  return t.split(/[\s:–-]+/).filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase();
}
function gradFromId(id){
  const h = (id*37)%360, h2 = (h+40)%360;
  return `linear-gradient(160deg, hsla(${h},70%,45%,.9), hsla(${h2},70%,35%,.9))`;
}
function makeCard(m){
  const card = document.createElement('div'); card.className='card'; card.title=m.title;

  const poster = document.createElement('div'); poster.className='poster';
  poster.style.setProperty('--grad', gradFromId(m.id));
  poster.textContent = initials(m.title);

  if (typeof m.score === 'number'){
    const badge = document.createElement('div'); badge.className='badge';
    badge.textContent = `${Math.round(m.score*100)}% match`;
    poster.appendChild(badge);
  }

  const meta = document.createElement('div'); meta.className='meta';
  const h = document.createElement('p'); h.className='title'; h.textContent = m.title;
  const chips = document.createElement('div'); chips.className='chips';
  m.genres.slice(0,3).forEach(g=>{
    const s = document.createElement('span'); s.className='chip-mini'; s.textContent=g;
    chips.appendChild(s);
  });

  meta.appendChild(h); meta.appendChild(chips);
  card.appendChild(poster); card.appendChild(meta);

  // Click a card to run recs for that title
  card.addEventListener('click', ()=>{
    $('movie-select').value = String(m.id);
    getRecommendations();
    window.scrollTo({top:0,behavior:'smooth'});
  });

  return card;
}
function renderRow(containerId, list){
  const row = $(containerId);
  row.innerHTML = '';
  list.forEach(m => row.appendChild(makeCard(m)));
}

/* ---------- Search ---------- */
function setupSearch(){
  const input = $('search');
  const sec = $('search-sec');
  input.addEventListener('input', ()=>{
    const q = input.value.trim().toLowerCase();
    if (!q){
      sec.classList.add('hidden');
      renderRow('row-popular', topPicks());
      return;
    }
    const matches = movies.filter(m => m.title.toLowerCase().includes(q)).slice(0,30);
    sec.classList.toggle('hidden', matches.length===0);
    renderRow('row-search', matches);
  });
}

/* ---------- Main: COSINE recommendations ---------- */
function getRecommendations(){
  const res = $('result');
  const idStr = $('movie-select').value;

  if (!idStr){ res.textContent = 'Please select a movie first.'; return; }

  const likedId = parseInt(idStr,10);
  const liked = movies.find(m => m.id === likedId);
  if (!liked){ res.textContent = 'Selected movie not found.'; return; }

  // IMPORTANT: exclude the liked movie (fix for "only shows liked movie")
  const candidates = movies.filter(m => m.id !== likedId);

  const likedVec = movieVec.get(likedId) || toVec(liked.genres);
  const scored = candidates.map(c=>{
    const s = cosine(likedVec, movieVec.get(c.id) || toVec(c.genres));
    return { ...c, score: s };
  }).sort((a,b)=> b.score - a.score);

  // show top 18 even if scores are 0 (so the row is never empty)
  const top = scored.slice(0,18);
  renderRow('row-recs', top);

  $('recs-title').textContent = `Because you liked: ${liked.title}`;
  res.textContent = `Using Cosine Similarity • ${top.length} similar titles`;
}

/* ---------- Init ---------- */
window.onload = async ()=>{
  try{
    await loadData();          // from data.js
    precomputeVectors();       // build cosine vectors
    computeStats();            // for Top Picks

    populateMoviesDropdown();
    renderRow('row-popular', topPicks());
    setupSearch();

    const r = $('result');
    r.textContent = `Data loaded: ${movies.length} movies. Select a movie or search.`;
    r.classList.remove('muted');

    // If you still only see the liked movie, your u.item probably has no overlapping genres.
    // We still show top items (even with 0% badges) so the rail is not empty.
  }catch(e){
    /* data.js already shows a friendly error */
  }
};
