/* ================= Cosine similarity + Posters + List ================= */
let genreVocab = [];
const movieVec = new Map();
const itemStats = new Map();

/* 1) OPTIONAL: TMDb posters (set your API key) */
const TMDB_KEY = ""; // <-- put your TMDb API key here (or leave blank for placeholders)
const TMDB_BASE = "https://api.themoviedb.org/3/search/movie";
const TMDB_IMG  = "https://image.tmdb.org/t/p/w342";
const posterCache = new Map();

async function fetchPosterUrl(title){
  if (!TMDB_KEY) return null;
  if (posterCache.has(title)) return posterCache.get(title);
  try{
    const url = `${TMDB_BASE}?api_key=${TMDB_KEY}&query=${encodeURIComponent(title)}&include_adult=false`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("tmdb fetch error");
    const data = await res.json();
    const path = data?.results?.[0]?.poster_path || null;
    const full = path ? `${TMDB_IMG}${path}` : null;
    posterCache.set(title, full);
    return full;
  }catch(e){
    console.warn("Poster not found for:", title);
    posterCache.set(title, null);
    return null;
  }
}

/* 2) Vectors and cosine */
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
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

/* 3) Popularity for Top Picks */
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

/* 4) UI helpers */
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
function makeCard(m){
  const card = document.createElement('div'); card.className='card'; card.title=m.title;

  const poster = document.createElement('div'); poster.className='poster';
  // image (if available) else a simple gradient + initials
  if (m.posterUrl){
    const img = document.createElement('img'); img.alt = m.title; img.src = m.posterUrl;
    poster.appendChild(img);
  }else{
    poster.textContent = initials(m.title);
  }

  if (typeof m.score === 'number'){
    const badge = document.createElement('div'); badge.className='badge';
    // EXACT or nearly exact percentage (2 decimal places):
    badge.textContent = `${(m.score*100).toFixed(2)}% match`;
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

/* 5) Search */
function setupSearch(){
  const input = $('search');
  const sec = $('search-sec');
  input.addEventListener('input', async ()=>{
    const q = input.value.trim().toLowerCase();
    if (!q){
      sec.classList.add('hidden');
      renderRow('row-popular', await withPosters(topPicks()));
      return;
    }
    const matches = movies.filter(m => m.title.toLowerCase().includes(q)).slice(0,30);
    sec.classList.toggle('hidden', matches.length===0);
    renderRow('row-search', await withPosters(matches));
  });
}

/* 6) Attach posters to a list (if TMDB_KEY set) */
async function withPosters(list){
  if (!TMDB_KEY) return list; // no key, skip
  const out = [];
  for (const m of list){
    const posterUrl = await fetchPosterUrl(m.title);
    out.push({ ...m, posterUrl });
  }
  return out;
}

/* 7) Main: COSINE recommendations */
async function getRecommendations(){
  const res = $('result');
  const recList = $('rec-list');
  const idStr = $('movie-select').value;

  if (!idStr){ res.textContent = 'Please select a movie first.'; recList.classList.add('hidden'); return; }

  const likedId = parseInt(idStr,10);
  const liked = movies.find(m => m.id === likedId);
  if (!liked){ res.textContent = 'Selected movie not found.'; recList.classList.add('hidden'); return; }

  const likedVec = movieVec.get(likedId) || toVec(liked.genres);
  const candidates = movies.filter(m => m.id !== likedId);

  const scored = candidates.map(c=>{
    const s = cosine(likedVec, movieVec.get(c.id) || toVec(c.genres));
    return { ...c, score: s };
  }).sort((a,b)=> b.score - a.score);

  const top = scored.slice(0,18);
  const topWithPosters = await withPosters(top);

  renderRow('row-recs', topWithPosters);
  $('recs-title').textContent = `Because you liked: ${liked.title}`;
  res.textContent = `Using Cosine Similarity • ${top.length} similar titles`;

  // textual list with exact percentages (top 5)
  const listText = top.slice(0,5).map(m => `${m.title} (${(m.score*100).toFixed(2)}%)`).join('  •  ');
  recList.textContent = `Top matches: ${listText}`;
  recList.classList.remove('hidden');
}

/* 8) Init */
window.onload = async ()=>{
  try{
    await loadData();          // from data.js
    precomputeVectors();
    computeStats();

    populateMoviesDropdown();
    renderRow('row-popular', await withPosters(topPicks()));
    setupSearch();

    const r = $('result');
    r.textContent = `Data loaded: ${movies.length} movies. Select a movie or search.`;
    r.classList.remove('muted');
  }catch(e){
    /* data.js already shows a friendly error */
  }
};
