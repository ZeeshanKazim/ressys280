/**
 * script.js
 * -----------
 * UI setup and content-based recommendation logic using Jaccard similarity
 * over movie genres. Depends on globals from data.js: movies, ratings, loadData().
 */

window.onload = (async () => {
  const resultEl = document.getElementById('result');
  if (resultEl) resultEl.innerText = 'Loading data…';

  await loadData();

  if (!Array.isArray(movies) || movies.length === 0) {
    if (resultEl) resultEl.innerText = 'No movie data found. Please ensure u.item is present.';
    return;
  }

  populateMoviesDropdown();
  if (resultEl) resultEl.innerText = 'Data loaded. Please select a movie.';
})();

/**
 * Populate the #movie-select dropdown (sorted by title).
 */
function populateMoviesDropdown() {
  const select = document.getElementById('movie-select');
  if (!select) return;

  select.innerHTML = '';
  const sorted = [...movies].sort((a, b) =>
    a.title.localeCompare(b.title, undefined, { sensitivity: 'base' })
  );

  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.disabled = true;
  placeholder.selected = true;
  placeholder.innerText = 'Select a movie…';
  select.appendChild(placeholder);

  for (const m of sorted) {
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.innerText = m.title;
    select.appendChild(opt);
  }
}

/**
 * Helper: Generate a poster image URL for a given title.
 * Uses a placeholder image with the title text so the UI always has visuals.
 */
function posterFor(title) {
  const text = encodeURIComponent(title);
  // 300x450 looks nice in a grid; bold white text on darker bg
  return `https://via.placeholder.com/300x450.png?text=${text}`;
}

/**
 * Jaccard similarity between two Sets.
 */
function jaccard(setA, setB) {
  let intersection = 0;
  for (const val of setA) if (setB.has(val)) intersection++;
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Render recommendation cards into #recommendations container.
 * Expects an array of movie objects with optional score.
 */
function renderRecommendations(recs) {
  const grid = document.getElementById('recommendations');
  if (!grid) return;

  grid.innerHTML = '';

  recs.forEach((m, idx) => {
    const card = document.createElement('article');
    card.className = 'card';

    const img = document.createElement('img');
    img.src = posterFor(m.title);
    img.alt = `${m.title} poster`;
    img.loading = 'lazy';

    const badge = document.createElement('div');
    badge.className = 'badge';
    // Show rank and (if available) similarity as percentage
    const pct = (m.score != null) ? ` • ${(m.score * 100).toFixed(0)}%` : '';
    badge.innerText = `#${idx + 1}${pct}`;

    const meta = document.createElement('div');
    meta.className = 'card-meta';

    const t = document.createElement('div');
    t.className = 'title';
    t.innerText = m.title;

    const g = document.createElement('div');
    g.className = 'genres';
    g.innerText = m.genres && m.genres.length ? m.genres.join(' • ') : '—';

    meta.appendChild(t);
    meta.appendChild(g);

    card.appendChild(img);
    card.appendChild(badge);
    card.appendChild(meta);

    grid.appendChild(card);
  });
}

/**
 * Compute and display content-based recommendations using Jaccard similarity.
 * - Keeps the result sentence
 * - Renders a Netflix-like grid of poster cards in a separate section
 */
function getRecommendations() {
  const resultEl = document.getElementById('result');
  const select = document.getElementById('movie-select');
  if (!select) return;

  const selectedVal = select.value;
  if (!selectedVal) {
    if (resultEl) resultEl.innerText = 'Please choose a movie first.';
    renderRecommendations([]); // clear
    return;
  }
  const selectedId = parseInt(selectedVal, 10);

  const likedMovie = movies.find(m => m.id === selectedId);
  if (!likedMovie) {
    if (resultEl) resultEl.innerText = 'Selected movie not found. Try another selection.';
    renderRecommendations([]); // clear
    return;
  }

  const likedGenresSet = new Set(likedMovie.genres);
  const candidateMovies = movies.filter(m => m.id !== likedMovie.id);

  const scoredMovies = candidateMovies.map(candidate => {
    const candidateSet = new Set(candidate.genres);
    const score = jaccard(likedGenresSet, candidateSet);
    return { ...candidate, score };
  });

  // Sort by similarity desc, then title for stable ties
  scoredMovies.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.title.localeCompare(b.title, undefined, { sensitivity: 'base' });
  });

  // TOP visual picks (feel more “Netflix” if we show more than 2)
  const topVisual = scoredMovies.slice(0, 8);
  renderRecommendations(topVisual);

  // Keep the concise sentence highlighting the very top two (as originally specified)
  const topTwo = scoredMovies.slice(0, 2);
  if (topTwo.length > 0) {
    const recTitles = topTwo.map(m => m.title).join(', ');
    if (resultEl) {
      resultEl.innerText = `Because you liked "${likedMovie.title}", we recommend: ${recTitles}`;
    }
  } else {
    if (resultEl) {
      resultEl.innerText = `We couldn't find similar movies to "${likedMovie.title}". Try another selection.`;
    }
  }
}

/* Debug helpers (optional) */
window._debug = {
  movies: () => movies,
  ratings: () => ratings
};
