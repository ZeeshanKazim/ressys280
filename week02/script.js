/**
 * script.js
 * -----------
 * Handles UI setup and content-based recommendation logic using Jaccard similarity
 * over movie genres. Depends on globals from data.js: movies, ratings, loadData().
 *
 * NOTE (Homework prompt on page): For homework, change similarity to COSINE SIMILARITY.
 * This file currently uses Jaccard over binary genre vectors, as specified.
 */

/* ---- Optional poster mapping for nicer visuals (fallbacks provided) ----
 * Keys should match movie titles in u.item exactly. Add more entries as desired.
 */
const POSTER_MAP = {
  'Toy Story (1995)': 'https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg',
  'GoldenEye (1995)': 'https://image.tmdb.org/t/p/w500/bmScZqPzQkR2PaWY5Avzkdxl3pG.jpg',
  'Four Rooms (1995)': 'https://image.tmdb.org/t/p/w500/5W1chq9WlH0I1GtIsfRSAxEPPYU.jpg',
  'Get Shorty (1995)': 'https://image.tmdb.org/t/p/w500/1V4EPPGjBJOCaFMEGLjmVnTQ3kP.jpg',
  'Babe (1995)': 'https://image.tmdb.org/t/p/w500/3tajwQ5d2S7RMyCV2wAPYp4nH3L.jpg',
  'Heat (1995)': 'https://image.tmdb.org/t/p/w500/umSVjVdbVwtx5ryCA2QXL44Durm.jpg',
  'Sabrina (1995)': 'https://image.tmdb.org/t/p/w500/d4vGN28AL5soiceqd7qV3CyMfCV.jpg',
  'Sudden Death (1995)': 'https://image.tmdb.org/t/p/w500/8OYe7iH730cdpShUMHbOWcZHhGX.jpg'
  // ...extend as needed
};

/**
 * Build a safe fallback poster (SVG via data URI) if we don't have a mapped poster.
 */
function fallbackPoster(title) {
  const text = encodeURIComponent(title);
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' width='800' height='1200'>
      <defs>
        <linearGradient id='g' x1='0' x2='0' y1='0' y2='1'>
          <stop offset='0%' stop-color='%230a3d91'/>
          <stop offset='100%' stop-color='%23081c4a'/>
        </linearGradient>
      </defs>
      <rect width='100%' height='100%' fill='url(%23g)'/>
      <text x='50%' y='50%' font-family='Arial, Helvetica, sans-serif' font-size='48' fill='%23e5ecff' text-anchor='middle'>
        ${text}
      </text>
    </svg>`;
  return `data:image/svg+xml;utf8,${svg}`;
}

/**
 * Initialize app once the window loads:
 *  - Load data (u.item + u.data)
 *  - Populate dropdown
 *  - Set initial message
 */
window.onload = (async () => {
  const resultEl = document.getElementById('result');

  if (resultEl) resultEl.innerText = 'Loading data…';
  await loadData();

  if (!Array.isArray(movies) || movies.length === 0) {
    if (resultEl) {
      resultEl.innerText = 'No movie data found. Please ensure u.item is present.';
    }
    return;
  }

  populateMoviesDropdown();
  if (resultEl) {
    resultEl.innerText = 'Data loaded. Please select a movie.';
  }
})();

/**
 * Populate the #movie-select dropdown with sorted movie titles.
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
 * Compute and display the top 2 content-based recommendations using Jaccard similarity
 * over genre sets. Triggered by the button click.
 *
 * HOMEWORK HINT:
 * To switch to Cosine similarity over binary genre vectors:
 *  - Represent each movie as an 18-d vector of 0/1.
 *  - cosine = dot(A,B) / (||A|| * ||B||)
 *  - dot(A,B) is the count of overlapping genres (same as intersection size),
 *    ||A|| = sqrt(#genres in A), ||B|| = sqrt(#genres in B).
 */
function getRecommendations() {
  const resultEl = document.getElementById('result');
  const cardsEl = document.getElementById('cards');
  const select = document.getElementById('movie-select');

  if (!select || !cardsEl) return;

  // Step 1: Read user selection and coerce to integer
  const selectedVal = select.value;
  if (!selectedVal) {
    if (resultEl) resultEl.innerText = 'Please choose a movie first.';
    cardsEl.innerHTML = '';
    return;
  }
  const selectedId = parseInt(selectedVal, 10);

  // Step 2: Find the liked movie
  const likedMovie = movies.find(m => m.id === selectedId);
  if (!likedMovie) {
    if (resultEl) resultEl.innerText = 'Selected movie not found. Try another selection.';
    cardsEl.innerHTML = '';
    return;
  }

  // Step 3: Prepare genre sets and candidate list
  const likedGenresSet = new Set(likedMovie.genres);
  const candidateMovies = movies.filter(m => m.id !== likedMovie.id);

  // Step 4: Calculate Jaccard scores for each candidate
  const scoredMovies = candidateMovies.map(candidate => {
    const candidateSet = new Set(candidate.genres);

    let intersectionSize = 0;
    for (const g of candidateSet) {
      if (likedGenresSet.has(g)) intersectionSize++;
    }
    const unionSize = likedGenresSet.size + candidateSet.size - intersectionSize;
    const score = unionSize === 0 ? 0 : intersectionSize / unionSize;

    return { ...candidate, score, _overlap: intersectionSize };
  });

  // Step 5: Sort by score (desc), then by title
  scoredMovies.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.title.localeCompare(b.title, undefined, { sensitivity: 'base' });
  });

  // Step 6: Take top 2 recommendations (per spec)
  const topTwo = scoredMovies.slice(0, 2);

  // Step 7: Display result text and visual cards
  if (topTwo.length === 0) {
    if (resultEl) resultEl.innerText = `We couldn't find similar movies to "${likedMovie.title}". Try another selection.`;
    cardsEl.innerHTML = '';
    return;
  }

  if (resultEl) {
    resultEl.innerText = `Because you liked "${likedMovie.title}", we recommend:`;
  }

  // Render cards
  cardsEl.innerHTML = '';
  topTwo.forEach(movie => {
    const card = document.createElement('article');
    card.className = 'card';

    const img = document.createElement('img');
    img.className = 'poster';
    const poster = POSTER_MAP[movie.title] || fallbackPoster(movie.title);
    img.src = poster;
    img.alt = `${movie.title} poster`;

    const body = document.createElement('div');
    body.className = 'card-body';

    const h3 = document.createElement('h3');
    h3.className = 'card-title';
    h3.textContent = movie.title;

    const chips = document.createElement('div');
    chips.className = 'genres';
    (movie.genres && movie.genres.length ? movie.genres : ['Uncategorized'])
      .slice(0, 5)
      .forEach(g => {
        const c = document.createElement('span');
        c.className = 'chip';
        c.textContent = g;
        chips.appendChild(c);
      });

    body.appendChild(h3);
    body.appendChild(chips);

    card.appendChild(img);
    card.appendChild(body);

    cardsEl.appendChild(card);
  });
}

/* Optional: Expose for console debugging */
window._debug = {
  movies: () => movies,
  ratings: () => ratings
};
