/* UI + Cosine Similarity Recommender (Prime/Hotstar style) */
"use strict";

/** Small shared list for reconstructing vectors if any are missing */
const GENRES_18 = [
  "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
];

window.addEventListener("DOMContentLoaded", async () => {
  const resultEl = document.getElementById("result");
  if (resultEl) resultEl.textContent = "Initializing…";

  await loadData();              // from data.js

  populateMoviesDropdown();
  wireSearch();

  // Also wire the click in JS (in addition to inline) for belt & suspenders.
  const btn = document.getElementById("recommend-btn");
  if (btn) btn.addEventListener("click", getRecommendations);

  if (resultEl) resultEl.textContent = "Data loaded. Please select a movie.";
});

/** Guarantee a valid 18-dim vector even if missing on the object */
function ensureVector(movie) {
  if (Array.isArray(movie.genreVector) && movie.genreVector.length === 18) {
    return movie.genreVector;
  }
  const vec = new Array(18).fill(0);
  (movie.genres || []).forEach(g => {
    const idx = GENRES_18.indexOf(g);
    if (idx >= 0) vec[idx] = 1;
  });
  movie.genreVector = vec; // cache back
  return vec;
}

/** Populate dropdown (alphabetical); keep selection if still present */
function populateMoviesDropdown(list = movies) {
  const select = document.getElementById("movie-select");
  const resultEl = document.getElementById("result");
  if (!select) return;

  const prev = select.value || "";
  select.querySelectorAll("option:not([value=''])").forEach((opt) => opt.remove());

  if (!Array.isArray(list) || list.length === 0) {
    if (resultEl) resultEl.textContent = "No movie data available.";
    return;
  }

  const sorted = [...list].sort((a, b) => a.title.localeCompare(b.title));
  for (const m of sorted) {
    const opt = document.createElement("option");
    opt.value = String(m.id);
    opt.textContent = m.title;
    select.appendChild(opt);
  }
  if ([...select.options].some(o => o.value === prev)) select.value = prev;
}

/** Search input filters dropdown; Enter triggers recommend */
function wireSearch() {
  const input = document.getElementById("search-input");
  const select = document.getElementById("movie-select");
  const btn = document.getElementById("recommend-btn");
  if (!input || !select) return;

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    const filtered = q ? movies.filter(m => m.title.toLowerCase().includes(q)) : movies;
    populateMoviesDropdown(filtered);
    if (filtered.length === 1) select.value = String(filtered[0].id);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && btn) btn.click();
  });
}

/** Cosine Similarity helpers */
function dot(a, b) { let s = 0; const n = Math.min(a.length, b.length); for (let i=0;i<n;i++) s += a[i]*b[i]; return s; }
function norm(a) { let s = 0; for (let i=0;i<a.length;i++) s += a[i]*a[i]; return Math.sqrt(s); }
function cosineSimilarity(vecA, vecB) {
  const vA = ensureVector({ genreVector: vecA, genres: [] }); // guard
  const vB = ensureVector({ genreVector: vecB, genres: [] });
  const nA = norm(vA), nB = norm(vB);
  if (nA === 0 || nB === 0) return 0;
  const cos = dot(vA, vB) / (nA * nB);
  return Math.max(0, Math.min(1, cos));
}

/** Rendering helpers */
function clearRecommendations() {
  const grid = document.getElementById("recommendations");
  if (grid) grid.innerHTML = "";
}
function createMovieCard(movie, percentMatch) {
  const card = document.createElement("article");
  card.className = "movie-card";
  card.setAttribute("tabindex", "0");
  card.setAttribute("aria-label", `${movie.title} — ${percentMatch}% match`);

  const img = document.createElement("img");
  img.className = "poster";
  img.src = movie.poster;
  img.alt = movie.title;
  img.loading = "lazy";

  const footer = document.createElement("div");
  footer.className = "movie-footer";
  const h = document.createElement("h3");
  h.className = "movie-title";
  h.textContent = movie.title;

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = `${percentMatch}%`;
  if (percentMatch === 100) badge.classList.add("full");

  footer.appendChild(h);
  card.appendChild(img);
  card.appendChild(footer);
  card.appendChild(badge);
  return card;
}

/** Main: Recommendations (Cosine) */
function getRecommendations() {
  try {
    const select = document.getElementById("movie-select");
    const resultEl = document.getElementById("result");
    const grid = document.getElementById("recommendations");
    if (!select || !grid) return;

    // Safe selection read
    const opt = select.selectedOptions && select.selectedOptions[0] ? select.selectedOptions[0] : null;
    const selectedVal = opt ? opt.value : select.value;
    const selectedId = Number.parseInt(selectedVal, 10);

    if (!selectedVal || Number.isNaN(selectedId)) {
      if (resultEl) resultEl.textContent = "Please select a movie first.";
      clearRecommendations();
      return;
    }

    const likedMovie = movies.find(m => Number(m.id) === selectedId);
    if (!likedMovie) {
      if (resultEl) resultEl.textContent = "Selected movie not found.";
      clearRecommendations();
      return;
    }

    const likedVec = ensureVector(likedMovie);
    const candidates = movies.filter(m => Number(m.id) !== selectedId);

    const scored = candidates.map(cand => {
      const score = cosineSimilarity(likedVec, ensureVector(cand));
      return { movie: cand, score };
    });

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, 10);

    if (resultEl) resultEl.textContent = `Because you liked “${likedMovie.title}”, your similar picks:`;
    clearRecommendations();

    if (!top.length) {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = "No similar movies found.";
      grid.appendChild(p);
      return;
    }

    for (const { movie, score } of top) {
      const percent = Math.round(score * 100);
      grid.appendChild(createMovieCard(movie, percent));
    }
  } catch (err) {
    console.error(err);
    const resultEl = document.getElementById("result");
    if (resultEl) resultEl.textContent = `Error: ${err.message}`;
  }
}

/* Expose globally so inline onclick works everywhere */
window.getRecommendations = getRecommendations;
