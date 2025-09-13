/* UI & Recommendation Logic Module (Cosine Similarity + Prime-style UI)
 * --------------------------------------------------------------------
 * - Initializes after data load
 * - Populates dropdown
 * - Computes content-based recommendations using Cosine Similarity on
 *   18-dim binary genre vectors.
 * - Renders recommendations as beautiful poster tiles with % match.
 */

"use strict";

/** Init */
window.onload = async function () {
  const resultEl = document.getElementById("result");
  if (resultEl) resultEl.textContent = "Initializing…";

  await loadData();          // from data.js

  populateMoviesDropdown();
  if (resultEl) resultEl.textContent = "Data loaded. Please select a movie.";
};

/** Populate dropdown alphabetically */
function populateMoviesDropdown() {
  const select = document.getElementById("movie-select");
  const resultEl = document.getElementById("result");
  if (!select) return;

  // Clear existing options except placeholder
  select.querySelectorAll("option:not([value=''])").forEach((opt) => opt.remove());

  if (!Array.isArray(movies) || movies.length === 0) {
    if (resultEl) resultEl.textContent = "No movie data available.";
    return;
  }

  const sorted = [...movies].sort((a, b) => a.title.localeCompare(b.title));
  for (const m of sorted) {
    const opt = document.createElement("option");
    opt.value = String(m.id);
    opt.textContent = m.title;
    select.appendChild(opt);
  }
}

/* =================== Math Utils (Cosine) =================== */
/** Dot product of two equal-length numeric arrays */
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
/** L2 norm (magnitude) */
function norm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}
/** Cosine similarity ∈ [0,1] for non-negative vectors; falls back to 0 when a norm is 0. */
function cosineSimilarity(vecA, vecB) {
  const nA = norm(vecA);
  const nB = norm(vecB);
  if (nA === 0 || nB === 0) return 0;
  const d = dot(vecA, vecB);
  const cos = d / (nA * nB);
  // Guard tiny numerical noise
  return Math.max(0, Math.min(1, cos));
}

/* =================== Rendering =================== */
function clearRecommendations() {
  const grid = document.getElementById("recommendations");
  if (grid) grid.innerHTML = "";
}

/** Create a movie card element with poster, title, and % badge */
function createMovieCard(movie, percentMatch) {
  const card = document.createElement("article");
  card.className = "movie-card";
  card.setAttribute("tabindex", "0");
  card.setAttribute("aria-label", `${movie.title} — ${percentMatch}% match`);

  const img = document.createElement("img");
  img.className = "poster";
  img.src = movie.poster;
  img.alt = movie.title;

  const footer = document.createElement("div");
  footer.className = "movie-footer";

  const h = document.createElement("h3");
  h.className = "movie-title";
  h.textContent = movie.title;

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = `${percentMatch}%`;
  badge.setAttribute("aria-label", `${percentMatch}% similar`);

  footer.appendChild(h);
  card.appendChild(img);
  card.appendChild(footer);
  card.appendChild(badge);

  return card;
}

/* =================== Main: Recommendations =================== */
/**
 * Steps:
 *  1) Read selected movie ID
 *  2) Find liked movie
 *  3) Prepare candidate list
 *  4) Score candidates with Cosine Similarity over genreVector
 *  5) Sort desc
 *  6) Pick top N
 *  7) Render as cards + percentage
 */
function getRecommendations() {
  const select = document.getElementById("movie-select");
  const resultEl = document.getElementById("result");
  const grid = document.getElementById("recommendations");

  if (!select || !grid) return;

  if (!Array.isArray(movies) || movies.length === 0) {
    if (resultEl) resultEl.textContent = "Data not loaded.";
    return;
  }

  // 1) Input
  const selectedVal = select.value;
  const selectedId = Number.parseInt(selectedVal, 10);
  if (!selectedVal || Number.isNaN(selectedId)) {
    if (resultEl) resultEl.textContent = "Please select a movie first.";
    clearRecommendations();
    return;
  }

  // 2) Find liked
  const likedMovie = movies.find((m) => m.id === selectedId);
  if (!likedMovie) {
    if (resultEl) resultEl.textContent = "Selected movie not found.";
    clearRecommendations();
    return;
  }

  // 3) Candidates (exclude liked)
  const candidates = movies.filter((m) => m.id !== likedMovie.id);

  // 4) Score (cosine on binary genre vectors)
  const scored = candidates.map((cand) => {
    const score = cosineSimilarity(likedMovie.genreVector, cand.genreVector);
    return { movie: cand, score };
  });

  // 5) Sort
  scored.sort((a, b) => b.score - a.score);

  // 6) Top N (Prime-like feel -> show 8)
  const TOP_N = 8;
  const top = scored.slice(0, TOP_N);

  // 7) Render
  if (resultEl) {
    resultEl.textContent = `Because you liked “${likedMovie.title}”, here are your most similar picks (match %):`;
  }
  clearRecommendations();

  if (top.length === 0) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No similar movies found.";
    grid.appendChild(p);
    return;
  }

  for (const { movie, score } of top) {
    // Convert to percentage; emphasize “from how similar to full similar”
    // 0% = no overlap, 100% = identical genre vector
    const percent = Math.round(score * 100);
    const card = createMovieCard(movie, percent);
    grid.appendChild(card);
  }
}
