/* UI & Recommendation Logic Module
 * --------------------------------
 * Depends on data.js having loaded and parsed the datasets.
 * Exposes:
 *   - getRecommendations() : called by the button in index.html
 */

"use strict";

/** Initialize app after window load */
window.onload = async function () {
  const resultEl = document.getElementById("result");
  if (resultEl) resultEl.textContent = "Initializing…";

  await loadData(); // from data.js

  populateMoviesDropdown();
  if (resultEl) resultEl.textContent = "Data loaded. Please select a movie.";
};

/** Populate the movie dropdown, sorted alphabetically by title */
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

  // Sort by title
  const sorted = [...movies].sort((a, b) => a.title.localeCompare(b.title));

  // Populate
  for (const m of sorted) {
    const opt = document.createElement("option");
    opt.value = String(m.id);
    opt.textContent = m.title;
    select.appendChild(opt);
  }
}

/**
 * Compute Jaccard similarity between two Sets of strings.
 * J(A, B) = |A ∩ B| / |A ∪ B|
 */
function jaccardSimilarity(setA, setB) {
  const intersectionSize = (() => {
    let count = 0;
    for (const val of setA) if (setB.has(val)) count++;
    return count;
  })();
  const unionSize = new Set([...setA, ...setB]).size;
  return unionSize === 0 ? 0 : intersectionSize / unionSize;
}

/**
 * Main entry point for recommendations (invoked by the button).
 * Steps per specification:
 *  1) Read selected movie ID
 *  2) Find liked movie
 *  3) Prepare genre set and candidate list
 *  4) Score candidates by Jaccard similarity
 *  5) Sort descending by score
 *  6) Pick top 2
 *  7) Display result
 */
function getRecommendations() {
  const select = document.getElementById("movie-select");
  const resultEl = document.getElementById("result");

  if (!select) return;
  if (!Array.isArray(movies) || movies.length === 0) {
    if (resultEl) resultEl.textContent = "Data not loaded.";
    return;
  }

  // Step 1: Get user selection
  const selectedVal = select.value;
  const selectedId = Number.parseInt(selectedVal, 10);

  if (!selectedVal || Number.isNaN(selectedId)) {
    if (resultEl) resultEl.textContent = "Please select a movie first.";
    return;
  }

  // Step 2: Find the liked movie
  const likedMovie = movies.find((m) => m.id === selectedId);
  if (!likedMovie) {
    if (resultEl) resultEl.textContent = "Selected movie not found.";
    return;
  }

  // Step 3: Prepare sets and candidates
  const likedSet = new Set(likedMovie.genres || []);
  const candidateMovies = movies.filter((m) => m.id !== likedMovie.id);

  // Step 4: Score by Jaccard similarity
  const scoredMovies = candidateMovies.map((cand) => {
    const candSet = new Set(cand.genres || []);
    const score = jaccardSimilarity(likedSet, candSet);
    return { ...cand, score };
  });

  // Step 5: Sort descending
  scoredMovies.sort((a, b) => b.score - a.score);

  // Step 6: Top 2
  const top = scoredMovies.slice(0, 2);

  // Step 7: Display
  const prettyList =
    top.length > 0
      ? top.map((m) => `${m.title}${m.score ? ` (score: ${m.score.toFixed(2)})` : ""}`).join(", ")
      : "No similar movies found.";

  if (resultEl) {
    resultEl.textContent = `Because you liked “${likedMovie.title}”, we recommend: ${prettyList}`;
  }
}
