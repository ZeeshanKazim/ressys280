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
  for (let i = 0; i < a.length; i++) s +
