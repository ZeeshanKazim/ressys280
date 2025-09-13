/* UI + Cosine Similarity Recommender (Prime/Hotstar style)
 * -------------------------------------------------------- */
"use strict";

/** ====== Initialization ====== */
window.onload = async function () {
  const resultEl = document.getElementById("result");
  if (resultEl) resultEl.textContent = "Initializing…";

  await loadData();              // from data.js

  populateMoviesDropdown();
  wireSearch();                  // live filter for the dropdown

  if (resultEl) resultEl.textContent = "Data loaded. Please select a movie.";
};

/** ====== Dropdown population (alphabetical) ====== */
function populateMoviesDropdown(list = movies) {
  const select = document.getElementById("movie-select");
  const resultEl = document.getElementById("result");
  if (!select) return;

  // Clear all except placeholder
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
}

/** ====== Search that filters the dropdown options ====== */
function wireSearch() {
  const input = document.getElementById("search-input");
  const select = document.getElementById("movie-select");
  if (!input || !select) return;

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    if (q === "") {
      populateMoviesDropdown(movies);
      return;
    }
    const filtered = movies.filter(m => m.title.toLowerCase().includes(q));
    populateMoviesDropdown(filtered);
    // If exactly one match, preselect it for convenience
    if (filtered.length === 1) select.value = String(filtered[0].id);
  });

  // Pressing Enter in search triggers recommendations for current selection
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      getRecommendations();
    }
  });
}

/** ====== Math: Cosine Similarity ====== */
function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}
function cosineSimilarity(vecA, vecB) {
  const nA = norm(vecA);
  const nB = norm(vecB);
  if (nA === 0 || nB === 0) return 0;  // safe guard for empty genre vectors
  const cos = dot(vecA, vecB) / (nA * nB);
  return Math.max(0, Math.min(1, cos));
}

/** ====== Render helpers ====== */
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

/** ====== Recommendations (Cosine) ====== */
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
  const selectedId = Number(selectedVal);
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

  // 3) Candidates (exclude liked) — **bug-proofing**
  const candidates = movies.filter((m) => Number(m.id) !== Number(likedMovie.id));

  // 4) Score (cosine on binary genre vectors)
  const scored = candidates.map((cand) => {
    const score = cosineSimilarity(likedMovie.genreVector, cand.genreVector);
    return { movie: cand, score };
  });

  // 5) Sort
  scored.sort((a, b) => b.score - a.score);

  // 6) Top N
  const TOP_N = 10;
  let top = scored.slice(0, TOP_N);

  // If everything is 0 (rare), we still show top by title order so UI isn’t empty
  const maxScore = top.length ? top[0].score : 0;
  if (maxScore === 0 && scored.length > 0) {
    top = scored.slice(0, TOP_N); // already sorted; all 0 is fine visually
  }

  // 7) Render
  if (resultEl) {
    resultEl.textContent = `Because you liked “${likedMovie.title}”, your similar picks:`;
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
    const percent = Math.round(score * 100);
    const card = createMovieCard(movie, percent);
    grid.appendChild(card);
  }
}
