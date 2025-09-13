/* UI + Cosine Similarity Recommender (Prime/Hotstar style) */
"use strict";

/** ====== Initialization ====== */
window.addEventListener("DOMContentLoaded", async () => {
  const resultEl = document.getElementById("result");
  const btn = document.getElementById("recommend-btn");

  if (resultEl) resultEl.textContent = "Initializing…";

  await loadData();              // from data.js

  populateMoviesDropdown();
  wireSearch();

  // Wire the click in JS (more reliable than inline onclick)
  if (btn) btn.addEventListener("click", getRecommendations);

  if (resultEl) resultEl.textContent = "Data loaded. Please select a movie.";
});

/** ====== Dropdown population (alphabetical) ====== */
function populateMoviesDropdown(list = movies) {
  const select = document.getElementById("movie-select");
  const resultEl = document.getElementById("result");
  if (!select) return;

  const prev = select.value || "";

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

  // Preserve selection if still present after filtering
  if ([...select.options].some(o => o.value === prev)) {
    select.value = prev;
  }
}

/** ====== Search that filters the dropdown options ====== */
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

  // Enter triggers recommendations
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && btn) btn.click();
  });
}

/** ====== Math: Cosine Similarity ====== */
function dot(a, b) { let s = 0; const n = Math.min(a.length, b.length); for (let i=0;i<n;i++) s += a[i]*b[i]; return s; }
function norm(a) { let s = 0; for (let i=0;i<a.length;i++) s += a[i]*a[i]; return Math.sqrt(s); }
function cosineSimilarity(vecA, vecB) {
  const nA = norm(vecA), nB = norm(vecB);
  if (nA === 0 || nB === 0) return 0;
  const cos = dot(vecA, vecB) / (nA * nB);
  return Math.max(0, Math.min(1, cos));
}

/** ====== Helpers ====== */
function clearRecommendations() {
  const grid = document.getElementById("recommendations");
  if (grid) grid.innerHTML = "";
}

/** If nothing selected (e.g., placeholder), auto-select the first real option. */
function ensureSelection() {
  const select = document.getElementById("movie-select");
  if (!select) return null;
  const chosen = select.value;
  if (chosen) return chosen;
  // Pick first non-empty option if available
  const firstReal = [...select.options].find(o => o.value !== "");
  if (firstReal) {
    select.value = firstReal.value;
    return firstReal.value;
  }
  return null;
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

  try {
    // Ensure we have a valid selection
    const selectedVal = ensureSelection();
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

    // Candidates (exclude liked)
    const candidates = movies.filter(m => Number(m.id) !== selectedId);

    // Score
    const scored = candidates.map(cand => ({
      movie: cand,
      score: cosineSimilarity(likedMovie.genreVector, cand.genreVector)
    }));

    // Sort and take top N
    scored.sort((a, b) => b.score - a.score);
    const TOP_N = 10;
    const top = scored.slice(0, TOP_N);

    // Render
    if (resultEl) {
      resultEl.textContent = `Because you liked “${likedMovie.title}”, your similar picks:`;
    }
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
    if (resultEl) resultEl.textContent = `Error: ${err.message}`;
    clearRecommendations();
  }
}
