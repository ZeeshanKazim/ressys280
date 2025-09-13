/* UI + Cosine Similarity Recommender with Always-Visible Breakdown */
"use strict";

/** ====== Initialization ====== */
window.addEventListener("DOMContentLoaded", async () => {
  const resultEl = document.getElementById("result");
  const btn = document.getElementById("recommend-btn");

  if (resultEl) resultEl.textContent = "Initializing…";
  await loadData();              // from data.js

  populateMoviesDropdown();
  wireSearch();

  if (btn) btn.addEventListener("click", getRecommendations);
  if (resultEl) resultEl.textContent = "Data loaded. Please select a movie.";
});

/** ====== Dropdown population (alphabetical) ====== */
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

/** ====== Search ====== */
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

function ensureSelection() {
  const select = document.getElementById("movie-select");
  if (!select) return null;
  const chosen = select.value;
  if (chosen) return chosen;
  const firstReal = [...select.options].find(o => o.value !== "");
  if (firstReal) { select.value = firstReal.value; return firstReal.value; }
  return null;
}

/** Per-genre breakdown */
function computeBreakdown(liked, cand) {
  const likedSet = new Set(liked.genres || []);
  const candSet  = new Set(cand.genres  || []);

  const overlap   = [...likedSet].filter(g => candSet.has(g));
  const onlyLiked = [...likedSet].filter(g => !candSet.has(g));
  const onlyCand  = [...candSet].filter(g => !likedSet.has(g));

  const lenA = (liked.genres || []).length;
  const lenB = (cand.genres  || []).length;
  const nA = Math.sqrt(lenA);
  const nB = Math.sqrt(lenB);
  const perGenrePct = (nA > 0 && nB > 0) ? (100 / (nA * nB)) : 0;

  const contributions = overlap.map(g => ({
    genre: g,
    pct: Math.round(perGenrePct * 10) / 10
  }));

  return { overlap, onlyLiked, onlyCand, contributions };
}

function buildBreakdownElement(likedTitle, breakdown) {
  const det = document.createElement("details");
  det.className = "breakdown";
  det.open = true; // show by default

  const sum = document.createElement("summary");
  sum.innerHTML = `<span>Similarity details</span><span class="caret" aria-hidden="true"></span>`;
  det.appendChild(sum);

  const box = document.createElement("div");
  box.className = "breakdown-content";

  // Shared
  const g1 = document.createElement("div");
  g1.innerHTML = `<p class="group-title">Shared genres</p>`;
  const chips1 = document.createElement("div");
  chips1.className = "chips";
  (breakdown.overlap.length ? breakdown.overlap : ["— none —"]).forEach(txt => {
    const c = document.createElement("span"); c.className="chip"; c.textContent = txt; chips1.appendChild(c);
  });
  g1.appendChild(chips1);

  // Only liked
  const g2 = document.createElement("div");
  g2.innerHTML = `<p class="group-title">Only in “${likedTitle}”</p>`;
  const chips2 = document.createElement("div");
  chips2.className = "chips";
  (breakdown.onlyLiked.length ? breakdown.onlyLiked : ["— none —"]).forEach(txt => {
    const c = document.createElement("span"); c.className="chip"; c.textContent = txt; chips2.appendChild(c);
  });
  g2.appendChild(chips2);

  // Only cand
  const g3 = document.createElement("div");
  g3.innerHTML = `<p class="group-title">Only in recommendation</p>`;
  const chips3 = document.createElement("div");
  chips3.className = "chips";
  (breakdown.onlyCand.length ? breakdown.onlyCand : ["— none —"]).forEach(txt => {
    const c = document.createElement("span"); c.className="chip"; c.textContent = txt; chips3.appendChild(c);
  });
  g3.appendChild(chips3);

  // Contributions
  const g4 = document.createElement("div");
  g4.innerHTML = `<p class="group-title">Contribution to match</p>`;
  const list = document.createElement("div");
  list.className = "contrib";
  if (breakdown.contributions.length) {
    breakdown.contributions.forEach(({ genre, pct }) => {
      const row = document.createElement("div");
      row.className = "contrib-row";

      const label = document.createElement("div");
      label.className = "contrib-label";
      label.textContent = genre;

      const bar = document.createElement("div");
      bar.className = "contrib-bar";
      const fill = document.createElement("div");
      fill.className = "contrib-fill";
      fill.style.width = `${pct}%`;
      bar.appendChild(fill);

      const pctEl = document.createElement("div");
      pctEl.className = "contrib-pct";
      pctEl.textContent = `${pct}%`;

      row.appendChild(label);
      row.appendChild(bar);
      row.appendChild(pctEl);
      list.appendChild(row);
    });
  } else {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No shared genres, so cosine is 0%.";
    list.appendChild(p);
  }
  g4.appendChild(list);

  box.appendChild(g1); box.appendChild(g2); box.appendChild(g3); box.appendChild(g4);
  det.appendChild(box);

  // rotate caret on toggle (for visual feedback)
  det.addEventListener("toggle", () => {
    const caret = sum.querySelector(".caret");
    if (caret) caret.style.transform = det.open ? "rotate(180deg)" : "rotate(0deg)";
  });

  return det;
}

/** Build a tile (card + breakdown) */
function createMovieTile(movie, percentMatch, breakdown, likedTitle) {
  const tile = document.createElement("div");
  tile.className = "movie-tile";

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

  // breakdown
  const breakdownEl = buildBreakdownElement(likedTitle, breakdown);

  // Clicking the card toggles the details
  card.addEventListener("click", () => { breakdownEl.open = !breakdownEl.open; });

  tile.appendChild(card);
  tile.appendChild(breakdownEl);
  return tile;
}

/** ====== Recommendations (Cosine) ====== */
function getRecommendations() {
  const select = document.getElementById("movie-select");
  const resultEl = document.getElementById("result");
  const grid = document.getElementById("recommendations");
  if (!select || !grid) return;

  try {
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

    const candidates = movies.filter(m => Number(m.id) !== selectedId);

    const scored = candidates.map(cand => ({
      movie: cand,
      score: cosineSimilarity(likedMovie.genreVector, cand.genreVector)
    }));

    scored.sort((a, b) => b.score - a.score);
    const TOP_N = 8;
    const top = scored.slice(0, TOP_N);

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
      const breakdown = computeBreakdown(likedMovie, movie);
      grid.appendChild(createMovieTile(movie, percent, breakdown, likedMovie.title));
    }
  } catch (err) {
    console.error(err);
    if (resultEl) resultEl.textContent = `Error: ${err.message}`;
    clearRecommendations();
  }
}
