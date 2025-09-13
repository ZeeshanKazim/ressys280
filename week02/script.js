/* UI + Cosine Similarity Recommender with Animated Background + Inspector */
"use strict";

let currentLiked = null;

/** ====== Initialization ====== */
window.addEventListener("DOMContentLoaded", async () => {
  const resultEl = document.getElementById("result");
  const btn = document.getElementById("recommend-btn");

  if (resultEl) resultEl.textContent = "Initializing…";
  await loadData(); // from data.js

  populateMoviesDropdown();
  wireSearch();

  if (btn) btn.addEventListener("click", getRecommendations);

  // Animated Netflix-like background
  startBackgroundHueCycle();

  if (resultEl) resultEl.textContent = "Data loaded. Please select a movie.";
});

/** Rotate backdrop hue every few seconds for a Netflix-like vibe */
function startBackgroundHueCycle() {
  const hues = [0, 20, 45, 90, 150, 210, 260, 320];
  let i = 0;
  const apply = () =>
    document.documentElement.style.setProperty("--heroHue", `${hues[i]}deg`);
  apply();
  setInterval(() => { i = (i + 1) % hues.length; apply(); }, 4000);
}

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

/** ====== Per-genre breakdown ====== */
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

/** ====== Inspector rendering ====== */
function renderInspector(liked, movie, percentMatch) {
  const panel = document.getElementById("inspector");
  panel.innerHTML = ""; // reset

  const title = document.createElement("h3");
  title.className = "inspector__title";
  title.textContent = movie.title;

  const sub = document.createElement("p");
  sub.className = "inspector__subtitle";
  sub.textContent = `Match: ${percentMatch}% — because you liked “${liked.title}”`;

  panel.appendChild(title);
  panel.appendChild(sub);

  const breakdown = computeBreakdown(liked, movi
