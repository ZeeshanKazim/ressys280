/* UI + Cosine Similarity Recommender with Right-Side Inspector */
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

  const header = document.createElement("div");
  header.className = "inspector__header";

  const title = document.createElement("h3");
  title.className = "inspector__title";
  title.textContent = movie.title;

  const sub = document.createElement("p");
  sub.className = "inspector__subtitle";
  sub.textContent = `Match: ${percentMatch}% — because you liked “${liked.title}”`;

  header.appendChild(title);
  header.appendChild(sub);
  panel.appendChild(header);

  const breakdown = computeBreakdown(liked, movie);

  const makeSection = (heading, items) => {
    const t = document.createElement("p");
    t.className = "section-title";
    t.textContent = heading;
    const chips = document.createElement("div");
    chips.className = "chips";
    if (items && items.length) {
      items.forEach(g => { const c = document.createElement("span"); c.className="chip"; c.textContent=g; chips.appendChild(c); });
    } else {
      const c = document.createElement("span"); c.className="chip"; c.textContent="— none —"; chips.appendChild(c);
    }
    panel.appendChild(t); panel.appendChild(chips);
  };

  makeSection("Shared genres", breakdown.overlap);
  makeSection(`Only in “${liked.title}”`, breakdown.onlyLiked);
  makeSection("Only in recommendation", breakdown.onlyCand);

  const t4 = document.createElement("p");
  t4.className = "section-title";
  t4.textContent = "Contribution to match";
  panel.appendChild(t4);

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
  panel.appendChild(list);
}

/** Build one recommendation card */
function buildCard(movie, percentMatch, liked) {
  const card = document.createElement("article");
  card.className = "movie-card";
  card.tabIndex = 0;
  card.dataset.id = String(movie.id);

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

  const activate = () => {
    // highlight
    document.querySelectorAll(".movie-card.selected").forEach(el => el.classList.remove("selected"));
    card.classList.add("selected");
    // render inspector
    renderInspector(liked, movie, percentMatch);
  };

  card.addEventListener("click", activate);
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
  });

  return card;
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

    currentLiked = likedMovie;

    const candidates = movies.filter(m => Number(m.id) !== selectedId);

    const scored = candidates.map(cand => ({
      movie: cand,
      score: cosineSimilarity(likedMovie.genreVector, cand.genreVector)
    }));

    scored.sort((a, b) => b.score - a.score);
    const TOP_N = 10;
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

    // Fill grid and auto-select the top card to show details immediately
    let firstCard = null;
    for (const { movie, score } of top) {
      const percent = Math.round(score * 100);
      const card = buildCard(movie, percent, likedMovie);
      if (!firstCard) firstCard = { movie, percent, node: card };
      grid.appendChild(card);
    }
    if (firstCard) {
      firstCard.node.classList.add("selected");
      renderInspector(likedMovie, firstCard.movie, firstCard.percent);
    }
  } catch (err) {
    console.error(err);
    if (resultEl) resultEl.textContent = `Error: ${err.message}`;
    clearRecommendations();
  }
}
