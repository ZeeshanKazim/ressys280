/* Data Handling Module (poster URLs & vectors) */
"use strict";

let movies = [];  // { id, title, genres[], genreVector[18], poster }
let ratings = [];

/** Deterministic placeholder poster (no API key required). */
function posterUrlFor(id) {
  const w = 300, h = 450;
  return `https://picsum.photos/seed/mov_${encodeURIComponent(String(id))}/${w}/${h}`;
}

/** Load and parse local data files (u.item first, then u.data). */
async function loadData() {
  const statusEl = document.getElementById("result");
  try {
    if (statusEl) statusEl.textContent = "Loading data files…";

    const itemResp = await fetch("u.item");
    if (!itemResp.ok) throw new Error(`Failed to load u.item (${itemResp.status})`);
    const itemText = await itemResp.text();
    parseItemData(itemText);

    const dataResp = await fetch("u.data");
    if (!dataResp.ok) throw new Error(`Failed to load u.data (${dataResp.status})`);
    const dataText = await dataResp.text();
    parseRatingData(dataText);

    if (statusEl) statusEl.textContent = "Data loaded. Please select a movie.";
  } catch (err) {
    console.error(err);
    if (statusEl) statusEl.textContent = `Error loading data: ${err.message}`;
  }
}

/** Parse u.item into movies[] with genres[] and genreVector[18]. */
function parseItemData(text) {
  const GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
  ];

  movies = [];
  const lines = text.split("\n");

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    const parts = line.split("|");
    if (parts.length < 6) continue;

    const id = Number(parts[0]);
    const title = parts[1];

    // Last 19 fields: unknown + 18 named flags
    const flags = parts.slice(-19);
    if (flags.length !== 19) continue;

    const genres = [];
    const genreVector = new Array(18).fill(0);

    for (let i = 1; i < flags.length; i++) {   // skip index 0 (unknown)
      const bit = flags[i].trim() === "1" ? 1 : 0;
      genreVector[i - 1] = bit;
      if (bit === 1) genres.push(GENRES[i - 1]);
    }

    movies.push({
      id,
      title,
      genres,
      genreVector,
      poster: posterUrlFor(id)
    });
  }
}

/** Parse u.data into ratings[]. */
function parseRatingData(text) {
  ratings = [];
  const lines = text.split("\n");

  for (const rawLine of
