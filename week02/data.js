/* Data Handling Module
 * --------------------
 * Responsible for fetching and parsing local MovieLens files:
 *  - u.item : movie metadata with genre flags
 *  - u.data : user ratings
 * Exposes:
 *   - globals: movies[], ratings[]
 *   - async function: loadData()
 *   - parsers: parseItemData(text), parseRatingData(text)
 */

"use strict";

// Global state (required by spec)
let movies = [];
let ratings = [];

/**
 * Load and parse data files from the same directory.
 * Uses sequential awaits per specification.
 */
async function loadData() {
  const statusEl = document.getElementById("result");
  try {
    if (statusEl) statusEl.textContent = "Loading data files…";

    // Fetch and parse u.item first
    const itemResp = await fetch("u.item");
    if (!itemResp.ok) throw new Error(`Failed to load u.item (${itemResp.status})`);
    const itemText = await itemResp.text();
    parseItemData(itemText);

    // Then fetch and parse u.data
    const dataResp = await fetch("u.data");
    if (!dataResp.ok) throw new Error(`Failed to load u.data (${dataResp.status})`);
    const dataText = await dataResp.text();
    parseRatingData(dataText);

    if (statusEl) statusEl.textContent = "Data loaded. Please select a movie.";
  } catch (err) {
    console.error(err);
    if (statusEl) {
      statusEl.textContent = `Error loading data: ${err.message}`;
    }
  }
}

/**
 * Parse u.item content.
 * Each line is pipe-delimited. The last 19 fields are genre flags:
 * [unknown, Action, Adventure, ..., Western].
 * The specification asks to define the 18 genre names from "Action" to "Western"
 * and build the movie's genres array where the flag is '1', skipping 'unknown'.
 */
function parseItemData(text) {
  // 18 named genres, skipping "unknown"
  const GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
  ];

  movies = []; // reset if reloaded
  const lines = text.split("\n");

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    const parts = line.split("|");
    if (parts.length < 6) continue; // malformed

    const id = Number(parts[0]);
    const title = parts[1];

    // The last 19 fields are the genre flags (unknown + 18 named)
    const flags = parts.slice(-19);
    if (flags.length !== 19) continue;

    // Skip index 0 (unknown). Map indices 1..18 to the 18 names above.
    const genres = [];
    for (let i = 1; i < flags.length; i++) {
      if (flags[i] === "1") {
        const genreName = GENRES[i - 1];
        if (genreName) genres.push(genreName);
      }
    }

    movies.push({ id, title, genres });
  }
}

/**
 * Parse u.data content.
 * Each line is tab-delimited: userId \t itemId \t rating \t timestamp
 */
function parseRatingData(text) {
  ratings = []; // reset if reloaded
  const lines = text.split("\n");

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    const parts = line.split("\t");
    if (parts.length < 4) continue;

    const userId = Number(parts[0]);
    const itemId = Number(parts[1]);
    const rating = Number(parts[2]);
    const timestamp = Number(parts[3]);

    ratings.push({ userId, itemId, rating, timestamp });
  }
}
