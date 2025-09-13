/**
 * data.js
 * ----------
 * Responsible for fetching and parsing MovieLens-style data files:
 *  - u.item : movie metadata (title + 19 binary genre flags)
 *  - u.data : user ratings  (userId, itemId, rating, timestamp)
 *
 * Exposes:
 *  - Global arrays: movies, ratings
 *  - loadData(): async function to fetch & parse the files
 *  - parseItemData(text), parseRatingData(text)
 */

let movies = [];  // Array<{ id: number, title: string, genres: string[] }>
let ratings = []; // Array<{ userId: number, itemId: number, rating: number, timestamp: number }>

/**
 * Fetch and parse both data files. This returns once both are loaded & parsed.
 * In case of failure, it writes a user-friendly error message to #result.
 */
async function loadData() {
  const resultEl = document.getElementById('result');

  try {
    // --- Fetch u.item (movie metadata) ---
    const itemResp = await fetch('u.item');
    if (!itemResp.ok) throw new Error(`Failed to load u.item (${itemResp.status})`);
    const itemText = await itemResp.text();
    parseItemData(itemText);

    // --- Fetch u.data (ratings) ---
    const dataResp = await fetch('u.data');
    if (!dataResp.ok) throw new Error(`Failed to load u.data (${dataResp.status})`);
    const dataText = await dataResp.text();
    parseRatingData(dataText);
  } catch (err) {
    console.error(err);
    if (resultEl) {
      resultEl.innerText = `Error: Unable to load data files. ${err.message}`;
    }
  }
}

/**
 * Parse contents of u.item
 * Format (MovieLens 100k style):
 *   movieId | title | release_date | video_release_date | IMDb_URL | unknown | Action | Adventure | ... | Western
 *
 * Spec: define the 18 genres from "Action" to "Western" (ignore 'unknown'),
 * and iterate over the last 19 fields to map genre flags.
 */
function parseItemData(text) {
  const genreNames = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
  ];

  movies = []; // reset if re-called

  const lines = text.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;

    const parts = line.split('|');
    if (parts.length < 6 + 19) continue;

    const id = parseInt(parts[0], 10);
    const title = parts[1];

    const flagsStart = parts.length - 19;

    const genres = [];
    for (let i = 1; i <= 18; i++) {
      const flag = parts[flagsStart + i];
      if (flag === '1') {
        genres.push(genreNames[i - 1]);
      }
    }

    movies.push({ id, title, genres });
  }
}

/**
 * Parse contents of u.data
 * Format (tab-separated):
 *   userId \t itemId \t rating \t timestamp
 */
function parseRatingData(text) {
  ratings = []; // reset if re-called

  const lines = text.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;

    const parts = line.split('\t');
    if (parts.length < 4) continue;

    const userId = parseInt(parts[0], 10);
    const itemId = parseInt(parts[1], 10);
    const rating = parseInt(parts[2], 10);
    const timestamp = parseInt(parts[3], 10);

    ratings.push({ userId, itemId, rating, timestamp });
  }
}
