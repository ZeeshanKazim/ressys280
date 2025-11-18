"use strict";

/**
 * script.js
 *
 * UI + recommendation logic.
 * Depends on:
 *  - loadData(), rawMovies, movieFeatures, masterSubGenres, masterThemes
 *    from data.js
 */

/**
 * Initialize app once the window has fully loaded
 * (HTML and CSS available).
 */
window.onload = async function () {
  const resultElement = document.getElementById("result");

  if (resultElement) {
    resultElement.textContent = "Initializing, please wait...";
  }

  // Load CSV + feature-engineer everything
  await loadData();

  if (!Array.isArray(movieFeatures) || movieFeatures.length === 0) {
    if (resultElement) {
      resultElement.textContent =
        "No movie data available. Please check that 'data/movies_metadata.csv' exists.";
    }
    return;
  }

  // Populate dropdown after data is ready
  populateMoviesDropdown();

  if (resultElement) {
    resultElement.textContent =
      'Data loaded. Please select a movie and click "Get Recommendations".';
  }
};

/**
 * Populate the #movie-select dropdown using movieFeatures,
 * sorted alphabetically by title.
 */
function populateMoviesDropdown() {
  const selectElement = document.getElementById("movie-select");
  if (!selectElement) return;

  // Clear any placeholder options
  selectElement.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "-- Select a movie --";
  selectElement.appendChild(placeholder);

  const sortedMovies = [...movieFeatures].sort((a, b) =>
    a.title.localeCompare(b.title)
  );

  sortedMovies.forEach((movie) => {
    const option = document.createElement("option");
    option.value = movie.id; // id is a string from the CSV
    option.textContent = movie.title;
    selectElement.appendChild(option);
  });
}

/**
 * Compute Jaccard similarity between two sets.
 * Jaccard(A, B) = |A ∩ B| / |A ∪ B|
 */
function jaccardSimilarity(setA, setB) {
  if (!(setA instanceof Set) || !(setB instanceof Set)) {
    return 0;
  }

  if (setA.size === 0 && setB.size === 0) {
    return 0;
  }

  let intersectionCount = 0;
  for (const value of setA) {
    if (setB.has(value)) {
      intersectionCount++;
    }
  }

  const unionSize = setA.size + setB.size - intersectionCount;
  if (unionSize === 0) {
    return 0;
  }

  return intersectionCount / unionSize;
}

/**
 * Main recommendation function, called by "Get Recommendations" button.
 *
 * Steps:
 * 1. Read selected movie id from dropdown.
 * 2. Find likedMovie in movieFeatures.
 * 3. Build a feature set from subGenres + themes of likedMovie.
 * 4. For every other movie, compute Jaccard similarity on feature sets.
 * 5. Sort by similarity (descending).
 * 6. Take top 2.
 * 7. Display results in #result.
 */
function getRecommendations() {
  const resultElement = document.getElementById("result");
  const selectElement = document.getElementById("movie-select");

  if (!Array.isArray(movieFeatures) || movieFeatures.length === 0) {
    if (resultElement) {
      resultElement.textContent =
        "Movie data is not loaded yet. Please wait or refresh.";
    }
    return;
  }

  if (!selectElement) return;

  const selectedId = selectElement.value;

  if (!selectedId) {
    if (resultElement) {
      resultElement.textContent = "Please select a movie first.";
    }
    return;
  }

  // Find the liked movie
  const likedMovie = movieFeatures.find((m) => m.id === selectedId);

  if (!likedMovie) {
    if (resultElement) {
      resultElement.textContent =
        "Could not find the selected movie in the dataset.";
    }
    return;
  }

  // Build a unified feature set: sub-genres + themes
  const likedFeatureSet = new Set([
    ...likedMovie.subGenres,
    ...likedMovie.themes
  ]);

  // Candidate movies = all others
  const candidates = movieFeatures.filter((m) => m.id !== likedMovie.id);

  const scored = candidates.map((movie) => {
    const featureSet = new Set([...movie.subGenres, ...movie.themes]);
    const score = jaccardSimilarity(likedFeatureSet, featureSet);
    return {
      ...movie,
      score
    };
  });

  // Sort descending by score
  scored.sort((a, b) => b.score - a.score);

  // Take top 2 with positive similarity
  const top = scored.filter((m) => m.score > 0).slice(0, 2);

  if (!resultElement) {
    return;
  }

  if (top.length === 0) {
    resultElement.textContent = `We couldn't find any movies that share sub-genres/themes with "${likedMovie.title}". Try another movie.`;
    return;
  }

  const recTitles = top.map((m) => m.title);
  const details = top
    .map(
      (m) =>
        `${m.title} (Sub-genres: ${m.subGenres.join(
          ", "
        )} | Themes: ${m.themes.join(", ") || "None"})`
    )
    .join(" • ");

  resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend: ${recTitles.join(
    ", "
  )}. 
${details}`;
}
