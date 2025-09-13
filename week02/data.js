<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Content-Based Movie Recommender</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header class="nav">
    <div class="brand">Prime • Hotstar Style</div>
    <input id="search" class="search" placeholder="Search movies..." />
  </header>

  <main class="wrap">
    <section class="hero">
      <h1>Find your next favorite</h1>
      <p class="muted">Homework: using <b>Cosine Similarity</b> over genres.</p>

      <div class="controls">
        <select id="movie-select" aria-label="Select a movie">
          <option value="">— Select a Movie —</option>
        </select>
        <button id="recommend-btn" onclick="getRecommendations()">Get Recommendations</button>
      </div>

      <div id="result-box">
        <p id="result" class="muted">Loading data…</p>
        <!-- textual list of recommended titles with exact % -->
        <p id="rec-list" class="hidden"></p>
      </div>
    </section>

    <!-- 1) RECOMMENDATIONS ABOVE TOP PICKS -->
    <section>
      <h2 class="row-title" id="recs-title">Because you liked…</h2>
      <div id="row-recs" class="row"></div>
    </section>

    <!-- 2) TOP PICKS BELOW -->
    <section>
      <h2 class="row-title">Top Picks</h2>
      <div id="row-popular" class="row"></div>
    </section>

    <!-- Search results rail -->
    <section id="search-sec" class="hidden">
      <h2 class="row-title">Search Results</h2>
      <div id="row-search" class="row"></div>
    </section>
  </main>

  <!-- Load order matters -->
  <script src="data.js"></script>
  <script src="script.js"></script>
</body>
</html>
