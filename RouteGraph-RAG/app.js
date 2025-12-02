// app.js
(function (global) {
  "use strict";

  const U = global.Utils;
  const EDA = global.EDA;

  const STATUS_EL_ID = "load-status";

  function setStatus(text, type) {
    const el = document.getElementById(STATUS_EL_ID);
    if (!el) return;
    el.textContent = text;
    el.classList.remove("badge-warning", "badge-success", "badge-danger");
    if (type === "success") el.classList.add("badge-success");
    else if (type === "error") el.classList.add("badge-danger");
    else el.classList.add("badge-warning");
  }

  // Load one dataset, trying multiple possible paths
  async function safeLoad(label, paths, required) {
    let lastError = null;
    for (const p of paths) {
      try {
        console.log(`[EDA] Loading ${label} from "${p}"…`);
        const data = await U.loadCsv(p);
        console.log(
          `[EDA] Loaded ${label} from "${p}" – ${data.length.toLocaleString()} rows`
        );
        return data;
      } catch (err) {
        console.warn(`[EDA] Failed to load ${label} from "${p}"`, err);
        lastError = err;
      }
    }

    if (required) {
      throw new Error(
        `Required dataset "${label}" could not be loaded from any of: ${paths.join(
          ", "
        )}`
      );
    }

    console.warn(
      `[EDA] Optional dataset "${label}" is missing. Continuing with empty array.`
    );
    return [];
  }

  async function loadAllData() {
    setStatus("Loading CSV files from /data…", "warning");

    try {
      const train = await safeLoad("train", ["data/train.csv"], true);
      const test = await safeLoad("test", ["data/test.csv"], false);
      const routes = await safeLoad(
        "routes",
        ["data/Full_Merge_of_All_Unique_Routes.csv"],
        false
      );
      const airports = await safeLoad(
        "airports",
        [
          "data/Full_Merge_of_All_Unique_Airports.csv",
          "data/Full_Merge_of_All_Unique Airports.csv",
        ],
        false
      );
      const reviews = await safeLoad(
        "reviews",
        ["data/Airline_Reviews.csv"],
        false
      );

      setStatus("Data loaded. Running EDA…", "warning");
      EDA.run({ train, test, routes, airports, reviews }
