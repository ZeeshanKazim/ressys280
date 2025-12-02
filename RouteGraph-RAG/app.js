// js/app.js
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

  async function loadAllData() {
    setStatus("Loading CSV files from /data…", "warning");

    const airportsPath1 = "data/Full_Merge_of_All_Unique_Airports.csv";
    const airportsPath2 = "data/Full_Merge_of_All_Unique Airports.csv";

    try {
      const [train, test, routes, airports, reviews] = await Promise.all([
        U.loadCsv("data/train.csv"),
        U.loadCsv("data/test.csv"),
        U.loadCsv("data/Full_Merge_of_All_Unique_Routes.csv"),
        U.loadCsv(airportsPath1).catch(() => U.loadCsv(airportsPath2)),
        U.loadCsv("data/Airline_Reviews.csv"),
      ]);

      setStatus("Data loaded. Running EDA…", "warning");
      EDA.run({ train, test, routes, airports, reviews });
      setStatus("EDA complete ✓", "success");
    } catch (err) {
      console.error("Failed to load data or run EDA:", err);
      setStatus("Error loading data – check console & paths.", "error");
    }
  }

  function init() {
    document
      .getElementById("rerun-btn")
      .addEventListener("click", () => loadAllData());

    loadAllData();
  }

  window.addEventListener("DOMContentLoaded", init);
})(window);
