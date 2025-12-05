// cp2/webapp/app.js
// Entry point for the RouteGraph-RAG CP2 demo.
// - Loads all data
// - Initializes UI + state
// - Handles "Recommend flights" actions

import { loadAllData } from "./dataStore.js";
import { initState, getCurrentConstraints } from "./state.js";
import { getRecommendations } from "./ranking.js";
import { initUI, renderRecommendations } from "./ui.js";

async function main() {
  const loadingEl = document.getElementById("loading-indicator");
  const errorEl = document.getElementById("error-indicator");

  try {
    if (loadingEl) {
      loadingEl.style.display = "block";
    }

    // 1) Load data (flights_small, graph, reviews, carrier_metadata)
    await loadAllData();

    // 2) Init UI containers
    initUI();

    // 3) Init state and wiring to "Recommend" button
    initState(handleRecommend);

    // No initial auto-recommendation: user must pick origin/destination first.
    if (loadingEl) {
      loadingEl.style.display = "none";
    }
    if (errorEl) {
      errorEl.style.display = "none";
    }
  } catch (err) {
    console.error("[app] Failed to initialize app:", err);
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) {
      errorEl.style.display = "block";
      errorEl.textContent =
        "Something went wrong while loading data. Please refresh the page.";
    }
  }
}

/**
 * Called whenever the user clicks "Recommend flights" in the UI.
 */
function handleRecommend(constraints) {
  // Require user to choose origin and destination
  if (!constraints.origin || !constraints.dest) {
    window.alert("Please select both origin and destination.");
    return;
  }

  const result = getRecommendations(constraints);

  renderRecommendations({
    baseline: result.baseline,
    hybrid: result.hybrid,
    hardConstraintsSatisfied: result.hardConstraintsSatisfied,
    constraints
  });
}

// Run once DOM is ready
document.addEventListener("DOMContentLoaded", main);
