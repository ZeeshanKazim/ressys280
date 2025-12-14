// cp2/webapp/app.js
// Entry point for the RouteGraph-RAG demo.
// - Loads all data
// - Initializes UI + state
// - Handles "Recommend flights" actions

import { loadAllData } from "./dataStore.js";
import {
  initState,
  getCurrentConstraints,
  getSortFilterOptions
} from "./state.js";
import { getRecommendations } from "./ranking.js";
import { initUI, renderRecommendations } from "./ui.js";

let lastResult = null;

async function main() {
  const loadingEl = document.getElementById("loading-indicator");
  const errorEl = document.getElementById("error-indicator");

  try {
    if (loadingEl) {
      loadingEl.style.display = "block";
    }

    // 1) Load data (flights, graph, reviews, carrier metadata)
    await loadAllData();

    // 2) Init UI containers
    initUI();

    // 3) Init state and wiring to "Recommend" button
    initState(handleRecommend);

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
function handleRecommend(constraintsFromUI) {
  // Require user to choose origin and destination
  if (!constraintsFromUI.origin || !constraintsFromUI.dest) {
    window.alert("Please select both origin and destination.");
    return;
  }

  const sortFilter = getSortFilterOptions();
  const constraints = {
    ...constraintsFromUI,
    maxPrice: sortFilter.maxPrice // apply price slider as hard constraint
  };

  const baseResult = getRecommendations(constraints);
  lastResult = baseResult;

  const adjusted = applySortAndFilter(baseResult, sortFilter);

  renderRecommendations({
    baseline: adjusted.baseline,
    hybrid: adjusted.hybrid,
    hardConstraintsSatisfied: baseResult.hardConstraintsSatisfied,
    constraints
  });
}

/**
 * Apply UI-level sort & filter on top of baseline + hybrid rankings.
 */
function applySortAndFilter(result, sortFilter) {
  let baseline = [...result.baseline];
  let hybrid = [...result.hybrid];

  const { sortBy, maxPrice } = sortFilter;

  // Price filter (on top of model's constraints)
  if (maxPrice != null) {
    baseline = baseline.filter((f) => f.price <= maxPrice);
    hybrid = hybrid.filter((f) => f.price <= maxPrice);
  }

  // Sorting
  if (sortBy === "price") {
    baseline.sort((a, b) => a.price - b.price);
    hybrid.sort((a, b) => a.price - b.price);
  } else if (sortBy === "duration") {
    baseline.sort((a, b) => a.durationMinutes - b.durationMinutes);
    hybrid.sort((a, b) => a.durationMinutes - b.durationMinutes);
  } // sortBy === "model" -> keep model ranking order

  return { baseline, hybrid };
}

// Run once DOM is ready
document.addEventListener("DOMContentLoaded", main);
