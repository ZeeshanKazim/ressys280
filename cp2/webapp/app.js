// cp2/webapp/app.js
// Entry point for the RouteGraph-RAG demo.

import { loadAllData } from "./dataStore.js";
import {
  initState,
  getCurrentConstraints,
  getSortFilterOptions
} from "./state.js";
import { getRecommendations } from "./ranking.js";
import { initUI, renderRecommendations } from "./ui.js";

async function main() {
  const loadingEl = document.getElementById("loading-indicator");
  const errorEl = document.getElementById("error-indicator");

  try {
    if (loadingEl) loadingEl.style.display = "block";

    await loadAllData();
    initUI();
    initState(handleRecommend);

    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) errorEl.style.display = "none";
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
 * Called whenever the user clicks "Recommend flights".
 */
function handleRecommend(constraintsFromUI) {
  if (!constraintsFromUI.origin || !constraintsFromUI.dest) {
    window.alert("Please select both origin and destination.");
    return;
  }

  const sortFilter = getSortFilterOptions();
  const constraints = {
    ...constraintsFromUI,
    maxPrice: sortFilter.maxPrice
  };

  const baseResult = getRecommendations(constraints);
  const adjusted = applySortAndFilter(baseResult, sortFilter);

  renderRecommendations({
    baseline: adjusted.baseline,
    hybrid: adjusted.hybrid,
    hardConstraintsSatisfied: baseResult.hardConstraintsSatisfied,
    constraints
  });
}

/**
 * Apply UI-level sorting and filtering (price + time-of-day) on the
 * baseline + hybrid rankings.
 */
function applySortAndFilter(result, sortFilter) {
  let baseline = [...result.baseline];
  let hybrid = [...result.hybrid];

  const { sortBy, maxPrice, departureBucket, arrivalBucket } = sortFilter;

  // Price filter
  if (maxPrice != null) {
    baseline = baseline.filter((f) => f.price <= maxPrice);
    hybrid = hybrid.filter((f) => f.price <= maxPrice);
  }

  // Time-of-day filters
  if (departureBucket || arrivalBucket) {
    const applyTimeFilter = (flight) => {
      ensureHours(flight);
      const depBucket = hourToBucket(flight._depHour);
      const arrBucket = hourToBucket(flight._arrHour);

      if (departureBucket && depBucket !== departureBucket) return false;
      if (arrivalBucket && arrBucket !== arrivalBucket) return false;
      return true;
    };

    baseline = baseline.filter(applyTimeFilter);
    hybrid = hybrid.filter(applyTimeFilter);
  }

  // Sort
  if (sortBy === "price") {
    baseline.sort((a, b) => a.price - b.price);
    hybrid.sort((a, b) => a.price - b.price);
  } else if (sortBy === "duration") {
    baseline.sort((a, b) => a.durationMinutes - b.durationMinutes);
    hybrid.sort((a, b) => a.durationMinutes - b.durationMinutes);
  }
  // sortBy === "model" -> keep original ranking

  return { baseline, hybrid };
}

/**
 * Attach departure/arrival hour-of-day on the flight object
 * (computed from departureIso + durationMinutes).
 */
function ensureHours(flight) {
  if (flight._depHour != null && flight._arrHour != null) return;

  try {
    const dep = new Date(flight.departureIso);
    if (Number.isNaN(dep.getTime())) {
      flight._depHour = null;
      flight._arrHour = null;
      return;
    }

    const durationMins = Number(flight.durationMinutes || 0);
    const arr = new Date(dep.getTime() + durationMins * 60000);

    flight._depHour = dep.getHours();
    flight._arrHour = Number.isNaN(arr.getTime()) ? null : arr.getHours();
  } catch (e) {
    flight._depHour = null;
    flight._arrHour = null;
  }
}

/**
 * Map an hour-of-day to a bucket key.
 */
function hourToBucket(hour) {
  if (!Number.isFinite(hour)) return null;
  if (hour < 6) return "early";
  if (hour < 12) return "morning";
  if (hour < 18) return "afternoon";
  return "evening";
}

// Run once DOM is ready
document.addEventListener("DOMContentLoaded", main);
