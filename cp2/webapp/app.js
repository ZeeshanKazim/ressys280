// app.js
// Entry point: load data, create state, bind UI.

import { loadAllData } from "./dataStore.js";
import { createInitialState } from "./state.js";
import { bindUI } from "./ui.js";
import { rankBaseline, rankEnhanced } from "./ranking.js";

async function main() {
  try {
    console.log("RouteGraph-RAG: loading data...");
    const data = await loadAllData();
    const state = createInitialState(data);
    console.log("Data loaded, state initialised:", state);

    bindUI({
      state,
      rankBaseline,
      rankEnhanced,
    });
  } catch (err) {
    console.error("Failed to initialise app:", err);
    alert(
      "Error loading RouteGraph-RAG data. Check console for details and that JSON files exist in data/processed/."
    );
  }
}

document.addEventListener("DOMContentLoaded", main);
