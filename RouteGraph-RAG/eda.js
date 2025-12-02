// js/eda.js
(function (global) {
  "use strict";

  const U = global.Utils;

  // ---- Feature engineering helpers ----

  function parseDurationMinutes(value) {
    if (value === null || value === undefined || value === "") return null;
    const s = String(value);
    try {
      // Examples: "07:25:00" or "1.00:30:00"
      let days = 0;
      let timePart = s;
      if (s.includes(".") && s.includes(":")) {
        const parts = s.split(".");
        if (parts.length === 2 && /^\d+$/.test(parts[0])) {
          days = parseInt(parts[0], 10);
          timePart = parts[1];
        }
      }
      const [hh, mm, ss] = timePart.split(":").map((x) => parseInt(x || "0", 10));
      const totalMinutes = (days * 24 + hh) * 60 + mm + (ss || 0) / 60;
      if (!Number.isFinite(totalMinutes)) return null;
      return totalMinutes;
    } catch (e) {
      console.warn("Failed to parse duration:", value, e);
      return null;
    }
  }

  const SEG_DEP_COLS = [
    "legs0_segments0_departureFrom_airport_iata",
    "legs0_segments1_departureFrom_airport_iata",
    "legs0_segments2_departureFrom_airport_iata",
    "legs0_segments3_departureFrom_airport_iata",
  ];

  const SEG_ARR_COLS = [
    "legs0_segments0_arrivalTo_airport_iata",
    "legs0_segments1_arrivalTo_airport_iata",
    "legs0_segments2_arrivalTo_airport_iata",
    "legs0_segments3_arrivalTo_airport_iata",
  ];

  function addBasicFeatures(rows) {
    return rows.map((row) => {
      const r = { ...row };

      // datetime parsing
      let depart = null;
      let request = null;
      if (row.legs0_departureAt) {
        const d = new Date(row.legs0_departureAt);
        if (!isNaN(d.getTime())) depart = d;
      }
      if (row.requestDate) {
        const d = new Date(row.requestDate);
        if (!isNaN(d.getTime())) request = d;
      }

      if (depart) {
        r.depart_hour = depart.getHours();
        r.depart_dow = depart.getDay(); // 0=Sunday
      }

      if (depart && request) {
        const diffMs = depart.getTime() - request.getTime();
        r.days_to_departure = diffMs / (1000 * 60 * 60 * 24);
      }

      // duration
      if (row.legs0_duration !== undefined) {
        r.duration_minutes = parseDurationMinutes(row.legs0_duration);
      }

      // carrier
      r.carrier =
        row.legs0_segments0_marketingCarrier_code ||
        row.legs0_segments0_operatingCarrier_code ||
        row.carrier ||
        null;

      // origin / dest via segments
      let origin = null;
      for (const col of SEG_DEP_COLS) {
        if (row[col]) {
          origin = row[col];
          break;
        }
      }
      let dest = null;
      for (let i = SEG_ARR_COLS.length - 1; i >= 0; i--) {
        const col = SEG_ARR_COLS[i];
        if (row[col]) {
          dest = row[col];
          break;
        }
      }
      r.origin = origin;
      r.dest = dest;

      if (origin && dest) {
        r.route_key = `${origin}-${dest}`;
      } else {
        r.route_key = r.route_key || null;
      }

      if (r.carrier && r.route_key) {
        r.carrier_route_key = `${r.carrier}_${r.route_key}`;
      }

      // stops
      let segCount = 0;
      for (const col of SEG_DEP_COLS) {
        if (row[col]) segCount += 1;
      }
      if (segCount === 0) segCount = 1;
      r.num_segments = segCount;
      r.stops = segCount - 1;

      // red-eye
      if (typeof r.depart_hour === "number") {
        r.is_red_eye = r.depart_hour >= 22 || r.depart_hour < 5;
      }

      // price + log price
      const price = Number(row.totalPrice);
      if (Number.isFinite(price)) {
        r.totalPrice = price;
        r.log_totalPrice = Math.log1p(price);
      }

      // label
      if (row.selected !== undefined) {
        r.selected = Number(row.selected);
      }

      return r;
    });
  }

  // ---- EDA renderers ----

  function renderOverview(trainFeat, testFeat) {
    const nRowsTrain = trainFeat.length;
    const nRowsTest = testFeat.length;
    const nColsTrain =
      nRowsTrain > 0 ? Object.keys(trainFeat[0]).length : 0;
    const nColsTest = nRowsTest > 0 ? Object.keys(testFeat[0]).length : 0;

    const html = `
      <p><strong>Train</strong>: ${nRowsTrain.toLocaleString()} rows, ~${nColsTrain} columns</p>
      <p><strong>Test</strong>: ${nRowsTest.toLocaleString()} rows, ~${nColsTest} columns</p>
      <p class="muted">
        Feature engineering in-browser adds columns such as
        <code>duration_minutes</code>, <code>route_key</code>,
        <code>carrier_route_key</code>, <code>depart_hour</code>,
        <code>days_to_departure</code>, <code>stops</code>, <code>is_red_eye</code>, and
        <code>log_totalPrice</code>.
      </p>
    `;
    U.setHtml("eda-overview", html);
  }

  function renderLabelDistribution(trainFeat) {
    const hasSelected = trainFeat.length > 0 && "selected" in trainFeat[0];
    if (!hasSelected) {
      U.setHtml(
        "eda-labels",
        '<p class="muted">Column <code>selected</code> not found, skipping label analysis.</p>'
      );
      return;
    }

    const counts = U.countBy(trainFeat, (r) => r.selected);
    const total = trainFeat.length;
    const rows = Object.entries(counts).map(([label, count]) => ({
      label,
      count,
      frac: count / total,
    }));

    const htmlIntro = `<p>Label distribution for <code>selected</code> (train set):</p>`;
    U.setHtml("eda-labels", htmlIntro);
    U.renderTable(
      "eda-labels",
      rows,
      [
        { key: "label", label: "Label" },
        {
          key: "count",
          label: "Count",
          format: (v) => v.toLocaleString(),
        },
        {
          key: "frac",
          label: "Proportion",
          format: (v) => v.toFixed(4),
        },
      ]
    );
  }

  function renderQueryStats(trainFeat) {
    if (!("ranker_id" in (trainFeat[0] || {}))) {
      U.setHtml(
        "eda-query-stats",
        '<p class="muted">Column <code>ranker_id</code> not found, skipping query-level stats.</p>'
      );
      return;
    }

    const groups = U.groupBy(trainFeat, (r) => r.ranker_id);
    const stats = [];
    groups.forEach((rows, key) => {
      const numCandidates = rows.length;
      const numSelected = rows.filter((r) => r.selected === 1).length;
      const posRate = numSelected / numCandidates;
      stats.push({
        ranker_id: key,
        numCandidates,
        numSelected,
        posRate,
      });
    });

    const candCounts = stats.map((s) => s.numCandidates);
    const posCounts = stats.map((s) => s.numSelected);
    const candSummary = U.numericSummary(candCounts);
    const posSummary = U.numericSummary(posCounts);

    let html = `
      <p>Each <code>ranker_id</code> is a search session (query group).</p>
      <pre>${[
        "Candidates per query:",
        U.formatSummary(candSummary),
        "",
        "Positives per query (selected=1):",
        U.formatSummary(posSummary),
      ].join("\n")}</pre>
    `;

    U.setHtml("eda-query-stats", html);

    // show first 10 groups as a table
    const top = stats
      .slice()
      .sort((a, b) => b.numCandidates - a.numCandidates)
      .slice(0, 10);

    U.appendHtml(
      "eda-query-stats",
      '<p>Example queries (top 10 by number of candidates):</p>'
    );
    U.renderTable("eda-query-stats", top, [
      { key: "ranker_id", label: "ranker_id" },
      {
        key: "numCandidates",
        label: "# candidates",
        format: (v) => v.toLocaleString(),
      },
      {
        key: "numSelected",
        label: "# selected",
        format: (v) => v.toLocaleString(),
      },
      {
        key: "posRate",
        label: "Pos rate",
        format: (v) => v.toFixed(3),
      },
    ]);
  }

  function renderPriceAndDuration(trainFeat) {
    const prices = trainFeat.map((r) => r.totalPrice);
    const logPrices = trainFeat.map((r) => r.log_totalPrice);
    const durations = trainFeat.map((r) => r.duration_minutes);
    const stops = trainFeat.map((r) => r.stops);

    const priceSummary = U.numericSummary(prices);
    const logSummary = U.numericSummary(logPrices);
    const durSummary = U.numericSummary(durations);
    const corrStopsDur = U.correlation(stops, durations);

    let text = "";
    text += "Price (totalPrice)\n";
    text += U.formatSummary(priceSummary) + "\n\n";
    text += "Log price (log_totalPrice)\n";
    text += U.formatSummary(logSummary) + "\n\n";
    text += "Duration (minutes)\n";
    text += U.formatSummary(durSummary) + "\n\n";
    text += "Correlation(stops, duration_minutes): " +
      (corrStopsDur === null ? "NA" : corrStopsDur.toFixed(3));

    U.setHtml(
      "eda-price-duration",
      `<pre>${text}</pre><p class="muted">You can describe these in your report as heavy-tailed prices, non-linear duration vs. stops, etc.</p>`
    );
  }

  function renderCarrierAndRoute(trainFeat) {
    // carriers
    const carrierGroups = U.groupBy(trainFeat, (r) => r.carrier || "UNKNOWN");
    const carrierStats = [];
    carrierGroups.forEach((rows, key) => {
      const prices = rows.map((r) => r.totalPrice);
      const durs = rows.map((r) => r.duration_minutes);
      const sel = rows.map((r) => r.selected);
      const n = rows.length;
      const priceMean = U.numericSummary(prices)?.mean ?? null;
      const durMean = U.numericSummary(durs)?.mean ?? null;
      const selMean = U.numericSummary(sel)?.mean ?? null;
      carrierStats.push({
        carrier: key,
        numOptions: n,
        meanPrice: priceMean,
        meanDuration: durMean,
        selectionRate: selMean,
      });
    });

    carrierStats.sort((a, b) => b.numOptions - a.numOptions);
    const topCarriers = carrierStats.slice(0, 10);

    U.setHtml(
      "eda-carriers",
      `<p>Top carriers by number of options (train set):</p>`
    );
    U.renderTable("eda-carriers", topCarriers, [
      { key: "carrier", label: "Carrier" },
      {
        key: "numOptions",
        label: "# options",
        format: (v) => v.toLocaleString(),
      },
      {
        key: "meanPrice",
        label: "Mean price",
        format: (v) => (v == null ? "" : v.toFixed(2)),
      },
      {
        key: "meanDuration",
        label: "Mean duration (min)",
        format: (v) => (v == null ? "" : v.toFixed(1)),
      },
      {
        key: "selectionRate",
        label: "Selection rate",
        format: (v) => (v == null ? "" : v.toFixed(3)),
      },
    ]);

    // Chart for carriers
    const canvas = document.getElementById("carrier-chart");
    if (canvas && global.Chart && topCarriers.length > 0) {
      const labels = topCarriers.map((c) => c.carrier);
      const counts = topCarriers.map((c) => c.numOptions);
      const selRates = topCarriers.map((c) => c.selectionRate || 0);

      new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "# options (train)",
              data: counts,
              yAxisID: "y",
            },
            {
              label: "Selection rate",
              data: selRates,
              type: "line",
              yAxisID: "y1",
            },
          ],
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true,
              title: { display: true, text: "# options" },
            },
            y1: {
              beginAtZero: true,
              position: "right",
              title: { display: true, text: "Selection rate" },
            },
          },
          plugins: {
            legend: { display: true },
          },
        },
      });
    }

    // Routes
    const routeGroups = U.groupBy(trainFeat, (r) => r.route_key || "UNKNOWN");
    const routeStats = [];
    routeGroups.forEach((rows, key) => {
      const carriers = new Set(rows.map((r) => r.carrier));
      const prices = rows.map((r) => r.totalPrice);
      const durs = rows.map((r) => r.duration_minutes);
      routeStats.push({
        route_key: key,
        numOptions: rows.length,
        nCarriers: carriers.size,
        meanPrice: U.numericSummary(prices)?.mean ?? null,
        meanDuration: U.numericSummary(durs)?.mean ?? null,
      });
    });

    routeStats.sort((a, b) => b.numOptions - a.numOptions);
    const topRoutes = routeStats.slice(0, 10);

    U.setHtml(
      "eda-routes",
      `<p>Top routes by number of options (train set):</p>`
    );
    U.renderTable("eda-routes", topRoutes, [
      { key: "route_key", label: "Route" },
      {
        key: "numOptions",
        label: "# options",
        format: (v) => v.toLocaleString(),
      },
      {
        key: "nCarriers",
        label: "# carriers",
        format: (v) => v.toLocaleString(),
      },
      {
        key: "meanPrice",
        label: "Mean price",
        format: (v) => (v == null ? "" : v.toFixed(2)),
      },
      {
        key: "meanDuration",
        label: "Mean duration (min)",
        format: (v) => (v == null ? "" : v.toFixed(1)),
      },
    ]);
  }

  function renderExternalCoverage(trainFeat, routes, airports) {
    if (!routes || routes.length === 0 || !airports || airports.length === 0) {
      U.setHtml(
        "eda-external",
        '<p class="muted">Routes or airports CSV empty/missing, skipping external coverage analysis.</p>'
      );
      return;
    }

    // route coverage
    const trainRoutes = new Set(
      trainFeat.map((r) => r.route_key).filter((x) => x)
    );

    // try to find columns that look like departure/destination
    const routeCols = Object.keys(routes[0]);
    const depCol =
      routeCols.find((c) => c.toLowerCase().includes("departure")) ||
      routeCols[0];
    const destCol =
      routeCols.find((c) => c.toLowerCase().includes("destination")) ||
      routeCols[1] ||
      routeCols[0];

    const extRoutes = new Set();
    for (const row of routes) {
      const dep = row[depCol];
      const dest = row[destCol];
      if (dep && dest) {
        extRoutes.add(`${dep}-${dest}`);
      }
    }

    const overlapRoutes = new Set(
      [...trainRoutes].filter((r) => extRoutes.has(r))
    );
    const routeCoverage =
      trainRoutes.size === 0 ? 0 : overlapRoutes.size / trainRoutes.size;

    // airport coverage
    const airportsCols = Object.keys(airports[0]);
    const airportCodeCol =
      airportsCols.find((c) => c.toLowerCase().includes("iata")) ||
      airportsCols.find((c) => c.toLowerCase().includes("id")) ||
      airportsCols[0];

    const extAirports = new Set(
      airports
        .map((row) => row[airportCodeCol])
        .filter((x) => x !== null && x !== undefined && x !== "")
    );
    const trainAirports = new Set();
    for (const r of trainFeat) {
      if (r.origin) trainAirports.add(r.origin);
      if (r.dest) trainAirports.add(r.dest);
    }

    const overlapAirports = new Set(
      [...trainAirports].filter((a) => extAirports.has(a))
    );
    const airportCoverage =
      trainAirports.size === 0
        ? 0
        : overlapAirports.size / trainAirports.size;

    const html = `
      <pre>${[
        "Route coverage vs external route list:",
        `Train routes:    ${trainRoutes.size}`,
        `External routes: ${extRoutes.size}`,
        `Overlap:         ${overlapRoutes.size} (${(routeCoverage * 100).toFixed(
          1
        )}%)`,
        "",
        "Airport coverage vs external airport list:",
        `Train airports:  ${trainAirports.size}`,
        `External airports: ${extAirports.size}`,
        `Overlap:         ${overlapAirports.size} (${(
          airportCoverage * 100
        ).toFixed(1)}%)`,
      ].join("\n")}</pre>
    `;
    U.setHtml("eda-external", html);
  }

  function renderReviews(reviews) {
    if (!reviews || reviews.length === 0) {
      U.setHtml(
        "eda-reviews",
        '<p class="muted">Airline_Reviews.csv is empty or missing.</p>'
      );
      return;
    }

    const cols = Object.keys(reviews[0]);
    const hasOverall = cols.includes("Overall_Rating");
    const overallValues = hasOverall
      ? reviews.map((r) => Number(r.Overall_Rating))
      : [];

    let html = `<p>Reviews dataset: ${reviews.length.toLocaleString()} rows.</p>`;

    html += `<p>Example rows (first 5):</p>`;
    const sample = reviews.slice(0, 5);
    const subsetCols = cols.slice(0, 6);
    U.setHtml("eda-reviews", html);
    U.renderTable(
      "eda-reviews",
      sample,
      subsetCols.map((c) => ({ key: c, label: c }))
    );

    if (hasOverall) {
      const summary = U.numericSummary(overallValues);
      U.appendHtml(
        "eda-reviews",
        `<p>Overall rating summary:</p><pre>${U.formatSummary(
          summary
        )}</pre>`
      );
    }

    // top airlines by count if column exists
    const airlineCol =
      cols.find((c) => c.toLowerCase().includes("airline")) || null;
    if (airlineCol) {
      const counts = U.countBy(reviews, (r) => r[airlineCol]);
      const rowsCounts = Object.entries(counts)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);
      U.appendHtml(
        "eda-reviews",
        `<p>Top airlines by number of reviews (<code>${airlineCol}</code>):</p>`
      );
      U.renderTable("eda-reviews", rowsCounts, [
        { key: "name", label: "Airline" },
        {
          key: "count",
          label: "# reviews",
          format: (v) => v.toLocaleString(),
        },
      ]);
    }
  }

  // ---- Public entrypoint ----

  function run({ train, test, routes, airports, reviews }) {
    const trainFeat = addBasicFeatures(train);
    const testFeat = addBasicFeatures(test);

    renderOverview(trainFeat, testFeat);
    renderLabelDistribution(trainFeat);
    renderQueryStats(trainFeat);
    renderPriceAndDuration(trainFeat);
    renderCarrierAndRoute(trainFeat);
    renderExternalCoverage(trainFeat, routes, airports);
    renderReviews(reviews);
  }

  global.EDA = { run };
})(window);
