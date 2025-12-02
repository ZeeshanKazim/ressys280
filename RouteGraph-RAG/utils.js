// utils.js
(function (global) {
  "use strict";

  // ---- CSV loading ----

  function loadCsv(path) {
    return new Promise((resolve, reject) => {
      Papa.parse(path, {
        download: true,
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          if (results.errors && results.errors.length > 0) {
            console.warn("CSV parse warnings for", path, results.errors);
          }
          resolve(results.data || []);
        },
        error: (err) => reject(err),
      });
    });
  }

  // ---- small helpers ----

  function groupBy(arr, keyFn) {
    const map = new Map();
    for (const item of arr) {
      const key = keyFn(item);
      const k = key === undefined || key === null ? "__MISSING__" : String(key);
      if (!map.has(k)) map.set(k, []);
      map.get(k).push(item);
    }
    return map;
  }

  function countBy(arr, keyFn) {
    const counts = {};
    for (const item of arr) {
      const key = keyFn(item);
      const k = key === undefined || key === null ? "null" : String(key);
      counts[k] = (counts[k] || 0) + 1;
    }
    return counts;
  }

  function numericSummary(values) {
    const arr = values
      .filter((v) => typeof v === "number" && !Number.isNaN(v))
      .slice();
    if (arr.length === 0) return null;
    arr.sort((a, b) => a - b);
    const n = arr.length;
    const mean = arr.reduce((s, v) => s + v, 0) / n;
    const min = arr[0];
    const max = arr[n - 1];
    const q = (p) => {
      const idx = (n - 1) * p;
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      if (lo === hi) return arr[lo];
      const w = idx - lo;
      return arr[lo] * (1 - w) + arr[hi] * w;
    };
    const median = q(0.5);
    const q1 = q(0.25);
    const q3 = q(0.75);
    const sq = arr.reduce((s, v) => s + (v - mean) * (v - mean), 0);
    const std = Math.sqrt(sq / (n - 1));

    return { n, mean, std, min, q1, median, q3, max };
  }

  function formatSummary(summary) {
    if (!summary) return "No numeric data.";
    const s = summary;
    const f = (x) =>
      x === null || x === undefined || Number.isNaN(x) ? "NA" : x.toFixed(3);
    return [
      `count : ${s.n}`,
      `mean  : ${f(s.mean)}`,
      `std   : ${f(s.std)}`,
      `min   : ${f(s.min)}`,
      `q25   : ${f(s.q1)}`,
      `median: ${f(s.median)}`,
      `q75   : ${f(s.q3)}`,
      `max   : ${f(s.max)}`,
    ].join("\n");
  }

  function correlation(xs, ys) {
    const arr = [];
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      const y = ys[i];
      if (
        typeof x === "number" &&
        typeof y === "number" &&
        !Number.isNaN(x) &&
        !Number.isNaN(y)
      ) {
        arr.push([x, y]);
      }
    }
    if (arr.length < 2) return null;
    const n = arr.length;
    const meanX = arr.reduce((s, [x]) => s + x, 0) / n;
    const meanY = arr.reduce((s, [, y]) => s + y, 0) / n;
    let num = 0,
      denX = 0,
      denY = 0;
    for (const [x, y] of arr) {
      const dx = x - meanX;
      const dy = y - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    if (denX === 0 || denY === 0) return null;
    return num / Math.sqrt(denX * denY);
  }

  // ---- DOM helpers ----

  function setHtml(id, html) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = html;
  }

  function appendHtml(id, html) {
    const el = document.getElementById(id);
    if (!el) return;
    el.insertAdjacentHTML("beforeend", html);
  }

  function renderTable(id, rows, columns) {
    const root = document.getElementById(id);
    if (!root) return;
    if (!rows || rows.length === 0) {
      root.insertAdjacentHTML(
        "beforeend",
        '<p class="muted">No data to display.</p>'
      );
      return;
    }
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const trHead = document.createElement("tr");
    columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col.label;
      trHead.appendChild(th);
    });
    thead.appendChild(trHead);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((col) => {
        const td = document.createElement("td");
        let value = row[col.key];
        if (typeof col.format === "function") {
          value = col.format(value, row);
        }
        td.textContent = value === undefined || value === null ? "" : value;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    const wrapper = document.createElement("div");
    wrapper.className = "table-wrapper";
    wrapper.appendChild(table);
    root.appendChild(wrapper);
  }

  global.Utils = {
    loadCsv,
    groupBy,
    countBy,
    numericSummary,
    formatSummary,
    correlation,
    setHtml,
    appendHtml,
    renderTable,
  };
})(window);
