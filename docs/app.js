/* ============================================================
   Interactive figures for "Estimating Time Spent on Work Tasks".
   All numbers come from the exported JSON (real computed data) or
   from the paper's tables — nothing is fabricated or interpolated.
   ============================================================ */

const COLORS = {
  purple: "#5b2a86",
  purpleSoft: "#b89ae8",
  red: "#d6453d",
  faint: "#847e93",
};

const fmtPct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");

/* ---------- shared fetch ---------- */
async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error("Failed to load " + path);
  return r.json();
}

/* ============================================================
   FIGURE 1 — occupation task time-share explorer
   ============================================================ */
const F1 = { data: null, code: null, measure: "rubric" };

function f1Render() {
  const occ = F1.data.occupations[F1.code];
  const m = "rubric";
  const shares = occ[m];
  document.getElementById("f1TaskShare").textContent = fmtPct(shares.share_tasks);
  document.getElementById("f1TimeShare").textContent = fmtPct(shares.share_time);

  document.getElementById("f1Caption").innerHTML =
    `<b>Figure 1.</b> Constituent tasks for ${occ.title} from O*NET. ` +
    `Bar length shows our estimated share of working time per task; tasks measured as exposed to AI under ` +
    `the rubric-based measure of Eloundou et al. [2024] are highlighted in red.`;

  const host = document.getElementById("f1Bars");
  host.innerHTML = "";
  for (const t of occ.tasks) {
    const exposed = t[m] === 1;
    const row = document.createElement("div");
    row.className = "bar-row";
    const w = t.share * 100;
    row.innerHTML =
      `<div class="bar-label${exposed ? " exp" : ""}" title="${t.label.replace(/"/g, "&quot;")}">${t.label}</div>` +
      `<div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${exposed ? COLORS.red : COLORS.purpleSoft}"></div></div>` +
      `<div class="bar-val">${(t.share * 100).toFixed(1)}%</div>`;
    host.appendChild(row);
  }
}

async function initFigure1() {
  F1.data = await loadJSON("data/occupations.json");
  F1.code = F1.data.default;
  f1Render();
}

/* ============================================================
   FIGURE 4 — survival CDF of exposure shares
   ============================================================ */
const F4 = { data: null, thresh: "0.25", measure: "rubric" };

function f4Series(measure, weighting, thresh) {
  const id = `${measure}_${weighting}_${thresh}`;
  return F4.data.series.find((s) => s.id === id || s.id === id.replace(/0$/, ""));
}

function f4Render() {
  const host = document.getElementById("f4Chart");
  host.innerHTML = "";
  const t = parseFloat(F4.thresh);
  const tKey = t === 0.5 ? "0.5" : "0.25";
  const ours = F4.data.series.find((s) => s.measure === F4.measure && s.weighting === "time_share" && s.threshold === t);
  const base = F4.data.series.find((s) => s.measure === F4.measure && s.weighting === "task_category" && s.threshold === t);

  // dimensions
  const W = 860, H = 460, m = { t: 18, r: 20, b: 52, l: 60 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;

  const allX = [...ours.points, ...base.points].map((p) => p.x);
  const xMax = Math.max(0.05, Math.max(...allX));
  const xS = (x) => m.l + (x / xMax) * iw;
  const yS = (y) => m.t + (1 - y) * ih;

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");

  // gridlines + y axis (0..1)
  const gGrid = document.createElementNS(NS, "g");
  gGrid.setAttribute("class", "grid");
  for (let i = 0; i <= 5; i++) {
    const yv = i / 5, yy = yS(yv);
    const ln = document.createElementNS(NS, "line");
    ln.setAttribute("x1", m.l); ln.setAttribute("x2", m.l + iw);
    ln.setAttribute("y1", yy); ln.setAttribute("y2", yy);
    gGrid.appendChild(ln);
    const tx = document.createElementNS(NS, "text");
    tx.setAttribute("x", m.l - 10); tx.setAttribute("y", yy + 4);
    tx.setAttribute("text-anchor", "end"); tx.setAttribute("fill", COLORS.faint);
    tx.setAttribute("font-size", "11"); tx.textContent = Math.round(yv * 100) + "%";
    gGrid.appendChild(tx);
  }
  svg.appendChild(gGrid);

  // x axis ticks
  const gAx = document.createElementNS(NS, "g");
  gAx.setAttribute("class", "axis");
  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {
    const xv = (xMax * i) / nTicks, xx = xS(xv);
    const tx = document.createElementNS(NS, "text");
    tx.setAttribute("x", xx); tx.setAttribute("y", H - m.b + 20);
    tx.setAttribute("text-anchor", "middle"); tx.setAttribute("fill", COLORS.faint);
    tx.setAttribute("font-size", "11"); tx.textContent = Math.round(xv * 100) + "%";
    gAx.appendChild(tx);
  }
  svg.appendChild(gAx);

  // axis labels
  const xl = document.createElementNS(NS, "text");
  xl.setAttribute("x", m.l + iw / 2); xl.setAttribute("y", H - 8);
  xl.setAttribute("text-anchor", "middle"); xl.setAttribute("fill", "#4a4458");
  xl.setAttribute("font-size", "13"); xl.setAttribute("font-weight", "600");
  xl.textContent = "Share of tasks exposed to AI";
  svg.appendChild(xl);

  const yl = document.createElementNS(NS, "text");
  yl.setAttribute("transform", `translate(16,${m.t + ih / 2}) rotate(-90)`);
  yl.setAttribute("text-anchor", "middle"); yl.setAttribute("fill", "#4a4458");
  yl.setAttribute("font-size", "13"); yl.setAttribute("font-weight", "600");
  yl.textContent = "Fraction of occupations at or above";
  svg.appendChild(yl);

  // step path builder (survival curve, where=post)
  function stepPath(points) {
    const pts = [...points].sort((a, b) => a.x - b.x);
    let d = `M ${xS(0)} ${yS(1)}`;
    let prevY = 1;
    for (const p of pts) {
      d += ` L ${xS(p.x)} ${yS(prevY)} L ${xS(p.x)} ${yS(p.y)}`;
      prevY = p.y;
    }
    d += ` L ${xS(xMax)} ${yS(prevY)}`;
    return d;
  }

  function addCurve(series, color, width, dash) {
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", stepPath(series.points));
    p.setAttribute("fill", "none");
    p.setAttribute("stroke", color);
    p.setAttribute("stroke-width", width);
    if (dash) p.setAttribute("stroke-dasharray", dash);
    p.setAttribute("stroke-linejoin", "round");
    svg.appendChild(p);
  }
  addCurve(base, COLORS.faint, 2.2, "6 4");
  addCurve(ours, COLORS.purple, 2.8, null);

  host.appendChild(svg);

  // tooltip + hover guideline
  const tip = document.createElement("div");
  tip.className = "cdf-tooltip";
  host.appendChild(tip);
  const guide = document.createElementNS(NS, "line");
  guide.setAttribute("stroke", COLORS.purpleSoft);
  guide.setAttribute("stroke-width", "1");
  guide.setAttribute("y1", m.t); guide.setAttribute("y2", m.t + ih);
  guide.style.opacity = 0;
  svg.appendChild(guide);

  function survivalAt(points, x) {
    // fraction at or above share x  == y of last point with px <= x ; default 1
    const pts = [...points].sort((a, b) => a.x - b.x);
    let y = 1;
    for (const p of pts) { if (p.x <= x) y = p.y; else break; }
    return y;
  }
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    if (px < m.l || px > m.l + iw) { tip.style.opacity = 0; guide.style.opacity = 0; return; }
    const xv = ((px - m.l) / iw) * xMax;
    guide.setAttribute("x1", px); guide.setAttribute("x2", px); guide.style.opacity = 1;
    const yo = survivalAt(ours.points, xv), yb = survivalAt(base.points, xv);
    tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 170) + "px";
    tip.style.top = (e.clientY - r.top - 10) + "px";
    tip.innerHTML = `≥ ${Math.round(xv * 100)}% of tasks exposed<br><b>Ours:</b> ${Math.round(yo * 100)}% of occ.<br>Core/supp: ${Math.round(yb * 100)}% of occ.`;
  });
  svg.addEventListener("mouseleave", () => { tip.style.opacity = 0; guide.style.opacity = 0; });
}

async function initFigure4() {
  F4.data = await loadJSON("data/fig4_cdf.json");
  document.querySelectorAll("#f4thresh button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#f4thresh button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active"); F4.thresh = b.dataset.t; f4Render();
    });
  });
  document.querySelectorAll("#f4measure button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#f4measure button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active"); F4.measure = b.dataset.m; f4Render();
    });
  });
  f4Render();
}

/* ---------- boot ---------- */
initFigure1();
initFigure4();
