import { DatabaseSync } from "node:sqlite";
import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(scriptDir, "../..");
const dbPath = resolve(root, ".codegraph/codegraph.db");
const outputPath = resolve(root, ".scratch/codegraph-map/codegraph-map.html");

const db = new DatabaseSync(dbPath, { readOnly: true });

const files = db
  .prepare(
    `
    select path, language, size, node_count
    from files
    order by path
    `,
  )
  .all();

const edgeRows = db
  .prepare(
    `
    select
      e.kind,
      source.file_path as source_file,
      target.file_path as target_file,
      source.qualified_name as source_symbol,
      target.qualified_name as target_symbol
    from edges e
    join nodes source on source.id = e.source
    join nodes target on target.id = e.target
    where e.kind != 'contains'
    `,
  )
  .all();

function normalizePath(path) {
  return String(path).replaceAll("\\", "/");
}

function moduleName(filePath) {
  const parts = normalizePath(filePath).split("/");
  if (parts[0] === "src" && parts[1] === "lattice_studio") {
    if (parts[2] === "presentation" && parts[3] === "qt") {
      return parts.slice(0, Math.min(parts.length - 1, 5)).join("/");
    }
    if (parts[2] === "engine" && parts[3]) {
      return parts.slice(0, Math.min(parts.length - 1, 4)).join("/");
    }
    return parts.slice(0, Math.min(parts.length - 1, 3)).join("/");
  }
  if (parts[0] === "tests") {
    return parts.slice(0, Math.min(parts.length - 1, 3)).join("/");
  }
  if (parts[0] === ".scratch") {
    return parts.slice(0, Math.min(parts.length - 1, 2)).join("/");
  }
  if (parts.length > 1) {
    return parts.slice(0, Math.min(parts.length - 1, 2)).join("/");
  }
  return "(root)";
}

function addNode(map, id, extra = {}) {
  if (!map.has(id)) {
    map.set(id, {
      id,
      label: id.split("/").slice(-2).join("/"),
      files: 0,
      symbols: 0,
      size: 0,
      incoming: 0,
      outgoing: 0,
      kinds: {},
      ...extra,
    });
  }
  return map.get(id);
}

function graphFromFiles() {
  const nodes = new Map();
  const links = new Map();

  for (const file of files) {
    addNode(nodes, normalizePath(file.path), {
      label: normalizePath(file.path).split("/").pop(),
      group: moduleName(file.path),
      files: 1,
      symbols: file.node_count ?? 0,
      size: file.size ?? 0,
      language: file.language,
    });
  }

  for (const edge of edgeRows) {
    const source = normalizePath(edge.source_file);
    const target = normalizePath(edge.target_file);
    if (!source || !target || source === target) continue;
    const key = `${source}\u0000${target}\u0000${edge.kind}`;
    const link = links.get(key) ?? { source, target, kind: edge.kind, weight: 0, examples: [] };
    link.weight += 1;
    if (link.examples.length < 4) {
      link.examples.push(`${edge.source_symbol} -> ${edge.target_symbol}`);
    }
    links.set(key, link);
  }

  return finalizeGraph(nodes, [...links.values()]);
}

function graphFromModules() {
  const nodes = new Map();
  const fileToModule = new Map();
  const links = new Map();

  for (const file of files) {
    const path = normalizePath(file.path);
    const module = moduleName(path);
    fileToModule.set(path, module);
    const node = addNode(nodes, module, { group: module.split("/")[0] || "(root)" });
    node.files += 1;
    node.symbols += file.node_count ?? 0;
    node.size += file.size ?? 0;
  }

  for (const edge of edgeRows) {
    const source = fileToModule.get(normalizePath(edge.source_file));
    const target = fileToModule.get(normalizePath(edge.target_file));
    if (!source || !target || source === target) continue;
    const key = `${source}\u0000${target}\u0000${edge.kind}`;
    const link = links.get(key) ?? { source, target, kind: edge.kind, weight: 0, examples: [] };
    link.weight += 1;
    if (link.examples.length < 5) {
      link.examples.push(`${edge.source_symbol} -> ${edge.target_symbol}`);
    }
    links.set(key, link);
  }

  return finalizeGraph(nodes, [...links.values()]);
}

function finalizeGraph(nodes, links) {
  for (const link of links) {
    const source = nodes.get(link.source);
    const target = nodes.get(link.target);
    if (!source || !target) continue;
    source.outgoing += link.weight;
    target.incoming += link.weight;
    source.kinds[link.kind] = (source.kinds[link.kind] ?? 0) + link.weight;
    target.kinds[link.kind] = (target.kinds[link.kind] ?? 0) + link.weight;
  }
  return {
    nodes: [...nodes.values()],
    links: links
      .filter((link) => nodes.has(link.source) && nodes.has(link.target))
      .sort((a, b) => b.weight - a.weight),
  };
}

const data = {
  generatedAt: new Date().toISOString(),
  stats: {
    files: files.length,
    edges: edgeRows.length,
    modules: new Set(files.map((file) => moduleName(file.path))).size,
  },
  moduleGraph: graphFromModules(),
  fileGraph: graphFromFiles(),
};

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CodeGraph Map</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101214;
  --panel: #171b1f;
  --panel-2: #20262b;
  --text: #edf2f4;
  --muted: #9aa8b2;
  --line: #3b454d;
  --accent: #69c0a3;
  --accent-2: #f0b35a;
  --danger: #ef6f6c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  overflow: hidden;
}
.app {
  display: grid;
  grid-template-columns: 320px 1fr 340px;
  height: 100vh;
}
aside {
  background: var(--panel);
  border-right: 1px solid #293038;
  padding: 18px;
  overflow: auto;
}
.details {
  border-left: 1px solid #293038;
  border-right: 0;
}
h1, h2 {
  margin: 0;
  letter-spacing: 0;
}
h1 { font-size: 21px; }
h2 { font-size: 13px; margin-top: 22px; color: var(--muted); text-transform: uppercase; }
.subtle { color: var(--muted); }
.stat-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 16px 0 18px;
}
.stat {
  background: var(--panel-2);
  border: 1px solid #303941;
  border-radius: 8px;
  padding: 10px;
}
.stat strong { display: block; font-size: 18px; }
label { display: block; margin: 12px 0 6px; color: var(--muted); }
select, input[type="search"], input[type="range"] {
  width: 100%;
  border: 1px solid #35414a;
  border-radius: 7px;
  background: #0f1316;
  color: var(--text);
  padding: 9px 10px;
}
.checks {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}
.check {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--panel-2);
  border: 1px solid #303941;
  border-radius: 7px;
  padding: 8px;
}
.legend {
  display: grid;
  gap: 8px;
  margin-top: 10px;
}
.legend span { display: flex; align-items: center; gap: 8px; color: var(--muted); }
.swatch { width: 16px; height: 4px; border-radius: 999px; display: inline-block; }
main {
  position: relative;
  min-width: 0;
}
svg {
  display: block;
  width: 100%;
  height: 100%;
  background:
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px),
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px);
  background-size: 28px 28px;
}
.toolbar {
  position: absolute;
  left: 16px;
  top: 16px;
  display: flex;
  gap: 8px;
}
button {
  background: #182126;
  color: var(--text);
  border: 1px solid #34414a;
  border-radius: 7px;
  padding: 8px 10px;
  cursor: pointer;
}
button:hover { border-color: var(--accent); }
.node circle {
  stroke: rgba(255,255,255,.78);
  stroke-width: 1.2;
  cursor: grab;
}
.node text {
  fill: var(--text);
  paint-order: stroke;
  stroke: rgba(16,18,20,.95);
  stroke-width: 4px;
  font-size: 11px;
  pointer-events: none;
}
.link {
  fill: none;
  stroke-linecap: round;
  opacity: .42;
}
.link.active { opacity: .9; }
.node.dim, .link.dim { opacity: .08; }
.pill {
  display: inline-flex;
  margin: 5px 5px 0 0;
  padding: 4px 7px;
  border-radius: 999px;
  background: #222a30;
  color: var(--muted);
  font-size: 12px;
}
.item {
  padding: 10px 0;
  border-bottom: 1px solid #283038;
  word-break: break-word;
}
.item strong { display: block; color: var(--text); }
.empty { color: var(--muted); margin-top: 18px; }
@media (max-width: 1050px) {
  .app { grid-template-columns: 280px 1fr; }
  .details { display: none; }
}
</style>
</head>
<body>
<div class="app">
  <aside>
    <h1>CodeGraph Map</h1>
    <div class="subtle">Generated <span id="generated"></span></div>
    <div class="stat-grid">
      <div class="stat"><strong id="fileCount"></strong><span class="subtle">files</span></div>
      <div class="stat"><strong id="moduleCount"></strong><span class="subtle">modules</span></div>
      <div class="stat"><strong id="edgeCount"></strong><span class="subtle">edges</span></div>
    </div>

    <label for="graphMode">Graph</label>
    <select id="graphMode">
      <option value="moduleGraph">Module level</option>
      <option value="fileGraph">File level</option>
    </select>

    <label for="search">Search nodes</label>
    <input id="search" type="search" placeholder="workbench, implicit, tests...">

    <label for="minWeight">Minimum edge weight: <span id="minWeightValue">1</span></label>
    <input id="minWeight" type="range" min="1" max="25" value="1">

    <h2>Edge kinds</h2>
    <div class="checks" id="kindChecks"></div>

    <h2>Legend</h2>
    <div class="legend">
      <span><i class="swatch" style="background:#69c0a3"></i> calls</span>
      <span><i class="swatch" style="background:#78a6ff"></i> imports</span>
      <span><i class="swatch" style="background:#f0b35a"></i> references</span>
      <span><i class="swatch" style="background:#de8cff"></i> instantiates</span>
      <span><i class="swatch" style="background:#ef6f6c"></i> extends</span>
    </div>
  </aside>
  <main>
    <div class="toolbar">
      <button id="reset">Reset view</button>
      <button id="settle">Settle layout</button>
    </div>
    <svg id="map" role="img" aria-label="CodeGraph dependency map"></svg>
  </main>
  <aside class="details">
    <h1>Selection</h1>
    <div id="selection" class="empty">Click a node or edge.</div>
    <h2>Strongest links</h2>
    <div id="topLinks"></div>
  </aside>
</div>
<script>
const DATA = ${JSON.stringify(data)};
const COLORS = {
  calls: "#69c0a3",
  imports: "#78a6ff",
  references: "#f0b35a",
  instantiates: "#de8cff",
  extends: "#ef6f6c",
};

const svg = document.getElementById("map");
const graphMode = document.getElementById("graphMode");
const search = document.getElementById("search");
const minWeight = document.getElementById("minWeight");
const minWeightValue = document.getElementById("minWeightValue");
const kindChecks = document.getElementById("kindChecks");
const selection = document.getElementById("selection");
const topLinks = document.getElementById("topLinks");

document.getElementById("generated").textContent = new Date(DATA.generatedAt).toLocaleString();
document.getElementById("fileCount").textContent = DATA.stats.files;
document.getElementById("moduleCount").textContent = DATA.stats.modules;
document.getElementById("edgeCount").textContent = DATA.stats.edges;

const edgeKinds = Object.keys(COLORS);
const enabledKinds = new Set(edgeKinds);
for (const kind of edgeKinds) {
  const label = document.createElement("label");
  label.className = "check";
  label.innerHTML = '<input type="checkbox" checked value="' + kind + '"><span>' + kind + '</span>';
  kindChecks.appendChild(label);
}

let view = { scale: 1, x: 0, y: 0 };
let nodes = [];
let links = [];
let selected = null;
let dragNode = null;
let panning = null;
let animation = null;

function radius(node) {
  return Math.max(5, Math.min(24, 4 + Math.sqrt(node.symbols + node.incoming + node.outgoing) * 0.55));
}

function cloneGraph() {
  const graph = DATA[graphMode.value];
  const allowed = new Set([...enabledKinds]);
  const min = Number(minWeight.value);
  const filteredLinks = graph.links.filter((link) => allowed.has(link.kind) && link.weight >= min);
  const linkedIds = new Set(filteredLinks.flatMap((link) => [link.source, link.target]));
  const query = search.value.trim().toLowerCase();
  nodes = graph.nodes
    .filter((node) => linkedIds.has(node.id) || query)
    .filter((node) => !query || node.id.toLowerCase().includes(query) || node.label.toLowerCase().includes(query))
    .map((node, index) => ({
      ...node,
      x: (svg.clientWidth / 2) + Math.cos(index * 2.399) * (80 + index * 3),
      y: (svg.clientHeight / 2) + Math.sin(index * 2.399) * (80 + index * 3),
      vx: 0,
      vy: 0,
    }));
  const ids = new Set(nodes.map((node) => node.id));
  links = filteredLinks
    .filter((link) => ids.has(link.source) && ids.has(link.target))
    .map((link) => ({ ...link }));
  for (const link of links) {
    link.sourceNode = nodes.find((node) => node.id === link.source);
    link.targetNode = nodes.find((node) => node.id === link.target);
  }
  selected = null;
  runLayout(180);
  render();
  renderTopLinks();
}

function runLayout(iterations = 1) {
  const width = svg.clientWidth || 800;
  const height = svg.clientHeight || 600;
  const centerX = width / 2;
  const centerY = height / 2;
  for (let step = 0; step < iterations; step += 1) {
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j];
        const dx = a.x - b.x || 0.1;
        const dy = a.y - b.y || 0.1;
        const dist2 = dx * dx + dy * dy;
        const force = Math.min(6, 900 / dist2);
        a.vx += dx * force * 0.012;
        a.vy += dy * force * 0.012;
        b.vx -= dx * force * 0.012;
        b.vy -= dy * force * 0.012;
      }
    }
    for (const link of links) {
      const a = link.sourceNode;
      const b = link.targetNode;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.hypot(dx, dy) || 1;
      const target = 95 + Math.max(0, 11 - Math.log2(link.weight + 1)) * 12;
      const force = (dist - target) * 0.006 * Math.min(4, Math.log2(link.weight + 1));
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }
    for (const node of nodes) {
      node.vx += (centerX - node.x) * 0.0008;
      node.vy += (centerY - node.y) * 0.0008;
      node.vx *= 0.82;
      node.vy *= 0.82;
      if (!node.fixed) {
        node.x += node.vx;
        node.y += node.vy;
      }
    }
  }
}

function startAnimation(frames = 140) {
  cancelAnimationFrame(animation);
  let remaining = frames;
  function tick() {
    runLayout(1);
    render();
    remaining -= 1;
    if (remaining > 0) animation = requestAnimationFrame(tick);
  }
  tick();
}

function edgePath(link) {
  const a = link.sourceNode;
  const b = link.targetNode;
  return "M" + a.x + "," + a.y + " L" + b.x + "," + b.y;
}

function transformPoint(event) {
  const rect = svg.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left - view.x) / view.scale,
    y: (event.clientY - rect.top - view.y) / view.scale,
  };
}

function render() {
  svg.replaceChildren();
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.setAttribute("transform", "translate(" + view.x + " " + view.y + ") scale(" + view.scale + ")");
  svg.appendChild(g);

  const activeIds = new Set();
  if (selected?.type === "node") {
    activeIds.add(selected.id);
    for (const link of links) {
      if (link.source === selected.id) activeIds.add(link.target);
      if (link.target === selected.id) activeIds.add(link.source);
    }
  }

  for (const link of links) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const active = selected?.type === "link" && selected.link === link;
    const related = selected?.type === "node" && (link.source === selected.id || link.target === selected.id);
    path.setAttribute("d", edgePath(link));
    path.setAttribute("class", "link" + (active || related ? " active" : "") + (selected && !active && !related ? " dim" : ""));
    path.setAttribute("stroke", COLORS[link.kind] || "#888");
    path.setAttribute("stroke-width", String(Math.max(1, Math.min(9, Math.sqrt(link.weight)))));
    path.addEventListener("click", (event) => {
      event.stopPropagation();
      selected = { type: "link", link };
      renderSelection();
      render();
    });
    g.appendChild(path);
  }

  for (const node of nodes) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const isDim = selected?.type === "node" && !activeIds.has(node.id);
    group.setAttribute("class", "node" + (isDim ? " dim" : ""));
    group.setAttribute("transform", "translate(" + node.x + " " + node.y + ")");
    group.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
      dragNode = node;
      node.fixed = true;
      svg.setPointerCapture(event.pointerId);
    });
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      selected = { type: "node", id: node.id, node };
      renderSelection();
      render();
    });

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", String(radius(node)));
    circle.setAttribute("fill", colorForGroup(node.group || node.id));
    group.appendChild(circle);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", String(radius(node) + 5));
    text.setAttribute("y", "4");
    text.textContent = node.label;
    group.appendChild(text);
    g.appendChild(group);
  }
}

function colorForGroup(group) {
  let hash = 0;
  for (let i = 0; i < group.length; i += 1) hash = (hash * 31 + group.charCodeAt(i)) >>> 0;
  const hue = hash % 360;
  return "hsl(" + hue + " 48% 52%)";
}

function renderSelection() {
  if (!selected) {
    selection.className = "empty";
    selection.textContent = "Click a node or edge.";
    return;
  }
  selection.className = "";
  if (selected.type === "node") {
    const node = selected.node;
    selection.innerHTML = [
      '<div class="item"><strong>' + escapeHtml(node.id) + '</strong><span class="subtle">' + escapeHtml(node.group || "") + '</span></div>',
      '<span class="pill">' + node.files + ' files</span>',
      '<span class="pill">' + node.symbols + ' symbols</span>',
      '<span class="pill">' + node.incoming + ' in</span>',
      '<span class="pill">' + node.outgoing + ' out</span>',
      '<h2>Edge mix</h2>',
      Object.entries(node.kinds).sort((a, b) => b[1] - a[1]).map(([kind, count]) => '<span class="pill">' + kind + ': ' + count + '</span>').join(""),
    ].join("");
  } else {
    const link = selected.link;
    selection.innerHTML = [
      '<div class="item"><strong>' + escapeHtml(link.source) + '</strong><span class="subtle">to</span><strong>' + escapeHtml(link.target) + '</strong></div>',
      '<span class="pill">' + escapeHtml(link.kind) + '</span>',
      '<span class="pill">' + link.weight + ' edges</span>',
      '<h2>Examples</h2>',
      link.examples.map((example) => '<div class="item">' + escapeHtml(example) + '</div>').join(""),
    ].join("");
  }
}

function renderTopLinks() {
  topLinks.innerHTML = links
    .slice()
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 12)
    .map((link) => '<div class="item"><strong>' + escapeHtml(link.source) + ' -> ' + escapeHtml(link.target) + '</strong><span class="pill">' + escapeHtml(link.kind) + '</span><span class="pill">' + link.weight + '</span></div>')
    .join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

svg.addEventListener("pointermove", (event) => {
  if (dragNode) {
    const point = transformPoint(event);
    dragNode.x = point.x;
    dragNode.y = point.y;
    dragNode.vx = 0;
    dragNode.vy = 0;
    render();
  } else if (panning) {
    view.x = event.clientX - panning.x;
    view.y = event.clientY - panning.y;
    render();
  }
});

svg.addEventListener("pointerup", () => {
  if (dragNode) dragNode.fixed = false;
  dragNode = null;
  panning = null;
});

svg.addEventListener("pointerdown", (event) => {
  selected = null;
  renderSelection();
  panning = { x: event.clientX - view.x, y: event.clientY - view.y };
  render();
});

svg.addEventListener("wheel", (event) => {
  event.preventDefault();
  const delta = event.deltaY > 0 ? 0.9 : 1.1;
  const rect = svg.getBoundingClientRect();
  const sx = event.clientX - rect.left;
  const sy = event.clientY - rect.top;
  const before = { x: (sx - view.x) / view.scale, y: (sy - view.y) / view.scale };
  view.scale = Math.max(0.2, Math.min(4, view.scale * delta));
  view.x = sx - before.x * view.scale;
  view.y = sy - before.y * view.scale;
  render();
}, { passive: false });

document.getElementById("reset").addEventListener("click", () => {
  view = { scale: 1, x: 0, y: 0 };
  cloneGraph();
});
document.getElementById("settle").addEventListener("click", () => startAnimation(180));
graphMode.addEventListener("change", cloneGraph);
search.addEventListener("input", cloneGraph);
minWeight.addEventListener("input", () => {
  minWeightValue.textContent = minWeight.value;
  cloneGraph();
});
kindChecks.addEventListener("change", (event) => {
  if (event.target.matches("input[type=checkbox]")) {
    if (event.target.checked) enabledKinds.add(event.target.value);
    else enabledKinds.delete(event.target.value);
    cloneGraph();
  }
});
window.addEventListener("resize", () => render());

cloneGraph();
startAnimation(120);
</script>
</body>
</html>`;

writeFileSync(outputPath, html, "utf8");
console.log(outputPath);
