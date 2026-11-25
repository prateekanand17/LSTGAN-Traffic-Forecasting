// Graph topology analysis for tour design. Usage: node ua-tour-analyze.js <input.json> <output.json>
const fs = require('fs');
const path = require('path');

const ENTRY_NAMES = new Set(['index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js','mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php','App.swift','Application.kt','main.cpp','main.c']);

function main() {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) throw new Error('Usage: ua-tour-analyze.js <input> <output>');
  const { nodes, edges, layers } = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const known = new Map(nodes.map(n => [n.id, n]));

  // Edges may point at function:/class: children, so roll them up to their owning file-level node.
  const rollUp = id => {
    if (known.has(id)) return id;
    const m = /^(?:function|class|method):(.+?):[^:]+$/.exec(id);
    if (m && known.has('file:' + m[1])) return 'file:' + m[1];
    return null;
  };

  const seen = new Set();
  const fileEdges = [];
  let dropped = 0;
  for (const e of edges) {
    const source = rollUp(e.source), target = rollUp(e.target);
    if (!source || !target) { dropped++; continue; }
    if (source === target) continue;
    const key = `${source}|${target}|${e.type}`;
    if (seen.has(key)) continue;
    seen.add(key);
    fileEdges.push({ source, target, type: e.type });
  }

  const inSets = new Map(nodes.map(n => [n.id, new Set()]));
  const outSets = new Map(nodes.map(n => [n.id, new Set()]));
  for (const e of fileEdges) {
    outSets.get(e.source).add(e.target);
    inSets.get(e.target).add(e.source);
  }
  const fanIn = id => inSets.get(id).size;
  const fanOut = id => outSets.get(id).size;

  const rank = (fn, key) => nodes
    .map(n => ({ id: n.id, [key]: fn(n.id), name: n.name }))
    .filter(r => r[key] > 0)
    .sort((a, b) => b[key] - a[key] || a.id.localeCompare(b.id))
    .slice(0, 20);
  const fanInRanking = rank(fanIn, 'fanIn');
  const fanOutRanking = rank(fanOut, 'fanOut');

  const sortedOut = nodes.map(n => fanOut(n.id)).sort((a, b) => b - a);
  const sortedIn = nodes.map(n => fanIn(n.id)).sort((a, b) => a - b);
  const outCut = sortedOut[Math.max(0, Math.ceil(nodes.length * 0.1) - 1)] || 0;
  const inCut = sortedIn[Math.max(0, Math.ceil(nodes.length * 0.25) - 1)] || 0;

  const scored = nodes.map(n => {
    let score = 0;
    const fp = (n.filePath || n.name).replace(/\\/g, '/');
    const depth = fp.split('/').length - 1;
    if (n.type === 'document') {
      if (depth === 0 && /^readme\.md$/i.test(n.name)) score += 5;
      else if (depth === 0 && /\.md$/i.test(n.name)) score += 2;
    } else if (n.type === 'file') {
      if (ENTRY_NAMES.has(n.name)) score += 3;
      if (depth <= 1) score += 1;
      if (outCut > 0 && fanOut(n.id) >= outCut) score += 1;
      if (fanIn(n.id) <= inCut) score += 1;
    }
    return { id: n.id, score, name: n.name, summary: n.summary, type: n.type };
  }).sort((a, b) => b.score - a.score || fanOut(b.id) - fanOut(a.id));
  const entryPointCandidates = scored.slice(0, 5);

  const codeEntry = scored.find(c => c.type === 'file' && !/__init__/.test(c.name));
  const adj = new Map(nodes.map(n => [n.id, []]));
  for (const e of fileEdges) if (e.type === 'imports' || e.type === 'calls') adj.get(e.source).push(e.target);
  const order = [], depthMap = {}, byDepth = {};
  if (codeEntry) {
    const q = [codeEntry.id];
    depthMap[codeEntry.id] = 0;
    while (q.length) {
      const cur = q.shift();
      order.push(cur);
      const d = depthMap[cur];
      (byDepth[d] = byDepth[d] || []).push(cur);
      for (const nx of [...new Set(adj.get(cur))].sort()) {
        if (!(nx in depthMap)) { depthMap[nx] = d + 1; q.push(nx); }
      }
    }
  }

  const pick = n => ({ id: n.id, name: n.name, summary: n.summary });
  const ofType = types => nodes.filter(n => types.includes(n.type)).map(pick);
  const nonCodeFiles = {
    documentation: ofType(['document']),
    infrastructure: ofType(['service', 'pipeline', 'resource']),
    data: ofType(['table', 'schema', 'endpoint']),
    config: ofType(['config']),
  };

  // Clusters: seed from bidirectional imports/calls pairs, expand with nodes touching 2+ members.
  const dep = new Set(fileEdges.filter(e => e.type === 'imports' || e.type === 'calls').map(e => `${e.source}|${e.target}`));
  const neighbours = new Map(nodes.map(n => [n.id, new Set([...inSets.get(n.id), ...outSets.get(n.id)])]));
  const clusters = [];
  const ids = nodes.map(n => n.id);
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = ids[i], b = ids[j];
      if (!(dep.has(`${a}|${b}`) && dep.has(`${b}|${a}`))) continue;
      const members = new Set([a, b]);
      for (const c of ids) {
        if (members.size >= 5) break;
        if (members.has(c)) continue;
        let links = 0;
        for (const m of members) if (neighbours.get(c).has(m)) links++;
        if (links >= 2) members.add(c);
      }
      const list = [...members].sort();
      const key = list.join('|');
      if (clusters.some(c => c.key === key)) continue;
      const edgeCount = fileEdges.filter(e => members.has(e.source) && members.has(e.target)).length;
      clusters.push({ key, nodes: list, edgeCount });
    }
  }
  const clusterOut = clusters.sort((a, b) => b.edgeCount - a.edgeCount).slice(0, 10).map(({ nodes, edgeCount }) => ({ nodes, edgeCount }));

  const edgeTypeCounts = {};
  for (const e of fileEdges) edgeTypeCounts[e.type] = (edgeTypeCounts[e.type] || 0) + 1;

  const nodeSummaryIndex = {};
  for (const n of nodes) nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary };

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: { startNode: codeEntry ? codeEntry.id : null, order, depthMap, byDepth },
    nonCodeFiles,
    clusters: clusterOut,
    layers: { count: layers.length, list: layers.map(l => ({ id: l.id, name: l.name, description: l.description })) },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
    fileLevelEdges: fileEdges.length,
    droppedEdges: dropped,
    edgeTypeCounts,
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 1));
}

try { main(); } catch (err) { console.error(err.stack || String(err)); process.exit(1); }
