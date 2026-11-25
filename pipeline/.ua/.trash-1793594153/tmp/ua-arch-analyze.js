// Structural analysis for layer identification. Args: <input.json> <output.json>
const fs = require('fs');
const path = require('path');

const DIR_PATTERNS = {
  api: ['routes', 'api', 'controllers', 'endpoints', 'handlers', 'serializers', 'controller', 'routers', 'blueprints'],
  service: ['services', 'core', 'lib', 'domain', 'logic', 'signals', 'internal', 'composables', 'mailers', 'jobs', 'channels'],
  data: ['models', 'db', 'data', 'persistence', 'repository', 'entities', 'migrations', 'entity', 'sql', 'database', 'schema'],
  ui: ['components', 'views', 'pages', 'ui', 'layouts', 'screens'],
  middleware: ['middleware', 'plugins', 'interceptors', 'guards'],
  utility: ['utils', 'helpers', 'common', 'shared', 'tools', 'templatetags', 'pkg'],
  config: ['config', 'constants', 'env', 'settings', 'management', 'commands'],
  test: ['__tests__', 'test', 'tests', 'spec', 'specs'],
  types: ['types', 'interfaces', 'schemas', 'contracts', 'dtos', 'dto', 'request', 'response'],
  hooks: ['hooks'],
  state: ['store', 'state', 'reducers', 'actions', 'slices'],
  assets: ['assets', 'static', 'public'],
  entry: ['cmd', 'bin'],
  documentation: ['docs', 'documentation', 'wiki'],
  infrastructure: ['deploy', 'deployment', 'infra', 'infrastructure', 'k8s', 'kubernetes', 'helm', 'charts', 'terraform', 'tf', 'docker'],
  'ci-cd': ['.github', '.gitlab', '.circleci'],
};

function dirLabel(name) {
  for (const [label, names] of Object.entries(DIR_PATTERNS)) {
    if (names.includes(name)) return label;
  }
  return null;
}

function fileLabel(p) {
  const b = path.posix.basename(p);
  if (/\.(test|spec)\./.test(b) || /^test_.*\.py$/.test(b) || /_test\.go$/.test(b)) return 'test';
  if (/\.d\.ts$/.test(b)) return 'types';
  if (['index.ts', 'index.js', '__init__.py'].includes(b)) return 'entry';
  if (b === 'manage.py' && !p.includes('/')) return 'entry';
  if (['wsgi.py', 'asgi.py'].includes(b)) return 'config';
  if (/^(Dockerfile|docker-compose\.)/.test(b) || /\.(tf|tfvars)$/.test(b) || b === 'Makefile') return 'infrastructure';
  if (/\.sql$/.test(b)) return 'data';
  if (/\.(graphql|gql|proto)$/.test(b)) return 'types';
  if (/\.(md|rst)$/.test(b)) return 'documentation';
  return null;
}

function main() {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) throw new Error('usage: ua-arch-analyze.js <input> <output>');
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.fileNodes;
  const ids = new Set(nodes.map((n) => n.id));
  const allEdges = (input.allEdges || []).filter((e) => ids.has(e.source) && ids.has(e.target));
  const importEdges = (input.importEdges || []).filter((e) => ids.has(e.source) && ids.has(e.target));

  // A. Directory groups: first segment; root-level files form the "root" group.
  const segs = nodes.map((n) => n.filePath.split('/'));
  let prefix = 0;
  if (segs.length && segs.every((s) => s.length > 1)) {
    while (segs.every((s) => s.length > prefix + 1 && s[prefix] === segs[0][prefix])) prefix++;
  }
  const groupOf = {};
  const directoryGroups = {};
  nodes.forEach((n, i) => {
    const rest = segs[i].slice(prefix);
    const g = rest.length > 1 ? rest[0] : 'root';
    groupOf[n.id] = g;
    (directoryGroups[g] = directoryGroups[g] || []).push(n.id);
  });

  // B. Node type groups
  const nodeTypeGroups = {};
  for (const n of nodes) (nodeTypeGroups[n.type] = nodeTypeGroups[n.type] || []).push(n.id);

  // C. Fan-in / fan-out
  const fileFanIn = {};
  const fileFanOut = {};
  const adjacency = {};
  for (const e of importEdges) {
    fileFanOut[e.source] = (fileFanOut[e.source] || 0) + 1;
    fileFanIn[e.target] = (fileFanIn[e.target] || 0) + 1;
    (adjacency[e.source] = adjacency[e.source] || []).push(e.target);
  }

  // D. Cross-category edges
  const typeOf = Object.fromEntries(nodes.map((n) => [n.id, n.type]));
  const cross = {};
  for (const e of allEdges) {
    if (typeOf[e.source] === typeOf[e.target]) continue;
    const k = `${typeOf[e.source]}|${typeOf[e.target]}|${e.type}`;
    cross[k] = (cross[k] || 0) + 1;
  }
  const crossCategoryEdges = Object.entries(cross).map(([k, count]) => {
    const [fromType, toType, edgeType] = k.split('|');
    return { fromType, toType, edgeType, count };
  });

  // E + F. Inter-group frequency and intra-group density
  const inter = {};
  const internal = {};
  const total = {};
  for (const e of importEdges) {
    const a = groupOf[e.source];
    const b = groupOf[e.target];
    total[a] = (total[a] || 0) + 1;
    if (a === b) {
      internal[a] = (internal[a] || 0) + 1;
    } else {
      total[b] = (total[b] || 0) + 1;
      const k = `${a}|${b}`;
      inter[k] = (inter[k] || 0) + 1;
    }
  }
  const interGroupImports = Object.entries(inter)
    .map(([k, count]) => ({ from: k.split('|')[0], to: k.split('|')[1], count }))
    .sort((x, y) => y.count - x.count);
  const intraGroupDensity = {};
  for (const g of Object.keys(directoryGroups)) {
    const t = total[g] || 0;
    const i = internal[g] || 0;
    intraGroupDensity[g] = { internalEdges: i, totalEdges: t, density: t ? +(i / t).toFixed(3) : 0 };
  }

  // G. Pattern matching
  const patternMatches = {};
  for (const g of Object.keys(directoryGroups)) {
    const l = dirLabel(g);
    if (l) patternMatches[g] = l;
  }
  const filePatternMatches = {};
  for (const n of nodes) {
    const l = fileLabel(n.filePath);
    if (l) filePatternMatches[n.id] = l;
  }

  // H. Deployment topology
  const paths = nodes.map((n) => n.filePath);
  const infraFiles = paths.filter((p) => /(^|\/)(Dockerfile[^/]*|docker-compose[^/]*|Makefile|Jenkinsfile|\.gitlab-ci\.yml)$/.test(p) || /\.(tf|tfvars)$/.test(p) || /^\.github\/workflows\//.test(p) || /(^|\/)(k8s|kubernetes|helm)\//.test(p));
  const deploymentTopology = {
    hasDockerfile: paths.some((p) => /(^|\/)Dockerfile/.test(p)),
    hasCompose: paths.some((p) => /(^|\/)docker-compose/.test(p)),
    hasK8s: paths.some((p) => /(^|\/)(k8s|kubernetes|helm)\//.test(p)),
    hasTerraform: paths.some((p) => /\.(tf|tfvars)$/.test(p)),
    hasCI: paths.some((p) => /^\.github\/workflows\/|\.gitlab-ci\.yml$|Jenkinsfile$/.test(p)),
    infraFiles,
  };

  // I. Data pipeline
  const dataPipeline = {
    schemaFiles: paths.filter((p) => /\.(sql|graphql|gql|proto|prisma)$/.test(p)),
    migrationFiles: paths.filter((p) => /(^|\/)migrations\//.test(p)),
    dataModelFiles: nodes.filter((n) => (n.tags || []).includes('data-model')).map((n) => n.filePath),
    apiHandlerFiles: nodes.filter((n) => (n.tags || []).includes('api-handler')).map((n) => n.filePath),
  };

  // J. Documentation coverage
  const docEdges = (input.allEdges || []).filter((e) => e.type === 'documents' && ids.has(e.source));
  const documented = new Set();
  for (const n of nodes) {
    if (n.type === 'document') {
      const dir = path.posix.dirname(n.filePath);
      if (dir !== '.') documented.add(dir.split('/')[0]);
    }
  }
  for (const e of docEdges) if (groupOf[e.target]) documented.add(groupOf[e.target]);
  const groups = Object.keys(directoryGroups);
  const docCoverage = {
    groupsWithDocs: groups.filter((g) => documented.has(g)).length,
    totalGroups: groups.length,
    coverageRatio: groups.length ? +(groups.filter((g) => documented.has(g)).length / groups.length).toFixed(2) : 0,
    undocumentedGroups: groups.filter((g) => !documented.has(g)),
  };

  // K. Dependency direction
  const dependencyDirection = [];
  const seen = new Set();
  for (const k of Object.keys(inter)) {
    const [a, b] = k.split('|');
    const pair = [a, b].sort().join('|');
    if (seen.has(pair)) continue;
    seen.add(pair);
    const ab = inter[`${a}|${b}`] || 0;
    const ba = inter[`${b}|${a}`] || 0;
    if (ab === ba) dependencyDirection.push({ dependent: a, dependsOn: b, mutual: true, counts: [ab, ba] });
    else if (ab > ba) dependencyDirection.push({ dependent: a, dependsOn: b, counts: [ab, ba] });
    else dependencyDirection.push({ dependent: b, dependsOn: a, counts: [ba, ab] });
  }

  const edgeTypeCounts = {};
  for (const e of input.allEdges || []) edgeTypeCounts[e.type] = (edgeTypeCounts[e.type] || 0) + 1;

  const result = {
    scriptCompleted: true,
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    filePatternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    importAdjacency: adjacency,
    edgeTypeCounts,
    fileStats: {
      totalFileNodes: nodes.length,
      fileLevelImportEdges: importEdges.length,
      filesPerGroup: Object.fromEntries(groups.map((g) => [g, directoryGroups[g].length])),
      nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([t, v]) => [t, v.length])),
    },
    fileFanIn,
    fileFanOut,
  };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
}

try {
  main();
} catch (err) {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
}
