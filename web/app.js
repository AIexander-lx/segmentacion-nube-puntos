'use strict';

const $ = id => document.getElementById(id);
const canvas = $('canvas');
const gl = canvas.getContext('webgl', {antialias: true, preserveDrawingBuffer: true});
let program, pointBuffer, meshBuffer, pointCount = 0, meshCount = 0;
let metadata, yaw = -0.75, pitch = 0.7, scale = 1, fitScale = 1, center = [0, 0, 0], extent = 1, bounds = null, pointSize = 2;
const classVisibility = {};
const vertexShader = `attribute vec3 position;attribute vec3 color;uniform vec3 center;uniform vec3 right;uniform vec3 up;uniform vec3 forward;uniform float scale;uniform float aspect;uniform float depthRange;uniform float pointSize;varying vec3 vColor;void main(){vec3 p=position-center;gl_Position=vec4(dot(p,right)/(scale*aspect),dot(p,up)/scale,dot(p,forward)/depthRange,1.0);gl_PointSize=pointSize;vColor=color;}`;
const fragmentShader = `precision mediump float;varying vec3 vColor;uniform float alpha;uniform bool points;void main(){if(points&&length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,alpha);}`;

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
function initGL() {
  if (!gl) throw new Error('WebGL no está disponible en este navegador.');
  program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShader));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShader));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program); pointBuffer = gl.createBuffer(); meshBuffer = gl.createBuffer();
  gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
}
function bind(buffer) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const pos = gl.getAttribLocation(program, 'position'), color = gl.getAttribLocation(program, 'color');
  gl.enableVertexAttribArray(pos); gl.enableVertexAttribArray(color);
  gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 24, 0);
  gl.vertexAttribPointer(color, 3, gl.FLOAT, false, 24, 12);
}
function render() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  gl.viewport(0, 0, width, height); gl.clearColor(27 / 255, 37 / 255, 48 / 255, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const set3 = (name, value) => gl.uniform3fv(gl.getUniformLocation(program, name), value);
  set3('center', center); set3('right', [cy, sy, 0]); set3('up', [-sy * sp, cy * sp, cp]); set3('forward', [sy * cp, -cy * cp, sp]);
  gl.uniform1f(gl.getUniformLocation(program, 'scale'), scale);
  gl.uniform1f(gl.getUniformLocation(program, 'aspect'), width / height);
  gl.uniform1f(gl.getUniformLocation(program, 'depthRange'), Math.max(extent * 2.2, .001));
  gl.uniform1f(gl.getUniformLocation(program, 'pointSize'), pointSize * dpr);
  if (pointCount) { bind(pointBuffer); gl.uniform1f(gl.getUniformLocation(program, 'alpha'), 1); gl.uniform1i(gl.getUniformLocation(program, 'points'), 1); gl.drawArrays(gl.POINTS, 0, pointCount); }
  if (meshCount && $('showMesh').checked) { bind(meshBuffer); gl.uniform1f(gl.getUniformLocation(program, 'alpha'), .48); gl.uniform1i(gl.getUniformLocation(program, 'points'), 0); gl.drawArrays(gl.TRIANGLES, 0, meshCount); }
  requestAnimationFrame(render);
}
function fit(data) {
  if (!data.length) return;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < data.length; i += 6) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], data[i + k]); hi[k] = Math.max(hi[k], data[i + k]); }
  bounds = {lo, hi}; center = lo.map((value, k) => (value + hi[k]) / 2);
  extent = Math.max(...hi.map((value, k) => value - lo[k]), .001); fitCamera(true);
}
function fitCamera(reset = false) {
  if (!bounds) return;
  const oldFit = fitScale || 1, cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const right = [cy, sy, 0], up = [-sy * sp, cy * sp, cp];
  let left = Infinity, rightEdge = -Infinity, bottom = Infinity, top = -Infinity;
  for (let mask = 0; mask < 8; mask++) {
    const p = [0, 1, 2].map(k => (mask & (1 << k)) ? bounds.hi[k] : bounds.lo[k]);
    const rel = p.map((value, k) => value - center[k]);
    const x = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2];
    const y = rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2];
    left = Math.min(left, x); rightEdge = Math.max(rightEdge, x); bottom = Math.min(bottom, y); top = Math.max(top, y);
  }
  const aspect = Math.max(canvas.clientWidth, 1) / Math.max(canvas.clientHeight, 1);
  const projectedWidth = Math.max(rightEdge - left, .001), projectedHeight = Math.max(top - bottom, .001);
  // Ajusta ambos ejes: deja margen visible y conserva la relación de aspecto del modelo.
  fitScale = Math.max(projectedWidth / (2 * .72 * aspect), projectedHeight / (2 * .80), extent * .015);
  scale = reset ? fitScale : fitScale * (scale / oldFit);
}
async function failResponse(response) {
  let body; try { body = await response.json(); } catch (_) { body = {error: response.statusText}; }
  throw new Error(body.error || 'No se pudo completar la solicitud.');
}
function applyClassVisibility(data) {
  if ($('mode').value !== 'semantic' || !metadata) return data;
  const enabled = Object.entries(metadata.classes).filter(([name]) => classVisibility[name] !== false).map(([, rgb]) => rgb);
  if (enabled.length === Object.keys(metadata.classes).length) return data;
  const rows = [];
  for (let i = 0; i < data.length; i += 6) {
    const color = [data[i + 3], data[i + 4], data[i + 5]];
    if (enabled.some(rgb => Math.abs(color[0] - rgb[0]) < .005 && Math.abs(color[1] - rgb[1]) < .005 && Math.abs(color[2] - rgb[2]) < .005)) {
      rows.push(data[i], data[i + 1], data[i + 2], ...color);
    }
  }
  return new Float32Array(rows);
}
async function loadCloud() {
  $('loading').style.display = 'grid';
  try {
    const params = new URLSearchParams({resolution: $('resolution').value, mode: $('mode').value,
      limit: $('limit').value, hide_unknown: $('hideUnknown').checked ? '1' : '0', result: $('result').value});
    const response = await fetch('/api/cloud?' + params);
    if (!response.ok) await failResponse(response);
    const data = applyClassVisibility(new Float32Array(await response.arrayBuffer()));
    if (data.length % 6) throw new Error('La respuesta de puntos está incompleta.');
    pointCount = data.length / 6; gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); fit(data);
    const total = Number(response.headers.get('X-Source-Count'));
    $('pointCount').textContent = `${pointCount.toLocaleString('es-PE')} visibles / ${total.toLocaleString('es-PE')} puntos`;
    $('statPoints').textContent = pointCount.toLocaleString('es-PE'); $('statTotal').textContent = 'Muestreo reproducible · semilla 42';
    const cloudName = $('resolution').selectedOptions[0].textContent.split(' · ')[0];
    $('viewTitle').innerHTML = `${cloudName} <span>/ ${$('mode').selectedOptions[0].textContent}</span>`;
    updateExportLink(); updateClassSummary();
    if ($('mode').value === 'semantic') $('message').textContent = 'Clases heurísticas sin anotaciones humanas. Gris indica unknown; no se asigna por defecto a “other”.';
    else $('message').textContent = metadata.warning;
  } catch (error) {
    $('message').textContent = error.message; $('pointCount').textContent = 'No se pudo cargar la visualización'; pointCount = 0;
  } finally { $('loading').style.display = 'none'; }
}
async function loadMesh() {
  if (!$('showMesh').checked || !$('mesh').value) { meshCount = 0; return; }
  try {
    const response = await fetch('/api/mesh?' + new URLSearchParams({name: $('mesh').value}));
    if (!response.ok) await failResponse(response);
    const xyz = new Float32Array(await response.arrayBuffer());
    if (xyz.length % 9) throw new Error('La malla está incompleta.');
    const data = new Float32Array(xyz.length * 2);
    for (let i = 0, j = 0; i < xyz.length; i += 3, j += 6) { data[j] = xyz[i]; data[j + 1] = xyz[i + 1]; data[j + 2] = xyz[i + 2]; data[j + 3] = .94; data[j + 4] = .52; data[j + 5] = .18; }
    meshCount = xyz.length / 3; gl.bindBuffer(gl.ARRAY_BUFFER, meshBuffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    $('message').textContent = 'Superficie reconstruida inferida; compárala con los puntos observados.';
  } catch (error) { meshCount = 0; $('message').textContent = error.message; }
}
function setView(name) {
  document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === name));
  if (name === 'top') { yaw = 0; pitch = Math.PI / 2; }
  else if (name === 'front') { yaw = 0; pitch = 0; }
  else if (name === 'side') { yaw = Math.PI / 2; pitch = 0; }
  else { yaw = -.75; pitch = .7; }
  fitCamera(true);
}
function updateExportLink() {
  const value = $('result').value;
  $('exportPly').href = '/api/export?' + new URLSearchParams({result: value});
  $('exportPly').classList.toggle('disabled', value !== 'baseline' && !value.includes('v3'));
}
function updateClassSummary() {
  const stats = metadata && metadata.stats || {};
  const v3 = (metadata.result_stats || {})[$('result').value] || stats.semantic_v3 || {};
  const counts = $('result').value === 'baseline' ? ((stats.semantic || {}).coverage || {}).original?.counts : v3.class_counts_original;
  $('classSummary').textContent = counts ? Object.entries(counts).map(([name, count]) => `${name}: ${Number(count).toLocaleString('es-PE')}`).join(' · ') : 'Elige una vista semántica para consultar la leyenda.';
}
function renderLegend() {
  const host = $('legend'); host.replaceChildren();
  if (!metadata) return;
  for (const [name, rgb] of Object.entries(metadata.classes)) {
    if (!(name in classVisibility)) classVisibility[name] = true;
    const label = document.createElement('label'), checkbox = document.createElement('input'), swatch = document.createElement('i'), text = document.createElement('span');
    checkbox.type = 'checkbox'; checkbox.checked = classVisibility[name]; checkbox.onchange = () => { classVisibility[name] = checkbox.checked; loadCloud(); };
    swatch.style.background = `rgb(${rgb.map(value => Math.round(value * 255)).join(',')})`; text.textContent = name;
    label.append(checkbox, swatch, text); host.append(label);
  }
}
async function refresh() {
  try {
    const response = await fetch('/api/metadata'); if (!response.ok) await failResponse(response); metadata = await response.json();
    const results = metadata.results || ['baseline'];
    if (Array.from($('result').options).map(option => option.value).join('|') !== results.join('|')) {
      const current = $('result').value; $('result').replaceChildren();
      for (const result of results) { const option = document.createElement('option'); option.value = result; option.textContent = result === 'baseline' ? 'V1 · Baseline' : (result.includes('v3') ? 'V3 · Refinamiento local' : result); $('result').append(option); }
      $('result').value = results.includes(current) ? current : 'baseline';
    }
    for (const option of $('resolution').options) option.disabled = !metadata.clouds[option.value];
    if ($('resolution').selectedOptions[0].disabled) $('resolution').value = 'original';
    $('mode').querySelector('option[value="instances"]').disabled = !metadata.instances_available;
    $('mode').querySelector('option[value="semantic"]').disabled = !metadata.semantic_available;
    if ($('mode').selectedOptions[0].disabled) $('mode').value = 'rgb';
    renderLegend();
    const currentMesh = $('mesh').value; $('mesh').replaceChildren(new Option('Sin malla', ''));
    for (const mesh of metadata.meshes) $('mesh').append(new Option(mesh, mesh)); $('mesh').value = currentMesh;
    const geometry = (metadata.stats.geometry || {}), semantic = (metadata.stats.semantic || {}), geometryCounts = geometry.counts || {};
    const selected = $('result').value, v3 = (metadata.result_stats || {})[selected] || metadata.stats.semantic_v3 || {};
    if (selected.includes('v3') && v3.counts) {
      $('statSegments').textContent = Number(v3.counts.regions_with_owner_points || 0).toLocaleString('es-PE');
      $('statSegments').nextElementSibling.textContent = 'Regiones locales · tiles';
    } else {
      $('statSegments').textContent = geometryCounts.planes != null ? Number(geometryCounts.planes + geometryCounts.clusters).toLocaleString('es-PE') : '—';
      $('statSegments').nextElementSibling.textContent = 'Planos y componentes';
    }
    let coverage;
    if (selected.includes('v3') && v3.comparison_v3) { coverage = v3.comparison_v3.original_coverage_percent; $('coverageLabel').textContent = 'V3 · heurística'; }
    else { const originalCoverage = ((semantic.coverage || {}).original || {}); coverage = originalCoverage.unknown_percent == null ? null : 100 - originalCoverage.unknown_percent; $('coverageLabel').textContent = 'V1 · heurística'; }
    $('statCoverage').textContent = Number.isFinite(coverage) ? `${coverage.toFixed(2)}%` : '—';
    const timing = selected.includes('v3') ? (v3.timings_seconds || {}) : (geometry.timings_seconds || {});
    const seconds = Object.values(timing).filter(Number.isFinite).reduce((sum, value) => sum + value, 0);
    $('statTime').textContent = seconds ? `${seconds.toFixed(1)} s` : '—';
    $('regionList').textContent = metadata.meshes.length ? metadata.meshes.join(' · ') : 'No hay mallas reconstruidas disponibles.';
    $('job').textContent = metadata.job.message + (metadata.job.elapsed_seconds != null ? ` · ${metadata.job.elapsed_seconds} s` : '') + (metadata.job.output ? ` · ${metadata.job.output}` : '');
    updateExportLink(); updateClassSummary();
  } catch (error) { $('message').textContent = error.message; }
}
async function startJob(kind) {
  try {
    const payload = {kind};
    if (kind === 'reconstruction' || kind === 'hybrid_roi') {
      payload.roi = $('roi').value.trim().split(/\s+/).map(Number);
      if (payload.roi.length !== 6 || payload.roi.some(value => !Number.isFinite(value))) throw new Error('Introduce seis límites XYZ válidos.');
    }
    const response = await fetch('/api/jobs', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Project-Token': metadata.token}, body: JSON.stringify(payload)});
    if (!response.ok) await failResponse(response);
    const job = await response.json(); $('job').textContent = job.message;
  } catch (error) { $('job').textContent = error.message; }
}
async function setInput(path) {
  try {
    const response = await fetch('/api/input', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Project-Token': metadata.token}, body: JSON.stringify({path})});
    if (!response.ok) await failResponse(response); await refresh(); $('resolution').value = path ? 'original' : 'global'; $('mode').value = 'rgb'; await loadCloud();
  } catch (error) { $('message').textContent = error.message; }
}

let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('pointerdown', event => { dragging = true; lastX = event.clientX; lastY = event.clientY; canvas.setPointerCapture(event.pointerId); });
canvas.addEventListener('pointermove', event => { if (dragging) { yaw += (event.clientX - lastX) * .008; pitch = Math.max(-1.55, Math.min(1.55, pitch + (event.clientY - lastY) * .008)); lastX = event.clientX; lastY = event.clientY; fitCamera(false); } });
canvas.addEventListener('pointerup', () => dragging = false);
canvas.addEventListener('wheel', event => { event.preventDefault(); scale = Math.max(fitScale * .22, Math.min(fitScale * 12, scale * Math.exp(event.deltaY * .001))); }, {passive: false});

const layout = $('layout'), toolsPanel = $('toolsPanel'), toggleTools = $('toggleTools'), drawerBackdrop = $('drawerBackdrop');
function setToolsCollapsed(collapsed) {
  layout.classList.toggle('tools-collapsed', collapsed);
  toggleTools.setAttribute('aria-expanded', String(!collapsed));
  toggleTools.textContent = collapsed ? 'Mostrar herramientas' : 'Ocultar herramientas';
  toggleTools.setAttribute('aria-label', collapsed ? 'Mostrar panel de herramientas' : 'Ocultar panel de herramientas');
  if (!collapsed && window.matchMedia('(max-width: 900px)').matches) toolsPanel.scrollTop = 0;
  requestAnimationFrame(() => fitCamera(false));
}
const compactLayout = window.matchMedia('(max-width: 900px)');
setToolsCollapsed(compactLayout.matches);
toggleTools.onclick = () => setToolsCollapsed(!layout.classList.contains('tools-collapsed'));
drawerBackdrop.onclick = () => setToolsCollapsed(true);
compactLayout.addEventListener('change', event => setToolsCollapsed(event.matches));
if ('ResizeObserver' in window) new ResizeObserver(() => fitCamera(false)).observe(canvas);
window.addEventListener('resize', () => fitCamera(false));

document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => {
  document.querySelectorAll('[data-tab]').forEach(tab => tab.classList.toggle('active', tab === button));
  document.querySelectorAll('[data-tab-panel]').forEach(panel => { const active = panel.dataset.tabPanel === button.dataset.tab; panel.hidden = !active; panel.classList.toggle('active', active); });
});
document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => setView(button.dataset.view));
$('load').onclick = loadCloud; $('resolution').onchange = loadCloud; $('mode').onchange = loadCloud; $('result').onchange = async () => { updateExportLink(); await loadCloud(); };
$('limit').onchange = loadCloud; $('hideUnknown').onchange = loadCloud; $('mesh').onchange = loadMesh; $('showMesh').onchange = loadMesh;
$('reset').onclick = () => { setView('iso'); scale = fitScale; };
$('snapshot').onclick = () => { const link = document.createElement('a'); link.download = 'fabrica_segmentacion_3d.png'; link.href = canvas.toDataURL('image/png'); link.click(); };
$('pointSize').oninput = () => { pointSize = Number($('pointSize').value); $('pointSizeValue').value = pointSize; };
$('sampleJob').onclick = () => startJob('sample_segmentation'); $('reconJob').onclick = () => startJob('reconstruction'); $('hybridJob').onclick = () => startJob('hybrid_roi'); $('semanticV3Job').onclick = () => startJob('semantic_v3'); $('fullJob').onclick = () => startJob('full_segmentation');
$('openInput').onclick = () => setInput($('inputPath').value.trim()); $('restoreInput').onclick = () => { $('inputPath').value = ''; setInput(''); };
(async () => { try { initGL(); await refresh(); await loadCloud(); requestAnimationFrame(render); setInterval(refresh, 5000); } catch (error) { $('loading').style.display = 'none'; $('message').textContent = error.message; } })();
