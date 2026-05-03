const dom = {
  status: document.getElementById('status'),
  apiBase: document.getElementById('apiBase'),
  saveApiBase: document.getElementById('saveApiBase'),
  refreshDatasets: document.getElementById('refreshDatasets'),
  datasetUpload: document.getElementById('datasetUpload'),
  datasetPath: document.getElementById('datasetPath'),
  registerPath: document.getElementById('registerPath'),
  datasetSelect: document.getElementById('datasetSelect'),
  previewDataset: document.getElementById('previewDataset'),
  datasetPreview: document.getElementById('datasetPreview'),
  targetMode: document.querySelectorAll('input[name="targetMode"]'),
  targetCol: document.getElementById('targetCol'),
  minNumericRatio: document.getElementById('minNumericRatio'),
  excludeCols: document.getElementById('excludeCols'),
  forceInclude: document.getElementById('forceInclude'),
  prepareFeatures: document.getElementById('prepareFeatures'),
  featurePreview: document.getElementById('featurePreview'),
  trainModels: document.getElementById('trainModels'),
  trainingResults: document.getElementById('trainingResults'),
  modelLabel: document.getElementById('modelLabel'),
  gbEstimators: document.getElementById('gbEstimators'),
  gbLearningRate: document.getElementById('gbLearningRate'),
  gbMaxDepth: document.getElementById('gbMaxDepth'),
  singleInput: document.getElementById('singleInput'),
  runSingle: document.getElementById('runSingle'),
  runBatch: document.getElementById('runBatch'),
  batchUpload: document.getElementById('batchUpload'),
  runBatchUpload: document.getElementById('runBatchUpload'),
  inferenceResults: document.getElementById('inferenceResults'),
}

const state = {
  apiBase: '',
  datasets: [],
  selectedDataset: '',
  preview: null,
  featurePrep: null,
  runResult: null,
  activeModelId: '',
}

const defaultApiBase = document.querySelector('meta[name="api-base"]')?.content || 'http://127.0.0.1:8000'

function setStatus(message, tone = 'idle') {
  dom.status.textContent = message
  dom.status.dataset.tone = tone
}

function setBusy(isBusy) {
  document.body.dataset.busy = isBusy ? 'true' : 'false'
}

function parseList(value) {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

const targetDefaults = {
  phaseA: {
    targetCol: 'pm_direction_up',
    excludeCols: 'trade_date,open_price,close_price,pm_return,pm_direction_up',
  },
  phaseB: {
    targetCol: 'target_close_higher_prev',
    excludeCols: 'trade_date,target_close_higher_prev,end_price,pm_direction_up,pm_direction_up_synth,up_price_final,down_price_final',
  },
}

function applyTargetDefaults(mode) {
  const defaults = targetDefaults[mode]
  if (!defaults) {
    return
  }
  dom.targetCol.value = defaults.targetCol
  dom.excludeCols.value = defaults.excludeCols
}

function getTargetMode() {
  const selected = Array.from(dom.targetMode).find((input) => input.checked)
  return selected ? selected.value : 'phaseA'
}

function resolveTargetCol() {
  const mode = getTargetMode()
  const defaults = targetDefaults[mode]
  if (defaults?.targetCol) {
    dom.targetCol.value = defaults.targetCol
    return defaults.targetCol
  }
  return dom.targetCol.value.trim()
}

function resolveExcludeCols() {
  const mode = getTargetMode()
  const defaults = targetDefaults[mode]
  if (defaults?.excludeCols) {
    dom.excludeCols.value = defaults.excludeCols
    return defaults.excludeCols
  }
  return dom.excludeCols.value
}

async function request(path, options = {}) {
  const response = await fetch(`${state.apiBase}${path}`, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.detail || 'Request failed')
  }
  return payload
}

function hydrateApiBase() {
  const saved = localStorage.getItem('apiBase')
  state.apiBase = saved || defaultApiBase
  dom.apiBase.value = state.apiBase
}

function saveApiBase() {
  const next = dom.apiBase.value.trim()
  if (!next) {
    setStatus('API base cannot be empty', 'error')
    return
  }
  state.apiBase = next
  localStorage.setItem('apiBase', next)
  setStatus('API base saved', 'success')
}

function renderDatasetOptions() {
  dom.datasetSelect.innerHTML = '<option value="">Select dataset...</option>'
  state.datasets.forEach((dataset) => {
    const option = document.createElement('option')
    option.value = dataset.dataset_id
    option.textContent = `${dataset.name} (${dataset.source})`
    dom.datasetSelect.appendChild(option)
  })
  if (state.selectedDataset) {
    dom.datasetSelect.value = state.selectedDataset
  }
}

function renderPreview(preview) {
  if (!preview) {
    dom.datasetPreview.innerHTML = '<p class="muted">No preview loaded yet.</p>'
    return
  }

  const columns = preview.columns || []
  const rows = preview.rows || []
  const missing = preview.missing_ratio || {}

  let html = `<div class="preview-meta">Rows: ${preview.shape?.[0] || 0} | Columns: ${preview.shape?.[1] || 0}</div>`
  html += '<div class="table-wrap"><table><thead><tr>'
  columns.slice(0, 8).forEach((col) => {
    html += `<th>${col}</th>`
  })
  html += '</tr></thead><tbody>'
  rows.slice(0, 5).forEach((row) => {
    html += '<tr>'
    columns.slice(0, 8).forEach((col) => {
      html += `<td>${row[col] ?? ''}</td>`
    })
    html += '</tr>'
  })
  html += '</tbody></table></div>'

  const missingKeys = Object.keys(missing)
  if (missingKeys.length) {
    const topMissing = missingKeys.slice(0, 6).map((key) => `${key}: ${missing[key]}`)
    html += `<p class="muted">Missing ratio (top): ${topMissing.join(', ')}</p>`
  }

  dom.datasetPreview.innerHTML = html
}

function renderFeaturePrep(prep) {
  if (!prep) {
    dom.featurePreview.innerHTML = ''
    return
  }
  const classDist = prep.class_distribution || {}
  const distParts = Object.keys(classDist).map((key) => `${key}: ${classDist[key]}`)
  dom.featurePreview.innerHTML = `
    <div class="preview-meta">Rows used: ${prep.rows} | Feature count: ${prep.feature_count}</div>
    <p class="muted">Class distribution: ${distParts.join(', ') || 'n/a'}</p>
  `
}

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(4) : 'n/a'
}

function renderTrainingResults(run) {
  if (!run) {
    dom.trainingResults.innerHTML = ''
    return
  }

  let html = `<div class="preview-meta">Run ID: ${run.run_id} | Features: ${run.feature_count}</div>`
  html += '<div class="table-wrap"><table><thead><tr>'
  html += '<th>Model</th><th>Accuracy</th><th>Balanced Accuracy</th><th>F1 Weighted</th>'
  html += '</tr></thead><tbody>'
  run.models.forEach((model) => {
    html += '<tr>'
    html += `<td>${model.name}</td>`
    html += `<td>${formatMetric(model.accuracy_mean)}</td>`
    html += `<td>${formatMetric(model.balanced_accuracy_mean)}</td>`
    html += `<td>${formatMetric(model.f1_weighted_mean)}</td>`
    html += '</tr>'
  })
  html += '</tbody></table></div>'
  dom.trainingResults.innerHTML = html
}

function updateModelLabel(run) {
  if (!run?.models?.length) {
    dom.modelLabel.textContent = 'Train the best model to enable inference.'
    dom.modelLabel.classList.remove('active')
    state.activeModelId = ''
    return
  }

  const best = run.models[0]
  state.activeModelId = best.model_id
  dom.modelLabel.textContent = `${best.name} (${best.model_id})`
  dom.modelLabel.classList.add('active')
}

function renderInferenceResults(payload) {
  if (!payload) {
    dom.inferenceResults.innerHTML = ''
    return
  }
  const text = JSON.stringify(payload, null, 2)
  dom.inferenceResults.innerHTML = `<pre>${text}</pre>`
}

function collectBestModelConfig() {
  return {
    selectedModels: ['Gradient Boosting'],
    hyperparams: {
      'Gradient Boosting': {
        n_estimators: Number(dom.gbEstimators.value),
        learning_rate: Number(dom.gbLearningRate.value),
        max_depth: Number(dom.gbMaxDepth.value),
      },
    },
  }
}

function collectBaseConfig() {
  return {
    dataset_id: dom.datasetSelect.value,
    target_col: resolveTargetCol(),
    exclude_cols: parseList(resolveExcludeCols()),
    min_numeric_ratio: Number(dom.minNumericRatio.value),
    force_include_features: parseList(dom.forceInclude.value),
  }
}

async function loadDatasets() {
  setBusy(true)
  try {
    const res = await request('/api/datasets/list')
    state.datasets = res.datasets || []
    if (!state.selectedDataset && state.datasets.length) {
      state.selectedDataset = state.datasets[0].dataset_id
    }
    renderDatasetOptions()
    setStatus('Dataset list updated', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function uploadDataset(event) {
  const file = event.target.files?.[0]
  if (!file) {
    return
  }
  setBusy(true)
  try {
    setStatus('Uploading dataset...', 'info')
    const form = new FormData()
    form.append('file', file)
    await request('/api/datasets/upload', { method: 'POST', body: form })
    await loadDatasets()
    setStatus('Upload complete', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function registerDatasetPath() {
  const pathValue = dom.datasetPath.value.trim()
  if (!pathValue) {
    setStatus('Provide a dataset path', 'error')
    return
  }
  setBusy(true)
  try {
    await request('/api/datasets/register-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: pathValue }),
    })
    await loadDatasets()
    setStatus('Path registered', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function previewDataset() {
  const datasetId = dom.datasetSelect.value
  if (!datasetId) {
    setStatus('Select a dataset first', 'error')
    return
  }
  setBusy(true)
  try {
    const res = await request(`/api/datasets/${encodeURIComponent(datasetId)}/preview?limit=8`)
    state.preview = res
    renderPreview(res)
    setStatus('Preview loaded', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function prepareFeatures() {
  const config = collectBaseConfig()
  if (!config.dataset_id) {
    setStatus('Select a dataset first', 'error')
    return
  }
  if (!config.target_col) {
    setStatus('Target column is required', 'error')
    return
  }
  setBusy(true)
  try {
    const res = await request('/api/features/prepare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    state.featurePrep = res
    renderFeaturePrep(res)
    setStatus('Features prepared', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function trainModels() {
  const config = collectBaseConfig()
  if (!config.dataset_id) {
    setStatus('Select a dataset first', 'error')
    return
  }

  const modelConfig = collectBestModelConfig()

  const payload = {
    ...config,
    selected_models: modelConfig.selectedModels,
    hyperparams: modelConfig.hyperparams,
    cv: {
      type: 'RepeatedStratifiedKFold',
      n_splits: 5,
      n_repeats: 3,
      random_state: 42,
    },
  }

  setBusy(true)
  try {
    setStatus('Training models...', 'info')
    const res = await request('/api/models/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    state.runResult = res
    renderTrainingResults(res)
    updateModelLabel(res)
    setStatus('Training complete', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function runSinglePrediction() {
  const modelId = state.activeModelId
  if (!modelId) {
    setStatus('Train the best model first', 'error')
    return
  }
  let features = {}
  try {
    features = JSON.parse(dom.singleInput.value)
  } catch (error) {
    setStatus('Single query JSON is invalid', 'error')
    return
  }
  setBusy(true)
  try {
    const res = await request('/api/predictions/single', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, features }),
    })
    renderInferenceResults(res)
    setStatus('Single prediction complete', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function runBatchPrediction() {
  const modelId = state.activeModelId
  const datasetId = dom.datasetSelect.value
  if (!modelId || !datasetId) {
    setStatus('Train the best model and select a dataset first', 'error')
    return
  }
  setBusy(true)
  try {
    const res = await request('/api/predictions/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, dataset_id: datasetId }),
    })
    renderInferenceResults({ count: res.count, sample: res.rows?.slice(0, 8) })
    setStatus('Batch prediction complete', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

async function runBatchUpload() {
  const modelId = state.activeModelId
  const file = dom.batchUpload.files?.[0]
  if (!modelId || !file) {
    setStatus('Train the best model and choose a CSV file', 'error')
    return
  }
  setBusy(true)
  try {
    const form = new FormData()
    form.append('file', file)
    const res = await request(`/api/predictions/batch-upload?model_id=${encodeURIComponent(modelId)}`, {
      method: 'POST',
      body: form,
    })
    renderInferenceResults({ count: res.count, sample: res.rows?.slice(0, 8) })
    setStatus('Batch upload prediction complete', 'success')
  } catch (error) {
    setStatus(error.message, 'error')
  } finally {
    setBusy(false)
  }
}

function bindEvents() {
  dom.saveApiBase.addEventListener('click', saveApiBase)
  dom.refreshDatasets.addEventListener('click', loadDatasets)
  dom.datasetUpload.addEventListener('change', uploadDataset)
  dom.registerPath.addEventListener('click', registerDatasetPath)
  dom.previewDataset.addEventListener('click', previewDataset)
  dom.prepareFeatures.addEventListener('click', prepareFeatures)
  dom.trainModels.addEventListener('click', trainModels)
  dom.runSingle.addEventListener('click', runSinglePrediction)
  dom.runBatch.addEventListener('click', runBatchPrediction)
  dom.runBatchUpload.addEventListener('click', runBatchUpload)
  dom.targetMode.forEach((input) => {
    input.addEventListener('change', (event) => {
      applyTargetDefaults(event.target.value)
    })
  })
  dom.datasetSelect.addEventListener('change', () => {
    state.selectedDataset = dom.datasetSelect.value
  })
}

function init() {
  hydrateApiBase()
  applyTargetDefaults('phaseA')
  bindEvents()
  loadDatasets()
}

init()
