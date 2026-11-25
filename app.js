/**
 * LLM Pipeline Explorer
 * Comprehensive implementation of:
 * 1. Vocabulary Generation with Unknown Token (<UNK> = 0)
 * 2. Sentence Tokenization with Character Decomposition for UNK (# separated, ID 0)
 * 3. 3D Vector Embedding Space (Spelling-based vs. Meaning-based) with PCA & 3D Interactive Plot
 */

// -------------------------------------------------------------
// State Management
// -------------------------------------------------------------
const state = {
  corpusLines: [
    "Language models process natural language by breaking text into tokens.",
    "Tokens represent words, subwords, or individual characters in vocabulary.",
    "Every unique token receives a numeric token identifier in the vocabulary.",
    "Neural networks transform these token sequences into dense continuous vectors.",
    "Dense vectors capture semantic relationships and contextual meaning between words.",
    "Unknown words fall back to special tokens or character representations.",
    "Exploring this pipeline helps demystify how language models reason."
  ],
  vocabulary: new Map(), // word -> { id, count }
  vocabList: [], // array of { id, word, count }
  sentences: [
    { text: "The king ruled his kingdom with great wisdom and honor.", cat: "Royal / Governance" },
    { text: "The queen governed her royal court with grace and power.", cat: "Royal / Governance" },
    { text: "A playful kitten chased the yarn ball across the floor.", cat: "Animals / Pets" },
    { text: "A cute puppy jumped over the ball in the grassy yard.", cat: "Animals / Pets" },
    { text: "Artificial intelligence algorithms train on neural networks.", cat: "Technology / AI" },
    { text: "Deep learning models compute gradients using backpropagation.", cat: "Technology / AI" },
    { text: "Fresh Italian pasta was tossed in garlic olive oil.", cat: "Food / Culinary" },
    { text: "Delicious warm pizza baked inside a traditional brick oven.", cat: "Food / Culinary" }
  ],
  embeddingMode: "meaning", // "meaning" | "spelling"
  isPlotlyAvailable: false,
  canvas3d: null
};

// -------------------------------------------------------------
// Semantic Concept Dictionary for Meaning-Based Embeddings
// 6 Core Semantic Dimensions:
// [0: Royalty/Leadership, 1: Animals/Pets, 2: AI/Tech/Math, 3: Food/Culinary, 4: Motion/Action, 5: Nature/Space]
// -------------------------------------------------------------
const SEMANTIC_VECTORS = {
  // Royalty / Leadership
  "king": [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],
  "queen": [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],
  "ruled": [0.95, 0.0, 0.0, 0.0, 0.3, 0.0],
  "governed": [0.95, 0.0, 0.0, 0.0, 0.2, 0.0],
  "kingdom": [1.0, 0.0, 0.0, 0.0, 0.0, 0.3],
  "royal": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "court": [0.85, 0.0, 0.0, 0.0, 0.0, 0.2],
  "wisdom": [0.75, 0.0, 0.25, 0.0, 0.0, 0.0],
  "honor": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
  "grace": [0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
  "power": [0.85, 0.0, 0.3, 0.0, 0.2, 0.0],

  // Animals / Pets
  "kitten": [0.0, 1.0, 0.0, 0.0, 0.5, 0.3],
  "puppy": [0.0, 1.0, 0.0, 0.0, 0.5, 0.3],
  "playful": [0.0, 0.85, 0.0, 0.0, 0.7, 0.2],
  "cute": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
  "chased": [0.0, 0.6, 0.0, 0.0, 1.0, 0.3],
  "jumped": [0.0, 0.6, 0.0, 0.0, 1.0, 0.3],
  "yarn": [0.0, 0.7, 0.0, 0.0, 0.2, 0.0],
  "ball": [0.0, 0.65, 0.0, 0.0, 0.6, 0.0],
  "floor": [0.0, 0.3, 0.0, 0.0, 0.0, 0.4],
  "grassy": [0.0, 0.3, 0.0, 0.0, 0.0, 0.95],
  "yard": [0.0, 0.4, 0.0, 0.0, 0.0, 0.9],

  // Technology / AI / Math
  "artificial": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  "intelligence": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  "algorithms": [0.0, 0.0, 1.0, 0.0, 0.3, 0.0],
  "train": [0.0, 0.0, 0.85, 0.0, 0.4, 0.0],
  "neural": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  "networks": [0.0, 0.0, 1.0, 0.0, 0.2, 0.0],
  "deep": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
  "learning": [0.0, 0.0, 0.95, 0.0, 0.1, 0.0],
  "models": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
  "compute": [0.0, 0.0, 0.95, 0.0, 0.3, 0.0],
  "gradients": [0.0, 0.0, 1.0, 0.0, 0.1, 0.0],
  "backpropagation": [0.0, 0.0, 1.0, 0.0, 0.2, 0.0],

  // Food / Culinary
  "pasta": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  "pizza": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  "fresh": [0.0, 0.0, 0.0, 0.75, 0.0, 0.3],
  "italian": [0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
  "tossed": [0.0, 0.0, 0.0, 0.6, 0.8, 0.0],
  "garlic": [0.0, 0.0, 0.0, 1.0, 0.0, 0.1],
  "olive": [0.0, 0.0, 0.0, 0.9, 0.0, 0.2],
  "oil": [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
  "delicious": [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
  "warm": [0.0, 0.0, 0.0, 0.6, 0.0, 0.0],
  "baked": [0.0, 0.0, 0.0, 0.95, 0.3, 0.0],
  "traditional": [0.3, 0.0, 0.0, 0.5, 0.0, 0.0],
  "brick": [0.0, 0.0, 0.0, 0.4, 0.0, 0.3],
  "oven": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
};

// -------------------------------------------------------------
// Spelling Features (Alphabet letters + Common English Bigrams)
// -------------------------------------------------------------
const ALPHABET = Array.from({ length: 26 }, (_, i) => String.fromCharCode(97 + i));
const COMMON_BIGRAMS = [
  "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
  "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar", "st", "to", "nt"
];

// -------------------------------------------------------------
// Math & Dimensionality Reduction (Pure JS PCA)
// -------------------------------------------------------------
function dotProduct(v1, v2) {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) sum += v1[i] * v2[i];
  return sum;
}

function vectorNorm(v) {
  return Math.sqrt(dotProduct(v, v));
}

function normalizeVector(v) {
  const norm = vectorNorm(v);
  if (norm === 0) return v.slice();
  return v.map(val => val / norm);
}

function computePCA3D(dataMatrix, iterations = 100) {
  const n = dataMatrix.length;
  if (n === 0) return [];
  const d = dataMatrix[0].length;

  // 1. Mean-center the matrix
  const means = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      means[j] += dataMatrix[i][j];
    }
  }
  for (let j = 0; j < d; j++) means[j] /= n;

  const centered = dataMatrix.map(row => row.map((val, j) => val - means[j]));

  // 2. Power iteration with deflation to find top 3 principal components
  const components = [];
  const xCurr = centered.map(row => row.slice());

  for (let c = 0; c < Math.min(3, d); c++) {
    // Initial deterministic non-zero seed
    let v = Array.from({ length: d }, (_, j) => Math.sin((j + 1) * (c + 1)));
    v = normalizeVector(v);

    for (let iter = 0; iter < iterations; iter++) {
      // Multiply: xv = X * v (length n)
      const xv = new Array(n);
      for (let i = 0; i < n; i++) {
        xv[i] = dotProduct(xCurr[i], v);
      }
      // Multiply: xt_xv = X^T * xv (length d)
      const xt_xv = new Array(d).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          xt_xv[j] += xCurr[i][j] * xv[i];
        }
      }
      v = normalizeVector(xt_xv);
    }
    components.push(v);

    // Deflate: xCurr = xCurr - (X * v) * v^T
    const xv = new Array(n);
    for (let i = 0; i < n; i++) xv[i] = dotProduct(xCurr[i], v);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        xCurr[i][j] -= xv[i] * v[j];
      }
    }
  }

  // 3. Project centered data onto top 3 components
  const projected = centered.map(row => {
    const coords = components.map(comp => dotProduct(row, comp));
    while (coords.length < 3) coords.push(0);
    return coords;
  });

  // 4. Normalize coordinates to [-1, 1] range for clean visualization
  let maxAbs = 0;
  for (let i = 0; i < projected.length; i++) {
    for (let j = 0; j < 3; j++) {
      if (Math.abs(projected[i][j]) > maxAbs) {
        maxAbs = Math.abs(projected[i][j]);
      }
    }
  }
  if (maxAbs > 0) {
    for (let i = 0; i < projected.length; i++) {
      for (let j = 0; j < 3; j++) {
        projected[i][j] = Number((projected[i][j] / maxAbs).toFixed(3));
      }
    }
  }

  return projected;
}

// -------------------------------------------------------------
// Section 1: Vocabulary Generation
// -------------------------------------------------------------
function extractWords(text) {
  const matches = text.match(/\b[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\b/g);
  return matches || [];
}

function generateVocabulary() {
  const corpusText = document.getElementById("corpusInput").value;
  const words = extractWords(corpusText);

  const freqMap = new Map();
  for (const w of words) {
    const lower = w.toLowerCase();
    freqMap.set(lower, (freqMap.get(lower) || 0) + 1);
  }

  state.vocabulary.clear();
  state.vocabList = [];

  // Token ID 0 is strictly reserved for Unknown Token (<UNK>)
  state.vocabulary.set("<unk>", { id: 0, word: "<UNK>", count: 0, isUnk: true });
  state.vocabList.push({ id: 0, word: "<UNK>", count: 0, isUnk: true });

  // Unique tokens sorted alphabetically (starting from Token ID 1)
  const sortedUniqueWords = Array.from(freqMap.keys()).sort();
  let nextId = 1;

  for (const word of sortedUniqueWords) {
    const count = freqMap.get(word);
    state.vocabulary.set(word, { id: nextId, word: word, count: count, isUnk: false });
    state.vocabList.push({ id: nextId, word: word, count: count, isUnk: false });
    nextId++;
  }

  // Update Stats Banner
  document.getElementById("statTotalTokens").textContent = words.length;
  document.getElementById("statUniqueVocab").textContent = state.vocabList.length;
  document.getElementById("statVocabMaxId").textContent = nextId - 1;

  renderVocabTable();

  // Also auto-retokenize sentence in Section 2 if input is present
  tokenizeSentence();
}

function renderVocabTable() {
  const filterQuery = (document.getElementById("vocabSearch")?.value || "").trim().toLowerCase();
  const tableBody = document.getElementById("vocabTableBody");
  if (!tableBody) return;

  tableBody.innerHTML = "";

  const filtered = state.vocabList.filter(item => {
    if (!filterQuery) return true;
    return item.word.toLowerCase().includes(filterQuery) || String(item.id).includes(filterQuery);
  });

  document.getElementById("vocabCountDisplay").textContent = `Showing ${filtered.length} of ${state.vocabList.length} tokens`;

  filtered.forEach(item => {
    const tr = document.createElement("tr");
    if (item.id === 0) {
      tr.className = "unk-row";
      tr.innerHTML = `
        <td><span class="token-id-badge red">ID 0</span></td>
        <td><strong>&lt;UNK&gt;</strong> (Reserved for Unknown Words)</td>
        <td><span class="primary-chip red">Special</span></td>
      `;
    } else {
      tr.innerHTML = `
        <td><span class="token-id-badge blue">${item.id}</span></td>
        <td><code>${escapeHtml(item.word)}</code></td>
        <td>${item.count}</td>
      `;
    }
    tableBody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// Section 2: Sentence Tokenization & Unknown Decomposition
// -------------------------------------------------------------
function tokenizeSentence() {
  const sentenceInput = document.getElementById("sentenceInput");
  if (!sentenceInput) return;
  const sentence = sentenceInput.value.trim();

  const tokenChipsContainer = document.getElementById("tokenChipsContainer");
  const tokenSequenceBox = document.getElementById("tokenSequenceBox");
  const sentenceTableBody = document.getElementById("sentenceTokenTableBody");
  const statTokensUsed = document.getElementById("statTokensUsed");
  const statKnownTokens = document.getElementById("statKnownTokens");
  const statUnknownTokens = document.getElementById("statUnknownTokens");

  if (!sentence) {
    tokenChipsContainer.innerHTML = `<span style="color: #94a3b8; font-size: 0.9rem;">Type a sentence above to view token breakdown.</span>`;
    tokenSequenceBox.textContent = `[]`;
    sentenceTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#94a3b8; padding: 20px;">No sentence entered.</td></tr>`;
    statTokensUsed.textContent = "0";
    statKnownTokens.textContent = "0";
    statUnknownTokens.textContent = "0";
    return;
  }

  // If vocabulary hasn't been generated yet, generate it first
  if (state.vocabulary.size === 0) {
    generateVocabulary();
  }

  const rawWords = extractWords(sentence);
  let totalTokenUnits = 0;
  let knownCount = 0;
  let unknownCount = 0;

  const sequenceIds = [];
  tokenChipsContainer.innerHTML = "";
  sentenceTableBody.innerHTML = "";

  rawWords.forEach((word, index) => {
    const lower = word.toLowerCase();
    const vocabEntry = state.vocabulary.get(lower);

    if (vocabEntry && vocabEntry.id !== 0) {
      // Known token
      totalTokenUnits += 1;
      knownCount += 1;
      sequenceIds.push({ id: vocabEntry.id, isUnk: false });

      // Visual Chip
      const chip = document.createElement("div");
      chip.className = "token-chip-known";
      chip.innerHTML = `<span>${escapeHtml(word)}</span><span class="badge-id">ID: ${vocabEntry.id}</span>`;
      tokenChipsContainer.appendChild(chip);

      // Table Row
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>#${index + 1}</td>
        <td><strong>${escapeHtml(word)}</strong></td>
        <td><span class="primary-chip blue">Known Word</span></td>
        <td><span class="token-id-badge blue">ID ${vocabEntry.id}</span></td>
        <td><code>${escapeHtml(word)}</code></td>
      `;
      sentenceTableBody.appendChild(row);
    } else {
      // Unknown token (Token ID 0)
      // Requirement: "if 0 is token then the token should be selected in each character with # between them and should assign every character 0 token id."
      unknownCount += 1;
      const chars = word.split("");
      totalTokenUnits += chars.length;

      // Group chip for unknown word with # separated characters
      const unkGroup = document.createElement("div");
      unkGroup.className = "token-chip-unknown-group";

      chars.forEach((ch, chIdx) => {
        sequenceIds.push({ id: 0, isUnk: true });

        const charBadge = document.createElement("span");
        charBadge.className = "char-badge";
        charBadge.innerHTML = `${escapeHtml(ch)} <span class="badge-zero">ID 0</span>`;
        unkGroup.appendChild(charBadge);

        if (chIdx < chars.length - 1) {
          const hashSpan = document.createElement("span");
          hashSpan.className = "hash-separator";
          hashSpan.textContent = "#";
          unkGroup.appendChild(hashSpan);
        }
      });

      tokenChipsContainer.appendChild(unkGroup);

      // Table Row for Unknown Word
      const row = document.createElement("tr");
      row.className = "unk-row";
      const charBreakdownDisplay = chars.map(ch => `${ch} (ID: 0)`).join(" <strong style='color:#ca8a04;'>#</strong> ");
      row.innerHTML = `
        <td>#${index + 1}</td>
        <td><strong>${escapeHtml(word)}</strong></td>
        <td><span class="primary-chip red">Unknown Token (0)</span></td>
        <td><span class="token-id-badge red">All chars ID 0</span></td>
        <td>${charBreakdownDisplay}</td>
      `;
      sentenceTableBody.appendChild(row);
    }
  });

  // Update stats
  statTokensUsed.textContent = totalTokenUnits;
  statKnownTokens.textContent = knownCount;
  statUnknownTokens.textContent = unknownCount;

  // Format Token ID Sequence box
  tokenSequenceBox.innerHTML = `[ ` + sequenceIds.map(item => {
    return item.isUnk
      ? `<span class="id-zero">0</span>`
      : `<span class="id-known">${item.id}</span>`;
  }).join(", ") + ` ]`;
}

// -------------------------------------------------------------
// Section 3: Vector Embeddings (Spelling-based & Meaning-based)
// -------------------------------------------------------------
function getSentenceSpellingVector(text) {
  const clean = text.toLowerCase();
  const total = Math.max(clean.length, 1);
  const vec = [];

  // 1. Single character relative frequencies (a-z)
  for (const char of ALPHABET) {
    let count = 0;
    for (let i = 0; i < clean.length; i++) {
      if (clean[i] === char) count++;
    }
    vec.push(count / total);
  }

  // 2. Common bigram frequencies
  for (const bg of COMMON_BIGRAMS) {
    let count = 0;
    let pos = 0;
    while ((pos = clean.indexOf(bg, pos)) !== -1) {
      count++;
      pos += 1;
    }
    vec.push((count * 2) / total);
  }

  // 3. Length feature
  vec.push(Math.min(clean.length / 80, 1.0));

  return vec;
}

function getSentenceMeaningVector(text) {
  const words = extractWords(text);
  const vectors = [];

  for (const w of words) {
    const lower = w.toLowerCase();
    if (SEMANTIC_VECTORS[lower]) {
      vectors.push(SEMANTIC_VECTORS[lower]);
    }
  }

  // If no content words matched directly, use a default fallback
  if (vectors.length === 0) {
    return [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
  }

  // Mean semantic vector
  const d = vectors[0].length;
  const meanVec = new Array(d).fill(0);
  for (const vec of vectors) {
    for (let j = 0; j < d; j++) {
      meanVec[j] += vec[j];
    }
  }
  for (let j = 0; j < d; j++) {
    meanVec[j] /= vectors.length;
  }
  return meanVec;
}

function embedAndPlot() {
  const sentencesList = readSentencesFromInputs();
  state.sentences = sentencesList;

  let highDimMatrix;
  if (state.embeddingMode === "spelling") {
    highDimMatrix = sentencesList.map(s => getSentenceSpellingVector(s.text));
  } else {
    highDimMatrix = sentencesList.map(s => getSentenceMeaningVector(s.text));
  }

  // Compute 3D coordinates using PCA
  const coords3D = computePCA3D(highDimMatrix);

  // Update UI mode tag
  const modeTag = document.getElementById("chartModeTag");
  if (modeTag) {
    if (state.embeddingMode === "spelling") {
      modeTag.className = "chart-mode-pill spelling";
      modeTag.textContent = "Spelling-Based Projection";
    } else {
      modeTag.className = "chart-mode-pill meaning";
      modeTag.textContent = "Meaning-Based Semantic Projection";
    }
  }

  // Try Plotly first, or fall back to interactive Canvas 3D
  if (typeof Plotly !== "undefined") {
    plotWithPlotly(sentencesList, coords3D);
  } else {
    plotWithCanvas(sentencesList, coords3D);
  }

  updateEmbeddingsSummary(sentencesList, coords3D);
}

function readSentencesFromInputs() {
  const rows = document.querySelectorAll(".sentence-input-item");
  const list = [];
  rows.forEach((row, i) => {
    const input = row.querySelector("input[type='text']");
    const cat = row.getAttribute("data-category") || "General";
    list.push({
      text: input ? input.value : state.sentences[i]?.text || `Sentence ${i + 1}`,
      cat: cat
    });
  });
  return list.length > 0 ? list : state.sentences;
}

// -------------------------------------------------------------
// Plotly 3D Scatter Rendering
// -------------------------------------------------------------
function plotWithPlotly(sentences, coords) {
  const chartDiv = document.getElementById("chart3d");
  if (!chartDiv) return;

  if (!chartDiv.classList.contains("js-plotly-plot")) {
    chartDiv.innerHTML = "";
    state.canvas3d = null;
  }

  // Primary Color mapping by category
  const categoryColors = {
    "Royal / Governance": "#1d4ed8", // Primary Blue
    "Animals / Pets": "#ca8a04",    // Primary Yellow
    "Technology / AI": "#2563eb",   // Deep Blue
    "Food / Culinary": "#dc2626"    // Primary Red
  };

  const x = coords.map(c => c[0]);
  const y = coords.map(c => c[1]);
  const z = coords.map(c => c[2]);
  const text = sentences.map((s, i) => `<b>[S${i + 1}] ${s.cat}</b><br>${s.text}<br>Coord: (${coords[i][0]}, ${coords[i][1]}, ${coords[i][2]})`);
  const colors = sentences.map(s => categoryColors[s.cat] || "#1d4ed8");

  const trace = {
    x: x,
    y: y,
    z: z,
    mode: "markers+text",
    type: "scatter3d",
    text: sentences.map((_, i) => `S${i + 1}`),
    textposition: "top center",
    textfont: {
      family: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
      size: 13,
      color: "#0f172a"
    },
    hoverinfo: "text",
    hovertext: text,
    marker: {
      size: 9,
      color: colors,
      opacity: 0.95,
      line: {
        color: "#ffffff",
        width: 2
      }
    }
  };

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    scene: {
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 }
      },
      xaxis: {
        title: "Dim 1",
        gridcolor: "#e2e8f0",
        backgroundcolor: "#ffffff",
        showbackground: true,
        zerolinecolor: "#cbd5e1"
      },
      yaxis: {
        title: "Dim 2",
        gridcolor: "#e2e8f0",
        backgroundcolor: "#ffffff",
        showbackground: true,
        zerolinecolor: "#cbd5e1"
      },
      zaxis: {
        title: "Dim 3",
        gridcolor: "#e2e8f0",
        backgroundcolor: "#ffffff",
        showbackground: true,
        zerolinecolor: "#cbd5e1"
      }
    },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["toImage", "sendDataToCloud"]
  };

  Plotly.react(chartDiv, [trace], layout, config);
}

// -------------------------------------------------------------
// Pure HTML5 Canvas 3D Fallback Engine (Zero-dependency Orbiting 3D)
// -------------------------------------------------------------
class Canvas3DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.sentences = [];
    this.coords = [];
    this.rotX = 0.45;
    this.rotY = -0.65;
    this.scale = 140;
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.hoverIndex = -1;

    this.initEvents();
  }

  setData(sentences, coords) {
    this.sentences = sentences;
    this.coords = coords;
    this.render();
  }

  initEvents() {
    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = (this.canvas.getBoundingClientRect && this.canvas.getBoundingClientRect()) || { left: 0, top: 0 };
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      if (this.isDragging) {
        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;
        this.rotY += dx * 0.008;
        this.rotX += dy * 0.008;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.render();
      } else {
        this.checkHover(mouseX, mouseY);
      }
    });

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.scale = Math.max(60, Math.min(300, this.scale - e.deltaY * 0.15));
      this.render();
    }, { passive: false });
  }

  project(x, y, z, cx, cy) {
    // Rotate around Y
    const cosY = Math.cos(this.rotY);
    const sinY = Math.sin(this.rotY);
    const x1 = x * cosY + z * sinY;
    const z1 = -x * sinY + z * cosY;

    // Rotate around X
    const cosX = Math.cos(this.rotX);
    const sinX = Math.sin(this.rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    // Perspective projection
    const distance = 4;
    const fov = distance / (distance + z2 * 0.8);
    const screenX = cx + x1 * this.scale * fov;
    const screenY = cy - y2 * this.scale * fov;

    return { x: screenX, y: screenY, z: z2, fov: fov };
  }

  checkHover(mx, my) {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    let foundIdx = -1;

    for (let i = 0; i < this.coords.length; i++) {
      const p = this.project(this.coords[i][0], this.coords[i][1], this.coords[i][2], cx, cy);
      const dist = Math.hypot(mx - p.x, my - p.y);
      if (dist < 14) {
        foundIdx = i;
        break;
      }
    }

    if (foundIdx !== this.hoverIndex) {
      this.hoverIndex = foundIdx;
      this.render();
      this.updateTooltip(foundIdx, mx, my);
    }
  }

  updateTooltip(idx, mx, my) {
    let tooltip = document.getElementById("canvasTooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "canvasTooltip";
      tooltip.className = "canvas-overlay-tooltip";
      this.canvas.parentElement.appendChild(tooltip);
    }

    if (idx === -1) {
      tooltip.style.display = "none";
      return;
    }

    const s = this.sentences[idx];
    const c = this.coords[idx];
    tooltip.style.display = "block";
    tooltip.style.left = `${mx + 15}px`;
    tooltip.style.top = `${my + 15}px`;
    tooltip.innerHTML = `
      <div style="font-weight:700; color:#93c5fd; margin-bottom:2px;">[S${idx + 1}] ${escapeHtml(s.cat)}</div>
      <div style="font-size:0.85rem; line-height:1.4;">${escapeHtml(s.text)}</div>
      <div style="font-size:0.75rem; color:#cbd5e1; margin-top:4px;">Coord: (${c[0]}, ${c[1]}, ${c[2]})</div>
    `;
  }

  render() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.ctx;
    const cx = width / 2;
    const cy = height / 2;

    ctx.clearRect(0, 0, width, height);

    // 1. Draw 3D Axes
    const axes = [
      { name: "Dim 1", end: [1.3, 0, 0], color: "#94a3b8" },
      { name: "Dim 2", end: [0, 1.3, 0], color: "#94a3b8" },
      { name: "Dim 3", end: [0, 0, 1.3], color: "#94a3b8" }
    ];
    const center = this.project(0, 0, 0, cx, cy);

    axes.forEach(axis => {
      const p = this.project(axis.end[0], axis.end[1], axis.end[2], cx, cy);
      ctx.beginPath();
      ctx.moveTo(center.x, center.y);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = axis.color;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = "#64748b";
      ctx.font = "11px sans-serif";
      ctx.fillText(axis.name, p.x + 4, p.y + 4);
    });

    // 2. Project all points and sort by depth (z)
    const projectedPoints = this.coords.map((coord, i) => {
      const p = this.project(coord[0], coord[1], coord[2], cx, cy);
      return { ...p, index: i, sentence: this.sentences[i] };
    });
    projectedPoints.sort((a, b) => b.z - a.z);

    const categoryColors = {
      "Royal / Governance": "#1d4ed8",
      "Animals / Pets": "#ca8a04",
      "Technology / AI": "#2563eb",
      "Food / Culinary": "#dc2626"
    };

    // 3. Draw Points
    projectedPoints.forEach(pt => {
      const isHovered = pt.index === this.hoverIndex;
      const radius = (isHovered ? 12 : 8) * pt.fov;
      const color = categoryColors[pt.sentence.cat] || "#1d4ed8";

      // Dropline to bottom plane
      const planePt = this.project(this.coords[pt.index][0], -1.0, this.coords[pt.index][2], cx, cy);
      ctx.beginPath();
      ctx.moveTo(pt.x, pt.y);
      ctx.lineTo(planePt.x, planePt.y);
      ctx.strokeStyle = "rgba(226, 232, 240, 0.7)";
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Point circle
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = "#0f172a";
      ctx.font = isHovered ? "bold 13px sans-serif" : "600 11px sans-serif";
      ctx.fillText(`S${pt.index + 1}`, pt.x + radius + 4, pt.y + 4);
    });
  }
}

function plotWithCanvas(sentences, coords) {
  const chartDiv = document.getElementById("chart3d");
  if (!chartDiv) return;

  let canvas = document.getElementById("fallbackCanvas3d");
  if (!canvas) {
    chartDiv.innerHTML = "";
    canvas = document.createElement("canvas");
    canvas.id = "fallbackCanvas3d";
    chartDiv.appendChild(canvas);
  }

  // Adjust canvas pixel resolution to container
  const rect = (chartDiv.getBoundingClientRect && chartDiv.getBoundingClientRect()) || { width: 800, height: 520 };
  canvas.width = rect.width || 800;
  canvas.height = rect.height || 520;

  if (!state.canvas3d) {
    state.canvas3d = new Canvas3DRenderer(canvas);
  } else {
    state.canvas3d.canvas = canvas;
    state.canvas3d.ctx = canvas.getContext("2d");
  }

  state.canvas3d.setData(sentences, coords);
}

function updateEmbeddingsSummary(sentences, coords) {
  const summaryBox = document.getElementById("embeddingsInsight");
  if (!summaryBox) return;

  if (state.embeddingMode === "meaning") {
    summaryBox.innerHTML = `
      <strong>Semantic Vector Insight:</strong> Notice how <em>S1 & S2 (Royalty)</em> cluster tightly together, 
      <em>S3 & S4 (Animals/Pets)</em> group closely, <em>S5 & S6 (Technology/AI)</em> form a distinct cluster, and 
      <em>S7 & S8 (Food/Italian Cuisine)</em> group together. Even though the word spellings differ substantially, 
      the language model vectors preserve semantic proximity!
    `;
  } else {
    summaryBox.innerHTML = `
      <strong>Spelling-Based Vector Insight:</strong> Clustering is now strictly driven by surface orthography 
      (character frequencies, n-grams, word length). Sentences with similar letter distributions move closer together, 
      while semantic meaning is completely disregarded.
    `;
  }
}

// -------------------------------------------------------------
// Event Listeners & UI Binding
// -------------------------------------------------------------
function setupEventListeners() {
  // Vocabulary Generate Button
  const btnGenVocab = document.getElementById("btnGenerateVocab");
  if (btnGenVocab) {
    btnGenVocab.addEventListener("click", () => {
      generateVocabulary();
    });
  }

  // Vocabulary Reset Button
  const btnResetCorpus = document.getElementById("btnResetCorpus");
  if (btnResetCorpus) {
    btnResetCorpus.addEventListener("click", () => {
      document.getElementById("corpusInput").value = state.corpusLines.join("\n");
      generateVocabulary();
    });
  }

  // Vocab Search Filter
  const vocabSearch = document.getElementById("vocabSearch");
  if (vocabSearch) {
    vocabSearch.addEventListener("input", () => {
      renderVocabTable();
    });
  }

  // Sentence Input Live Tokenizer
  const sentenceInput = document.getElementById("sentenceInput");
  if (sentenceInput) {
    sentenceInput.addEventListener("input", () => {
      tokenizeSentence();
    });
  }

  // Sample Sentence Pills
  const samplePills = document.querySelectorAll(".sample-sentence-pill");
  samplePills.forEach(pill => {
    pill.addEventListener("click", () => {
      const text = pill.getAttribute("data-text");
      const input = document.getElementById("sentenceInput");
      if (input) {
        input.value = text;
        tokenizeSentence();
      }
    });
  });

  // Embedding Options Radio Buttons
  const radioSpelling = document.getElementById("optSpelling");
  const radioMeaning = document.getElementById("optMeaning");

  if (radioSpelling && radioMeaning) {
    const updateRadioStyles = () => {
      const labelSpelling = radioSpelling.closest(".radio-label");
      const labelMeaning = radioMeaning.closest(".radio-label");
      if (radioSpelling.checked) {
        state.embeddingMode = "spelling";
        labelSpelling.classList.add("active");
        labelMeaning.classList.remove("active");
      } else {
        state.embeddingMode = "meaning";
        labelMeaning.classList.add("active");
        labelSpelling.classList.remove("active");
      }
    };

    radioSpelling.addEventListener("change", () => {
      updateRadioStyles();
      embedAndPlot();
    });
    radioMeaning.addEventListener("change", () => {
      updateRadioStyles();
      embedAndPlot();
    });
  }

  // Embed and Plot Button
  const btnEmbedPlot = document.getElementById("btnEmbedPlot");
  if (btnEmbedPlot) {
    btnEmbedPlot.addEventListener("click", () => {
      embedAndPlot();
    });
  }

  // Window Resize
  window.addEventListener("resize", () => {
    if (typeof Plotly !== "undefined" && document.getElementById("chart3d")) {
      Plotly.Plots.resize(document.getElementById("chart3d"));
    } else if (state.canvas3d) {
      const chartDiv = document.getElementById("chart3d");
      if (chartDiv) {
        const rect = (chartDiv.getBoundingClientRect && chartDiv.getBoundingClientRect()) || { width: 800, height: 520 };
        state.canvas3d.canvas.width = rect.width || 800;
        state.canvas3d.canvas.height = rect.height || 520;
        state.canvas3d.render();
      }
    }
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  // Populate default 6-7 lines
  const corpusInput = document.getElementById("corpusInput");
  if (corpusInput) {
    corpusInput.value = state.corpusLines.join("\n");
  }

  // Populate sentences in section 3
  renderSentenceInputs();

  // Setup listeners
  setupEventListeners();

  // Initial vocabulary generation & sentence tokenization
  generateVocabulary();

  // Initial Embedding Plot
  embedAndPlot();
});

function renderSentenceInputs() {
  const container = document.getElementById("sentencesEditorList");
  if (!container) return;

  container.innerHTML = "";
  const catClasses = {
    "Royal / Governance": "cat-royal",
    "Animals / Pets": "cat-pets",
    "Technology / AI": "cat-tech",
    "Food / Culinary": "cat-food"
  };

  state.sentences.forEach((s, idx) => {
    const row = document.createElement("div");
    row.className = "sentence-row sentence-input-item";
    row.setAttribute("data-category", s.cat);
    const catClass = catClasses[s.cat] || "cat-tech";

    row.innerHTML = `
      <div class="sentence-index">${idx + 1}</div>
      <span class="sentence-category-tag ${catClass}">${escapeHtml(s.cat)}</span>
      <input type="text" class="input-clean" value="${escapeHtml(s.text)}" />
    `;
    container.appendChild(row);
  });
}
