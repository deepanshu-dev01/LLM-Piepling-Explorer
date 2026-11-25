# LLM Pipeline Explorer

An interactive web application exploring the core internal mechanisms of Large Language Models (LLMs):
1. **Vocabulary Generation & Indexing** (`<UNK>` assigned Token ID 0)
2. **Sentence Tokenization & Unknown Token Decomposition** (Character-level fallback separated with `#`, assigned Token ID 0)
3. **3D Vector Embedding Space** (Comparing *Spelling-based* orthographic projections vs. *Meaning-based* semantic projections)

---

## 🎨 Design Philosophy
- **Light Theme**: Clean, minimalist white and soft slate surfaces (`#ffffff`, `#f8fafc`).
- **Primary Colors Only**:
  - 🔵 **Primary Blue (`#1d4ed8`)**: Main interface elements, buttons, known word tokens.
  - 🔴 **Primary Red (`#dc2626`)**: Unknown token (`<UNK>`) indicators and ID 0 character decompositions.
  - 🟡 **Primary Yellow (`#ca8a04`)**: `#` token separators and highlight accents.
- **Uncluttered & Focused**: Every section is cleanly spaced with clear metrics and interactive feedback.

---

## 🚀 How to Run

You can open `index.html` directly in any web browser, or launch a lightweight local server:

```bash
cd /Users/deepanshu/.gemini/antigravity/scratch/llm-pipeline-explorer
python3 -m http.server 8000
```
Then navigate to: **[http://localhost:8000](http://localhost:8000)**

---

## 🧠 Features

### 1. Vocabulary Test
- Pre-loaded with 7 sample lines explaining tokenization and neural representations.
- **Generate vocabulary** button parses all words, counts frequencies, and constructs an indexed vocabulary table:
  - `Token ID 0` is strictly reserved for `<UNK>` (Unknown Token).
  - Unique words are assigned `Token ID 1, 2, ..., N`.
- Search and filter tokens in real-time.

### 2. Sentence Tokenization & Encoding
- Enter any custom sentence or pick from pre-configured test scenarios.
- Displays:
  - **Total Tokens Used**
  - **Visual Token Stream**: Known words appear as blue badges with their Token ID; unknown words (ID 0) are decomposed into individual characters separated by `#` (e.g., `r [0] # o [0] # b [0] # o [0] # t [0] # i [0] # c [0] # s [0]`).
  - **Full Token ID Sequence Array** (e.g. `[13, 17, 22, 0, 0, 0, 0, 0, 0, 0, 0]`).
  - **Token Inspection Table** with order, status, and breakdown.

### 3. Vector Embedding (3D Projection)
- 8 pre-configured sentences spanning 4 semantic domains:
  1. *Royal / Governance* (King & Queen)
  2. *Animals / Pets* (Kitten & Puppy)
  3. *Technology / AI* (Neural networks & Backpropagation)
  4. *Food / Culinary* (Italian pasta & Brick oven pizza)
- Choose between:
  - **Meaning based**: Embeds sentences using semantic concept vectors, placing conceptual pairs (King & Queen, Pasta & Pizza) right next to each other.
  - **Spelling based**: Embeds sentences purely based on character distributions and letter n-grams, breaking semantic clusters.
- **Embed and plot** renders an interactive 3D scatter plot (Plotly.js with automatic Canvas 3D fallback) supporting rotation, zoom, tooltips, and category color-coding.
