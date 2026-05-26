# Tool-meme

Official implementation of **Tool-meme: Tool-Learning Framework for Multimodal Meme Detection**.

Tool-meme treats harmful meme detection as adaptive evidence acquisition: it retrieves socio-cultural context, plans meme-specific semantic checks, routes only the necessary LMM-callable tools under budget constraints, and fuses the evidence into a calibrated prediction.

## Setup

```bash
conda create -n tool_meme python=3.10 -y
conda activate tool_meme
pip install -r requirements.txt
```

Set the API key for LMM-based modules:

```bash
export OPENAI_API_KEY="your_api_key"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Optional environment variables: `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_MAX_TOKENS`, `OPENAI_TEMPERATURE`, `OPENAI_TOP_P`, `OPENAI_TIMEOUT`, `OPENAI_PROXY`, `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`.

## Data

```text
data/
  FHM/
    images/
    train.jsonl
    test.jsonl
  HarM/
    images/
    train.jsonl
    test.jsonl
  MAMI/
    images/
    train.jsonl
    test.jsonl
```

## Run

Build CAMR retrieval files:

```bash
python CAMR.py --datasets HarM FHM MAMI --k 10 --output_dir CAMR_output --batch_size 128
```

Run Tool-meme:

```bash
python Tool_meme.py --datasets HarM FHM MAMI
```

Outputs are written to:

```text
results/{dataset}_ToolMeme.jsonl
```

## Optional Preparation

Generate optional image embeddings and image-text alignment scores:

```bash
python prepare_inputs.py --datasets HarM FHM MAMI --split test --image_embeddings --alignment_scores
```

Generate event contexts and merge them into CAMR:

```bash
python prepare_inputs.py --datasets HarM FHM MAMI --event_contexts --event_split train --event_output_suffix with_event
python prepare_inputs.py --datasets HarM FHM MAMI --merge_events --camr_dir CAMR_output --out_dir CAMR_output
```

Train the RL-guided ATR router:

```bash
python -m rl_atr.train_tool_router \
  --logs results/HarM_ToolMeme.jsonl results/FHM_ToolMeme.jsonl results/MAMI_ToolMeme.jsonl \
  --output rl_atr/router_policy.pt
```

Run with the trained router:

```bash
python Tool_meme.py --datasets HarM FHM MAMI --atr_mode rl --router_path rl_atr/router_policy.pt
```

## Main Switches

```bash
# Major modules
python Tool_meme.py --no_mcp
python Tool_meme.py --no_atr
python Tool_meme.py --no_mpre
python Tool_meme.py --no_cbdf

# Individual cognitive tools
python Tool_meme.py --no_cross_modal_aligner
python Tool_meme.py --no_semantic_dissector
python Tool_meme.py --no_rhetorical_scanner
python Tool_meme.py --no_knowledge_grounding
python Tool_meme.py --no_expectation_deviator
python Tool_meme.py --no_inconsistency_amplifier
python Tool_meme.py --no_cultural_decoder
python Tool_meme.py --no_evidence_comparator

# ATR routing
python Tool_meme.py --atr_mode dag
python Tool_meme.py --atr_mode heuristic
python Tool_meme.py --atr_mode random
python Tool_meme.py --atr_mode all_tools
python Tool_meme.py --atr_mode rl --router_path rl_atr/router_policy.pt

# Paper defaults / common ablations
python Tool_meme.py --k 10 --tool_budget 6 --short_circuit_threshold 0.80 --mcp_max_depth 4 --decision_source cbdf
python Tool_meme.py --camr_dir CAMR_output --output_dir results
python Tool_meme.py --decision_source lmm
python Tool_meme.py --cbdf_use_lmm
python Tool_meme.py --no_check_images
```

## Repository Layout

```text
Tool_meme.py        Full CAMR -> MCP -> ATR -> MPRE -> CBDF inference
CAMR.py             CAMR retrieval construction
prepare_inputs.py   Optional embeddings, alignment scores, and event contexts
MCP.py              Meta-cognitive planning
ATR.py              Adaptive tool routing
MPRE.py             Evidence aggregation
CBDF.py             Calibrated final decision
tools/              Cognitive tools
rl_atr/             RL router
utils/              Data, CAMR, OpenAI, and configuration utilities
CAMR_output/        CAMR retrieval output
results/            Final Tool-meme predictions
embeddings/         Optional offline feature cache
```
