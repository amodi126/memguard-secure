#!/bin/bash
# MemGuard-Secure: Full experiment runbook for IEEE Access paper
# Run from: ~/Documents/Research_Spring2026_Gtech/memguard_secure/letta_eval/
#
# Prerequisites:
#   - Docker running with letta-server container
#   - Python venv activated: source memguard/bin/activate
#   - API keys set in ../docker_inject/memguard_keys.json
#   - pip install scipy numpy

set -e

INJECT_DIR="../docker_inject"
CONFIG_FILE="${INJECT_DIR}/memguard_keys.json"
CONTAINER_CONFIG="/app/memguard_keys.json"

# ─── Helper: update JSON config and redeploy to container ──────────────────
# Usage: update_config '{"LETTA_MEMGUARD_ENABLED": "1", "MEMGUARD_VOTERS": "...", ...}'
# Merges the provided keys into the existing config file, then copies to container.
update_config() {
    python3 -c "
import json, sys
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
updates = json.loads(sys.argv[1])
cfg.update(updates)
with open('${CONFIG_FILE}', 'w') as f:
    json.dump(cfg, f, indent=4)
" "$1"
    docker cp "${CONFIG_FILE}" "letta-server:${CONTAINER_CONFIG}"
}

# ─── Helper: deploy all code files to container ───────────────────────────
deploy_code() {
    echo "  Deploying MemGuard code to container..."
    docker cp "${INJECT_DIR}/memguard_hook.py" letta-server:/app/letta/services/memguard_hook.py
    docker cp "${INJECT_DIR}/memguard_context.py" letta-server:/app/letta/services/memguard_context.py
    docker cp "${INJECT_DIR}/core_tool_executor.py" letta-server:/app/letta/services/tool_executor/core_tool_executor.py
    docker cp "${INJECT_DIR}/letta_agent_v3.py" letta-server:/app/letta/agents/letta_agent_v3.py
    docker cp "${CONFIG_FILE}" "letta-server:${CONTAINER_CONFIG}"
}

# ─── Helper: restart container and wait ────────────────────────────────────
restart_and_wait() {
    docker restart letta-server
    echo "  Waiting for container to start..."
    sleep 20
    # Verify container is running
    if ! docker ps --filter "name=letta-server" --filter "status=running" -q | grep -q .; then
        echo "ERROR: letta-server is not running after restart!"
        exit 1
    fi
    echo "  Container ready."
}

echo "=== MemGuard-Secure Experiment Runner ==="
echo "Make sure letta-server is running: docker start letta-server"
echo ""

# ─── Phase 0: Quick sanity check (5 min) ─────────────────────────────────────
echo "--- Phase 0: Sanity check with 1 conversation ---"

# Generate 1 enhanced conversation to test the pipeline
python attack_generator.py --scenario travel_planner --num_conversations 1 \
    --output attacks_sanity/ --enhanced

# Run baseline (defense off)
update_config '{"LETTA_MEMGUARD_ENABLED": "0", "MEMGUARD_SINGLE_MODEL": "", "MEMGUARD_VOTERS": "", "MEMGUARD_THRESHOLD": ""}'
deploy_code
restart_and_wait

python benchmark_runner.py --attacks attacks_sanity/ \
    --output results/sanity_baseline/ --verbose

# Run defended (defense on)
update_config '{"LETTA_MEMGUARD_ENABLED": "1"}'
restart_and_wait

LETTA_MEMGUARD_ENABLED=1 python benchmark_runner.py --attacks attacks_sanity/ \
    --output results/sanity_defended/ --defended --verbose --capture-voter-data

echo ""
echo "=== Phase 0 complete. Check sanity results before proceeding! ==="
echo "Press Enter to continue or Ctrl+C to abort..."
read -r

# ─── Phase 1: Generate all attack conversations (~15 min) ────────────────────
echo "--- Phase 1: Generate attack conversations ---"

# 30 standard conversations (3 attack types each)
python attack_generator.py --scenario all --num_conversations 30 \
    --output attacks_30/

# 30 enhanced conversations (5 attack types each)
python attack_generator.py --scenario all --num_conversations 30 \
    --output attacks_30_enhanced/ --enhanced

echo "Attack generation complete."
echo ""

# ─── Phase 2: Baseline runs (~2 hours) ───────────────────────────────────────
echo "--- Phase 2: Baseline (no defense) ---"

# Disable defense via config file
update_config '{"LETTA_MEMGUARD_ENABLED": "0", "MEMGUARD_SINGLE_MODEL": "", "MEMGUARD_VOTERS": "", "MEMGUARD_THRESHOLD": ""}'
restart_and_wait

# Standard attacks — baseline
python benchmark_runner.py --attacks attacks_30/ \
    --output results/baseline_30/ --verbose

# Enhanced attacks — baseline
python benchmark_runner.py --attacks attacks_30_enhanced/ \
    --output results/baseline_30_enhanced/ --verbose

echo "Baseline runs complete."
echo ""

# ─── Phase 3: Defended runs (~2 hours) ───────────────────────────────────────
echo "--- Phase 3: Defended (cross-model consensus) ---"

# Enable defense with default config (all 3 cross-model voters, majority threshold)
update_config '{"LETTA_MEMGUARD_ENABLED": "1", "MEMGUARD_SINGLE_MODEL": "", "MEMGUARD_VOTERS": "memory_consistency,request_classification,instruction_detection", "MEMGUARD_THRESHOLD": ""}'
restart_and_wait

# Standard attacks — defended
LETTA_MEMGUARD_ENABLED=1 python benchmark_runner.py --attacks attacks_30/ \
    --output results/defended_30/ --defended --verbose --capture-voter-data

# Enhanced attacks — defended
LETTA_MEMGUARD_ENABLED=1 python benchmark_runner.py --attacks attacks_30_enhanced/ \
    --output results/defended_30_enhanced/ --defended --verbose --capture-voter-data

echo "Defended runs complete."
echo ""

# ─── Phase 4: Ablation runs (~4 hours) ──────────────────────────────────────
echo "--- Phase 4: Ablation study ---"

# 4a: Single-voter ablations
for voter in memory_consistency request_classification instruction_detection; do
    echo "  Ablation: ${voter} only"
    update_config "{\"LETTA_MEMGUARD_ENABLED\": \"1\", \"MEMGUARD_VOTERS\": \"${voter}\", \"MEMGUARD_THRESHOLD\": \"1\", \"MEMGUARD_SINGLE_MODEL\": \"\"}"
    restart_and_wait

    LETTA_MEMGUARD_ENABLED=1 python benchmark_runner.py --attacks attacks_30/ \
        --output "results/ablation_${voter}/" --defended --capture-voter-data
done

# 4b: Same-model baseline (all GPT-4o-mini, no cross-model diversity)
echo "  Ablation: same-model (all gpt-4o-mini)"
update_config '{"LETTA_MEMGUARD_ENABLED": "1", "MEMGUARD_SINGLE_MODEL": "gpt-4o-mini", "MEMGUARD_VOTERS": "memory_consistency,request_classification,instruction_detection", "MEMGUARD_THRESHOLD": ""}'
restart_and_wait

LETTA_MEMGUARD_ENABLED=1 python benchmark_runner.py --attacks attacks_30/ \
    --output results/ablation_same_model/ --defended --capture-voter-data

# 4c: Restore default config
echo "  Restoring default config..."
update_config '{"LETTA_MEMGUARD_ENABLED": "1", "MEMGUARD_SINGLE_MODEL": "", "MEMGUARD_VOTERS": "", "MEMGUARD_THRESHOLD": ""}'
restart_and_wait

echo "Ablation runs complete."
echo ""

# ─── Phase 5: Voter injection (~4 hours) ─────────────────────────────────────
echo "--- Phase 5: Voter injection attack ---"

# Phase 5a: Hardened prompts (with CRITICAL anti-injection directive) — expected ASR ~0%
echo "  Phase 5a: Voter injection vs. hardened prompts"
update_config '{"LETTA_MEMGUARD_ENABLED": "1", "MEMGUARD_HARDENED": "1", "MEMGUARD_SINGLE_MODEL": "", "MEMGUARD_VOTERS": "", "MEMGUARD_THRESHOLD": ""}'
restart_and_wait

python voter_injection_runner.py \
    --input attacks_30_enhanced/ \
    --output-attacks attacks_30_voter_injection/ \
    --results results/voter_injection_hardened/ \
    --verbose

# Phase 5b: Unhardened prompts (CRITICAL directive removed) — expected ASR ~63.3%
# This is the ablation condition reported in Section V-D of the paper.
echo "  Phase 5b: Voter injection vs. unhardened prompts (ablation)"
update_config '{"MEMGUARD_HARDENED": "0"}'
restart_and_wait

python voter_injection_runner.py \
    --input attacks_30_enhanced/ \
    --output-attacks attacks_30_voter_injection/ \
    --results results/voter_injection_unhardened/ \
    --verbose

# Restore hardened config
update_config '{"MEMGUARD_HARDENED": "1"}'

echo "Voter injection runs complete."
echo ""

# ─── Phase 6: Adaptive adversary (~4 hours) ──────────────────────────────────
echo "--- Phase 6: Adaptive adversary ---"

# Requires OPENAI_API_KEY in the local environment (used to generate follow-up attacks)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set in local terminal — adaptive attacks will fail."
    echo "  Export it first: export OPENAI_API_KEY=<your key>"
fi

python adaptive_attack_runner.py \
    --output results/adaptive_attacks/ \
    --num_conversations 30

echo "Adaptive adversary runs complete."
echo ""

# ─── Phase 7: Analysis ───────────────────────────────────────────────────────
echo "--- Phase 7: Statistical analysis ---"

# Main analysis: standard attacks
python analysis.py \
    --baseline results/baseline_30/ \
    --defended results/defended_30_standard_hardened/ \
    --ablation results/ablation_memory_consistency_hardened/ \
              results/ablation_request_classification_hardened/ \
              results/ablation_instruction_detection_hardened/ \
              results/ablation_same_model_hardened/ \
    --output results/analysis_standard.json

# Enhanced attacks analysis
python analysis.py \
    --baseline results/baseline_30_enhanced/ \
    --defended results/defended_30_enhanced/ \
    --output results/analysis_enhanced.json

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Results saved to results/"
echo "Analysis reports: results/analysis_standard.json, results/analysis_enhanced.json"
