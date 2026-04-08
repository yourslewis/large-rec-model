#!/usr/bin/env bash
# Requires bash 4+ for associative arrays.
# On macOS: brew install bash && use /opt/homebrew/bin/bash
# =============================================================================
# LRM PR Validation — Test Runner & Results Collector
# =============================================================================
# PR: https://github.com/yourslewis/large-rec-model/pull/1
# Branches: astrov6-local-changes → upstream-latest
#
# Usage:
#   ./run_tests.sh setup           # Prepare data split (holdout shard 15)
#   ./run_tests.sh launch <test>   # Launch a specific test (2A|2B|3A|3B|3D)
#   ./run_tests.sh monitor <test>  # Tail the log for a running test
#   ./run_tests.sh status          # GPU utilization + running processes
#   ./run_tests.sh metrics <test>  # Extract eval metrics from a test's log
#   ./run_tests.sh report          # Full comparison table across all runs
#   ./run_tests.sh kill            # Kill all training processes
#
# Tests:
#   2A — Upstream baseline (single-GPU, 75K iters)
#   2B — Our code baseline (single-GPU, target 75K iters → ran to 197K)
#   3A — Ablation F2: event type emb removed (DDP×2, 49K iters)
#   3B — Ablation F1: embedding proj swapped (DDP×2, 107K iters)
#   3D — RotateInDomain negatives (DDP×2, 39K iters)
#
# Environment:
#   GPU_HOST    — SSH target (default: yourslewis@192.168.0.23)
#   GPU_BASE    — Base dir on GPU server (default: /home/yourslewis/lrm_validation)
#   DATA_DIR    — Astrov6 data root (default: /home/yourslewis/lrm_astrov6_split)
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
GPU_HOST="${GPU_HOST:-yourslewis@192.168.0.23}"
GPU_BASE="${GPU_BASE:-/home/yourslewis/lrm_validation}"
DATA_DIR="${DATA_DIR:-/home/yourslewis/lrm_astrov6_split}"
SEED=42
RESULTS_DIR="${GPU_BASE}/results"

# Lookup functions (bash 3.x compatible)
get_workspace() {
    case "$1" in
        2A) echo "${GPU_BASE}/2A_upstream" ;;
        2B) echo "${GPU_BASE}/2B_local" ;;
        3A) echo "${GPU_BASE}/3A_no_event_type" ;;
        3B) echo "${GPU_BASE}/3B_proj_swap" ;;
        3D) echo "${GPU_BASE}/3D_rotate_negatives" ;;
        *)  echo "" ;;
    esac
}

get_config() {
    case "$1" in
        2A) echo "validation_2A.gin" ;;
        2B) echo "validation_2B.gin" ;;
        3A) echo "validation_3A.gin" ;;
        3B) echo "validation_3B.gin" ;;
        3D) echo "validation_3D.gin" ;;
        *)  echo "" ;;
    esac
}

get_description() {
    case "$1" in
        2A) echo "Upstream baseline (event type emb + MLP proj + InBatch)" ;;
        2B) echo "Our code baseline (no event type emb + Linear-SwishLN proj + InBatch)" ;;
        3A) echo "Ablation F2: upstream code, event type emb REMOVED" ;;
        3B) echo "Ablation F1: upstream code, our embedding proj swapped in" ;;
        3D) echo "RotateInDomain negatives (1280 negs, same-domain sampling)" ;;
        *)  echo "Unknown" ;;
    esac
}

get_gpu_mode() {
    case "$1" in
        2A|2B) echo "single" ;;
        3A|3B|3D) echo "ddp" ;;
        *)  echo "" ;;
    esac
}

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ─── Helpers ─────────────────────────────────────────────────────────────────

log_info()  { echo -e "${CYAN}[INFO]${NC}  $(date +%H:%M:%S) $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $(date +%H:%M:%S) $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date +%H:%M:%S) $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date +%H:%M:%S) $*"; }

ssh_cmd() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$GPU_HOST" "$@"
}

validate_test_id() {
    local test_id="$1"
    local ws
    ws=$(get_workspace "$test_id")
    if [[ -z "$ws" ]]; then
        log_error "Unknown test: $test_id (valid: 2A 2B 3A 3B 3D)"
        exit 1
    fi
}

# ─── Data Setup ──────────────────────────────────────────────────────────────

do_setup() {
    log_info "═══ DATA SETUP ═══"

    ssh_cmd bash <<REMOTE
set -e
echo "Checking data directory..."
ls -la ${DATA_DIR}/train/train_*.parquet | wc -l | xargs -I{} echo "Train shards: {}"

if [ ! -d "${DATA_DIR}/eval" ]; then
    mkdir -p "${DATA_DIR}/eval"
fi

if [ ! -e "${DATA_DIR}/eval/eval_0.parquet" ]; then
    echo "Creating eval symlink: train_15.parquet → eval_0.parquet"
    ln -s "${DATA_DIR}/train/train_15.parquet" "${DATA_DIR}/eval/eval_0.parquet"
else
    echo "Eval shard already exists"
fi

echo "Eval dir:"
ls -la ${DATA_DIR}/eval/

# Create results dirs
for d in 2A 2B 3A 3B 3D; do
    mkdir -p ${RESULTS_DIR}/\$d
done
echo "Results directories ready at ${RESULTS_DIR}"

# Verify workspaces
for d in 2A_upstream 2B_local 3A_no_event_type 3B_proj_swap 3D_rotate_negatives; do
    if [ -d "${GPU_BASE}/\$d" ]; then
        echo "✓ Workspace: \$d"
    else
        echo "✗ MISSING: \$d"
    fi
done
REMOTE

    log_ok "Data setup complete"
}

# ─── Launch ──────────────────────────────────────────────────────────────────

do_launch() {
    local test_id="$1"
    validate_test_id "$test_id"

    local workspace config mode desc
    workspace=$(get_workspace "$test_id")
    config=$(get_config "$test_id")
    mode=$(get_gpu_mode "$test_id")
    desc=$(get_description "$test_id")
    local logfile="${RESULTS_DIR}/${test_id}/train.log"

    log_info "═══ LAUNCHING: Test $test_id ═══"
    log_info "Description: $desc"
    log_info "Workspace:   $workspace"
    log_info "Config:      $config"
    log_info "Mode:        $mode"
    log_info "Log:         $logfile"

    if [[ "$mode" == "single" ]]; then
        ssh_cmd bash <<REMOTE
set -e
# Kill any existing training
pkill -f "torchrun.*main.py" 2>/dev/null || true
pkill -f "python.*main.py.*gin" 2>/dev/null || true
sleep 2

cd ${workspace}
echo "Starting single-GPU training for ${test_id}..."

nohup bash -c "
    cd ${workspace} && \\
    CUDA_VISIBLE_DEVICES=0 \\
    torchrun --nproc_per_node=1 \\
        src/hstu_retrieval/main.py \\
        --gin_config_file=PR-validation/configs/${config} \\
        --mode=local \\
        2>&1
" > ${logfile} 2>&1 &

echo "PID: \$!"
echo "Training launched. Monitor with: ./run_tests.sh monitor ${test_id}"
REMOTE
    else
        ssh_cmd bash <<REMOTE
set -e
# Kill any existing training
pkill -f "torchrun.*main.py" 2>/dev/null || true
pkill -f "python.*main.py.*gin" 2>/dev/null || true
sleep 2

cd ${workspace}
echo "Starting DDP×2 training for ${test_id}..."

nohup bash -c "
    cd ${workspace} && \\
    CUDA_VISIBLE_DEVICES=0,1 \\
    torchrun --nproc_per_node=2 \\
        src/hstu_retrieval/main.py \\
        --gin_config_file=PR-validation/configs/${config} \\
        --mode=local \\
        2>&1
" > ${logfile} 2>&1 &

echo "PID: \$!"
echo "Training launched. Monitor with: ./run_tests.sh monitor ${test_id}"
REMOTE
    fi

    log_ok "Test $test_id launched"
}

# ─── Monitor ─────────────────────────────────────────────────────────────────

do_monitor() {
    local test_id="$1"
    validate_test_id "$test_id"
    local logfile="${RESULTS_DIR}/${test_id}/train.log"

    log_info "Tailing log for $test_id: $logfile"
    ssh_cmd "tail -f ${logfile}"
}

# ─── Status ──────────────────────────────────────────────────────────────────

do_status() {
    log_info "═══ GPU STATUS ═══"
    ssh_cmd bash <<'REMOTE'
echo "=== GPU Utilization ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo ""
echo "=== Training Processes ==="
ps aux | grep -E "torchrun|main.py" | grep -v grep || echo "(none running)"
echo ""
echo "=== Recent Log Activity ==="
for d in 2A 2B 3A 3B 3D; do
    logfile=~/lrm_validation/results/$d/train.log
    if [ -f "$logfile" ]; then
        last_line=$(tail -1 "$logfile" 2>/dev/null)
        last_eval=$(grep 'eval @' "$logfile" 2>/dev/null | tail -1 | grep -oP "train_iteration' \K[0-9]+" || echo "N/A")
        mod_time=$(stat -c %Y "$logfile" 2>/dev/null || stat -f %m "$logfile" 2>/dev/null)
        now=$(date +%s)
        age=$(( (now - mod_time) / 60 ))
        echo "  $d: last_eval_iter=$last_eval, log_age=${age}min"
    else
        echo "  $d: no log"
    fi
done
REMOTE
}

# ─── Metrics ─────────────────────────────────────────────────────────────────

do_metrics() {
    local test_id="$1"
    validate_test_id "$test_id"
    local logfile="${RESULTS_DIR}/${test_id}/train.log"

    log_info "═══ METRICS: Test $test_id — $(get_description "$test_id") ═══"

    ssh_cmd bash -s "$logfile" <<'REMOTE'
logfile="$1"
if [ ! -f "$logfile" ]; then
    echo "ERROR: Log file not found: $logfile"
    exit 1
fi

echo "Last 5 eval checkpoints:"
echo "─────────────────────────────────────────────────────────────────────────"
printf "%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n" "Iter" "NDCG@10" "HR@10" "MRR" "log_pplx" "pos_sim" "neg_sim"
echo "─────────────────────────────────────────────────────────────────────────"

grep 'eval @' "$logfile" | tail -5 | while read -r line; do
    iter=$(echo "$line" | grep -oP "train_iteration' \K[0-9]+")
    ndcg=$(echo "$line" | grep -oP "ndcg_10 \K[0-9.]+")
    hr=$(echo "$line" | grep -oP "hr_10 \K[0-9.]+")
    mrr=$(echo "$line" | grep -oP "mrr \K[0-9.]+")
    pplx=$(echo "$line" | grep -oP "log_pplx \K[0-9.]+")
    pos_sim=$(echo "$line" | grep -oP "positive_similarity \K[0-9.-]+")
    neg_sim=$(echo "$line" | grep -oP "negative_similarity \K[0-9.-]+")
    printf "%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n" "$iter" "$ndcg" "$hr" "$mrr" "$pplx" "$pos_sim" "$neg_sim"
done

echo ""
echo "Total eval checkpoints: $(grep -c 'eval @' "$logfile")"

# Best NDCG@10
echo ""
echo "Best NDCG@10:"
grep 'eval @' "$logfile" | python3 -c "
import sys
best_ndcg = 0
best_line = ''
for line in sys.stdin:
    try:
        parts = line.split()
        for i, p in enumerate(parts):
            if p == 'ndcg_10':
                val = float(parts[i+1].rstrip(','))
                if val > best_ndcg:
                    best_ndcg = val
                    for j, q in enumerate(parts):
                        if q == \"'train_iteration'\":
                            best_iter = parts[j+1].strip(\"'\").rstrip(',')
                            break
                    best_line = f'  iter={best_iter} ndcg_10={best_ndcg:.4f}'
                break
    except:
        pass
print(best_line)
"
REMOTE
}

# ─── Report ──────────────────────────────────────────────────────────────────

do_report() {
    log_info "═══ FULL RESULTS REPORT ═══"

    ssh_cmd bash <<'REMOTE'
RESULTS_DIR=~/lrm_validation/results

echo ""
echo "# LRM PR Validation — Results Comparison"
echo "# PR: https://github.com/yourslewis/large-rec-model/pull/1"
echo "# Generated: $(date -u +%Y-%m-%d\ %H:%M\ UTC)"
echo ""

# Phase 1 reference iters
ITER_2A=75000
ITER_2B=75000

# Phase 2 — we compare at matched DDP iters. 
# DDP×2: each iter = 2× samples. 37.5K DDP iters ≈ 75K single-GPU.
# But runs went beyond that, so we report at multiple checkpoints.
ITER_3A_DDP=37000   # closest to 37.5K target
ITER_3B_DDP=37000
ITER_3D_DDP=37000

echo "## Phase 1: Baselines (single-GPU, @75K iters)"
echo ""
echo "| Run | Description | NDCG@10 | HR@10 | MRR | log_pplx |"
echo "|-----|-------------|---------|-------|-----|----------|"

for test_id in 2A 2B; do
    logfile="${RESULTS_DIR}/${test_id}/train.log"
    target_iter=$ITER_2A
    
    if [ -f "$logfile" ]; then
        # Get metrics at target iteration
        line=$(grep "eval @" "$logfile" | grep "'train_iteration' ${target_iter}" | head -1)
        if [ -z "$line" ]; then
            # Fallback: last eval
            line=$(grep "eval @" "$logfile" | tail -1)
            actual_iter=$(echo "$line" | grep -oP "'train_iteration' \K[0-9]+")
            note=" (@ iter ${actual_iter})"
        else
            note=""
        fi
        
        ndcg=$(echo "$line" | grep -oP "ndcg_10 \K[0-9.]+")
        hr=$(echo "$line" | grep -oP "hr_10 \K[0-9.]+")
        mrr=$(echo "$line" | grep -oP "mrr \K[0-9.]+")
        pplx=$(echo "$line" | grep -oP "log_pplx \K[0-9.]+")
        
        case $test_id in
            2A) desc="Upstream baseline" ;;
            2B) desc="Our code baseline" ;;
        esac
        
        echo "| ${test_id} | ${desc}${note} | ${ndcg} | ${hr} | ${mrr} | ${pplx} |"
    else
        echo "| ${test_id} | - | N/A | N/A | N/A | N/A |"
    fi
done

echo ""

# Compute delta
ndcg_2a=$(grep "eval @" "${RESULTS_DIR}/2A/train.log" | grep "'train_iteration' ${ITER_2A}" | head -1 | grep -oP "ndcg_10 \K[0-9.]+")
ndcg_2b=$(grep "eval @" "${RESULTS_DIR}/2B/train.log" | grep "'train_iteration' ${ITER_2B}" | head -1 | grep -oP "ndcg_10 \K[0-9.]+")

if [ -n "$ndcg_2a" ] && [ -n "$ndcg_2b" ]; then
    delta=$(python3 -c "print(f'{(${ndcg_2a} - ${ndcg_2b}) / ${ndcg_2a} * 100:.2f}%')")
    echo "**2A vs 2B delta (NDCG@10):** ${ndcg_2a} vs ${ndcg_2b} = ${delta} gap"
else
    echo "**2A vs 2B delta:** Could not compute (missing data)"
fi

echo ""
echo "## Phase 2: Ablations (DDP×2)"
echo ""
echo "Comparing at matched sample count: 37K DDP iters ≈ 74K single-GPU iters"
echo ""
echo "| Run | Feature Ablated | NDCG@10 | HR@10 | MRR | log_pplx | Δ NDCG vs 2A |"
echo "|-----|----------------|---------|-------|-----|----------|--------------|"

for test_id in 3A 3B 3D; do
    logfile="${RESULTS_DIR}/${test_id}/train.log"
    target_iter=37000
    
    if [ -f "$logfile" ]; then
        line=$(grep "eval @" "$logfile" | grep "'train_iteration' ${target_iter}" | head -1)
        if [ -z "$line" ]; then
            line=$(grep "eval @" "$logfile" | tail -1)
            actual_iter=$(echo "$line" | grep -oP "'train_iteration' \K[0-9]+")
            note=" (@${actual_iter})"
        else
            note=""
        fi
        
        ndcg=$(echo "$line" | grep -oP "ndcg_10 \K[0-9.]+")
        hr=$(echo "$line" | grep -oP "hr_10 \K[0-9.]+")
        mrr=$(echo "$line" | grep -oP "mrr \K[0-9.]+")
        pplx=$(echo "$line" | grep -oP "log_pplx \K[0-9.]+")
        
        if [ -n "$ndcg_2a" ] && [ -n "$ndcg" ]; then
            delta_ndcg=$(python3 -c "d=(${ndcg}-${ndcg_2a})/${ndcg_2a}*100; sign='+' if d>0 else ''; print(f'{sign}{d:.2f}%')")
        else
            delta_ndcg="N/A"
        fi
        
        case $test_id in
            3A) desc="F2: event type emb removed${note}" ;;
            3B) desc="F1: proj → Linear-SwishLN(1024)${note}" ;;
            3D) desc="RotateInDomain negs (1280)${note}" ;;
        esac
        
        echo "| ${test_id} | ${desc} | ${ndcg} | ${hr} | ${mrr} | ${pplx} | ${delta_ndcg} |"
    else
        echo "| ${test_id} | - | N/A | N/A | N/A | N/A | N/A |"
    fi
done

echo ""
echo "## Phase 2: Extended Training (3B ran to 107K iters)"
echo ""
echo "3B continued training well past the 37.5K DDP target. Metrics at later checkpoints:"
echo ""
echo "| Iter (DDP) | ~Equiv Single-GPU | NDCG@10 | HR@10 | MRR | log_pplx |"
echo "|------------|-------------------|---------|-------|-----|----------|"

logfile="${RESULTS_DIR}/3B/train.log"
for iter in 37000 50000 75000 100000 107000; do
    line=$(grep "eval @" "$logfile" | grep "'train_iteration' ${iter}" | head -1)
    if [ -n "$line" ]; then
        equiv=$((iter * 2))
        ndcg=$(echo "$line" | grep -oP "ndcg_10 \K[0-9.]+")
        hr=$(echo "$line" | grep -oP "hr_10 \K[0-9.]+")
        mrr=$(echo "$line" | grep -oP "mrr \K[0-9.]+")
        pplx=$(echo "$line" | grep -oP "log_pplx \K[0-9.]+")
        echo "| ${iter} | ~${equiv} | ${ndcg} | ${hr} | ${mrr} | ${pplx} |"
    fi
done

echo ""
echo "## Attribution Analysis"
echo ""
echo "| Feature | Ablation | NDCG@10 | vs 2A Baseline | Impact |"
echo "|---------|----------|---------|----------------|--------|"

# 2A baseline for comparison
echo "| (baseline) | 2A upstream | ${ndcg_2a:-N/A} | — | Reference |"

# 3A: remove event type → how much does event type matter?
ndcg_3a=$(grep "eval @" "${RESULTS_DIR}/3A/train.log" | grep "'train_iteration' 37000" | head -1 | grep -oP "ndcg_10 \K[0-9.]+")
if [ -n "$ndcg_3a" ] && [ -n "$ndcg_2a" ]; then
    # Compare 3A (37K DDP ≈ 74K single) vs 2A (75K single)
    delta_3a=$(python3 -c "d=(${ndcg_3a}-${ndcg_2a})/${ndcg_2a}*100; sign='+' if d>0 else ''; print(f'{sign}{d:.2f}%')")
    if python3 -c "exit(0 if ${ndcg_3a} < ${ndcg_2a} else 1)"; then
        impact_3a="Event type emb helps (+$(python3 -c "print(f'{${ndcg_2a}-${ndcg_3a}:.4f}')"))"
    else
        impact_3a="Event type emb hurts or neutral"
    fi
    echo "| F2: Event type emb | 3A (removed) | ${ndcg_3a} | ${delta_3a} | ${impact_3a} |"
fi

# 3B: swap projection
ndcg_3b=$(grep "eval @" "${RESULTS_DIR}/3B/train.log" | grep "'train_iteration' 37000" | head -1 | grep -oP "ndcg_10 \K[0-9.]+")
if [ -n "$ndcg_3b" ] && [ -n "$ndcg_2a" ]; then
    delta_3b=$(python3 -c "d=(${ndcg_3b}-${ndcg_2a})/${ndcg_2a}*100; sign='+' if d>0 else ''; print(f'{sign}{d:.2f}%')")
    echo "| F1: Emb projection | 3B (our proj) | ${ndcg_3b} | ${delta_3b} | Projection arch effect |"
fi

# 3D: RotateInDomain
ndcg_3d=$(grep "eval @" "${RESULTS_DIR}/3D/train.log" | grep "'train_iteration' 37000" | head -1 | grep -oP "ndcg_10 \K[0-9.]+")
if [ -n "$ndcg_3d" ] && [ -n "$ndcg_2a" ]; then
    delta_3d=$(python3 -c "d=(${ndcg_3d}-${ndcg_2a})/${ndcg_2a}*100; sign='+' if d>0 else ''; print(f'{sign}{d:.2f}%')")
    echo "| Neg sampling | 3D (RotateInDomain) | ${ndcg_3d} | ${delta_3d} | Negative sampling effect |"
fi

echo ""
echo "## Notes"
echo "- All runs use seed=42, astrov6 data, eval on held-out shard 15"
echo "- 2A/2B: single-GPU, batch=128"
echo "- 3A/3B/3D: DDP×2 GPUs, batch=128/GPU (256 effective)"
echo "- DDP runs: 37K DDP iters ≈ 74K single-GPU iters (matched sample count)"
echo "- 2B ran significantly longer (197K iters) — use 75K checkpoint for fair comparison"
REMOTE
}

# ─── Kill ────────────────────────────────────────────────────────────────────

do_kill() {
    log_warn "Killing all training processes on GPU server..."
    ssh_cmd bash <<'REMOTE'
pkill -f "torchrun.*main.py" 2>/dev/null && echo "Killed torchrun processes" || echo "No torchrun processes"
pkill -f "python.*main.py.*gin" 2>/dev/null && echo "Killed python training processes" || echo "No python training processes"
sleep 2
echo "Remaining:"
ps aux | grep -E "torchrun|main.py" | grep -v grep || echo "(none)"
REMOTE
    log_ok "Done"
}

# ─── Loss Curve ──────────────────────────────────────────────────────────────

do_loss_curve() {
    local test_id="$1"
    validate_test_id "$test_id"
    local logfile="${RESULTS_DIR}/${test_id}/train.log"

    log_info "═══ LOSS CURVE: Test $test_id ═══"

    ssh_cmd bash <<REMOTE
logfile="${logfile}"
echo "Iter    Loss (train)"
echo "────    ────────────"
grep "batch-stat (train)" "\$logfile" | grep "rank: 0" | \
    awk -F'[: ]+' '{
        for(i=1;i<=NF;i++) {
            if(\$i=="iteration") iter=\$(i+1);
            if(\$i ~ /[0-9]+\.[0-9]{6}/) loss=\$i;
        }
        if(iter % 5000 == 0) printf "%-8s %s\n", iter, loss;
    }'
REMOTE
}

# ─── Main ────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
LRM PR Validation Test Runner

Usage: $0 <command> [args]

Commands:
  setup              Prepare data split and directories
  launch <test>      Launch a training run (2A|2B|3A|3B|3D)
  monitor <test>     Tail the training log
  status             GPU utilization + running processes
  metrics <test>     Show eval metrics from a test's log
  report             Full comparison table across all runs
  loss <test>        Show loss curve (sampled every 5K iters)
  kill               Kill all training processes

Examples:
  $0 setup           # First-time data setup
  $0 launch 3A       # Launch test 3A
  $0 metrics 2A      # Check 2A results
  $0 report          # Generate full comparison report
EOF
}

main() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi

    local cmd="$1"
    shift

    case "$cmd" in
        setup)    do_setup ;;
        launch)   [[ $# -lt 1 ]] && { log_error "Usage: $0 launch <test_id>"; exit 1; }; do_launch "$1" ;;
        monitor)  [[ $# -lt 1 ]] && { log_error "Usage: $0 monitor <test_id>"; exit 1; }; do_monitor "$1" ;;
        status)   do_status ;;
        metrics)  [[ $# -lt 1 ]] && { log_error "Usage: $0 metrics <test_id>"; exit 1; }; do_metrics "$1" ;;
        report)   do_report ;;
        loss)     [[ $# -lt 1 ]] && { log_error "Usage: $0 loss <test_id>"; exit 1; }; do_loss_curve "$1" ;;
        kill)     do_kill ;;
        *)        log_error "Unknown command: $cmd"; usage; exit 1 ;;
    esac
}

main "$@"
