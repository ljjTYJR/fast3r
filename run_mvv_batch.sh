#!/bin/bash
# Fast3R multi-view pose estimation batch processor
# Processes all scenes in MultiCamVideo-Dataset with auto-detected cameras

# set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEMO_SCRIPT="${SCRIPT_DIR}/demo_mvv.py"

# Default configuration
BASE_DIR="${1:-/media/shuo/T7/multiple_views/MultiCamVideo-Dataset/processed_videos}"
CHECKPOINT="${CHECKPOINT:-jedyang97/Fast3R_ViT_Large_512}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
SAMPLING_INTERVAL="${SAMPLING_INTERVAL:-1}"

die() {
    echo "Error: $*" >&2
    exit 1
}

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

process_scene() {
    local scene_dir="$1"
    local scene_name output_dir

    scene_name="$(basename "${scene_dir}")"
    output_dir="${scene_dir}/fast3r_poses"

    # Skip if already processed
    if [[ -f "${output_dir}/summary.json" ]]; then
        log "Skipping ${scene_name} (already processed)"
        return 0
    fi

    log "Processing ${scene_name}..."

    if python3 "${DEMO_SCRIPT}" \
        --data_dir "${scene_dir}" \
        --output_dir "${output_dir}" \
        --checkpoint "${CHECKPOINT}" \
        --sampling_interval "${SAMPLING_INTERVAL}" \
        --image_size "${IMAGE_SIZE}"; then
        log "Success: ${scene_name}"
        return 0
    else
        log "Failed: ${scene_name}" >&2
        return 1
    fi
}

main() {
    log "Fast3R Multi-View Batch Processing"
    log "Base directory: ${BASE_DIR}"
    log "Sampling: ${SAMPLING_INTERVAL}, Image size: ${IMAGE_SIZE}"

    [[ -d "${BASE_DIR}" ]] || die "Base directory not found: ${BASE_DIR}"
    [[ -f "${DEMO_SCRIPT}" ]] || die "Demo script not found: ${DEMO_SCRIPT}"

    local total=0 success=0 failed=0 skipped=0

    # Process each scene directory
    for scene_dir in "${BASE_DIR}"/*/; do
        [[ -d "${scene_dir}" ]] || continue

        local scene_name
        scene_name="$(basename "${scene_dir}")"

        # Skip non-scene directories
        [[ "${scene_name}" =~ ^scene[0-9]+$ ]] || continue

        ((total++))

        if [[ -f "${scene_dir}/fast3r_poses/summary.json" ]]; then
            ((skipped++))
            log "Skipping ${scene_name} (already processed)"
            continue
        fi

        if process_scene "${scene_dir}"; then
            ((success++))
        else
            ((failed++))
        fi
    done

    log "=========================================="
    log "Batch processing complete"
    log "Total: ${total}, Success: ${success}, Failed: ${failed}, Skipped: ${skipped}"
    log "=========================================="

    [[ ${failed} -eq 0 ]] || exit 1
}

main "$@"
