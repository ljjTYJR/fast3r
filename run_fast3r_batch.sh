#!/bin/bash
# Batch processing script for Fast3R pose estimation and depth prediction
# This script processes all subdirectories in a given parent directory

set -e  # Exit on error

# Configuration
BASE_DIR="/media/shuo/T7/robolab/scripts/processed_clips/11_10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_SCRIPT="${SCRIPT_DIR}/demo.py"

# Model settings
CHECKPOINT="jedyang97/Fast3R_ViT_Large_512"
IMAGE_SIZE=512
SAMPLING_INTERVAL=2

# Camera settings - adjust if your data has different camera names
CAMERAS="cam01 cam02"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Timing summary file
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TIMING_SUMMARY="${BASE_DIR}/fast3r_batch_timing_${TIMESTAMP}.csv"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fast3R Batch Processing Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Base directory: ${BASE_DIR}"
echo "Sampling interval: ${SAMPLING_INTERVAL}"
echo "Image size: ${IMAGE_SIZE}"
echo "Cameras: ${CAMERAS}"
echo ""

# Initialize timing summary CSV file
echo "scene_name,model_load_time,processing_time,total_time,status" > "${TIMING_SUMMARY}"
echo "Timing summary will be saved to: ${TIMING_SUMMARY}"
echo ""

# Check if base directory exists
if [ ! -d "${BASE_DIR}" ]; then
    echo -e "${RED}Error: Base directory does not exist: ${BASE_DIR}${NC}"
    exit 1
fi

# Process each subdirectory
for subdir in "${BASE_DIR}"/*/ ; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")

        # Skip if it's a results directory
        if [[ "$subdir_name" == *"results"* ]] || [[ "$subdir_name" == *"summary"* ]]; then
            echo -e "${BLUE}Skipping: ${subdir_name}${NC}"
            continue
        fi

        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Processing: ${subdir_name}${NC}"
        echo -e "${GREEN}========================================${NC}"

        # Record start time (wall-clock)
        scene_start_time=$(date +%s.%N)

        source 3renv/bin/activate
        # Run Fast3R
        python3 "${DEMO_SCRIPT}" \
            --data_dir "${subdir}" \
            --output_dir "${subdir}/fast3r_poses" \
            --cameras ${CAMERAS} \
            --checkpoint "${CHECKPOINT}" \
            --sampling_interval ${SAMPLING_INTERVAL} \
            --image_size ${IMAGE_SIZE}

        exit_code=$?

        # Record end time
        scene_end_time=$(date +%s.%N)

        # Calculate total time
        total_time=$(echo "$scene_end_time - $scene_start_time" | bc)

        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}✓ Successfully processed: ${subdir_name}${NC}"

            # Extract timing info from timing.json
            timing_json="${subdir}/fast3r_poses/timing.json"
            if [ -f "${timing_json}" ]; then
                # Parse JSON using python for reliability, format to 3 decimal places
                model_load_time=$(python3 -c "import json; print(f\"{json.load(open('${timing_json}'))['model_load_time']:.3f}\")" 2>/dev/null || echo "N/A")
                processing_time=$(python3 -c "import json; print(f\"{json.load(open('${timing_json}'))['total_time']:.3f}\")" 2>/dev/null || echo "N/A")

                # Format total_time to 3 decimal places
                total_time_formatted=$(LC_NUMERIC=C printf "%.3f" ${total_time})

                # Append to CSV
                echo "${subdir_name},${model_load_time},${processing_time},${total_time_formatted},success" >> "${TIMING_SUMMARY}"
            else
                echo -e "${RED}Warning: timing.json not found for ${subdir_name}${NC}"
                # Format total_time to 3 decimal places
                total_time_formatted=$(LC_NUMERIC=C printf "%.3f" ${total_time})
                echo "${subdir_name},N/A,N/A,${total_time_formatted},success_no_timing" >> "${TIMING_SUMMARY}"
            fi
        else
            echo -e "${RED}✗ Failed to process: ${subdir_name}${NC}"
            # Format total_time to 3 decimal places
            total_time_formatted=$(printf "%.3f" ${total_time})
            echo "${subdir_name},N/A,N/A,${total_time_formatted},failed" >> "${TIMING_SUMMARY}"
        fi
        echo ""
    fi
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Batch processing completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Timing summary saved to: ${TIMING_SUMMARY}"
echo ""
echo "Summary statistics:"
# Count successes and failures
total_scenes=$(grep -c "," "${TIMING_SUMMARY}" 2>/dev/null || echo 0)
total_scenes=$((total_scenes - 1))  # Subtract header line
success_count=$(grep -c ",success$" "${TIMING_SUMMARY}" 2>/dev/null || echo 0)
failed_count=$(grep -c ",failed$" "${TIMING_SUMMARY}" 2>/dev/null || echo 0)

echo "  Total scenes: ${total_scenes}"
echo "  Successful: ${success_count}"
echo "  Failed: ${failed_count}"
echo ""
