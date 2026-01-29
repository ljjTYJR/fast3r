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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fast3R Batch Processing Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Base directory: ${BASE_DIR}"
echo "Sampling interval: ${SAMPLING_INTERVAL}"
echo "Image size: ${IMAGE_SIZE}"
echo "Cameras: ${CAMERAS}"
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
        source 3renv/bin/activate
        # Run Fast3R
        python3 "${DEMO_SCRIPT}" \
            --data_dir "${subdir}" \
            --output_dir "${subdir}/fast3r_poses" \
            --cameras ${CAMERAS} \
            --checkpoint "${CHECKPOINT}" \
            --sampling_interval ${SAMPLING_INTERVAL} \
            --image_size ${IMAGE_SIZE}

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Successfully processed: ${subdir_name}${NC}"
        else
            echo -e "${RED}✗ Failed to process: ${subdir_name}${NC}"
        fi
        echo ""
    fi
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Batch processing completed!${NC}"
echo -e "${BLUE}========================================${NC}"
