#!/bin/bash
# Single directory processing script for Fast3R pose estimation and depth prediction

set -e  # Exit on error

# Default configuration
DATA_DIR="/media/shuo/T7/robolab/scripts/processed_clips/11_08/robodog_01"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_SCRIPT="${SCRIPT_DIR}/demo.py"

# Model settings
CHECKPOINT="jedyang97/Fast3R_ViT_Large_512"
IMAGE_SIZE=512
SAMPLING_INTERVAL=2
CAMERAS="cam01 cam02"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --sampling_interval)
            SAMPLING_INTERVAL="$2"
            shift 2
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --cameras)
            CAMERAS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_dir DIR          Data directory (default: robodog_01)"
            echo "  --sampling_interval N   Use every Nth frame (default: 2)"
            echo "  --image_size SIZE       Image size (default: 512)"
            echo "  --cameras \"cam1 cam2\"   Camera names (default: \"cam01 cam02\")"
            echo "  --checkpoint PATH       Model checkpoint (default: jedyang97/Fast3R_ViT_Large_512)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --data_dir /path/to/data --sampling_interval 4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fast3R Processing${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${DATA_DIR}/fast3r_poses"
echo "Sampling interval: ${SAMPLING_INTERVAL}"
echo "Image size: ${IMAGE_SIZE}"
echo "Cameras: ${CAMERAS}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo -e "${RED}Error: Data directory does not exist: ${DATA_DIR}${NC}"
    exit 1
fi

# Run Fast3R
echo -e "${GREEN}Starting Fast3R processing...${NC}"
python3 "${DEMO_SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/fast3r_poses" \
    --cameras ${CAMERAS} \
    --checkpoint "${CHECKPOINT}" \
    --sampling_interval ${SAMPLING_INTERVAL} \
    --image_size ${IMAGE_SIZE}

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Processing completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Results saved to: ${DATA_DIR}/fast3r_poses"
    echo ""
    echo "Output files:"
    echo "  - Camera poses: *_poses.txt (TUM format with frame indices)"
    echo "  - Depth maps: *_depth/*.npy (raw depth values)"
    echo "  - Depth visualizations: *_depth/*.png (colored depth maps)"
    echo "  - Depth statistics: *_depth_stats.json"
    echo "  - Metadata: *_meta.json"
else
    echo -e "${RED}✗ Processing failed${NC}"
    exit 1
fi
