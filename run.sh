
#!/bin/bash

# Define the function for stitching photos
stitch_photos() {
    echo "Starting stitching for $1. Please wait..."
    python main.py "$1" -mbb
    echo "Finished stitching plz check $1/result."
}

# Run the stitching process for each set of photos
echo "============================================================"
echo "=== Stitching Photos ==="

# Stitch photos in the 'Base' folder
stitch_photos "Photos/Base"

# Stitch photos in the 'Challenge' folder
stitch_photos "Photos/Challenge"

echo "=== All stitching processes completed ==="
echo "============================================================"
