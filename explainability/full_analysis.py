import os
import sys
from .eye_blink_mismatch import main as eye_blink_main
from .iris_alignment import main as iris_main
from .eyebrow_mismatch import main as eyebrow_main
from .texture_analyzer import main as texture_main
from .flicker_detection import main as run_flicker

# Optional, if audio module exists
try:
    from .lip_sync_mismatch import main as lip_sync_main
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


def full_analysis(video_path, opts=None):
    """
    Runs the complete deepfake explainability pipeline on the given video.

    Parameters:
        video_path (str): Path to the input video file
        opts (list): List of modules to run.
                     Default: ['eye','iris','eyebrow','skin','flicker','lip']

    Returns:
        list: Results (strings) for each check
    """
    if opts is None:
        opts = ['eye', 'iris', 'eyebrow', 'skin', 'flicker', 'lip']

    results = [f"--- Full Analysis on {os.path.basename(video_path)} ---"]

    # Eye blink mismatch
    if 'eye' in opts:
        try:
            results.append("Eye Blink Mismatch: Running...")
            results.append(str(eye_blink_main(video_path)))
        except Exception as e:
            results.append(f"⚠ Eye blink analysis failed: {str(e)}")

    # Iris alignment
    if 'iris' in opts:
        try:
            results.append("Iris Alignment: Running...")
            results.append(str(iris_main(video_path)))
        except Exception as e:
            results.append(f"⚠ Iris alignment failed: {str(e)}")

    # Eyebrow mismatch
    if 'eyebrow' in opts:
        try:
            results.append("Eyebrow Mismatch: Running...")
            results.append(str(eyebrow_main(video_path)))
        except Exception as e:
            results.append(f"⚠ Eyebrow mismatch analysis failed: {str(e)}")

    # Skin texture analysis
    if 'skin' in opts:
        try:
            results.append("Skin Texture Analysis: Running...")
            results.append(str(texture_main(video_path)))
        except Exception as e:
            results.append(f"⚠ Skin texture analysis failed: {str(e)}")

    # Flicker detection
    if 'flicker' in opts:
        try:
            results.append("Flicker Detection: Running...")
            results.append(str(run_flicker(video_path)))
        except Exception as e:
            results.append(f"⚠ Flicker detection failed: {str(e)}")

    # Lip sync mismatch
    if 'lip' in opts:
        if HAS_AUDIO:
            try:
                results.append("Lip Sync Mismatch: Running...")
                results.append(str(lip_sync_main(video_path)))
            except Exception as e:
                results.append(f"⚠ Lip sync mismatch failed: {str(e)}")
        else:
            results.append("⚠ Lip sync check skipped: audio module not available.")

    return results


# ------------------------------
# Run via command line
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠ Usage: python scripts/full_analysis.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    output = full_analysis(video_path)
    for line in output:
        print(line)
