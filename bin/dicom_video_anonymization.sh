#!/usr/bin/env bash
# Lossless DICOM anonymization (mask top-left box) with parallelism
# Layout:
#   INPUT:  /root/<session_date>/<session_id>/.../*.dcm
#   OUTPUT: /out/<session_date>/<session_id>/{DICOM_ANON,ANON_QC,ANON_VID}
#
# Handles (UNCOMPRESSED only): RGB (planar 0/1), YBR_FULL, YBR_FULL_422,
# YBR_PARTIAL_422, YBR_PARTIAL_420, MONOCHROME1/2 (8/16), PALETTE COLOR.
#
# Requirements in container: bash, coreutils, findutils, dcmdump (dcmtk),
# ffmpeg, python3 + pydicom.

set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""
NUM_THREADS=4
OVERWRITE=0
KEEP_RAW="${KEEP_RAW:-0}"   # set to 1 to keep *.raw intermediates
NO_MKV="${NO_MKV:-0}"       # set via -V/--no-mkv to skip .mkv output
# Blur single-frame PNGs if there are at least this many black rows from the top
# These have been shown to contain PHI outside of the standard area we are checking.
BLUR_TOP_BLACK_ROWS="${BLUR_TOP_BLACK_ROWS:-5}"
# Choose lossless video codec: ffv1 (default) or x264rgb
VID_CODEC="${VID_CODEC:-ffv1}"  # or x264rgb
# Coerce/guard env knobs once up front
: "${ULTRASOUND_FALLBACK_FRACW:=2}"
# Must be a positive integer
[[ ! "$ULTRASOUND_FALLBACK_FRACW" =~ ^[1-9][0-9]*$ ]] && ULTRASOUND_FALLBACK_FRACW=2

: "${ULTRASOUND_FALLBACK_BOXH:=35}"
# Must be a positive integer
[[ ! "$ULTRASOUND_FALLBACK_BOXH" =~ ^[1-9][0-9]*$ ]] && ULTRASOUND_FALLBACK_BOXH=35

: "${MIN_NEAR_BLACK_PCT:=0.10}"
# Must be a valid floating-point >=0 <=1
case "$MIN_NEAR_BLACK_PCT" in
  ''|*[!0-9.eE+-]*) MIN_NEAR_BLACK_PCT=0.10 ;;
esac

: "${NEAR_BLACK_LUMA:=30}"
# Must be positive integer
[[ ! "$NEAR_BLACK_LUMA" =~ ^[1-9][0-9]*$ ]] && NEAR_BLACK_LUMA=30

export BLUR_TOP_BLACK_ROWS VID_CODEC KEEP_RAW OVERWRITE ULTRASOUND_FALLBACK_FRACW ULTRASOUND_FALLBACK_BOXH MIN_NEAR_BLACK_PCT NEAR_BLACK_LUMA

usage() {
  cat <<USAGE
Usage: $0 -i <input_dir> -o <output_dir> [-n <num_threads>] [--overwrite-existing] [--no-mkv|-V]


Notes:
  * Expects input files under: <input_dir>/<session_date>/<session_id>/.../*.dcm
  * Outputs:
      <output_dir>/<session_date>/<session_id>/DICOM_ANON/*.dcm
      <output_dir>/<session_date>/<session_id>/ANON_VID/*.mkv
      <output_dir>/<session_date>/<session_id>/ANON_IMG/*.png
      <output_dir>/<session_date>/<session_id>/ANON_QC/*.png
  * Only uncompressed transfer syntaxes are edited losslessly. Compressed are skipped.
  * Use --no-mkv / -V to skip ANON_VID .mkv creation (DICOM + QC PNG still produced).
USAGE
  exit 1
}

# ---------- Black-bar detector (decodes first frame to RGB ONLY FOR ANALYSIS) ----------
detect_black_rows_rgb24_from_dicom() {
  # args: dcm_path rows cols (rows/cols are not strictly needed now)
  local dcm="$1"

  python3 - "$dcm" <<'PY'
import sys, numpy as np
try:
    import pydicom
except Exception:
    print(0); sys.exit(0)

dcm = sys.argv[1]
try:
    ds = pydicom.dcmread(dcm, force=True, stop_before_pixels=False)
    a  = ds.pixel_array  # uses handlers for compressed & uncompressed
except Exception:
    print(0); sys.exit(0)

# Normalize to (F,R,C,3) uint8 **without** manual YCbCr math.
if a.ndim == 2:
    a = a[None, ...]                 # (1,R,C)
if a.ndim == 3:
    if a.shape[-1] == 3:             # (R,C,3)
        a = a[None, ...]             # (1,R,C,3)
    else:                            # (F,R,C) mono
        a = a[..., None]             # (F,R,C,1)

# (F,R,C,S)
if a.shape[-1] == 1:
    a = a.astype(np.uint8)
    a = np.repeat(a, 3, axis=-1)     # MONO→RGB
else:
    a = a[..., :3].astype(np.uint8)  # drop alpha if present

# Use luminance (Rec.601) to detect “black” rows
Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)
R = Y.shape[1]; C = Y.shape[2]
threshold = 30.0      # Y <= 30 is “black”
pct       = 0.90      # ≥90% pixels dark
need      = 3         # 3 consecutive rows

streak = 0
start  = -1
for y in range(R):
    dark = (Y[0,y] <= threshold).mean()
    if dark >= pct:
        if streak == 0: start = y
        streak += 1
        if streak >= need:
            print(start + streak)
            sys.exit(0)
    else:
        streak = 0
        start  = -1

print(0)
sys.exit(0)
PY
}

# ---------- Top-of-image black-run detector (rows from top only) ----------
detect_top_black_rows_rgb24_from_dicom() {
  # args: dcm_path
  local dcm="$1"
  python3 - "$dcm" <<'PY'
import sys, numpy as np
try:
    import pydicom
except Exception:
    print(0); sys.exit(0)

dcm = sys.argv[1]
try:
    ds = pydicom.dcmread(dcm, force=True, stop_before_pixels=False)
    a  = ds.pixel_array
except Exception:
    print(0); sys.exit(0)

# Normalize to (F,R,C,3) uint8
if a.ndim == 2:
    a = a[None, ...]
if a.ndim == 3:
    if a.shape[-1] == 3:
        a = a[None, ...]
    else:
        a = a[..., None]
if a.shape[-1] == 1:
    a = a.astype(np.uint8)
    a = np.repeat(a, 3, axis=-1)
else:
    a = a[..., :3].astype(np.uint8)

# Luma
Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)
R = Y.shape[1]
threshold = 30.0   # black
pct       = 0.90   # ≥90% pixels in a row are "black"

# Count consecutive black rows starting at the very top/Mask box

top_run = 0
for y in range(R):
    if (Y[0,y] <= threshold).mean() >= pct:
        top_run += 1
    else:
        break

print(top_run)
sys.exit(0)
PY
}

# ---------- Whole-frame near-black coverage detector (returns 0.0..1.0) ----------
detect_near_black_pct_from_dicom() {
  local dcm="$1" luma_thresh="$2"
  python3 - "$dcm" "$luma_thresh" <<'PY'
import sys, numpy as np
try:
    import pydicom
except Exception:
    print("0.0"); sys.exit(0)

def log(level, msg):
    print(f"[{level}] {msg}", file=sys.stderr)

dcm = sys.argv[1]
try:
    thr = float(sys.argv[2])
except Exception as e:
    thr = 30.0

try:
    ds = pydicom.dcmread(dcm, force=True, stop_before_pixels=False)
    a  = ds.pixel_array
except Exception:
    print("0.0"); sys.exit(0)

# Normalize to (F,R,C,3) uint8 quickly
if a.ndim == 2: a = a[None, ...]
if a.ndim == 3:
    if a.shape[-1] == 3: a = a[None, ...]
    else: a = a[..., None]
if a.shape[-1] == 1:
    # mono → uint8 view (scale for preview only)
    if a.dtype != np.uint8:
        a = (a.astype(np.float32) / max(float(a.max()),1.0) * 255.0).astype(np.uint8)
    a = np.repeat(a, 3, axis=-1)
else:
    a = a[..., :3].astype(np.uint8, copy=False)

# Compute luma on first frame only—enough for coverage
Y = (0.299*a[0,...,0] + 0.587*a[0,...,1] + 0.114*a[0,...,2]).astype(np.float32)
near_black = (Y <= thr).mean()
print(f"{near_black:.6f}")
PY
}

# ---------- Per-file worker ----------
process_one() {
  local dcm="$1" inroot="$2" outroot="$3"

  # Preserve full relative directory from input root
  local ddir rel_full
  ddir="$(dirname "$dcm")"
  rel_full="$(python3 - "$inroot" "$ddir" <<'PY'
import os,sys
root=os.path.realpath(sys.argv[1]); d=os.path.realpath(sys.argv[2])
try:
    print(os.path.relpath(d,root))
except Exception:
    print("")
PY
)"
  [[ -z "$rel_full" ]] && rel_full="unknown"

  # Output dirs: append category folders under the original relative path
  local out_base="$outroot/$rel_full"
  local out_dcm_dir="$out_base/DICOM_ANON"
  local out_qc_dir="$out_base/ANON_QC"
  local out_vid_dir="$out_base/ANON_VID"
  local out_img_dir="$out_base/ANON_IMG"
  # Always need DICOM, QC, and IMG dirs
  mkdir -p "$out_dcm_dir" "$out_qc_dir" "$out_img_dir"

  # Only create ANON_VID if we’re actually keeping video
  if [[ "${NO_MKV:-0}" -eq 0 ]]; then
    mkdir -p "$out_vid_dir"
  fi

  # Pull minimal metadata via dcmdump (fast & robust)
  tag() {
    local file="$1" key="$2"
    dcmdump -q +P "$key" "$file" 2>/dev/null | awk '
      {
        sub(/.*\) [A-Z][A-Z] ?/, "", $0);
        sub(/#.*/ , "", $0);
        gsub(/^[ \t]+|[ \t]+$/, "", $0);
        if ($0 ~ /^\[.*\]$/) { sub(/^\[/,""); sub(/\]$/ ,"") }
        sub(/^=[ \t]*/, "", $0);
        print $0; exit
      }'
  }

  normalize_tsuid() {
    case "$1" in
      LittleEndianImplicit|ImplicitVRLittleEndian) echo 1.2.840.10008.1.2 ;;
      LittleEndianExplicit|ExplicitVRLittleEndian) echo 1.2.840.10008.1.2.1 ;;
      BigEndianExplicit|ExplicitVRBigEndian)       echo 1.2.840.10008.1.2.2 ;;
      *) echo "$1" ;;
    esac
  }

  recompress_lossless() {
    local in_dcm="$1" out_tmp="${in_dcm}.tmp"

    if ! command -v gdcmconv >/dev/null 2>&1; then
      echo "  ⚠️  gdcmconv not found; skipping recompress (install package: gdcm-tools)."
      return 0
    fi
  
    # Read basic attributes to pick a codec
    local bits pi spp
    bits="$(dcmdump -q +P 0028,0100 "$in_dcm" 2>/dev/null | awk '{print $NF}')"
    pi="$(dcmdump -q +P 0028,0004 "$in_dcm" 2>/dev/null | awk '{print $NF}' | tr -d '[]' | tr '[:lower:]' '[:upper:]')"
    spp="$(dcmdump -q +P 0028,0002 "$in_dcm" 2>/dev/null | awk '{print $NF}')"
    bits="${bits:-8}"; spp="${spp:-1}"
  
    # Choose a sane, lossless target
    local target=()
    if [[ "$pi" == PAL* ]]; then
      # Palette Color → RLE is the most compatible
      target=(--rle)
    elif [[ "$bits" -gt 8 ]]; then
      # 16-bit paths → JPEG-LS lossless (or J2K lossless if you prefer)
      target=(--jpegls --lossless)
    else
      # 8-bit mono/color → JPEG-LS lossless by default
      target=(--jpegls --lossless)
    fi
  
    if gdcmconv "${target[@]}" "$in_dcm" "$out_tmp" 2>/dev/null; then
      mv -f "$out_tmp" "$in_dcm"
      echo "  🔄 Re-compressed lossless via gdcmconv ${target[*]}"
    else
      rm -f "$out_tmp" 2>/dev/null || true
      echo "  ⚠️  Re-compress failed; keeping uncompressed."
    fi
  }

  local rows cols bits pi frames planar tsuid
  rows="$(tag "$dcm" 0028,0010)"; cols="$(tag "$dcm" 0028,0011)"
  bits="$(tag "$dcm" 0028,0100)"; pi="$(tag "$dcm" 0028,0004)"
  frames="$(tag "$dcm" 0028,0008)"; planar="$(tag "$dcm" 0028,0006)"
  tsuid_raw="$(tag "$dcm" 0002,0010)"; tsuid="$(normalize_tsuid "$tsuid_raw")"
  [[ -z "${frames:-}" ]] && frames=1
  [[ -z "${planar:-}" ]] && planar=0

  echo -e "\n🧩 [$rel_full] $(basename "$dcm")"
  echo "  Rows=$rows Cols=$cols Bits=$bits PI=$pi Frames=$frames Planar=$planar TSUID=$tsuid"

  echo "  Normalized TSUID='$tsuid'"

  # Endianness from TS
  local endian="LE"
  [[ "$tsuid" == "1.2.840.10008.1.2.2" ]] && endian="BE"

  # Allow uncompressed and JPEG Baseline (decode once, no extra loss)
  case "$tsuid" in
    # --- Uncompressed syntaxes ---
    1.2.840.10008.1.2|1.2.840.10008.1.2.1|1.2.840.10008.1.2.2|\
    ImplicitVRLittleEndian|LittleEndianImplicit|\
    ExplicitVRLittleEndian|LittleEndianExplicit|\
    ExplicitVRBigEndian|BigEndianExplicit)
      codec="uncompressed"
      ;;
  
    # --- JPEG Baseline (Process 1) ---
    1.2.840.10008.1.2.4.50|JPEGBaseline|JPEGProcess1)
      codec="jpeg-baseline"
      ;;
  
    # --- JPEG-LS ---
    1.2.840.10008.1.2.4.80|1.2.840.10008.1.2.4.81|\
    JPEGLSLossless|JPEGLSNearLossless)
      codec="jpeg-ls"
      ;;
  
    # --- JPEG 2000 (Lossless/Lossy) ---
    1.2.840.10008.1.2.4.90|1.2.840.10008.1.2.4.91|\
    JPEG2000LosslessOnly|JPEG2000Lossless|JPEG2000|JPEG2000Lossy)
      codec="jpeg2000"
      ;;
  
    # --- RLE ---
    1.2.840.10008.1.2.5|RLELossless)
      codec="rle"
      ;;
  
    # --- Default: unsupported / vendor private ---
    *)
      echo "  ⚠️  SKIP (compressed TSUID=$tsuid not supported yet)."
      return 0
      ;;
  esac

  : "${codec:=__UNSET__}"
  if [[ "$codec" == "__UNSET__" ]]; then
    echo "  ❌ INTERNAL: codec not set; TSUID='$tsuid'"; return 1
  fi


  echo "  Pixel codec: $codec"

  # Compute mask box (detect top black rows by decoding 1 frame ONLY for analysis)
  local box_height box_width
  #box_height="$(detect_black_rows_rgb24_from_dicom "$dcm" "$rows" "$cols" || echo 0)"
  #box_height="$(detect_black_rows_rgb24_from_dicom "$dcm" "$rows" "$cols")"
  # strip any stray non-digits or newlines, default to 0
  box_height="$(detect_black_rows_rgb24_from_dicom "$dcm")" 
  box_height="${box_height//$'\n'/}"
  box_height="${box_height//[^0-9]/}"
  : "${box_height:=0}"
  [[ -z "$box_height" ]] && box_height=0
  # Width heuristic (your existing logic)
  if (( box_height < 40 )); then box_width=$(( cols * 2 / 3 )); else box_width=$(( cols / 3 )); fi
  echo "  Mask box: height=$box_height width=$box_width"

  # For single-frame PNGs, decide if we should blur the whole image
  top_black="$(detect_top_black_rows_rgb24_from_dicom "$dcm")"
  top_black="${top_black//$'\n'/}"; top_black="${top_black//[^0-9]/}"; : "${top_black:=0}"
  
  phi_blur=0
  if [[ "${frames:-1}" -le 1 ]]; then
    if [[ "$top_black" -ge "$BLUR_TOP_BLACK_ROWS" ]]; then
      phi_blur=1
    fi
  fi
  echo "  PHI risk (single-frame): phi_blur=$phi_blur (top_black_rows=$top_black, threshold_rows=$BLUR_TOP_BLACK_ROWS)"

  # ---- Corner-case rules ----
  # Compute near-black coverage (0..1) on first frame
  near_black_pct="$(detect_near_black_pct_from_dicom "$dcm" "$NEAR_BLACK_LUMA" || echo 0.0)"
  near_black_pct="${near_black_pct//[^0-9.eE+-]/}"
  [[ -z "$near_black_pct" ]] && near_black_pct="0.0"
  echo "  Near-black coverage: $(printf '%.3f' "$near_black_pct") (threshold=$(printf '%.3f' "$MIN_NEAR_BLACK_PCT"))"

  # --- Safe integer parsing ---
  # cols may come back with stray chars; coerce to integer >=1
  cols_int="${cols//[^0-9]/}"
  [[ -z "$cols_int" || "$cols_int" -lt 1 ]] && cols_int=1

  # Fraction for left width: clamp to >=1 (default 3)
  fallback_frac="${ULTRASOUND_FALLBACK_FRACW:-2}"
  # keep only digits; if empty or 0, set 3
  fallback_frac="${fallback_frac//[^0-9]/}"
  [[ -z "$fallback_frac" || "$fallback_frac" -lt 1 ]] && fallback_frac=2

  # Fallback mask geometry: 35px tall × left (cols / fallback_frac)
  fallback_boxh="${ULTRASOUND_FALLBACK_BOXH:-35}"
  # coerce height to >=1
  fallback_boxh="${fallback_boxh//[^0-9]/}"
  [[ -z "$fallback_boxh" || "$fallback_boxh" -lt 1 ]] && fallback_boxh=35

  fallback_boxw=$(( cols_int / fallback_frac ))
  (( fallback_boxw < 1 )) && fallback_boxw=1

  # (1) Ultrasound videos with no top band → enforce fallback mask
  if (( frames > 1 )) && (( box_height <= 10 )); then
    echo "  ⚑ Fallback (video,no-top-band): masking ${fallback_boxh}x${fallback_boxw} (top-left)"
    box_height="$fallback_boxh"
    box_width="$fallback_boxw"
  fi

  # (2) Single-frame screenshots/light UIs with low near-black content → fallback
  awk_cmp=$(awk -v a="$near_black_pct" -v b="$MIN_NEAR_BLACK_PCT" 'BEGIN{print (a<b)?"1":"0"}')
  if (( frames <= 1 )) && [[ "$awk_cmp" == "1" ]]; then
    echo "  ⚑ Fallback (single-frame,low near-black): blur image"
    phi_blur=1
  fi

  echo "  Final Mask box (after fallbacks): height=$box_height width=$box_width"

  # Echo final box for transparency
  echo "  Final Mask box (after fallbacks): height=$box_height width=$box_width"

  
  local base; base="$(basename "${dcm%.*}")"
  local out_dcm="$out_dcm_dir/${base}_anon.dcm"
  local out_vid="$out_vid_dir/${base}_anon.mkv"
  local out_png="$out_qc_dir/${base}_anon_mid.png"
  local out_img="$out_img_dir/${base}_anon.png"

  # Skip already-processed unless --overwrite-existing
  if [[ "$OVERWRITE" -eq 0 ]]; then
    if [[ "${frames:-1}" -le 1 ]]; then
      if [[ -s "$out_dcm" && -s "$out_img" ]]; then
        echo "  ⏭  Skip (exists: DCM & IMG)"
        return 0
      fi
    else
      if [[ "$NO_MKV" -eq 1 ]]; then
        # When MKV is disabled, only require DCM + QC PNG
        if [[ -s "$out_dcm" && -s "$out_png" ]]; then
          echo "  ⏭  Skip (exists: DCM & QC PNG; MKV disabled)"
          return 0
        fi
      else
        # Default behavior: require DCM + MKV + QC PNG
        if [[ -s "$out_dcm" && -s "$out_vid" && -s "$out_png" ]]; then
          echo "  ⏭  Skip (exists: DCM & MKV & PNG)"
          return 0
        fi
      fi
    fi
  fi

  # Try to read FrameTime (0018,1063) in ms → fps; default 30
  ft_ms_raw="$(tag "$dcm" 0018,1063 || true)"
  # strip non-numeric (keeps digits and dot)
  ft_ms="${ft_ms_raw//[^0-9.]}"
  if [[ -n "${ft_ms:-}" && "$ft_ms" != "." ]]; then
    fps="$(python3 - "$ft_ms" <<'PY'
import sys
try:
    ft = float(sys.argv[1])
    print(max(1, min(240, round(1000.0/ft))))  # clamp
except Exception:
    print(30)
PY
)"
  else
    fps=30
  fi
  echo "  FPS=$fps (from FrameTime=${ft_ms:-N/A})"

  # Path for rgb24 raw stream (all frames)
  # Keep it alongside QC artifacts so it exists even when MKV is disabled.
  raw_rgb="$out_qc_dir/${base}_anon.rgb24.raw"
  raw_gray16="$out_img.gray16le.raw"

  cleanup_raws() {
    [[ "$KEEP_RAW" -eq 1 ]] && return 0
    local deleted=0
    if [[ -n "${raw_rgb:-}"    && -e "$raw_rgb"    ]]; then rm -f -- "$raw_rgb";    deleted=1; fi
    if [[ -n "${raw_gray16:-}" && -e "$raw_gray16" ]]; then rm -f -- "$raw_gray16"; deleted=1; fi
    [[ $deleted -eq 1 ]] && echo "🧹 cleaned intermediate raw files"
  }
  
  trap cleanup_raws RETURN

  # Call Python to LOSSLESSLY mask native Pixel Data inside DICOM
  ROWS="$rows" COLS="$cols" BITS="$bits" FRAMES="$frames" \
  PI="$pi" TSUID="$tsuid" PLANAR="$planar" ENDIAN="$endian" \
  BOXH="$box_height" BOXW="$box_width" \
  SRC_DCM="$dcm" OUT_DCM="$out_dcm" \
  RAW_OUT="$raw_rgb" \
  RAW_GRAY16_OUT="$raw_gray16" \
  PHI_BLUR="$phi_blur" \
  python3 - <<'PY'
# ========= robust read + decode + geometry normalize + ROI clamp =========
import os, sys
import numpy as np
import pydicom
from pydicom.tag import Tag
from pydicom.uid import UID, ExplicitVRLittleEndian   # <-- fixes NameError

def log(level, msg):
    print(f"[{level}] {msg}", file=sys.stderr)

# --------- Env ----------
src     = os.environ["SRC_DCM"]
outp    = os.environ["OUT_DCM"]
raw_out = os.environ.get("RAW_OUT")
raw_gray16_out = os.environ.get("RAW_GRAY16_OUT")
boxh_env = int(os.environ.get("BOXH", "0") or 0)
boxw_env = int(os.environ.get("BOXW", "0") or 0)

# --------- Read dataset ----------
try:
    ds = pydicom.dcmread(src, force=True, stop_before_pixels=False)
    log("INFO", f"Loaded DICOM: {src}")
except Exception as e:
    log("ERROR", f"Failed to read DICOM: {e}")
    raise

# ---- Collect key tags (as-present) ----
tsuid = str(getattr(ds.file_meta, "TransferSyntaxUID", "") or "")
pi    = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
spp   = int(getattr(ds, "SamplesPerPixel", 1) or 1)
bits  = int(getattr(ds, "BitsAllocated", getattr(ds, "BitsStored", 8)) or 8)
planar= int(getattr(ds, "PlanarConfiguration", 0) or 0)
tag_rows  = int(getattr(ds, "Rows", 0) or 0)
tag_cols  = int(getattr(ds, "Columns", 0) or 0)
frames_attr = getattr(ds, "NumberOfFrames", 1)
try:
    frames = int(str(frames_attr))
except Exception:
    frames = 1

UNCOMP = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"}
compressed = tsuid not in UNCOMP
log("INFO", f"TSUID={tsuid} (compressed={compressed}) PI={pi} SPP={spp} Planar={planar} Bits={bits} Tags={tag_rows}x{tag_cols} FramesTag={frames}")

# ---- Helper: convert any decoded array to RGB uint8 (F, R, C, 3) ----
def to_rgb_uint8(arr):
    a = arr
    if a.ndim == 2:                      # (R,C)
        a = a[None, ...]                 # (1,R,C)
    if a.ndim == 3:
        if a.shape[-1] == 3:             # (R,C,3)
            a = a[None, ...]             # (1,R,C,3)
        else:                            # (F,R,C) mono-ish
            a = a[..., None]             # (F,R,C,1)
    # (F,R,C,S)
    if a.shape[-1] == 1:                 # MONO → RGB
        # If your MONO is 16-bit, handlers often yield uint16—OK to view, but we only
        # use this for preview/video, not for lossless masking of native bytes.
        if a.dtype != np.uint8:
            # scale down only for preview pipeline; native masking stays byte-true below
            a = (a / np.maximum(a.max(), 1)).astype(np.float32) * 255.0
            a = a.astype(np.uint8, copy=False)
        else:
            a = a.astype(np.uint8, copy=False)
        a = np.repeat(a, 3, axis=-1)
    else:
        a = a[..., :3].astype(np.uint8, copy=False)  # drop alpha if present
    return a

# ---- Try to decode pixel data once; measure true decoded dims ----
arr_rgb = None
dec_rows = dec_cols = None
decode_exception = None
try:
    arr_dec = ds.pixel_array
    # Compute decoded H,W,Frames
    if arr_dec.ndim == 2:                 # (R,C)
        dec_rows, dec_cols = arr_dec.shape
        frames_dec = 1
    elif arr_dec.ndim == 3:
        if arr_dec.shape[-1] == 3:        # (R,C,3)
            dec_rows, dec_cols = arr_dec.shape[0], arr_dec.shape[1]
            frames_dec = 1
        else:                             # (F,R,C)
            frames_dec = arr_dec.shape[0]
            dec_rows, dec_cols = arr_dec.shape[1], arr_dec.shape[2]
    elif arr_dec.ndim == 4:               # (F,R,C,S)
        frames_dec = arr_dec.shape[0]
        dec_rows, dec_cols = arr_dec.shape[1], arr_dec.shape[2]
    else:
        raise ValueError(f"Unsupported decoded ndim={arr_dec.ndim}")

    log("INFO", f"Decoded dims={dec_rows}x{dec_cols} frames_dec={frames_dec}")

    # For compressed content, **trust decoder** and normalize tags immediately
    if compressed and ((tag_rows != dec_rows) or (tag_cols != dec_cols)):
        log("WARN", f"Tag dims {tag_rows}x{tag_cols} != decoded {dec_rows}x{dec_cols}; normalizing tags")
        ds.Rows = dec_rows
        ds.Columns = dec_cols
        tag_rows, tag_cols = dec_rows, dec_cols

    if compressed:
        frames = frames_dec

    # Build normalized RGB preview once (used for MKV/PNG only)
    arr_rgb = to_rgb_uint8(arr_dec)
    dec_rows, dec_cols = arr_rgb.shape[1], arr_rgb.shape[2]
    log("INFO", f"arr_rgb ready: shape={arr_rgb.shape}")

except Exception as e:
    decode_exception = e
    log("NOTE", f"Pixel decode failed or inconsistent ({e}); proceeding without arr_rgb")

# ---- Canonical geometry (what we will WRITE) ----
# - COMPRESSED with successful decode: use decoded dims
# - UNCOMPRESSED: keep tag dims (native byte-level edits rely on them)
if compressed and (arr_rgb is not None):
    rows, cols = dec_rows, dec_cols
else:
    rows, cols = tag_rows, tag_cols

if rows <= 0 or cols <= 0:
    log("ERROR", f"Invalid geometry rows={rows} cols={cols}")
    sys.exit(3)

log("INFO", f"Canonical geometry rows={rows} cols={cols} frames={frames}")

# ---- ROI clamp (two passes) ----
# 1) clamp to canonical write geometry
boxw = max(0, min(boxw_env, cols))
boxh = max(0, min(boxh_env, rows))

# 2) we’ll clamp again after we finalize the actual RGB buffer

# ==== Build a *safe* RGB preview buffer we fully own and that matches (frames, rows, cols, 3) ====
F_tag = max(1, int(frames))

if arr_rgb is None:
    # No decoded preview (should only happen for compressed failure path which we already handled)
    arr_rgb = np.zeros((F_tag, rows, cols, 3), dtype=np.uint8)
    log("WARN", f"No decoded arr_rgb; allocated zeros {arr_rgb.shape}")
else:
    F_dec, H_dec, W_dec, _ = arr_rgb.shape
    # Make sure we own the memory and it’s C-contiguous (some handlers return odd views)
    if not arr_rgb.flags['C_CONTIGUOUS'] or arr_rgb.base is not None:
        log("INFO", f"Making arr_rgb contiguous (was contiguous={arr_rgb.flags['C_CONTIGUOUS']}, has_base={arr_rgb.base is not None})")
        arr_rgb = np.ascontiguousarray(arr_rgb, dtype=np.uint8)
    # If tag says more/less frames than decode, trust the decoder’s count for preview
    F_use = F_dec
    # Final owned buffer exactly matching the geometry we’ll write
    arr_safe = np.zeros((F_use, rows, cols, 3), dtype=np.uint8)
    h_copy = min(rows, H_dec)
    w_copy = min(cols, W_dec)
    log("INFO", f"arr_rgb before copy: shape={arr_rgb.shape}, "
                f"C_CONTIG={arr_rgb.flags.c_contiguous} base_is_None={arr_rgb.base is None}")
    log("INFO", f"arr_safe target: shape={arr_safe.shape} h_copy={h_copy} w_copy={w_copy}")
    arr_safe[:, 0:h_copy, 0:w_copy, :] = arr_rgb[:, 0:h_copy, 0:w_copy, :]
    if (H_dec != rows) or (W_dec != cols):
        log("WARN", f"Preview size {H_dec}x{W_dec} differs from write size {rows}x{cols}; copied {h_copy}x{w_copy} and padded with zeros")
    arr_rgb = arr_safe

# Re-clamp ROI to *actual* buffer bounds (defensive)
boxw = max(0, min(boxw, arr_rgb.shape[2]))
boxh = max(0, min(boxh, arr_rgb.shape[1]))
log("INFO", f"Final ROI (h x w) after safe buffer = {boxh} x {boxw}")

# ---- PARACHUTE: make ROI and write geometry impossible to fail ----
def apply_roi_zero_inplace(rgb, bh, bw):
    if bh > 0 and bw > 0:
        rgb[:, 0:int(bh), 0:int(bw), :] = 0

# First attempt
try:
    apply_roi_zero_inplace(arr_rgb, boxh, boxw)
except Exception as e:
    log("ERROR", f"ROI mask failed on arr_rgb={arr_rgb.shape}: {e}")
    # Build an owned, exact-size buffer using current tag geometry as a hint.
    rows_w = max(1, int(getattr(ds, "Rows", arr_rgb.shape[1]) or arr_rgb.shape[1]))
    cols_w = max(1, int(getattr(ds, "Columns", arr_rgb.shape[2]) or arr_rgb.shape[2]))
    padded = np.zeros((arr_rgb.shape[0], rows_w, cols_w, 3), dtype=np.uint8)
    hcpy = min(rows_w, arr_rgb.shape[1]); wcpy = min(cols_w, arr_rgb.shape[2])
    padded[:, 0:hcpy, 0:wcpy, :] = arr_rgb[:, 0:hcpy, 0:wcpy, :]
    arr_rgb = padded
    # Re-clamp ROI and retry once
    boxh = max(0, min(boxh, arr_rgb.shape[1]))
    boxw = max(0, min(boxw, arr_rgb.shape[2]))
    log("WARN", f"Parachute engaged; retrying ROI with (h x w) = {boxh} x {boxw} on {arr_rgb.shape}")
    apply_roi_zero_inplace(arr_rgb, boxh, boxw)

# Enforce write geometry from the buffer we will actually write
rows = arr_rgb.shape[1]
cols = arr_rgb.shape[2]
frames = arr_rgb.shape[0]
log("INFO", f"Write geometry enforced from buffer: rows={rows} cols={cols} frames={frames}")

# Set uncompressed RGB metadata *before* writing PixelData
ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
ds.PhotometricInterpretation = "RGB"
ds.SamplesPerPixel = 3
ds.PlanarConfiguration = 0
ds.BitsAllocated = 8
ds.BitsStored = 8
ds.HighBit = 7
ds.Rows = rows
ds.Columns = cols
if frames > 1:
    ds.NumberOfFrames = str(frames)
elif "NumberOfFrames" in ds:
    del ds.NumberOfFrames

# Ensure contiguous byte buffer
buf = np.ascontiguousarray(arr_rgb, dtype=np.uint8).tobytes(order="C")

# ---- Early bail for compressed-but-not-decodable: keep pass-through (your policy) ----
if compressed and (arr_rgb is None):
    log("ERROR", "Compressed PixelData could not be decoded; writing pass-through")
    pydicom.dcmwrite(outp, ds, write_like_original=True)
    print(f"Wrote DICOM (compressed pass-through; decode failed): {outp}")
    sys.exit(0)

# --------- If flagged to blur the WHOLE image (single-frame PHI risk) ----------
phi_blur = (os.environ.get("PHI_BLUR", "0") == "1")
if phi_blur:
    # We require a decoded array
    if arr_rgb is None:
       print("ERROR: PHI blur requested but pixel decode failed", file=sys.stderr)
       sys.exit(2)

    # Only defined for single frame per your policy
    if frames != 1:
        print("NOTE: PHI_BLUR set but NumberOfFrames != 1; skipping PHI blur path.", file=sys.stderr)
    else:
        # ---- Simple separable box blur (approx Gaussian) on RGB uint8 ----
        import numpy as _np
        def box_blur_channel(ch, k=5, passes=2):
            assert k % 2 == 1, "k must be odd"
            r = k // 2
            # Work in float for accuracy
            out = ch.astype(_np.float32, copy=False)
        
            for _ in range(passes):
                # ---- Horizontal pass (size-preserving) ----
                pad = _np.pad(out, ((0, 0), (r, r)), mode='edge')   # (H, W+2r)
                cs  = pad.cumsum(axis=1, dtype=_np.float64)
                # prefix sums need a leading zero for exact window k outputs
                cs  = _np.concatenate([_np.zeros((cs.shape[0], 1), dtype=cs.dtype), cs], axis=1)
                # window sums for j=0..W-1 → (H, W)
                out = (cs[:, (0 + k):(0 + k) + out.shape[1]] - cs[:, 0:out.shape[1]]) / k
        
                # ---- Vertical pass (size-preserving) ----
                pad = _np.pad(out, ((r, r), (0, 0)), mode='edge')   # (H+2r, W)
                cs  = pad.cumsum(axis=0, dtype=_np.float64)
                cs  = _np.concatenate([_np.zeros((1, cs.shape[1]), dtype=cs.dtype), cs], axis=0)
                out = (cs[(0 + k):(0 + k) + out.shape[0], :] - cs[0:out.shape[0], :]) / k
        
            return out.clip(0, 255).astype(_np.uint8, copy=False)

        img = arr_rgb[0]                # (H,W,3) uint8
        img_blur = _np.empty_like(img)
        img_blur[...,0] = box_blur_channel(img[...,0], k=5, passes=2)
        img_blur[...,1] = box_blur_channel(img[...,1], k=5, passes=2)
        img_blur[...,2] = box_blur_channel(img[...,2], k=5, passes=2)

        if img_blur.shape != img.shape:
            # center-pad or crop to match exactly
            H, W = img.shape[:2]
            hb = min(H, img_blur.shape[0]); wb = min(W, img_blur.shape[1])
            fixed = np.zeros_like(img)
            fixed[:hb, :wb] = img_blur[:hb, :wb]
            img_blur = fixed

        # Replace DICOM PixelData with blurred RGB, write as uncompressed Explicit VR Little Endian
        from pydicom.uid import ExplicitVRLittleEndian
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.Rows = rows
        ds.Columns = cols
        if "NumberOfFrames" in ds:
            del ds.NumberOfFrames  # ensure single-frame

        ds[Tag(0x7fe0,0x0010)] = pydicom.dataelem.DataElement(
            Tag(0x7fe0,0x0010), "OW", img_blur.astype(_np.uint8).tobytes(order="C")
        )
        pydicom.dcmwrite(outp, ds, write_like_original=False)
        print(f"Wrote DICOM (PHI-BLUR single-frame → uncompressed RGB): {outp}")

        # Also emit rgb24 raw to keep downstream PNG generation consistent
        if raw_out:
            try:
                with open(raw_out, "wb") as fh:
                    fh.write(img_blur.reshape(-1,3).tobytes(order="C"))
                print(f"Wrote rgb24 raw (blurred): {raw_out}")
            except Exception as e:
                print(f"ERROR: failed to write blurred rgb24 raw: {e}", file=sys.stderr)

        sys.exit(0)

# --------- UNCOMPRESSED path: keep your PI-specific byte masking & TS ---------
if not compressed:
    # Prepare raw bytearray for native masking (your existing logic)
    pb = ds.PixelData
    pix = bytearray(pb if isinstance(pb, (bytes, bytearray)) else bytes(pb))

    # Helpers
    def put16LE(buf, off, v): buf[off]=v&0xFF; buf[off+1]=(v>>8)&0xFF
    def put16BE(buf, off, v): buf[off]=(v>>8)&0xFF; buf[off+1]=v&0xFF
    big = (UID(tsuid) == UID("1.2.840.10008.1.2.2"))
    put16 = put16BE if big else put16LE

    # Per-frame size by PI
    def frame_bytes(pi_s):
        if pi_s in ("RGB","YBR_FULL"):
            if bits != 8: raise ValueError
            return rows*cols*3
        if pi_s in ("YBR_FULL_422","YBR_PARTIAL_422"):
            if bits != 8: raise ValueError
            return rows*cols*2
        if pi_s == "YBR_PARTIAL_420":
            if bits != 8: raise ValueError
            return rows*cols + (rows//2)*(cols//2)*2
        if pi_s in ("MONOCHROME1","MONOCHROME2"):
            return rows*cols*(1 if bits==8 else 2)
        if pi_s == "PALETTE COLOR":
            if bits != 8: raise ValueError
            return rows*cols
        raise ValueError(f"Unsupported PI={pi_s} Bits={bits}")

    try:
        pfb = frame_bytes(pi)
    except Exception as e:
        print(f"ERROR: Unsupported PI={pi} Bits={bits}", file=sys.stderr); sys.exit(2)

    total = len(pix)
    if frames * pfb != total:
        frames = total // pfb

    # If no mask requested, still produce MKV raw (from arr_rgb) and pass-through write
    if boxw == 0 or boxh == 0:
        if arr_rgb is not None and raw_out:
            try:
                with open(raw_out, "wb") as fh:
                    fh.write(arr_rgb.reshape(-1,3).tobytes(order="C"))
                print(f"Wrote rgb24 raw: {raw_out}")
            except Exception as e:
                print(f"ERROR: failed to write rgb24 raw: {e}", file=sys.stderr)
        pydicom.dcmwrite(outp, ds, write_like_original=True)
        print(f"Wrote DICOM (pass-through): {outp}")
        sys.exit(0)

    # Constants for YBR/mono masking
    YBLACK_FULL    = 0
    YBLACK_PARTIAL = 16
    MID            = 128
    maxmono        = (1<<bits) - 1

    # Byte-level mask (your existing cases)
    for f in range(frames):
        base = f*pfb
        if pi == "RGB":
            if int(planar) == 0:
                row_span = cols*3
                for y in range(boxh):
                    off = base + y*row_span
                    for i in range(off, off+boxw*3, 3):
                        pix[i]=0; pix[i+1]=0; pix[i+2]=0
            else:
                plane_sz = rows*cols
                R0 = base; G0 = base + plane_sz; B0 = base + 2*plane_sz
                for y in range(boxh):
                    for x in range(boxw):
                        p = y*cols + x
                        pix[R0+p] = 0; pix[G0+p] = 0; pix[B0+p] = 0

        elif pi == "YBR_FULL":
            row_span = cols*3
            for y in range(boxh):
                off = base + y*row_span
                for i in range(off, off+boxw*3, 3):
                    pix[i]   = YBLACK_FULL
                    pix[i+1] = MID
                    pix[i+2] = MID

        elif pi in ("YBR_FULL_422","YBR_PARTIAL_422"):
            row_span = cols*2
            yblack = YBLACK_PARTIAL if pi.endswith("_PARTIAL_422") else YBLACK_FULL
            for y in range(boxh):
                off = base + y*row_span
                for x in range(0, cols, 2):
                    m = off + (x//2)*4   # Y0 Cb Y1 Cr
                    in0 = x < boxw
                    in1 = (x+1) < boxw
                    if in0: pix[m+0] = yblack
                    if in1: pix[m+2] = yblack
                    if in0 or in1:
                        pix[m+1] = MID
                        pix[m+3] = MID

        elif pi == "YBR_PARTIAL_420":
            Ysz = rows*cols; Csz = (rows//2)*(cols//2)
            Y0 = base; Cb0 = base+Ysz; Cr0 = base+Ysz+Csz
            yblack = YBLACK_PARTIAL
            for y in range(boxh):
                off = Y0 + y*cols
                for i in range(off, off+boxw):
                    pix[i] = yblack
            cw = cols//2; ch = rows//2
            cw_mask = max(0, min(cw, (boxw+1)//2))
            ch_mask = max(0, min(ch, (boxh+1)//2))
            for cy in range(ch_mask):
                row_cb = Cb0 + cy*cw
                row_cr = Cr0 + cy*cw
                for cx in range(cw_mask):
                    pix[row_cb+cx] = MID
                    pix[row_cr+cx] = MID

        elif pi in ("MONOCHROME2","MONOCHROME1"):
            bps = 1 if bits==8 else 2
            row_span = cols*bps
            black = 0 if pi=="MONOCHROME2" else maxmono
            for y in range(boxh):
                off = base + y*row_span
                for x in range(boxw):
                    p = off + x*bps
                    if bps==1:
                        pix[p] = black & 0xFF
                    else:
                        put16(pix, p, black)

        elif pi == "PALETTE COLOR":
            for y in range(boxh):
                off = base + y*cols
                for i in range(off, off+boxw):
                    pix[i] = 0

        else:
            print(f"ERROR: PI={pi} not supported in masker", file=sys.stderr); sys.exit(2)

    # Write back WITHOUT altering transfer syntax/metadata
    ds[Tag(0x7fe0,0x0010)] = pydicom.dataelem.DataElement(Tag(0x7fe0,0x0010), "OW", bytes(pix))
    pydicom.dcmwrite(outp, ds, write_like_original=True)
    print(f"Wrote DICOM: {outp}")

    # Emit rgb24 raw for MKV from arr_rgb (masked earlier to match)
    if arr_rgb is not None and raw_out:
        try:
            with open(raw_out, "wb") as fh:
                fh.write(arr_rgb.reshape(-1,3).tobytes(order="C"))
            print(f"Wrote rgb24 raw: {raw_out}")
        except Exception as e:
            print(f"ERROR: failed to write rgb24 raw: {e}", file=sys.stderr)

    # If MONO 16-bit, also emit gray16 raw for single-frame PNG fidelity
    if raw_gray16_out and pi in ("MONOCHROME1","MONOCHROME2") and bits == 16:
        try:
            # After masking, pix contains the native bytes; extract first (or all) frame(s)
            total_pixels = rows * cols
            frame_bytes = total_pixels * 2
            # For single-frame PNG we only need the first frame; but writing all is fine.
            with open(raw_gray16_out, "wb") as fh:
                fh.write(pix[:frame_bytes])
            print(f"Wrote gray16 raw: {raw_gray16_out}")
        except Exception as e:
            print(f"ERROR: failed to write gray16 raw: {e}", file=sys.stderr)

    sys.exit(0)

# --------- COMPRESSED path (e.g., JPEG Baseline): decode → mask → write uncompressed RGB ----------
def _write_rgb_uncompressed(ds, rgb, outp, frames_note=""):
    """Write rgb (F,H,W,3) uint8 as uncompressed RGB DICOM (Explicit VR LE)."""
    F, H, W, _ = rgb.shape
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.Rows = H
    ds.Columns = W
    if F > 1:
        ds.NumberOfFrames = str(F)
    elif "NumberOfFrames" in ds:
        del ds.NumberOfFrames
    buf = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(order="C")
    ds[Tag(0x7fe0,0x0010)] = pydicom.dataelem.DataElement(Tag(0x7fe0,0x0010), "OW", buf)
    pydicom.dcmwrite(outp, ds, write_like_original=False)
    print(f"Wrote DICOM (uncompressed RGB{frames_note}): {outp}")

def _ensure_rgb_preview():
    """Return a safe RGB preview array (F,H,W,3) uint8; never None."""
    if arr_rgb is not None:
        return np.ascontiguousarray(arr_rgb, dtype=np.uint8)
    # last-ditch decode
    try:
        dec = ds.pixel_array
        rgb = to_rgb_uint8(dec)
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    except Exception as e:
        # absolute last resort: black frame based on tags
        H = max(1, int(getattr(ds, "Rows", 0) or 0))
        W = max(1, int(getattr(ds, "Columns", 0) or 0))
        F = max(1, int(str(getattr(ds, "NumberOfFrames", 1)) or "1"))
        log("WARN", f"Preview decode failed; emitting black {F}x{H}x{W} ({e})")
        return np.zeros((F, H, W, 3), dtype=np.uint8)

# If no mask requested, still produce MKV raw and write pass-through DICOM
if boxw == 0 or boxh == 0:
    rgb = _ensure_rgb_preview()
    if raw_out:
        try:
            with open(raw_out, "wb") as fh:
                fh.write(rgb.reshape(-1,3).tobytes(order="C"))
            print(f"Wrote rgb24 raw: {raw_out}")
        except Exception as e:
            log("ERROR", f"Failed writing rgb24 raw: {e}")
    # Keep original pixel data when no ROI; just normalize header if needed
    pydicom.dcmwrite(outp, ds, write_like_original=True)
    print(f"Wrote DICOM (pass-through): {outp}")
    sys.exit(0)

# Main compressed masking + write with FORCE fallback
try:
    # Start from a safe owned buffer matching our intended write geometry
    rgb = _ensure_rgb_preview()

    # Align geometry used for writing to this buffer
    rows, cols, frames = rgb.shape[1], rgb.shape[2], rgb.shape[0]

    # Clamp ROI to this buffer (double-defensive)
    bh = max(0, min(int(boxh), rows))
    bw = max(0, min(int(boxw), cols))
    log("INFO", f"Final ROI (h x w) after safe buffer = {bh} x {bw}")

    # Apply ROI (zeros)
    if bh > 0 and bw > 0:
        rgb[:, 0:bh, 0:bw, :] = 0

    # Optional PHI-blur for single-frame reviews (loss is OK per your policy)
    phi_blur = (os.environ.get("PHI_BLUR", "0") == "1")
    if phi_blur and frames == 1:
        import numpy as _np
        def box_blur_channel(ch, k=5, passes=2):
            assert k % 2 == 1, "k must be odd"
            r = k // 2
            # Work in float for accuracy
            out = ch.astype(_np.float32, copy=False)
        
            for _ in range(passes):
                # ---- Horizontal pass (size-preserving) ----
                pad = _np.pad(out, ((0, 0), (r, r)), mode='edge')   # (H, W+2r)
                cs  = pad.cumsum(axis=1, dtype=_np.float64)
                # prefix sums need a leading zero for exact window k outputs
                cs  = _np.concatenate([_np.zeros((cs.shape[0], 1), dtype=cs.dtype), cs], axis=1)
                # window sums for j=0..W-1 → (H, W)
                out = (cs[:, (0 + k):(0 + k) + out.shape[1]] - cs[:, 0:out.shape[1]]) / k
        
                # ---- Vertical pass (size-preserving) ----
                pad = _np.pad(out, ((r, r), (0, 0)), mode='edge')   # (H+2r, W)
                cs  = pad.cumsum(axis=0, dtype=_np.float64)
                cs  = _np.concatenate([_np.zeros((1, cs.shape[1]), dtype=cs.dtype), cs], axis=0)
                out = (cs[(0 + k):(0 + k) + out.shape[0], :] - cs[0:out.shape[0], :]) / k
        
            return out.clip(0, 255).astype(_np.uint8, copy=False)

        img = rgb[0]
        img_blur = np.empty_like(img)
        img_blur[...,0] = box_blur_channel(img[...,0], k=5, passes=2)
        img_blur[...,1] = box_blur_channel(img[...,1], k=5, passes=2)
        img_blur[...,2] = box_blur_channel(img[...,2], k=5, passes=2)

        if img_blur.shape != img.shape:
            # center-pad or crop to match exactly
            H, W = img.shape[:2]
            hb = min(H, img_blur.shape[0]); wb = min(W, img_blur.shape[1])
            fixed = np.zeros_like(img)
            fixed[:hb, :wb] = img_blur[:hb, :wb]
            img_blur = fixed

        rgb[0] = img_blur

    # Always emit raw for MKV even if DICOM write later fails
    if raw_out:
        try:
            with open(raw_out, "wb") as fh:
                fh.write(rgb.reshape(-1,3).tobytes(order="C"))
            print(f"Wrote rgb24 raw: {raw_out}")
        except Exception as e:
            log("ERROR", f"Failed writing rgb24 raw: {e}")

    # Write uncompressed RGB DICOM (decoded→masked→uncompressed)
    _write_rgb_uncompressed(ds, rgb, outp, frames_note=f", frames={frames}")

except Exception as e:
    # ===== FORCE OUTPUT SAFETY NET =====
    log("ERROR", f"Compressed path failed; forcing output: {e}")

    # Rebuild a guaranteed-good buffer based on tags or decoded dims
    try:
        dec = ds.pixel_array
        rgb = to_rgb_uint8(dec)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    except Exception:
        H = max(1, int(getattr(ds, "Rows", 0) or 0))
        W = max(1, int(getattr(ds, "Columns", 0) or 0))
        F = max(1, int(str(getattr(ds, "NumberOfFrames", 1)) or "1"))
        rgb = np.zeros((F, H, W, 3), dtype=np.uint8)

    # Pad/crop to a sane even-aligned size (helps some codecs/viewers)
    F, H, W, _ = rgb.shape
    He = H - (H % 2)
    We = W - (W % 2)
    if He == 0: He = 2
    if We == 0: We = 2
    if (He,We) != (H,W):
        tmp = np.zeros((F, He, We, 3), dtype=np.uint8)
        tmp[:, 0:min(He,H), 0:min(We,W), :] = rgb[:, 0:min(He,H), 0:min(We,W), :]
        rgb = tmp
        H, W = He, We
        log("WARN", f"FORCE: resized preview to even dims {H}x{W}")

    # Clamp and apply ROI again (never crash)
    bh = max(0, min(int(boxh), H))
    bw = max(0, min(int(boxw), W))
    if bh > 0 and bw > 0:
        rgb[:, 0:bh, 0:bw, :] = 0

    # Emit raw no matter what
    if raw_out:
        try:
            with open(raw_out, "wb") as fh:
                fh.write(rgb.reshape(-1,3).tobytes(order="C"))
            print(f"[FORCE] Wrote rgb24 raw: {raw_out}")
        except Exception as e2:
            log("ERROR", f"[FORCE] Failed writing rgb24 raw: {e2}")

    # Emit DICOM as uncompressed RGB so reviewing tools can open it
    try:
        _write_rgb_uncompressed(ds, rgb, outp, frames_note=f" [FORCE], frames={rgb.shape[0]}")
        print(f"[FORCE] Wrote DICOM for review: {outp}")
    except Exception as e3:
        # Absolute last resort: pass-through original (still produces MKV from raw)
        log("ERROR", f"[FORCE] DICOM write failed; falling back to pass-through: {e3}")
        pydicom.dcmwrite(outp, ds, write_like_original=True)
        print(f"[FORCE] Wrote DICOM (pass-through): {outp}")

# Emit rgb24 raw for MKV
if arr_rgb is not None and raw_out:
    try:
        with open(raw_out, "wb") as fh:
            fh.write(arr_rgb.reshape(-1,3).tobytes(order="C"))
        print(f"Wrote rgb24 raw: {raw_out}")
    except Exception as e:
        print(f"ERROR: failed to write rgb24 raw: {e}", file=sys.stderr)
PY

  recompress_lossless "$out_dcm"

  rows="$(tag "$out_dcm" 0028,0010)"
  cols="$(tag "$out_dcm" 0028,0011)"
  frames="$(tag "$out_dcm" 0028,0008)"; [[ -z "${frames:-}" ]] && frames=1

  # Build outputs depending on frame count
  if [[ "${frames:-1}" -le 1 ]]; then
    # Single-frame: either blur the entire image (PHI risk) or write lossless PNG.
    if [[ "$phi_blur" -eq 1 ]]; then
      # Prefer rgb24 raw if present; if not, fall back to gray16->rgb24 in ffmpeg.
      if [[ -s "$raw_rgb" ]]; then
        echo "  ffmpeg: BLUR PNG from rgb24 raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" \
          -i "$raw_rgb" \
          -vf "gblur=sigma=10" -frames:v 1 -pix_fmt rgb24 "$out_img"
      elif [[ -s "$raw_gray16" ]]; then
        echo "  ffmpeg: BLUR PNG from gray16le raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format gray16le -video_size "${cols}x${rows}" \
          -i "$raw_gray16" \
          -vf "format=rgb24,gblur=sigma=10" -frames:v 1 -pix_fmt rgb24 "$out_img"
      else
        echo "  ⚠️  No raw produced; skipping image"
      fi
    else
      # Not PHI-risk: write lossless (preserve 16-bit mono when available)
      if [[ -s "$raw_gray16" ]]; then
        echo "  ffmpeg: making 16-bit PNG from gray16le raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format gray16le -video_size "${cols}x${rows}" \
          -i "$raw_gray16" \
          -frames:v 1 -pix_fmt gray16le "$out_img"
      elif [[ -s "$raw_rgb" ]]; then
        echo "  ffmpeg: making PNG from rgb24 raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" \
          -i "$raw_rgb" \
          -frames:v 1 -pix_fmt rgb24 "$out_img"
      else
        echo "  ⚠️  No raw produced; skipping image"
      fi
    fi
   
    # For single-frame, we skip MKV and QC (png is the deliverable)
    echo "  ✅ Wrote:"
    echo "     - $out_dcm"
    [[ -s "$out_img" ]] && echo "     - $out_img"
  else
    # Multi-frame → MKV + QC (unless --no-mkv)
    out_vid="$out_vid_dir/${base}_anon.mkv"

    if [[ -s "$raw_rgb" ]]; then
      if [[ "$NO_MKV" -eq 0 ]]; then
        echo "  ffmpeg: making lossless video (${cols}x${rows} @ ${fps}fps) -> $VID_CODEC"
        if [[ "$VID_CODEC" == "ffv1" ]]; then
          ffmpeg -hide_banner -loglevel error -nostdin -y \
            -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" -r "$fps" \
            -i "$raw_rgb" \
            -c:v ffv1 -level 3 -g 1 -coder 1 -context 1 -slices 24 -slicecrc 1 \
            -pix_fmt rgb24 \
            "$out_vid"
        else
          ffmpeg -hide_banner -loglevel error -nostdin -y \
            -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" -r "$fps" \
            -i "$raw_rgb" \
            -c:v libx264rgb -crf 0 -preset veryslow -pix_fmt rgb24 \
            "$out_vid"
        fi
      else
        echo "  ⏭  MKV generation disabled (--no-mkv); skipping .mkv output"
      fi
    else
      echo "  ⚠️  No rgb24 raw produced; skipping video"
    fi

    # Middle frame for QC PNG
    local mid=0
    if [[ -n "$frames" && "$frames" =~ ^[0-9]+$ && "$frames" -gt 0 ]]; then
      mid=$(( (frames - 1) / 2 ))
    fi

    if [[ -s "$raw_rgb" ]]; then
      if [[ "$NO_MKV" -eq 0 ]]; then
        if [[ -s "$out_vid" ]]; then
          ffmpeg -hide_banner -loglevel error -nostdin -y \
            -i "$out_vid" \
            -vf "select=eq(n\,${mid})" -frames:v 1 "$out_png"
        else
          echo "  ⚠️  No video; skipping QC PNG"
        fi
      else
        echo "  ffmpeg: making QC PNG from rgb24 raw (no MKV) (${cols}x${rows} @ frame ${mid})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" -r "$fps" \
          -i "$raw_rgb" \
          -vf "select=eq(n\,${mid})" -frames:v 1 "$out_png"
      fi
    else
      echo "  ⚠️  No rgb24 raw; skipping QC PNG"
    fi

    echo "  ✅ Wrote:"
    echo "     - $out_dcm"
    if [[ "$NO_MKV" -eq 0 && -s "$out_vid" ]]; then
      echo "     - $out_vid"
    fi
    [[ -s "$out_png" ]] && echo "     - $out_png"
  fi

}

# ---------- Worker re-entry ----------
if [[ "${1:-}" == "--process-one" ]]; then
  shift
  process_one "$@"
  exit 0
fi

# ---------- CLI ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input-dir)  INPUT_DIR="$2"; shift 2;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2;;
    -n|--num-threads) NUM_THREADS="$2"; shift 2;;
    --overwrite-existing|-O) OVERWRITE=1; shift 1;;
    --no-mkv|-V) NO_MKV=1; shift 1;;
    -h|--help) usage;;
    *) echo "Unknown argument: $1" >&2; usage;;
  esac
done

[[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]] && { echo "❌ --input-dir and --output-dir are required."; usage; }
[[ ! -d "$INPUT_DIR" ]] && { echo "❌ Input directory not found: $INPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

echo "🔎 Input : $INPUT_DIR"
echo "💾 Output: $OUTPUT_DIR"
echo "🚀 Threads: $NUM_THREADS"

SCRIPT_PATH="$(readlink -f "$0")"

# ---------- Parallel dispatcher (streaming; no SIGPIPE aborts) ----------
export PYTHONUNBUFFERED=1
export NO_MKV
export -f process_one detect_black_rows_rgb24_from_dicom detect_top_black_rows_rgb24_from_dicom detect_near_black_pct_from_dicom

# Save & disable pipefail only for this pipeline
_prev_pipefail="$(set -o | awk '/pipefail/{print $3}')"
set +o pipefail
# If available, make the last pipeline command run in current shell:
shopt -s lastpipe 2>/dev/null || true

# FOR TESTING
#| grep -zE 'dvzhsi|dwjadv|ebxizz' \
#| grep -zE '25414729|36886005' \
#| grep -zE '15831723493|2041430769|21079754628' \
find "$INPUT_DIR" -type f -name '*.dcm' -print0 \
| xargs -0 -r -n 1 -P "$NUM_THREADS" -I{} bash -c '
  file="$1"; inroot="$2"; outroot="$3"
  echo ">>> START  $file"
  if process_one "$file" "$inroot" "$outroot"; then
    echo "<<< END OK  $file"
    exit 0
  else
    rc=$?
    echo "<<< END FAIL[$rc]  $file"
    exit "$rc"
  fi
' _ {} "$INPUT_DIR" "$OUTPUT_DIR"

# Restore previous pipefail state
[[ "$_prev_pipefail" = "on" ]] && set -o pipefail

