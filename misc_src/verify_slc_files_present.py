import os
from pathlib import Path

SLC_PUBLIC_PATH = Path("/media/joe/Elements/jc_work/Copernicus/SLC_Public")
REQUIRED_MEASUREMENT = [
    ("-iw1-", "-vh-"),
    ("-iw1-", "-vv-"),
    ("-iw2-", "-vh-"),
    ("-iw2-", "-vv-"),
    ("-iw3-", "-vh-"),
    ("-iw3-", "-vv-"),
]
REQUIRED_ANNOTATION = REQUIRED_MEASUREMENT
TIFF_MIN_SIZE = 500 * 1024 * 1024  # 500MB in bytes

def find_subdirs_with_names(root, required_names):
    """Find all subdirectories under root that match required_names."""
    found = {name: [] for name in required_names}
    for subdir in root.rglob("*"):
        if subdir.is_dir() and subdir.name in required_names:
            found[subdir.name].append(subdir)
    return found

def check_measurement_dir(measurement_dir):
    errors = []
    tiff_files = [f for f in measurement_dir.glob("*.tiff") if f.is_file()]
    if len(tiff_files) < 6:
        errors.append(f"Less than 6 .tiff files found ({len(tiff_files)})")
    for iw, pol in REQUIRED_MEASUREMENT:
        matches = [f for f in tiff_files if iw in f.name and pol in f.name and f.stat().st_size > TIFF_MIN_SIZE]
        if not matches:
            errors.append(f"Missing .tiff file with '{iw}' and '{pol}' and >500MB")
    return errors

def check_annotation_dir(annotation_dir):
    errors = []
    xml_files = [f for f in annotation_dir.glob("*.xml") if f.is_file()]
    if len(xml_files) < 6:
        errors.append(f"Less than 6 .xml files found ({len(xml_files)})")
    for iw, pol in REQUIRED_ANNOTATION:
        matches = [f for f in xml_files if iw in f.name and pol in f.name]
        if not matches:
            errors.append(f"Missing .xml file with '{iw}' and '{pol}'")
    return errors

def main():
    log = []
    for subdir in SLC_PUBLIC_PATH.iterdir():
        if not subdir.is_dir():
            continue
        found = find_subdirs_with_names(subdir, ["measurement", "annotation"])
        measurement_dirs = found["measurement"]
        annotation_dirs = found["annotation"]
        if not measurement_dirs or not annotation_dirs:
            log.append(f"{subdir}: Missing 'measurement' or 'annotation' subdirectory")
            continue
        # Check all found measurement and annotation dirs
        for m_dir in measurement_dirs:
            m_errors = check_measurement_dir(m_dir)
            if m_errors:
                log.append(f"{subdir} -> {m_dir}: " + "; ".join(m_errors))
        for a_dir in annotation_dirs:
            a_errors = check_annotation_dir(a_dir)
            if a_errors:
                log.append(f"{subdir} -> {a_dir}: " + "; ".join(a_errors))
    # Write the log
    with open("slc_public_check_log.txt", "w") as f:
        for line in log:
            f.write(line + "\n")
    print("Check complete. See slc_public_check_log.txt for results.")

if __name__ == "__main__":
    main()

