import os
import pandas as pd
import pydicom
from pathlib import Path


def create_strict_unique_full_mammo():
    """create strict unique full mammogram metadata by matching case descriptions with meta.csv"""

    meta_df = pd.read_csv('csv/meta.csv')

    # read case description CSVs
    calc_train = pd.read_csv('csv/calc_case_description_train_set.csv')
    calc_test = pd.read_csv('csv/calc_case_description_test_set.csv')
    mass_train = pd.read_csv('csv/mass_case_description_train_set.csv')
    mass_test = pd.read_csv('csv/mass_case_description_test_set.csv')

    # standardize column names
    case_dfs = []
    for df, ab_type, split in [(calc_train, 'calcification', 'train'),
                               (calc_test, 'calcification', 'test'),
                               (mass_train, 'mass', 'train'),
                               (mass_test, 'mass', 'test')]:
        df = df.copy()
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        df['abnormality_category'] = ab_type
        df['dataset_split'] = split
        case_dfs.append(df)

    combined_cases = pd.concat(case_dfs, ignore_index=True)

    # use a set to track processed case_id + view_position combinations
    processed_combinations = set()
    metadata_list = []

    for idx, case_row in combined_cases.iterrows():
        subject_id = case_row['image_file_path'].split('/')[0]  # eg: "Calc-Training_P_00005_RIGHT_CC"

        # extract case ID and view position
        case_id = case_row['patient_id']
        view_position = case_row['image_view']
        combination = f"{case_id}_{view_position}"  # eg: "P_00005_CC"

        # if already processed, skip
        if combination in processed_combinations:
            continue

        # find matching Subject ID in meta.csv
        matching_meta = meta_df[meta_df['Subject ID'] == subject_id]

        if len(matching_meta) > 0:
            # strict filtering: must be full mammogram and pass path validation
            full_mammo_candidates = []
            for _, meta_row in matching_meta.iterrows():
                if 'full mammogram' in meta_row['Series Description'].lower():
                    file_location = meta_row['File Location']
                    actual_path = find_and_validate_full_mammo(file_location, subject_id)

                    if actual_path and os.path.exists(actual_path):
                        # verify this is indeed a full mammogram (not ROI)
                        if validate_is_actually_full_mammo(actual_path):
                            full_mammo_candidates.append((meta_row, actual_path))

            # take only one full mammogram per case + view
            if full_mammo_candidates:
                meta_row, actual_path = full_mammo_candidates[0]  # take the first one

                record = {
                    'case_id': case_id,
                    'view_position': view_position,
                    'image_path': actual_path,
                    'pathology': case_row['pathology'],
                    'breast_density': case_row['breast_density'],
                    'left_or_right_breast': case_row['left_or_right_breast'],
                    'abnormality_id': case_row['abnormality_id'],
                    'abnormality_type': case_row['abnormality_type'],
                    'abnormality_category': case_row['abnormality_category'],
                    'assessment': case_row['assessment'],
                    'subtlety': case_row['subtlety'],
                    'dataset_split': case_row['dataset_split'],
                    'series_description': meta_row['Series Description'],
                    'file_location': file_location
                }
                metadata_list.append(record)
                processed_combinations.add(combination)

                if len(metadata_list) % 500 == 0:
                    print(f"already completed {len(metadata_list)} unique full mammograms")

    metadata_df = pd.DataFrame(metadata_list)

    print(f"\n=== strict unique matching results ===")
    print(f"unique full mammograms: {len(metadata_df)}")
    print(f"theoretical maximum (full mammograms in meta.csv): 3103")

    if len(metadata_df) > 0:
        print(f"pathology distribution:\n{metadata_df['pathology'].value_counts()}")
        print(f"view position distribution:\n{metadata_df['view_position'].value_counts()}")

    metadata_df.to_csv('strict_unique_full_mammo.csv', index=False)
    return metadata_df


def validate_is_actually_full_mammo(file_path):
    """validate file is indeed full mammogram and not ROI mask"""
    try:
        import pydicom
        ds = pydicom.dcmread(file_path)

        # Check 1: Series Description
        series_desc = getattr(ds, 'SeriesDescription', '').lower()
        if 'roi' in series_desc or 'mask' in series_desc:
            return False

        # Check 2: Image Size (full mammograms are usually larger)
        rows = getattr(ds, 'Rows', 0)
        cols = getattr(ds, 'Columns', 0)
        if rows < 1000 or cols < 1000:  # ROI masks are usually smaller
            return False

        # Check 3: File Path
        file_path_lower = file_path.lower()
        if 'roi' in file_path_lower or 'mask' in file_path_lower:
            return False

        return True

    except Exception as e:
        print(f"validation failed {file_path}: {e}")
        return False


def find_and_validate_full_mammo(file_location, subject_id):
    """find and validate full mammogram file"""

    # remove leading "./"
    if file_location.startswith('./'):
        file_location = file_location[2:]

    dicom_root = './dicom_data'
    full_dir_path = os.path.join(dicom_root, file_location)

    if os.path.exists(full_dir_path) and os.path.isdir(full_dir_path):
        for file in os.listdir(full_dir_path):
            if file.endswith('.dcm'):
                full_path = os.path.join(full_dir_path, file)
                if validate_is_actually_full_mammo(full_path):
                    return full_path

    return None


# run strict matching
strict_metadata = create_strict_unique_full_mammo()