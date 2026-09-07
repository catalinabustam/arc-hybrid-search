"""Build the "expanded" variant of the ARC catalog.

`user_list` / `multilist` rows get exploded into one row per list item, and
`radio` / `checkbox` rows get their cleaned answer options appended to the
`Question` text. This is what makes the "expanded" catalog match individual
list items and options directly, at the cost of no longer being a 1:1 copy
of the published ARC catalog (see the "raw" catalog for that).
"""

import os
import re

import pandas as pd


def create_expanded_arc_dataframe(arc_df: pd.DataFrame, lists_path: str) -> pd.DataFrame:
    """Expand `arc_df`, replacing list questions with individual list items.

    Parameters
    ----------
    arc_df : the raw ARC dataframe (as downloaded from `ARC.csv`).
    lists_path : directory containing the downloaded `ARC_Lists` subfolders
        (see `download.download_arc_lists`).
    """
    expanded_rows = []

    for _, row in arc_df.iterrows():
        question_type = str(row["Type"]).strip().lower()
        base_question = str(row["Question"]).strip()

        if question_type in ("user_list", "multilist") and pd.notna(row["List"]):
            list_identifier = str(row["List"]).strip()
            folder, file_name = list_identifier.split("_", 1)
            file_path = os.path.join(lists_path, folder, f"{file_name}.csv")

            if os.path.exists(file_path):
                list_df = pd.read_csv(file_path)
                items = list_df.iloc[:, 0].dropna().astype(str).tolist()
                for item in items:
                    new_row = row.copy()
                    new_row["Question"] = f"{base_question}, {file_name}: {item}"
                    expanded_rows.append(new_row)
            else:
                print(
                    f"Warning: List file '{file_path}' not found. Skipping expansion for this row."
                )
                expanded_rows.append(row)
        elif question_type in ("radio", "checkbox"):
            options = str(row["Answer Options"]).split("|")
            cleaned_options = [
                cleaned
                for option in options
                if (
                    cleaned := re.sub(
                        r"\d+|,|unknown|yes|no", "", option, flags=re.IGNORECASE
                    ).strip()
                )
            ]
            options_suffix = ", Options: " + ", ".join(cleaned_options) if cleaned_options else ""
            row["Question"] = row["Question"] + options_suffix
            expanded_rows.append(row)
        else:
            expanded_rows.append(row)

    return pd.DataFrame(expanded_rows).reset_index(drop=True)
