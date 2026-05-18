import pandas as pd
import os
import uuid
import random

def mergeTrain(list_of_file, output_file):
    result = []
    for file in list_of_file:
        df = pd.read_parquet(file)
        result.append(df)
    print(len(result))
    merged_df = pd.concat(result, ignore_index=True).sample(frac=1).reset_index(drop=True)
    print("Merge complete")
    if not os.path.exists(output_file):
        merged_df.to_parquet(output_file)
    else:
        new_output_path = output_file.replace(".parquet", f"_{uuid.uuid4().hex[:8]}.parquet")
        merged_df.to_parquet(new_output_path)

def mergeValidation(list_negative_file, positive_file, len_need, month, scale, output_file):
    result = []

    pos_df = pd.read_parquet(positive_file)
    pos_df["rounded_dt_up"] = pd.to_datetime(pos_df["rounded_dt_up"])

    pos_df = pos_df.loc[pos_df["rounded_dt_up"].dt.month == month]

    if len(pos_df) < len_need:
        raise ValueError(f"Not enough positive samples: {len(pos_df)} < {len_need}")

    pos_df = pos_df.sample(n=len_need, random_state=42)
    result.append(pos_df)

    total_need = int(len(pos_df) * scale)

    print("len pos_df:", len(pos_df))
    print("total_need:", total_need)

    for neg_file in list_negative_file:
        if total_need <= 0:
            break

        neg_df = pd.read_parquet(neg_file)

        if len(neg_df) <= total_need:
            result.append(neg_df)
            total_need -= len(neg_df)
        else:
            result.append(neg_df.sample(n=total_need, random_state=42))
            total_need = 0

    if total_need > 0:
        print(f"Warning: thiếu {total_need} negative samples")

    print("Take negative complete")

    merged_df = pd.concat(result, ignore_index=True).sample(frac=1).reset_index(drop=True)

    if not os.path.exists(output_file):
        print("Saving to:", output_file)
        merged_df.to_parquet(output_file)
    else:
        new_output_path = output_file.replace(".parquet", f"_{uuid.uuid4().hex[:8]}.parquet")
        print("Saving to:", new_output_path)
        merged_df.to_parquet(new_output_path)

if __name__ == "__main__":
    input_files = [
        "/sdd/Dubaoset/src/Thang/DataMB/Test/pos/train_new/batch_0001_from_00000_to_00000_new.parquet", # Pos
        "/sdd/Dubaoset/src/Thang/DataMB/Test/train/batch_0001_from_00000_to_00024.parquet" # Neg
    ]
    output_file = "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/train_dataset.parquet"
    mergeTrain(input_files, output_file)
    # list_negative_file = [os.path.join("/sdd/Dubaoset/src/Thang/DataMB/Test/6", f) for f in os.listdir("/sdd/Dubaoset/src/Thang/DataMB/Test/6")]
    # random.shuffle(list_negative_file)
    # positive_file = "/sdd/Dubaoset/src/Thang/DataMB/Test/pos/test/batch_0001_from_00000_to_00000.parquet"
    # len_need = 10000 # 295698
    # month = 6
    # scale = 544
    # output_file = "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/validation_dataset_10000.parquet"
    # mergeValidation(list_negative_file, positive_file, len_need, month, scale, output_file)