import logging
import random
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from rapidfuzz.distance.Levenshtein import distance as lev_dist
from rdkit.ML.Cluster.Butina import ClusterData

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M",
    format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
)


def cluster_seqs_in_df_by_lev_dist(
    df: DataFrame,
    dist_threshold=4,
    min_extra_sample_num=1,
    max_extra_sample_num=200,
    big_cluster_threshold=50,
    huge_cluster_threshold=100,
    extra_sample_ratio_for_huge_cluster=0.01,
    out_file=None,
    seq_col_name="Sequence",
    random_seed=1,
) -> DataFrame:
    """
    https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html

    Args:
        dist_threshold: if dist_threshold value is too small, the clusters may be too noisy.
        Advice dist_threshold as min(int(0.3 * min_seq_len), 3), that's >= 3. 
        If dist_threshold = 3 get less seqs, advice to generate more seqs and try again.
        e.g. the input len(df) num is 46533, min_len=8, max_len=10, with default configs,
        such as dist_threshold=4 etc., the output len(selected_df) is 975.

        extra_sample_num e.g.\n
        If a cluster has 45 samples, less than big_cluster_threshold, no extra sample, 
        just the first id in the cluster.
        If a cluster has 50 samples, equal or more than big_cluster_threshold, but less
        than huge_cluster_threshold, min_extra_sample_num is sampled.
        If a cluster has 300 samples, more than huge_cluster_threshold, extra sample num
        300 * 0.01 = 3
        If a cluster has 20000 samples, more than huge_cluster_threshold, extra sample 
        num is min(20000 * 0.01, max_extra_sample_num), that's 100 is sampled.
    
    """
    logger.info(
        f"Run cluster\ndist_threshold {dist_threshold}\n"
        f"min_extra_sample_num {min_extra_sample_num}\n"
        f"max_extra_sample_num {max_extra_sample_num}\n"
        f"big_cluster_threshold {big_cluster_threshold}\n"
        f"huge_cluster_threshold {huge_cluster_threshold}\n"
        f"extra_sample_ratio_for_huge_cluster {extra_sample_ratio_for_huge_cluster}"
    )
    clusters = ClusterData(
        df[seq_col_name].tolist(), len(df), dist_threshold, distFunc=lev_dist
    )
    logger.info(f"len(clusters) {len(clusters)}")
    idx = []
    random.seed(random_seed)

    big_cluster_num = 0
    for cluster in clusters:
        cluster_num = len(cluster)
        if cluster_num > 0:
            idx.append(cluster[0])

        if cluster_num >= big_cluster_threshold:
            extra_sample_num = min_extra_sample_num
            if cluster_num >= huge_cluster_threshold:
                extra_sample_num = max(
                    int(cluster_num * extra_sample_ratio_for_huge_cluster), min_extra_sample_num
                )
                extra_sample_num = min(max_extra_sample_num, extra_sample_num)
            logger.info(
                f"big_cluster len(cluster) {len(cluster)}, extra_sample_num {extra_sample_num}"
            )
            big_cluster_num += 1
            cluster_pos_tmp = list(cluster)
            cluster_pos_tmp.remove(cluster[0])
            idx_plus = random.sample(cluster_pos_tmp, extra_sample_num)
            idx.extend(idx_plus)

    selected_df = df.iloc[idx].copy().reset_index(drop=True)
    logger.info(
        f"len(selected_df) {len(selected_df)}, big_cluster_num {big_cluster_num}"
    )
    if out_file:
        selected_df.to_csv(out_file, index=False, sep=",")

    return selected_df


def length_filter(df, min_len=8, max_len=10):
    """ """
    logger.info(f"Pre run seq length filter, min_len={min_len}, max_len={max_len}")
    if "len" not in df.columns:
        df["len"] = df["Sequence"].map(len)
    df = df[(df["len"] >= min_len) & (df["len"] <= max_len)]
    logger.info(f"Seq length describe\n{df['len'].describe()}")
    logger.info(f"Post run seq length filter, len(df)={len(df)}")
    return df


if __name__ == "__main__":
    data_dir = Path(
        "/mnt/nas1/bio_drug_corpus/peptides/cyclic/GA_generation/PRLR/PRLR-classifier_prediction"
    )
    file = data_dir / "PRLR-generated_PRLR_all_pos.csv"
    input_df = pd.read_csv(file)
    logger.info(f"input file: {file}, len(input_df): {len(input_df)}")
    input_df = length_filter(input_df, min_len=8, max_len=10)  # optional

    cluster_seqs_in_df_by_lev_dist(
        input_df,
        out_file=file.with_stem(f"{file.stem}_min_len8_distThresh5"),
    )
