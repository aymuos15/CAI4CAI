import cc3d
import numpy as np
import pandas as pd
from brats import dice

def collect_legacy_metrics(pred_label_cc, gt_label_cc):
    legacy_metrics = []
    tp = []

    for gtcomp in range(1, np.max(gt_label_cc) + 1):
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1

        intersecting_cc = np.unique(pred_label_cc * gt_tmp)
        intersecting_cc = intersecting_cc[intersecting_cc != 0]
        for cc in intersecting_cc:
            tp.append(cc)

        if len(intersecting_cc) == 0:
            legacy_metrics.append({'GT': gtcomp, 'Pred': 0, 'Dice': 0})
        else:
            for predcomp in intersecting_cc:
                pred_tmp = np.zeros_like(pred_label_cc)
                pred_tmp[pred_label_cc == predcomp] = 1

                legacy_metrics.append({'GT': gtcomp, 'Pred': predcomp, 'Dice': dice(pred_tmp, gt_tmp)})

    fp = np.unique(pred_label_cc[np.isin(pred_label_cc, tp + [0], invert=True)])

    return legacy_metrics, fp

def find_overlapping_components(prediction_cc, gt_cc):
    overlapping_components = {}

    for i in range(prediction_cc.shape[0]):
        for j in range(prediction_cc.shape[1]):
            prediction_component = prediction_cc[i, j]
            gt_component = gt_cc[i, j]

            if prediction_component != 0 and gt_component != 0:
                if prediction_component not in overlapping_components:
                    overlapping_components[prediction_component] = set()
                overlapping_components[prediction_component].add(gt_component)

    overlapping_components = {k: v for k, v in overlapping_components.items() if len(v) > 1}

    return overlapping_components


def find_overlapping_components_inverse(prediction_cc, gt_cc):
    overlapping_components = {}

    for i in range(prediction_cc.shape[0]):
        for j in range(prediction_cc.shape[1]):
            prediction_component = prediction_cc[i, j]
            gt_component = gt_cc[i, j]

            if prediction_component != 0 and gt_component != 0:
                if gt_component not in overlapping_components:
                    overlapping_components[gt_component] = set()
                overlapping_components[gt_component].add(prediction_component)

    overlapping_components = {k: v for k, v in overlapping_components.items() if len(v) > 1}

    return overlapping_components


def generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components):
    overlap_metrics = []

    for pred_components, gt_components in overlapping_components.items():
        gtcomps = list(gt_components)

        pred_label_cc_tmp = pred_label_cc.copy()
        gt_label_cc_tmp = gt_label_cc.copy()

        pred_cc_tmp = (pred_label_cc_tmp == pred_components).astype(int)
        gt_cc_tmp = np.array([[1 if (x > 0 and x in gtcomps) else 0 for x in sublist] for sublist in gt_label_cc_tmp])

        overlap_metrics.append({'GT': gtcomps, 'Pred': pred_components, 'Dice': dice(pred_cc_tmp, gt_cc_tmp)})

    return overlap_metrics


def generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components):
    overlap_metrics = []

    for gt_components, pred_components in overlapping_components.items():
        predcomps = list(pred_components)

        pred_label_cc_tmp = pred_label_cc.copy()
        gt_label_cc_tmp = gt_label_cc.copy()

        gt_cc_tmp = (gt_label_cc_tmp == gt_components).astype(int)
        pred_cc_tmp = np.array([[1 if (x > 0 and x in predcomps) else 0 for x in sublist] for sublist in pred_label_cc_tmp])

        overlap_metrics.append({'GT': gt_components, 'Pred': predcomps, 'Dice': dice(pred_cc_tmp, gt_cc_tmp)})

    return overlap_metrics


def collect_metrics(pred_label_cc, gt_label_cc, overlapping_components):
    legacy_metrics, fp = collect_legacy_metrics(pred_label_cc, gt_label_cc)
    legacy_metrics = pd.DataFrame(legacy_metrics)
    overlap_metrics = generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components)
    overlap_metrics = pd.DataFrame(overlap_metrics)

    initial_metrics_df = pd.concat([legacy_metrics, overlap_metrics], ignore_index=True)

    pred_list = []
    pred_int = []

    for _, row in initial_metrics_df.iterrows():
        if isinstance(row['GT'], list):
            pred_list.append(row['Pred'])
        if isinstance(row['GT'], int):
            pred_int.append(row['Pred'])

    common_elements = set(pred_list).intersection(set(pred_int))

    final_metrics_df = initial_metrics_df.copy()

    for index, row in final_metrics_df.iterrows():
        if row['Pred'] in common_elements and isinstance(row['GT'], int):
            final_metrics_df.drop(index, inplace=True)

    return initial_metrics_df, fp


def collect_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components):
    legacy_metrics, fp = collect_legacy_metrics(pred_label_cc, gt_label_cc)
    legacy_metrics = pd.DataFrame(legacy_metrics)
    overlap_metrics = generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components)
    overlap_metrics = pd.DataFrame(overlap_metrics)

    initial_metrics_df = pd.concat([legacy_metrics, overlap_metrics], ignore_index=True)

    pred_list = []
    pred_int = []

    for _, row in initial_metrics_df.iterrows():
        if isinstance(row['Pred'], list):
            pred_list.append(row['GT'])
        if isinstance(row['Pred'], int):
            pred_int.append(row['GT'])

    common_elements = set(pred_list).intersection(set(pred_int))

    final_metrics_df = initial_metrics_df.copy()

    for index, row in final_metrics_df.iterrows():
        if row['GT'] in common_elements and isinstance(row['Pred'], int):
            final_metrics_df.drop(index, inplace=True)

    return initial_metrics_df, fp


def process_metric_df(df):
    gt_list = []
    pred_list = []

    for gt, pred in zip(df['GT'], df['Pred']):
        if isinstance(gt, list):
            gt_list.extend(gt)

        if isinstance(pred, list):
            pred_list.extend(pred)

    combined = set(gt_list + pred_list)

    indices_to_drop = []

    for idx, (gt, pred) in enumerate(zip(df['GT'], df['Pred'])):
        if isinstance(gt, int):
            if gt in combined:
                if isinstance(pred, int):
                    indices_to_drop.append(idx)

    df.drop(indices_to_drop, inplace=True)
    df['GT'] = df['GT'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df['Pred'] = df['Pred'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df.drop_duplicates(subset=['GT', 'Pred'], inplace=True)

    return df['Dice'].to_list()


def metric(pred, gt):
    gt_label_cc = cc3d.connected_components(gt)
    pred_label_cc = cc3d.connected_components(pred)

    overlapping_components = find_overlapping_components(pred_label_cc, gt_label_cc)
    overlapping_components_inverse = find_overlapping_components_inverse(pred_label_cc, gt_label_cc)

    metrics, fp = collect_metrics(pred_label_cc, gt_label_cc, overlapping_components)
    metrics_inverse, fp_inverse = collect_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components_inverse)

    final_metric = pd.concat([metrics, metrics_inverse], ignore_index=True)

    dice_score = process_metric_df(final_metric)
    dice_score = np.sum(dice_score) / (len(dice_score) + len(fp))

    return dice_score

# 1. Calcualte the indiviual lesion wise dice score for every GT lesion.
# 2. Find the cases where:
        # - One Prediction covers multiple GT lesions
        # - One GT lesion is covered by multiple predictions
# 3. Isolate the above cases and calculate the dice score for each of them.
# 4. Form a table of all the above cases and then drop the rows where:
# 5. Extract a set of the formed matches
        # - GT is a single component and in matched list
        # - Prediction is a single component.
# 6. Final Dice: Sum of all the dice scores in the table divided by the length of the table + length of false positives.