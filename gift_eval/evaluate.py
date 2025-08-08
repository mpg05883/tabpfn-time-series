import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))


import argparse
import csv
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import wandb
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality
from serde import serialize_forecasts

from gift_eval.data import Dataset
from gift_eval.dataset_definition import (
    DATASET_PROPERTIES_MAP,
    MED_LONG_DATASETS,
)
from gift_eval.tabpfn_ts_wrapper import TabPFNMode, TabPFNTSPredictor

# Instantiate the metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

metrics = [
    MSE(forecast_type="mean"),
]

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()
    
def get_dataset_config(dataset: Dataset) -> str:
    name = dataset.name
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    name = name.split("/")[0] if "/" in name else name
    cleaned_name = pretty_names.get(name.lower(), name.lower())
    cleaned_freq = dataset.freq.split("-")[0]
    return f"{cleaned_name}/{cleaned_freq}/{dataset.term.value}"


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


# Set up logging
logger = logging.getLogger(__name__)


def construct_evaluation_data(
    dataset_name: str,
    dataset_storage_path: Path,
    terms: str,
) -> List[Tuple[Dataset, dict]]:
    sub_datasets = []

    # Construct evaluation data
    ds_key = dataset_name.split("/")[0]
    for term in terms:
        if (
            term == "medium" or term == "long"
        ) and dataset_name not in MED_LONG_DATASETS:
            continue

        if "/" in dataset_name:
            ds_key = dataset_name.split("/")[0]
            ds_freq = dataset_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = dataset_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = DATASET_PROPERTIES_MAP[ds_key]["frequency"]

        print(f'term: {term}')
        # Initialize the dataset
        to_univariate = (
            False
            if Dataset(
                name=dataset_name,
                term=term,
                to_univariate=False,
                storage_path=dataset_storage_path,
            ).target_dim
            == 1
            else True
        )
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            storage_path=dataset_storage_path,
        )
        season_length = get_seasonality(dataset.freq)

        dataset_metadata = {
            "full_name": f"{ds_key}/{ds_freq}/{term}",
            "key": ds_key,
            "freq": ds_freq,
            "term": term,
            "season_length": season_length,
        }
        sub_datasets.append((dataset, dataset_metadata))

    return sub_datasets


def create_csv_file(csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(
            [
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                # "eval_metrics/MSE[0.5]",
                # "eval_metrics/MAE[0.5]",
                # "eval_metrics/MASE[0.5]",
                # "eval_metrics/MAPE[0.5]",
                # "eval_metrics/sMAPE[0.5]",
                # "eval_metrics/MSIS",
                # "eval_metrics/RMSE[mean]",
                # "eval_metrics/NRMSE[mean]",
                # "eval_metrics/ND[0.5]",
                # "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
            ]
        )


def append_results_to_csv(
    res,
    csv_file_path,
    dataset_metadata,
    model_name,
):
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                dataset_metadata["full_name"],
                model_name,
                res["MSE[mean]"][0],
                # res["MSE[0.5]"][0],
                # res["MAE[0.5]"][0],
                # res["MASE[0.5]"][0],
                # res["MAPE[0.5]"][0],
                # res["sMAPE[0.5]"][0],
                # res["MSIS"][0],
                # res["RMSE[mean]"][0],
                # res["NRMSE[mean]"][0],
                # res["ND[0.5]"][0],
                # res["mean_weighted_sum_quantile_loss"][0],
                DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["domain"],
                DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["num_variates"],
            ]
        )

    print(f"Results for {dataset_metadata['key']} have been written to {csv_file_path}")


def log_results_to_wandb(
    res,
    dataset_metadata,
):
    wandb_log_data = {
        "MSE_mean": res["MSE[mean]"][0],
        "MSE_0.5": res["MSE[0.5]"][0],
        "MAE_0.5": res["MAE[0.5]"][0],
        "MASE_0.5": res["MASE[0.5]"][0],
        "MAPE_0.5": res["MAPE[0.5]"][0],
        "sMAPE_0.5": res["sMAPE[0.5]"][0],
        "MSIS": res["MSIS"][0],
        "RMSE_mean": res["RMSE[mean]"][0],
        "NRMSE_mean": res["NRMSE[mean]"][0],
        "ND_0.5": res["ND[0.5]"][0],
        "mean_weighted_sum_quantile_loss": res["mean_weighted_sum_quantile_loss"][0],
        "domain": DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["domain"],
        "num_variates": DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["num_variates"],
        "term": dataset_metadata["term"],
    }
    wandb.log(wandb_log_data)


def main(args):
    metadata_path = Path("data") / "meta" / "train_test" / "metadata.csv"
    metadata_df = pd.read_csv(metadata_path, usecols=["name", "term"])
    name, term = metadata_df.iloc[args.task_index][["name", "term"]]

    cleaned_name = f"{name}_{term}".lower().replace("/", "_")

    # Initialize wandb
    wandb.init(
        project="EnsembleTS",
        entity="mpgee-usc",
        name=cleaned_name,
        tags=[
            "model=tabpfn_ts",
            "eval",
            f"dataset_name={name}",
            f"term={term}",
            "debug",
        ],
        # job_type="eval",
        job_type="debug",
        group="tabpfn_ts",
        mode="online",
    )

    table = wandb.Table(
        columns=[
            "model",
            "val_loss",
            "forecast_artifact",
        ]
    )

    output_dir = args.output_dir / args.model_name / get_dataset_config(Dataset(name=name, term=term, storage_path=Path(args.dataset_storage_path),),)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "results.csv"

    # Create CSV file
    create_csv_file(output_csv_path)

    # Construct evaluation data (i.e. sub-datasets) for this dataset
    # (some datasets contain different forecasting terms, e.g. short, medium, long)
    print(f'name: {name}')
    print(f'storage path: {args.dataset_storage_path}')
    
    print(f'term {term}')
    
    sub_datasets = construct_evaluation_data(
        name,
        Path(args.dataset_storage_path),
        [term],
    )

    # Evaluate model
    for i, (sub_dataset, dataset_metadata) in enumerate(sub_datasets):
        logger.info(
            f"Evaluating {i + 1}/{len(sub_datasets)} dataset {sub_dataset.name}"
        )
        logger.info(f"Dataset size: {len(sub_dataset.test_data)}")
        logger.info(f"Dataset freq: {sub_dataset.freq}")
        logger.info(f"Dataset term: {dataset_metadata['term']}")
        logger.info(f"Dataset prediction length: {sub_dataset.prediction_length}")
        logger.info(f"Dataset target dim: {sub_dataset.target_dim}")

        tabpfn_predictor = TabPFNTSPredictor(
            ds_prediction_length=sub_dataset.prediction_length,
            ds_freq=sub_dataset.freq,
            tabpfn_mode=TabPFNMode.LOCAL,
            # tabpfn_mode=TabPFNMode.CLIENT,
            context_length=4096,
            debug=args.debug,
        )

        forecasts = list(tabpfn_predictor.predict(sub_dataset.test_data))

        res = evaluate_forecasts(
            forecasts=forecasts,
            test_data=sub_dataset.test_data,
            metrics=metrics,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=dataset_metadata["season_length"],
        )

        parts = [
            "./results/forecasts",
            get_dataset_config(sub_dataset),
        ]

        forecasts_path = Path(*parts) / "tabpfn_ts.pkl"

        forecast_dict = serialize_forecasts(forecasts)

        with open(forecasts_path, "wb") as f:
            pickle.dump(forecast_dict, f)

        forecast_artifact = wandb.Artifact(
            name=f"{forecasts_path.stem}_forecasts",
            type="forecast",
        )
        forecast_artifact.add_file(forecasts_path)
        wandb.log_artifact(forecast_artifact)

        # Write results to csv
        append_results_to_csv(
            res=res,
            csv_file_path=output_csv_path,
            dataset_metadata=dataset_metadata,
            model_name=args.model_name,
        )

        forecast_artifact = None

        table.add_data(
            args.model_name,
            res.values[0].item(),
            forecast_artifact,
        )

    table_artifact = wandb.Artifact(
        name=f"{name.replace('/','_')}_{term}",
        type="evaluation",
    )
    table_artifact.add(table, name="evaluation")
    wandb.log_artifact(table_artifact)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tabpfn_ts")
    parser.add_argument("--task_index", type=int, required=True)


    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    parser.add_argument("--output_dir", type=str, default=str(output_dir))

    parser.add_argument(
        "--dataset_storage_path",
        type=str,
        default=str(Path(__file__).parent / "datasets" / "train_test"),
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.dataset_storage_path = Path(args.dataset_storage_path)
    args.output_dir = Path(args.output_dir)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Command Line Arguments: {vars(args)}")

    main(args)
