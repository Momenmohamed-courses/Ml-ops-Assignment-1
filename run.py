import argparse
import sys
from pathlib import Path

from production_app.logging_config import get_logger, setup_logging
from production_app.main import run_prediction, run_training
from production_app.model_service import InvalidInputError, ModelNotFoundError

logger = get_logger("run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breast cancer classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("train", help="Train, evaluate and save the model")

    predict_parser = commands.add_parser("predict", help="Predict with the saved model")
    predict_parser.add_argument(
        "--input", type=Path, required=True, help="CSV file with the feature columns"
    )

    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        if args.command == "train":
            run_training()
        else:
            run_prediction(args.input)
    except (ModelNotFoundError, InvalidInputError, FileNotFoundError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
