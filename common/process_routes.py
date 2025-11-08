import argparse
import random
import sys
import xml.etree.ElementTree as ET
from typing import Optional


def add_vehicle_types(
    input_file: str, output_file: str, av_mpr: float, seed: Optional[int] = None
):
    """
    Reads a SUMO route file and assigns a type to each vehicle based on the
    Market Penetration Rate (MPR) for Autonomous Vehicles (AVs).

    Args:
        input_file (str): Path to the input SUMO route file.
        output_file (str): Path to save the new route file.
        av_mpr (float): The market penetration rate for AVs (0.0 to 1.0).
        seed (Optional[int]): A seed for the random number generator for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    try:
        if not (0.0 <= av_mpr <= 1.0):
            raise ValueError("MPR must be a value between 0.0 and 1.0.")

        tree = ET.parse(input_file)
        root = tree.getroot()

        for vehicle in root.findall("vehicle"):
            if random.random() < av_mpr:
                vehicle.set("type", "AV")
            else:
                vehicle.set("type", "HV")

        if sys.version_info >= (3, 9):
            ET.indent(tree)

        tree.write(output_file, encoding="UTF-8", xml_declaration=True)

    except (FileNotFoundError, ET.ParseError) as e:
        print(f"Error processing XML file: {e}", file=sys.stderr)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process SUMO routes to create a mixed-traffic scenario."
    )
    parser.add_argument(
        "-i", "--input", default="city2.rou.xml", help="Input route file."
    )
    parser.add_argument(
        "-o", "--output", default="city2.mixed.rou.xml", help="Output route file."
    )
    parser.add_argument(
        "--mpr",
        type=float,
        default=0.5,
        help="AV Market Penetration Rate (a value from 0.0 to 1.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    add_vehicle_types(args.input, args.output, args.mpr, args.seed)
