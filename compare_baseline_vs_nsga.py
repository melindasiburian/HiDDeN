"""
Compare baseline HiDDeN vs NSGA-II optimized final embedding metrics.
Produces `comparison_metrics.csv` and prints a small summary.
"""
import argparse
import os
import json
import csv
import pandas as pd


def load_baseline_metrics(validation_csv_path: str):
    if not os.path.exists(validation_csv_path):
        raise FileNotFoundError(f"Baseline validation CSV not found: {validation_csv_path}")
    df = pd.read_csv(validation_csv_path)
    # aggregate final epoch metrics (assume last row is final)
    last = df.iloc[-1].to_dict()
    return last


def load_nsga_metrics(metrics_json_path: str):
    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(f"NSGA metrics file not found: {metrics_json_path}")
    with open(metrics_json_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs NSGA-II metrics')
    parser.add_argument('--baseline-csv', required=True, help='Path to baseline validation CSV (validation.csv)')
    parser.add_argument('--nsga-metrics', required=True, help='Path to NSGA metrics.json')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline = load_baseline_metrics(args.baseline_csv)
    nsga = load_nsga_metrics(args.nsga_metrics)

    comparison = {
        'psnr_baseline': baseline.get('psnr', None) or baseline.get('encoder_mse', None),
        'ssim_baseline': baseline.get('ssim', None),
        'ber_baseline': baseline.get('bitwise-error', None) or baseline.get('bitwise_error', None),
        'psnr_nsga': nsga.get('psnr'),
        'ssim_nsga': nsga.get('ssim'),
        'ber_nsga': nsga.get('ber'),
        'bpp_nsga': nsga.get('bpp'),
        'security_nsga': nsga.get('security')
    }

    out_csv = os.path.join(args.output_dir, 'comparison_metrics.csv')
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['metric', 'baseline', 'nsga'])
        writer.writerow(['psnr', comparison['psnr_baseline'], comparison['psnr_nsga']])
        writer.writerow(['ssim', comparison['ssim_baseline'], comparison['ssim_nsga']])
        writer.writerow(['ber', comparison['ber_baseline'], comparison['ber_nsga']])
        writer.writerow(['bpp', '', comparison['bpp_nsga']])
        writer.writerow(['security', '', comparison['security_nsga']])

    print('Comparison saved to', out_csv)


if __name__ == '__main__':
    main()
