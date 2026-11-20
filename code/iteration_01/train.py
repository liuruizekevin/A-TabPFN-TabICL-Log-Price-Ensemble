#!/usr/bin/env python3
"""Run the 21 first-iteration configurations in an isolated destination."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def historical_output(parser, value, label='--output-dir'):
    """Resolve an explicitly selected output without touching current evidence."""
    if value is None:
        parser.error(
            f'{label} is required; use --output-dir build/iteration_01.'
        )
    output = Path(value).expanduser().resolve()
    protected = [ROOT/name for name in (
        'features', 'models', 'submissions', 'Final_Submission',
        'data', 'docs', 'references', 'scripts', 'src', 'tests', '.git',
    )]
    for directory in protected:
        directory = directory.resolve()
        if output == directory or directory in output.parents or output in directory.parents:
            parser.error(f'{label} must be isolated from current source/evidence directories; rejected {output}. Use a new build/ subdirectory or another dedicated directory.')
    if output.exists() and not output.is_dir():
        parser.error(f'{label} is not a directory: {output}')
    if output.is_dir() and any(item.is_symlink() for item in output.rglob('*')):
        parser.error(f'{label} contains symlinks that could redirect writes; use a dedicated directory without symlinked contents: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--output-dir', type=Path, help='Required isolated destination for historical Phase 1 outputs.')
    parser.add_argument('--feature-output-dir', type=Path, help='Optional isolated feature destination; defaults beneath --output-dir.')
    args = parser.parse_args()
    output = historical_output(parser, args.output_dir)
    features = historical_output(parser, args.feature_output_dir, '--feature-output-dir') if args.feature_output_dir is not None else output/'processed_features'
    # Resolve safety and scope before importing or executing training code.
    sys.path.insert(0, str(ROOT/'src'))
    from house_prices.experiment import run_experiments
    run_experiments(args.data_dir, output, features)


if __name__ == '__main__':
    main()
