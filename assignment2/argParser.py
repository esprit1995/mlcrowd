from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path


def validate_args(args):
    if args.destroy_source and args.scenario_path is None:
        raise ValueError('Path to scenario not specified for destoy_source task')
    if args.destroy_source and args.trajectory_path is None:
        raise ValueError('Path to to trajectories not specified for destroy_source task')
    if args.destroy_source and args.output_dir is None:
        raise ValueError('Output directory not specified')
    if args.destroy_source and args.trajectory_file is None:
        raise ValueError('Output directory not specified')
    if args.destroy_source and args.scenario_file is None:
        raise ValueError('Output directory not specified')


def parse_args():
    ap = ArgumentParser()

    # mode arguments
    ap.add_argument('--destroy_source', dest='destroy_source',
                    help='whether to replace source with pedestrian entities', default=1)

    # destroy-source arguments
    ap.add_argument('--scenario_path', dest='scenario_path', help='path to scenario file')
    ap.add_argument('--scenario_file', dest='scenario_file', help='name of the scenario file')
    ap.add_argument('--trajectory_path', dest='trajectory_path', help='path to trajectory file')
    ap.add_argument('--trajectory_file', dest='trajectory_file', help='name of the trajectory file')
    ap.add_argument('--new_name', dest='new_name', help='how to name the new scenario file',
                    default='task5Updated.scenario')
    ap.add_argument('--output_dir', dest='output_dir', help='Where to put the resulting scenario file')

    args = ap.parse_args()
    validate_args(args)
    return args
