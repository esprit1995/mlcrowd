import jsonParserFuncs as jpf
import argParser as ap


def main():
    args = ap.parse_args()
    if args.destroy_source:
        print('=============================================')
        print('Destroying source for file: ' + str(args.scenario_file) + ' at ' + str(args.scenario_path))
        print('The new scenario filename will be in : ' + (str(args.output_dir) + ' and named ' +  str(args.new_name)))
        print('=============================================')

        jpf.replace_source_with_pedestrians(str(args.scenario_file),
                                            str(args.trajectory_file),
                                            str(args.scenario_path),
                                            str(args.trajectory_path),
                                            str(args.new_name),
                                            targets=[1])


if __name__ == '__main__':
    main()
