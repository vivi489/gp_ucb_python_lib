import argparse
from myenv import MyEnvironment
from gphypo.run_cmdenv import run_gmrf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw a loss figure')
    parser.add_argument('-i', '--input', type=str, default='parameter_gmrf.json', help='parameter_gp filename')

    args = parser.parse_args()
    print(args)
    run_gmrf(args.input, MyEnvironment)
