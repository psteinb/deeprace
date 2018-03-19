"""usage: deeprace list [options]
    -h, --help
    -v, --verbose        be verbose
"""

from docopt import docopt

def main():

    args = docopt(__doc__)
    print(args)

if __name__ == '__main__':
    main()
