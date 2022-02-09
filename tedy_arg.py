from argparse import ArgumentParser

parser = ArgumentParser(description="Un programme qui mange des patates", epilog="That's all folk")
parser.add_argument("-f", "--file", dest="filename",
                    help="write report to FILE", metavar="FILE")
parser.add_argument("-q", "--quiet",
                    action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")

args = parser.parse_args()
# print(args.accumulate(args.integers))
print(args.filename)