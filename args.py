from argparse import ArgumentParser

a_parser = ArgumentParser()

a_parser.add_argument('--save_or_load', type=str, default='load', choices=['save', 'load'])
a_parser.add_argument('--do_test', type=bool, default=True)

a_parser.add_argument('--system', type=str, default='A', choices=['A', 'B'])
a_parser.add_argument('--model_card', type=str, default='roberta-base')
a_parser.add_argument('--bpe_as_beginning', type=bool, default=True)

a_parser.add_argument('--epochs', type=int, default=2)
a_parser.add_argument('--batch_size', type=int, default=16)

a_parser.add_argument('--seed', type=int, default=333)
a_parser.add_argument('--cuda_device', type=str, default='cuda:0')


args = a_parser.parse_args()