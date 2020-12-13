from models.utils import (
    get_command_line_parser, 
    preprocess_args,
    pprint,
    set_seeds
)
from models.train import (
    Trainer
)

if __name__ == '__main__':
    set_seeds()
    """
    准备命令行参数
    """
    parser = get_command_line_parser()
    args = preprocess_args(parser.parse_args())
    pprint(vars(args))

    """
    Training
    """
    trainer = Trainer(args)
    trainer.fit()
    trainer.test()