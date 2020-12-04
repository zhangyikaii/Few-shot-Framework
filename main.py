from models.utils import (
    get_command_line_parser, 
    preprocess_args,
    pprint
)
from models.dataloader.mini_imagenet import (
    dataloader_main
)

if __name__ == '__main__':
    """
    准备命令行参数
    """
    parser = get_command_line_parser()
    args = preprocess_args(parser.parse_args())
    pprint(vars(args))

    """
    准备 Dataloader
    """
    dataloaders = dataloader_main(args)


    """
    准备 Model, Optimizer, loss_func
    """


    """
    Training
    """


