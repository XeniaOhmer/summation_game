
from egg.core.callbacks import *


class InteractionSaverLocal(InteractionSaver):

    def __init__(self,
                 train_epochs: Optional[List[int]] = None,
                 test_epochs: Optional[List[int]] = None,
                 checkpoint_dir: str = ""
                 ):
        super(InteractionSaverLocal, self).__init__(train_epochs, test_epochs, checkpoint_dir)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        rank = self.trainer.distributed_context.rank
        self.dump_interactions(train_logs, 'train', epoch, rank, self.checkpoint_dir)
        self.dump_interactions(test_logs, 'validation', epoch, rank, self.checkpoint_dir)
