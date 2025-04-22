from alpaca import Alpaca
from tokenizer import Tokenizer
import os
from tqdm import tqdm
from multiprocessing import freeze_support


def test():
    data = None
    data = Alpaca() if data is None else data
    print('')

    tokenizer = Tokenizer(os.getcwd() + '/1.3B')
    data.connect(tokenizer=tokenizer, batch_size=4, max_seq_length=2048)
    data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()

    batch_iterator = tqdm(train_dataloader, mininterval=0, colour="blue")
    for step, batch in enumerate(batch_iterator):
        print('')

if __name__ == "__main__":
    test()