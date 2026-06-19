import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ..model.bumblebee.bumblebee import Bumblebee
from .train import train
from ..model.loss import CrossEntropyLoss
from datasets.Wikitext.wikitext import WikiText
from models.utils import DEVICE, load_checkpoint


if __name__ == "__main__":
    # Load dataset
    vocab_json = 'datasets/Wikitext/Wikitext/wikitext103/vocab.json'
    wikitext2_train = WikiText(encoded_text_json='datasets/Wikitext/Wikitext/wikitext103/encoded_text_train.json', vocab_json=vocab_json)
    wikitext2_val = WikiText(encoded_text_json='datasets/Wikitext/Wikitext/wikitext103/encoded_text_val.json', vocab_json=vocab_json)
    wikitext2_test = WikiText(encoded_text_json='datasets/Wikitext/Wikitext/wikitext103/encoded_text_test.json', vocab_json=vocab_json)

    # Define params
    VOCAB_SIZE = wikitext2_train.get_vocab_size()
    BATCH_SIZE = 32

    # Create dataloaders
    train_loader = DataLoader(wikitext2_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(wikitext2_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(wikitext2_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    d_model = 512
    model = Bumblebee(vocab_size=VOCAB_SIZE, d_model=d_model)
    
    loss_func = CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)

    model = model.to(DEVICE)

    num_epochs = 4
    total_steps = len(train_loader) * num_epochs
    num_warmup_steps = total_steps // 10


    def get_scheduler(optimizer, num_warmup_steps, total_steps):
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/num_warmup_steps, total_iters=num_warmup_steps)

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-num_warmup_steps, eta_min=1e-5)

        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])
    
    scheduler = get_scheduler(optimizer, num_warmup_steps, total_steps)
    
    # start training (!!)
    train(model, train_loader, val_loader, test_loader, loss_func, optimizer, scheduler, 
          num_epochs=num_epochs,
          start_epoch=0,
          runs_dir='runs'
    )    



    