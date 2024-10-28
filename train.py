import torchvision
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import torch.nn.functional as F

def train(model, train_dataloader, val_dataloader, writer, epochs, validation_step, log_dir, logger):

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=epochs)
    loss_function = torch.nn.CrossEntropyLoss()


    for epoch in range(epochs):
        losses = []
        pbar = tqdm(train_dataloader)
        for step, data in enumerate(pbar):
            data = data.to('cuda:0')

            # Get the model prediction
            pred = model(data)

            pred = F.softmax(pred, dim=1)

            # Calculate the loss
            loss = loss_function(pred, data.y)
            loss.backward()
            losses.append(loss.item())

            writer.add_scalar('Iteration/Training loss', loss.mean(), step * epoch + step)

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()
        
        lr_scheduler.step()
        writer.add_scalar('Iteration/Learning rate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % validation_step == 0:
            model.eval()
            with torch.no_grad():
                loss_last_epoch = sum(losses) / len(train_dataloader)
                logger.info(f"Epoch : {epoch+1}, loss: {loss_last_epoch}")
                max_memory_allocated = torch.cuda.max_memory_allocated(device='cuda:0')
                logger.info(f"Max GPU Memory allocated : {max_memory_allocated / 10e8} Gb")
                #overfit_data = {'Train': torch.tensor(self.train_loss).mean().item(), 'Val': torch.tensor(self.val_loss).mean().item()}
                #writer.add_scalars('Epoch/Train_vs_val_loss', overfit_data, epoch)

                epoch_val_acc = []
                pbar = tqdm(val_dataloader)
                for step, data in enumerate(pbar):
                    data = data.to('cuda:0')

                    # Get the model prediction
                    pred = model(data)

                    pred = F.softmax(pred, dim=1).argmax(dim=1)

                    epoch_val_acc.append(sum(pred == data.y) / len(pred))

                
                    #pred = model(data).argmax(dim=1)
                    #correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                    #acc = int(correct) / int(data.test_mask.sum())
                
                acc = torch.stack(epoch_val_acc).mean(0)
                writer.add_scalar('Epoch/Accuracy', acc, epoch)

                torch.save(model.state_dict(), os.path.join(log_dir, 'weights.pth'))
                
    torch.save(model.state_dict(), os.path.join(log_dir, 'weights.pth'))