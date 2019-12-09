"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
import nice
import os
import matplotlib.pyplot as plt

N = 100


def train(flow, trainloader, optimizer, full_dim):
    flow.train()  # set to training mode
    running_loss = 0
    for n_batches, inputs in enumerate(trainloader,1):
        inputs = inputs.view(-1,full_dim) #change  shape from Bx1x28x28 to Bx784
        inputs = dequantize(inputs)
        loss = -flow(inputs).mean()
        running_loss += float(loss)
        loss.backward()
        optimizer.step()
    return running_loss / n_batches


def test(flow, testloader, epoch, filename, full_dim):
    flow.eval()  # set to inference mode
    running_loss = 0
    with torch.no_grad():
        samples = flow.sample(100).cpu()
        samples = samples.view(-1,1,28,28)
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        for n_batches, inputs in enumerate(testloader):
            inputs = inputs.view(-1,full_dim) #change  shape from Bx1x28x28 to Bx784
            inputs = dequantize(inputs)
            loss = -flow(inputs).mean()
            running_loss += float(loss)
    return running_loss / n_batches


def dequantize(x):
    '''Dequantize data.
    Add noise sampled from Uniform(0, 1) to each pixel (in [0, 255]).
    Args:
        x: input tensor.
        reverse: True in inference mode, False in training mode.
    Returns:
        dequantized data.
    '''
    noise = torch.distributions.Uniform(0., 1.).sample(x.size())
    return (x * 255. + noise) / 256.


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_dim = 28 * 28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)) #dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size = args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'coupling%d_' % args.coupling \
             + 'mid%d_' % args.mid_dim \
             + 'hidden%d_' % args.hidden \
             + '.pt'

    flow = nice.NICE(
                prior=args.prior,
                coupling=args.coupling,
                in_out_dim=full_dim, 
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                mask_config=1.,
                device=device,
                coup_type=args.coup_type).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    train_loss = []
    test_loss   = []
    for e in range(args.epochs):
        train_loss.append(train(flow, trainloader, optimizer, full_dim))
        test_loss.append(test(flow, testloader, e, filename="sampled"), full_dim)
        if e % args.save_every == 0:
            torch.save(flow.state_dict(), model_save_filename)
            print("#" * 15,"\n")
            print(f"Epoch {e}:  train loss: {train_loss[-1]} test loss: {test_loss[-1]}\n")

    fig, ax = plt.subplots()
    ax.plot(train_loss)
    ax.plot(test_loss)
    ax.set_title("Train/Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["train loss","test loss"])
    plt.savefig(os.path.join(os.getcwd(),"loss.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling',
                        help='.',
                        type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--save-every',
                        help='every how many epochs to save the model',
                        type=float,
                        default=1e+3)
    parser.add_argument('--coup-type',
                        help="coupling type",
                        type=str,
                        default="additive")
    args = parser.parse_args()
    main(args)