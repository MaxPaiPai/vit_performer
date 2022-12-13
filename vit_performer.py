from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from vit_pytorch.efficient import ViT as ViT_efficient
from vit_pytorch import ViT


from performer_pytorch import Performer
from torch import Tensor
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.expand(-1, 3, -1, -1) # to deal with greyscale MNIST data
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        writer.add_scalar('Loss/train', loss.item(), (epoch-1) * len(train_loader) + batch_idx)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.expand(-1, 3, -1, -1) # to deal with greyscale MNIST data
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with ViT Performer')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')

    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='which torchvision dataset to use', choices=['MNIST', 'Caltech256'])
    parser.add_argument('--kernel', type=str, default='softmax',
                        help='kernel types', choices=['non_performer', 'softmax', 'relu', 'quad', 'x^4'])
    parser.add_argument('--nb_features',  type=int, default=512,
                    help='number of random features')
    parser.add_argument('--redraw', type=int, default=None,
                        help='how many steps before redraw random features')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--tb_log_dir', type=str, default=None,
                        help='For tensorboard log directory')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if args.kernel == 'non_performer':
        tb_log_dir = f"./runs/{args.dataset}/{args.dataset}_vit_{args.kernel}"
    else:
        tb_log_dir = f"./runs/{args.dataset}/{args.dataset}_vit_performer_{args.kernel}_nb_features_{args.nb_features}_feature_redraw_interval_{args.redraw}"
    writer = SummaryWriter(tb_log_dir)
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 6,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    if args.dataset == 'MNIST':
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('./data', train=False,
                        transform=transform)

        # parameters
        num_classes = 10
        image_size = 28
        patch_size = 7
        d_channel = 512

    elif args.dataset == 'Caltech256':
        normalize = transforms.Normalize(
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225],
            mean=(0.485,),
            std=(0.229,),
        )
        transform = transforms.Compose([
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                normalize,
                                ])

        dataset1 = datasets.Caltech256('./data', download=True,
                            transform=transform)

        # remove greyscale images, these indices are of RGB images
        try:
            my_file = open("caltech256_rgb_indices.txt", "r")
            indices = my_file.read()
            indices = indices.split("\n")[:-1]
            indices = [int(x) for x in indices]
            my_file.close()
        except:
            indices = []
            for i in range(len(dataset1)):
                if dataset1[i][0].shape[0] == 3: # is RGB image
                    indices.append(i)
            with open("caltech256_rgb_indices.txt", 'w') as output:
                for row in indices:
                    output.write(str(row) + '\n')

        dataset1 = Subset(dataset1, indices)

        dataset1, dataset2 = torch.utils.data.random_split(dataset1, 
                    [int(len(dataset1)*0.8), len(dataset1)-int(len(dataset1)*0.8)],
                    generator=torch.Generator().manual_seed(args.seed))
        
        # parameters
        num_classes = 257
        image_size = 256
        patch_size = 8
        d_channel = 512
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    
    from torch.autograd import Variable

    class quad(nn.Module):

        def __init__(self):
            super(quad, self).__init__()

        def forward(self, x):
            squ = torch.pow(x, 2)
            return squ

    class biquad(nn.Module):

        def __init__(self):
            super(biquad, self).__init__()

        def forward(self, x):
            squ = torch.pow(x, 4)
            return squ

    if args.kernel == 'non_performer':
        model = ViT(
            dim = d_channel,
            mlp_dim = d_channel,
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            depth = 8,
            heads = 8,
            dropout = 0.,
            emb_dropout = 0.
        )
    else:
        if args.kernel == 'softmax':
            generalized_attention = False
            kernel_fn = None
        elif args.kernel == 'relu':
            generalized_attention = True
            kernel_fn = nn.ReLU()
        elif args.kernel == 'quad':
            generalized_attention = True
            kernel_fn = quad()
        elif args.kernel == 'x^4':
            generalized_attention = True
            kernel_fn = biquad()

        transformer = Performer(
            dim = d_channel,
            depth = 8,
            heads = 8,
            causal = False,
            dim_head = 64,
            ff_mult = 1,

            generalized_attention = generalized_attention,
            kernel_fn = kernel_fn,
            nb_features = args.nb_features, # if nb_features is 0, then use None as projection_matrix in generalized kernel function \
                                            # which means using determinisitc feature projection \
                                            # you need to first cd to "~/anaconda3/envs/vit_performer/lib/python3.8/site-packages/performer_pytorch" \
                                            # edit "performer_pytorch.py": add "if nb_full_blocks == 0: return None" after Line 143
            feature_redraw_interval = args.redraw
        )

        model = ViT_efficient(
            dim = d_channel,
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            transformer = transformer
        )
    
    model = model.to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of parameters in the model: {count_parameters(model)}")
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_time = 0
    test_time = 0

    for epoch in range(1, args.epochs + 1):
        epoch_train_start = time.time()
        train(args, model, device, train_loader, optimizer, epoch, writer)
        epoch_test_start = epoch_train_end = time.time()
        test(model, device, test_loader)
        epoch_test_end = time.time()
        scheduler.step()

        train_time += (epoch_train_end - epoch_train_start)
        test_time += (epoch_test_end - epoch_test_start)

    print(f"Average seconds for training 1 epoch: {train_time/epoch}")
    print(f"Average seconds for testing: {test_time/epoch}")

    if args.save_model:
        torch.save(model.state_dict(), "./models/mnist_vit_performer.pt")


if __name__ == '__main__':
    main()