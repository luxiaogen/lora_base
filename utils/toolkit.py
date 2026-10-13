def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()
