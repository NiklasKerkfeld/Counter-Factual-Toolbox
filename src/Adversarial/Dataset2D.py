from torch.utils.data import Dataset


class Dataset2D(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return index

   
if __name__ == '__main__':
    main()
