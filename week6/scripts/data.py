"""Read the original splits, preserving image/label order; reusable native-pixel plots."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import tarfile
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
ARTIFACTS = ROOT / 'artifacts'
for directory in (EVIDENCE, ARTIFACTS):
    directory.mkdir(exist_ok=True)
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
URLS = {f'mnist/{s}-{k}.gz': f'https://storage.googleapis.com/cvdf-datasets/mnist/{s}-{k}.gz'
        for s in ('train', 't10k') for k in ('images-idx3-ubyte', 'labels-idx1-ubyte')}
URLS['cifar10/cifar-10-binary.tar.gz'] = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz'


def download():
    for name, url in URLS.items():
        path = ROOT / 'inputs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix('.part')
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)


def mnist(split):
    prefix = 'train' if split == 'train' else 't10k'
    base = ROOT / 'inputs/mnist'
    with gzip.open(base / f'{prefix}-images-idx3-ubyte.gz', 'rb') as f:
        magic, n, h, w = struct.unpack('>4I', f.read(16))
        assert magic == 2051 and (h, w) == (28, 28)
        images = np.frombuffer(f.read(), np.uint8).reshape(n, h, w)
    with gzip.open(base / f'{prefix}-labels-idx1-ubyte.gz', 'rb') as f:
        magic, count = struct.unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), np.uint8)
        assert magic == 2049 and count == n == len(labels)
    return images, labels


def cifar(split):
    files = [f'data_batch_{i}.bin' for i in range(1, 6)] if split == 'train' else ['test_batch.bin']
    batches = []
    with tarfile.open(ROOT / 'inputs/cifar10/cifar-10-binary.tar.gz') as archive:
        for name in files:
            member = next(m for m in archive.getmembers() if m.name.endswith('/' + name))
            batches.append(np.frombuffer(archive.extractfile(member).read(), np.uint8).reshape(-1, 3073))
    records = np.concatenate(batches)
    return records[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), records[:, 0]


def show_image(ax, image, label=None, vmin=0, vmax=255):
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if label is not None:
        ax.set_title(str(label), fontsize=9)


def gallery(images, labels=None, columns=8, filename=None, title='', vmin=0, vmax=255):
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.3, rows * 1.5), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            show_image(ax, images[i], None if labels is None else labels[i], vmin, vmax)
        else:
            ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    if filename:
        fig.savefig(EVIDENCE / filename, dpi=160)
        plt.close(fig)
    return fig


def class_frequencies(labels, classes=10):
    return np.bincount(labels, minlength=classes) / len(labels)


def dataset_figure(loader, name, classes):
    train, labels = loader('train')
    test, test_labels = loader('test')
    fig = plt.figure(figsize=(14, 8), layout='constrained')
    grid = fig.add_gridspec(5, 10, height_ratios=[1, 1, 1, .2, 2])
    for c in range(10):
        for row, index in enumerate(np.flatnonzero(labels == c)[:3]):
            show_image(fig.add_subplot(grid[row, c]), train[index], classes[c] if row == 0 else None)
    ax = fig.add_subplot(grid[4, :])
    x = np.arange(10)
    ax.bar(x - .2, class_frequencies(labels), .4, label=f'Train (n={len(train):,})')
    ax.bar(x + .2, class_frequencies(test_labels), .4, label=f'Test (n={len(test):,})')
    ax.set(xticks=x, xticklabels=classes, ylabel='Fraction of all labels', ylim=(0, .14))
    ax.legend()
    fig.suptitle(f'{name}: first three training examples per class; original splits, native pixels')
    fig.savefig(EVIDENCE / f'{name.lower()}_distribution.png', dpi=160)
    plt.close(fig)
    return dict(train_shape=list(train.shape), test_shape=list(test.shape), dtype=str(train.dtype),
                pixel_range=[int(train.min()), int(train.max())],
                train_counts=np.bincount(labels).tolist(), test_counts=np.bincount(test_labels).tolist())


def main():
    download()
    summary = {'MNIST': dataset_figure(mnist, 'MNIST', list(map(str, range(10)))),
               'CIFAR10': dataset_figure(cifar, 'CIFAR10', CIFAR_CLASSES)}
    assert summary['MNIST']['train_shape'] == [60000, 28, 28]
    assert summary['MNIST']['test_shape'] == [10000, 28, 28]
    for split in ('train', 'test'):
        counts = summary['MNIST'][split + '_counts']
        assert np.argmax(counts) == 1 and np.argmin(counts) == 5
    assert summary['CIFAR10']['train_shape'] == [50000, 32, 32, 3]
    assert summary['CIFAR10']['test_shape'] == [10000, 32, 32, 3]
    assert summary['CIFAR10']['train_counts'] == [5000] * 10
    assert summary['CIFAR10']['test_counts'] == [1000] * 10
    summary['sources'] = {name: {'url': url, 'sha256': hashlib.sha256((ROOT / 'inputs' / name).read_bytes()).hexdigest()}
                          for name, url in URLS.items()}
    (ARTIFACTS / 'datasets.json').write_text(json.dumps(summary, indent=2))
    (EVIDENCE / 'datasets.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
