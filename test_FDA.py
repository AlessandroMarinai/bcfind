import argparse
from ipdb import launch_ipdb_on_exception
import tifffile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Myriad Pro"]
sns.set_context("paper", font_scale=1.6)
plt.switch_backend("GTK3Agg")

matplotlib.rc('image', cmap='gray')

def low_freq_mutate_np(amp_src, amp_trg, L, lmbda):
    amp_src = tf.signal.fftshift(amp_src, axes=(0, 1, 2))
    amp_trg = tf.signal.fftshift(amp_trg, axes=(0, 1, 2))

    amp_src1 = amp_src
    amp_trg1 = amp_trg

    d, h, w = amp_src.shape
    b = (np.floor(np.amin((h,w,d))*L)).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    c_z = np.floor(d/2.0).astype(int)

    print(b,d,h,w)
    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1
    d1, d2 = c_z - b//2, c_z + b//2 + 1

    amp_src = amp_src.numpy()
    amp_trg = amp_trg.numpy()
    amp_src[d1:d2,h1:h2,w1:w2] = lmbda * amp_trg[d1:d2,h1:h2,w1:w2] + (1-lmbda) * amp_src[d1:d2,h1:h2,w1:w2]
    amp_src = tf.convert_to_tensor(amp_src)
    amp_src = tf.signal.ifftshift(amp_src, axes=(0, 1, 2))
    return amp_src, amp_src1, amp_trg1


def FDA_source_to_target_np(src_img, trg_img, L, lmbda):
    fft_src = tf.signal.fft3d(tf.cast(src_img, tf.complex64))
    fft_trg = tf.signal.fft3d(tf.cast(trg_img, tf.complex64))

    # Get amplitude and phase of source and target
    amp_src, pha_src = tf.abs(fft_src), tf.math.angle(fft_src)
    amp_trg, _ = tf.abs(fft_trg), tf.math.angle(fft_trg)

    # mutate the amplitude part of source with target
    amp_src_, amp_src1, amp_trg1 = low_freq_mutate_np(amp_src, amp_trg, L, lmbda)

    # mutated fft of source
    fft_src_ = tf.complex(amp_src_, tf.zeros_like(amp_src_)) * tf.exp(tf.complex(tf.zeros_like(pha_src), pha_src))

    # Perform inverse FFT to get the transformed source image
    src_in_trg = tf.signal.ifft3d(fft_src_)

    # Return the transformed image and label
    return tf.math.real(src_in_trg), amp_src1, amp_trg1


def apply_fda_transformation_img_tf(src_img, y, trg_img, L, lmbda):
    ret, amp_src, amp_trg = FDA_source_to_target_np(src_img, trg_img, L, lmbda)

    return tf.convert_to_tensor(ret), y, amp_src, amp_trg


def compare_at_z(img, trg_img, z0, output_file):
    # plt.figure()
    tmp = np.array([img.numpy(), trg_img.numpy()])
    vmin = np.percentile(tmp, 2.5)
    vmin = 0
    vmax = np.percentile(tmp, 95)

    _Ls = [0.1,0.15,0.2,0.25,0.3]
    _lmbdas = [0.1,0.2,0.3,0.4,0.75]
    fig, axes = plt.subplots(len(_Ls), 2+len(_lmbdas), figsize=[20,20])

    i = 0
    for _L in _Ls:
        ax = sns.heatmap(img.numpy()[z0], vmin=vmin, vmax=vmax, ax=axes[i][0], cbar=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set(title='Source')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = sns.heatmap(trg_img.numpy()[z0], vmin=vmin, vmax=vmax, ax=axes[i][-1])
        ax.tick_params(axis='both', which='both', length=0)
        ax.set(title='Target')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        j = 1
        for _lmbda in _lmbdas:
            img_modified, _, amp_src, amp_trg = apply_fda_transformation_img_tf(
                img, 0, trg_img, L=_L, lmbda=_lmbda)
            ax = sns.heatmap(img_modified.numpy()[z0], vmin=vmin, vmax=vmax, ax=axes[i][j], cbar=False)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set(title=f'β={_L}, λ={_lmbda}')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            j += 1
        i += 1
    if output_file == 'screen':
        plt.show(block=False)
        plt.draw()
    else:
        plt.savefig(output_file, dpi=150)
    # plt.show()


def _get_image(path):
    img = tifffile.imread(path)
    img[img>2**15] = 2**15
    img = img/2**15
    img = tf.convert_to_tensor(img)
    return img

def main(opts):
    trg_img = _get_image(opts.target_image)
    img = _get_image(opts.source_image)
    compare_at_z(img, trg_img, opts.z0, opts.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_image")
    parser.add_argument("target_image")
    parser.add_argument("output_file")
    parser.add_argument("-z", "--z0", type=int, default=10)
    opts = parser.parse_args()
    with launch_ipdb_on_exception():
        main(opts)
