"""Датасет для распознавания топологического заряда (ТЗ) искажённого вихря.

Физика повторяет MATLAB-классы RichWolf / debay2DGeneral / circularPar:
острая фокусировка (интеграл Ричардса-Вольфа) апланатической линзой
плоской волны с правой круговой поляризацией и спиральной фазой exp(i*m*phi),
в которую внесено искажение - секторная диафрагма (интегрирование по
phi в пределах [beta + delta/2, beta + 2*pi - delta/2]).

Каждое изображение отличается:
  * физикой:    ширина сектора delta, ориентация сектора beta;
  * геометрией: сдвиг центра, масштаб, эллиптичность (растяжение), поворот;
  * шумами:     дробовой (Пуассон), гауссов, спекл, фон, размытие, битые пиксели.

Зеркальные отражения НЕ используются: они меняют знак ТЗ.

Запуск:
    python generate_tc_dataset.py                  # 4 класса x 500 изображений
    python generate_tc_dataset.py --n 50 --out test_ds
"""

import argparse
import csv
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------- параметры по умолчанию (как в main.m) ----------------
LAMBDA = 0.633          # длина волны, мкм
NA = 0.95               # числовая апертура
R_MAX = 1.0             # окно изображения [-R_MAX, R_MAX] мкм
TC_LIST = [1, 2, -1, -2]


# ======================= физика: интеграл Ричардса-Вольфа =======================

def focus_intensity(m, alpha_min, alpha_max, x, y, lam=LAMBDA, na=NA, z=0.0,
                    pol=(1, 1j), n_theta=64, n_phi=160):
    """Интенсивность |Ex|^2+|Ey|^2+|Ez|^2 в плоскости z на сетке x (столбцы) * y (строки).

    То же, что calcIntensityFocus в circularPar.m, но интеграл считается
    квадратурой Гаусса-Лежандра и сводится к одному матричному произведению.
    pol = (a, b): (1, 1j) - 'right', (1, -1j) - 'left', (1, 0) - 'linear x'.
    """
    k = 2 * np.pi / lam
    theta_max = np.arcsin(na)

    t, wt = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.5 * theta_max * (t + 1)
    wt = 0.5 * theta_max * wt
    p, wp = np.polynomial.legendre.leggauss(n_phi)
    phi = alpha_min + 0.5 * (alpha_max - alpha_min) * (p + 1)
    wp = 0.5 * (alpha_max - alpha_min) * wp

    th, ph = np.meshgrid(theta, phi, indexing='ij')
    th, ph = th.ravel(), ph.ravel()
    w = np.outer(wt, wp).ravel()

    a, b = pol
    # underinteq: sin(theta) * T(theta) * exp(i*m*phi) * exp(i*k*z*cos(theta))
    common = (w * np.sin(th) * np.sqrt(np.cos(th)) * np.exp(1j * m * ph)
              * np.exp(1j * k * z * np.cos(th)))
    # mMatrixEx / Ey / Ez
    cth, cph, sph = np.cos(th), np.cos(ph), np.sin(ph)
    mx = (1 + cph ** 2 * (cth - 1)) * a + sph * cph * (cth - 1) * b
    my = sph * cph * (cth - 1) * a + (1 + sph ** 2 * (cth - 1)) * b
    mz = -np.sin(th) * cph * a - np.sin(th) * sph * b

    # exp(i k r sin(theta) cos(phi - psi)) = exp(i k x u) * exp(i k y v)
    u = np.sin(th) * cph
    v = np.sin(th) * sph
    ex_x = np.exp(1j * k * np.outer(x, u))   # (Nx, Q)
    ex_y = np.exp(1j * k * np.outer(y, v))   # (Ny, Q)

    intensity = np.zeros((len(y), len(x)))
    for mm in (mx, my, mz):
        amp = -1j * (ex_y * (common * mm)) @ ex_x.T
        intensity += np.abs(amp) ** 2
    return intensity


# ============================ искажения и шумы ============================

def uniform(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def geometric_warp(img, rng, cfg):
    """Аффинное искажение: поворот, масштаб, эллиптичность, сдвиг (без отражений)."""
    n = img.shape[0]
    rot = uniform(rng, 0, 2 * np.pi)
    scale = uniform(rng, *cfg['scale'])
    aspect = uniform(rng, *cfg['aspect'])           # эллиптичность
    aspect_angle = uniform(rng, 0, np.pi)
    shift = rng.uniform(-cfg['shift'], cfg['shift'], size=2) * n

    def rmat(a):
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    stretch = rmat(aspect_angle) @ np.diag([aspect, 1 / aspect]) @ rmat(-aspect_angle)
    fwd = scale * stretch @ rmat(rot)               # det > 0 -> без зеркала
    inv = np.linalg.inv(fwd)
    c = np.array([(n - 1) / 2, (n - 1) / 2])
    offset = c - inv @ (c + shift)
    out = ndimage.affine_transform(img, inv, offset=offset, order=3,
                                   mode='constant', cval=0.0)
    out = np.clip(out, 0, None)
    params = dict(rot=rot, scale=scale, aspect=aspect, aspect_angle=aspect_angle,
                  shift_x=shift[1] / n, shift_y=-shift[0] / n)
    return out, params


def add_noise(img, rng, cfg):
    """img нормирован на [0, 1]. Возвращает зашумлённое изображение в [0, 1]."""
    n = img.shape[0]
    p = {}

    # размытие (несовершенная оптика / расфокусировка камеры)
    p['blur_sigma'] = uniform(rng, *cfg['blur'])
    if p['blur_sigma'] > 0.05:
        img = ndimage.gaussian_filter(img, p['blur_sigma'])

    # спекл (мультипликативный шум с корреляцией в несколько пикселей)
    p['speckle'] = uniform(rng, *cfg['speckle'])
    sp = ndimage.gaussian_filter(rng.standard_normal((n, n)), 1.5)
    sp /= sp.std() + 1e-12
    img = img * np.clip(1 + p['speckle'] * sp, 0, None)

    # фон: постоянная засветка + линейный градиент
    p['background'] = uniform(rng, *cfg['background'])
    g = rng.uniform(-1, 1, size=2) * p['background']
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    img = img + p['background'] + 0.5 * (g[0] * xx + g[1] * yy)
    img = np.clip(img, 0, None)

    # дробовой шум: число фотонов в максимуме
    p['photons'] = float(np.exp(rng.uniform(*np.log(cfg['photons']))))
    img = rng.poisson(img * p['photons']) / p['photons']

    # гауссов шум считывания
    p['gauss_sigma'] = uniform(rng, *cfg['gauss'])
    img = img + p['gauss_sigma'] * rng.standard_normal((n, n))

    # битые / горячие пиксели (salt & pepper)
    p['impulse_prob'] = uniform(rng, *cfg['impulse'])
    mask = rng.random((n, n))
    img[mask < p['impulse_prob'] / 2] = 0
    img[mask > 1 - p['impulse_prob'] / 2] = img.max()

    img = img - img.min()
    img = img / (img.max() + 1e-12)
    return img, p


# ================================ генерация ================================

CONFIG = dict(
    delta=(np.pi / 32, np.pi / 4),   # ширина сектора диафрагмы, рад
    scale=(0.85, 1.15),
    aspect=(0.9, 1.1),
    shift=0.08,                      # доля размера кадра
    blur=(0.0, 1.5),                 # пикс.
    speckle=(0.0, 0.15),
    background=(0.0, 0.1),
    photons=(30, 3000),              # лог-равномерно
    gauss=(0.0, 0.05),
    impulse=(0.0, 0.01),
)


def make_sample(m, rng, x, cfg):
    delta = uniform(rng, *cfg['delta'])
    beta = uniform(rng, 0, 2 * np.pi)    # ориентация препятствия
    base = focus_intensity(m, beta + delta / 2, beta + 2 * np.pi - delta / 2, x, x)
    base = base[::-1]                    # строка 0 - верх кадра (y вверх, как axis xy)
    base /= base.max()
    img, geo = geometric_warp(base, rng, cfg)
    img /= img.max() + 1e-12
    img, noise = add_noise(img, rng, cfg)
    meta = dict(tc=m, delta=delta, beta=beta, **geo, **noise)
    return img, meta


def viridis_lut():
    anchors = np.array([
        [0.267004, 0.004874, 0.329415],
        [0.253935, 0.265254, 0.529983],
        [0.163625, 0.471133, 0.558148],
        [0.134692, 0.658636, 0.517649],
        [0.477504, 0.821444, 0.318195],
        [0.993248, 0.906157, 0.143936]])
    xs = np.linspace(0, 1, len(anchors))
    q = np.linspace(0, 1, 256)
    return (np.stack([np.interp(q, xs, anchors[:, c]) for c in range(3)], 1) * 255).astype(np.uint8)


def save_image(img, path, colormap):
    u8 = np.round(img * 255).astype(np.uint8)
    if colormap == 'viridis':
        Image.fromarray(viridis_lut()[u8]).save(path)
    else:
        Image.fromarray(u8).save(path)   # 8-bit grayscale


def class_name(m):
    return f'tc_{m:+d}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='dataset_tc')
    ap.add_argument('--n', type=int, default=500, help='изображений на класс')
    ap.add_argument('--size', type=int, default=128, help='размер изображения, пикс.')
    ap.add_argument('--tc', type=int, nargs='+', default=TC_LIST)
    ap.add_argument('--colormap', choices=['gray', 'viridis'], default='gray')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = np.linspace(-R_MAX, R_MAX, args.size)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    t0 = time.time()
    for m in args.tc:
        folder = os.path.join(args.out, class_name(m))
        os.makedirs(folder, exist_ok=True)
        for i in range(1, args.n + 1):
            img, meta = make_sample(m, rng, x, CONFIG)
            fname = f'img_{i:04d}.png'
            save_image(img, os.path.join(folder, fname), args.colormap)
            rows.append(dict(file=f'{class_name(m)}/{fname}', **meta))
        print(f'{class_name(m)}: {args.n} изображений, {time.time() - t0:.0f} c')

    with open(os.path.join(args.out, 'metadata.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
    save_preview(args.out, args.tc, min(args.n, 8))
    print(f'Готово: {len(rows)} изображений в {os.path.abspath(args.out)}')


def save_preview(out, tc_list, n_cols):
    """preview.png: строка на класс, первые n_cols изображений."""
    rows = []
    for m in tc_list:
        imgs = [np.array(Image.open(os.path.join(out, class_name(m), f'img_{i:04d}.png')))
                for i in range(1, n_cols + 1)]
        rows.append(np.concatenate([np.pad(im, [(2, 2), (2, 2)] + [(0, 0)] * (im.ndim - 2),
                                           constant_values=255) for im in imgs], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out, 'preview.png'))


if __name__ == '__main__':
    main()
