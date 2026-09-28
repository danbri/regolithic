"""Joint multi-view height-field reconstruction by differentiable resampling.

Unknowns
  h(x, y)     terrain height on a regular grid (km, relative to the
              landing-site level used by the App. 3 altitudes)
  A(x, y)     surface brightness (albedo-like), shared by all views
  per exposure: small rotation (3) and position offset (3) applied to the
              App. 3 pose prior; the SLI, MRI and HRI images of one exposure
              share them
  per view:   gain and a linear-in-image-coordinates offset, which absorb
              haze and exposure differences

Forward model for view v at ground texel p = (x, y, h):
  project p through the posed gnomonic camera to image coordinates,
  sample the image there (bilinear), and compare with
      g_v * blur_v(A)(x, y) + b_v0 + b_v1 * xn + b_v2 * yn
  where blur_v matches the view's ground sampling distance and (xn, yn)
  are normalised image coordinates.

Occlusion is ignored; views are limited to nadir angles below NA_MAX,
where the gentle relief of the site cannot hide itself at DISR
resolution.

Losses: Charbonnier photometric misfit; pose priors (sigma per term);
second-difference smoothness on h; total variation on A.
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fn

from . import camera
from .pose import head_basis

torch.set_num_threads(4)


def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-9)
    K = skew(w)
    I = torch.eye(3, dtype=w.dtype).expand_as(K)
    return I + torch.sin(th) / th * K + (1 - torch.cos(th)) / th ** 2 * K @ K


def gauss_kernel(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / max(sigma, 1e-3)) ** 2)
    return k / k.sum()


def blur(img, sigma):
    """Separable Gaussian blur of a (1,1,H,W) tensor, reflect padding."""
    if sigma < 0.3:
        return img
    k = gauss_kernel(sigma)
    r = (len(k) - 1) // 2
    img = Fn.conv2d(Fn.pad(img, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return Fn.conv2d(Fn.pad(img, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


class Scene:
    def __init__(self, views, x0, x1, y0, y1, res, na_max=65.0):
        self.views = views
        self.res = res
        self.xs = torch.arange(x0 + res / 2, x1, res)
        self.ys = torch.arange(y0 + res / 2, y1, res)
        Y, X = torch.meshgrid(self.ys, self.xs, indexing="ij")
        self.X, self.Y = X, Y
        self.na_max = math.radians(na_max)
        exps = sorted({round(v["mt"], 1) for v in views})
        self.exp_index = {t: i for i, t in enumerate(exps)}
        self.ev = torch.tensor([self.exp_index[round(v["mt"], 1)] for v in views])
        self.B0 = torch.tensor(np.stack([head_basis(v["az"], v["pitch"], v["roll"]) for v in views]), dtype=torch.float32)
        self.C0 = torch.tensor(np.stack([v["C"] for v in views]), dtype=torch.float32)
        cams = []
        for v in views:
            a, r, u = camera.axes(v["imager"])
            cams.append(np.concatenate([a, r, u, [camera.SC[v["imager"]], camera.WIDTH[v["imager"]]]]))
        self.cam = torch.tensor(np.stack(cams), dtype=torch.float32)
        self.images = []
        for v in views:
            im = v["img"].astype(np.float32)
            im = im / np.median(im)
            self.images.append(torch.tensor(im)[None, None])
        # nominal ground sampling distance (km/pixel) at the image centre
        self.gsd = torch.tensor([camera.SC[v["imager"]] * v["C"][2] / math.cos(math.radians(camera.NAC[v["imager"]])) ** 2
                                 for v in views], dtype=torch.float32)

    def project(self, i, P, w, dC):
        """Grid points P (...,3) into view i -> normalised grid coords, valid mask."""
        R = so3_exp(w[self.ev[i]])
        B = R @ self.B0[i]
        C = self.C0[i] + dC[self.ev[i]]
        d = (P - C) @ B                      # components along F, R, Dn
        a, r, u = self.cam[i, 0:3], self.cam[i, 3:6], self.cam[i, 6:9]
        s, W = self.cam[i, 9], self.cam[i, 10]
        z = d @ a
        x = (W - 1) / 2 + (d @ r) / z / s
        y = 127.5 + (d @ u) / z / s
        na = torch.acos((d[..., 2] / d.norm(dim=-1)).clamp(-1, 1))
        valid = (z > 0) & (x > 2) & (x < W - 3) & (y > 2) & (y < 253) & (na < self.na_max)
        gx = x / (W - 1) * 2 - 1
        gy = y / 255 * 2 - 1
        return gx, gy, valid


class Model(torch.nn.Module):
    def __init__(self, scene, A0=None):
        super().__init__()
        n_exp = len(scene.exp_index)
        n_v = len(scene.views)
        ny, nx = scene.X.shape
        self.h = torch.nn.Parameter(torch.zeros(ny, nx))
        self.A = torch.nn.Parameter(torch.ones(ny, nx) if A0 is None else A0.clone())
        self.w = torch.nn.Parameter(torch.zeros(n_exp, 3))
        self.dC = torch.nn.Parameter(torch.zeros(n_exp, 3))
        self.gain = torch.nn.Parameter(torch.ones(n_v))
        self.off = torch.nn.Parameter(torch.zeros(n_v, 3))


def resample(t, shape):
    return Fn.interpolate(t[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def view_bbox(scene, i, margin=8):
    """Grid index window containing the view's footprint (prior pose, flat ground)."""
    with torch.no_grad():
        P = torch.stack([scene.X, scene.Y, torch.zeros_like(scene.X)], -1)
        z = torch.zeros(len(scene.exp_index), 3)
        _, _, valid = scene.project(i, P, z, z)
        if valid.sum() == 0:
            return None
        rr, cc = torch.where(valid)
        ny, nx = scene.X.shape
        return (max(int(rr.min()) - margin, 0), min(int(rr.max()) + margin + 1, ny),
                max(int(cc.min()) - margin, 0), min(int(cc.max()) + margin + 1, nx))


def view_residuals(scene, model, i, img_sigma=0.0, with_pred=False):
    if not hasattr(scene, "bbox"):
        scene.bbox = [view_bbox(scene, k) for k in range(len(scene.views))]
    bb = scene.bbox[i]
    if bb is None:
        e = torch.zeros(1, 1)
        return (e, e, e.bool()) if with_pred else (e, e.bool())
    r0, r1, c0, c1 = bb
    sig = float(scene.gsd[i]) / scene.res * (1 + img_sigma) * 0.6
    pad = max(1, int(math.ceil(3 * sig)))
    ny, nx = scene.X.shape
    R0, R1, C0, C1 = max(r0 - pad, 0), min(r1 + pad, ny), max(c0 - pad, 0), min(c1 + pad, nx)
    P = torch.stack([scene.X[r0:r1, c0:c1], scene.Y[r0:r1, c0:c1], model.h[r0:r1, c0:c1]], -1)
    gx, gy, valid = scene.project(i, P, model.w, model.dC)
    img = scene.images[i]
    if img_sigma > 0:
        img = blur(img, img_sigma)
    obs = Fn.grid_sample(img, torch.stack([gx, gy], -1)[None], mode="bilinear", align_corners=True)[0, 0]
    Ab = blur(model.A[R0:R1, C0:C1][None, None], sig)[0, 0][r0 - R0:r1 - R0, c0 - C0:c1 - C0]
    pred = model.gain[i] * Ab + model.off[i, 0] + model.off[i, 1] * gx + model.off[i, 2] * gy
    if HIGHPASS_SIGMA > 0 and not with_pred:
        s_hp = HIGHPASS_SIGMA * max(1.0, sig)
        return masked_highpass(obs, valid, s_hp) - masked_highpass(pred, valid, s_hp), valid
    if with_pred:
        full = lambda t, fill: Fn.pad(t, (c0, nx - c1, r0, ny - r1), value=fill)
        return full(obs, 0.0), full(pred, 0.0), full(valid.float(), 0.0) > 0
    return (obs - pred), valid


HIGHPASS_SIGMA = 0.0   # texels; > 0 compares locally high-passed obs and pred


def masked_highpass(t, m, sigma):
    """t - local mean of t within mask m (normalised convolution)."""
    mf = m.float()[None, None]
    num = blur(t[None, None] * mf, sigma)
    den = blur(mf, sigma).clamp_min(1e-3)
    return t - (num / den)[0, 0]


def charbonnier(r, eps=0.02):
    return torch.sqrt(r * r + eps * eps) - eps


def fit(scene, model, iters, lr=0.01, img_sigma=0.0, w_smooth=1.0, w_tv=0.01,
        sig_rot_deg=1.0, sig_pos_h=0.2, sig_pos_v=0.05, train=None, fix_geometry=False, log=print, w_prior=50.0, fix_height=False):
    train = list(range(len(scene.views))) if train is None else train
    geo = [model.h, model.w, model.dC]
    params = [model.A, model.gain, model.off] + ([] if fix_geometry else geo)
    opt = torch.optim.Adam([{"params": [model.A, model.gain, model.off], "lr": lr},
                            {"params": [] if (fix_geometry or fix_height) else [model.h], "lr": lr * 0.5},
                            {"params": [] if fix_geometry else [model.w, model.dC], "lr": lr * 0.1}])
    sr = math.radians(sig_rot_deg)
    for it in range(iters):
        opt.zero_grad()
        data, n = 0.0, 0
        for i in train:
            r, m = view_residuals(scene, model, i, img_sigma)
            if m.sum() < 50:
                continue
            data = data + charbonnier(r[m]).sum()
            n += int(m.sum())
        data = data / max(n, 1)
        h = model.h
        d2x = h[:, 2:] - 2 * h[:, 1:-1] + h[:, :-2]
        d2y = h[2:, :] - 2 * h[1:-1, :] + h[:-2, :]
        dxy = h[1:, 1:] - h[1:, :-1] - h[:-1, 1:] + h[:-1, :-1]
        smooth = (d2x.pow(2).mean() + d2y.pow(2).mean() + 2 * dxy.pow(2).mean()) / scene.res ** 2
        A = model.A
        tv = (A[:, 1:] - A[:, :-1]).abs().mean() + (A[1:, :] - A[:-1, :]).abs().mean()
        prior = ((model.w / sr) ** 2).sum() + ((model.dC[:, :2] / sig_pos_h) ** 2).sum() + ((model.dC[:, 2] / sig_pos_v) ** 2).sum()
        loss = data + w_smooth * smooth * 1e-3 + w_tv * tv + prior / max(n, 1) * w_prior
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.h -= model.h.mean()   # vertical datum is a free gauge; fix mean to 0
        if it % 25 == 0 or it == iters - 1:
            log("it %4d data %.5f smooth %.4g tv %.4f prior %.2f  h[p5,p95]=[%.0f,%.0f] m  rot rms %.2f deg  dC rms %.0f m" % (
                it, float(data.detach()), float(smooth.detach()), float(tv.detach()), float(prior.detach()),
                float(torch.quantile(model.h.detach(), 0.05)) * 1e3, float(torch.quantile(model.h.detach(), 0.95)) * 1e3,
                math.degrees(float(model.w.detach().norm(dim=1).pow(2).mean().sqrt())),
                float(model.dC.detach().norm(dim=1).pow(2).mean().sqrt()) * 1e3))
    return model


def upsample_model(scene_new, model_old):
    m = Model(scene_new)
    shape = scene_new.X.shape
    with torch.no_grad():
        m.h.copy_(resample(model_old.h.detach(), shape))
        m.A.copy_(resample(model_old.A.detach(), shape))
        m.w.copy_(model_old.w.detach())
        m.dC.copy_(model_old.dC.detach())
        m.gain.copy_(model_old.gain.detach())
        m.off.copy_(model_old.off.detach())
    return m


def init_albedo(scene, model, img_sigma=0.0):
    """Average of normalised ortho-samples on the current surface."""
    with torch.no_grad():
        acc = torch.zeros_like(scene.X)
        cnt = torch.zeros_like(scene.X)
        for i in range(len(scene.views)):
            obs, pred, m = view_residuals(scene, model, i, img_sigma, with_pred=True)
            if m.sum() < 50:
                continue
            o = obs / obs[m].mean()
            acc += torch.where(m, o, torch.zeros_like(o))
            cnt += m.float()
        A = torch.where(cnt > 0, acc / cnt.clamp_min(1), torch.ones_like(acc))
        model.A.copy_(A)
    return cnt


def fit_views_only(scene, model, idx, iters=150, img_sigmas=(2.0, 1.0, 0.0), lr=0.01,
                   sig_rot_deg=1.0, sig_pos_h=0.2, sig_pos_v=0.05, w_prior=50.0):
    """Refit pose (per exposure) and photometry (per view) for views idx only,
    with the surface (h, A) frozen. Used to score held-out views fairly."""
    exps = sorted({int(scene.ev[i]) for i in idx})
    emask = torch.zeros(len(scene.exp_index), 1)
    emask[exps] = 1
    vmask = torch.zeros(len(scene.views), 1)
    vmask[idx] = 1
    h_req, A_req = model.h.requires_grad, model.A.requires_grad
    model.h.requires_grad_(False); model.A.requires_grad_(False)
    opt = torch.optim.Adam([{"params": [model.w, model.dC], "lr": lr * 0.1},
                            {"params": [model.gain, model.off], "lr": lr}])
    sr = math.radians(sig_rot_deg)
    for isig in img_sigmas:
        for it in range(iters):
            opt.zero_grad()
            data, n = 0.0, 0
            for i in idx:
                r, m = view_residuals(scene, model, i, isig)
                if m.sum() < 50:
                    continue
                data = data + charbonnier(r[m]).sum()
                n += int(m.sum())
            w, dC = model.w[exps], model.dC[exps]
            prior = ((w / sr) ** 2).sum() + ((dC[:, :2] / sig_pos_h) ** 2).sum() + ((dC[:, 2] / sig_pos_v) ** 2).sum()
            loss = (data + prior * w_prior) / max(n, 1)
            loss.backward()
            model.w.grad *= emask; model.dC.grad *= emask
            model.gain.grad *= vmask[:, 0]; model.off.grad *= vmask
            opt.step()
    model.h.requires_grad_(h_req); model.A.requires_grad_(A_req)
