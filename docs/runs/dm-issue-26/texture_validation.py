import numpy as np, torch
torch.set_num_threads(1)
from deep import straight as st, nfixtures as nf
from motion import flow_fixtures as fx
paths, _ = st.synthetic_room(cache="runs/stage11-dry/room.npz", fov_frames=False)
looks128, truth, fan, refs, _ = paths["128"]
j = refs[0]
tex = [fx._texture_noise(30 + i) for i in range(7)]
X, Y, Z = np.eye(3)
planes = [fx._plane([0, 0, -1.6], Z, X, tex[0], "floor"), fx._plane([0, 0, 1.4], -Z, X, tex[1], "ceiling"),
          fx._plane([-2.0, 0, 0], X, Y, tex[2], "left"), fx._plane([2.5, 0, 0], -X, Y, tex[3], "right"),
          fx._plane([0, 4.0, 0], -Y, X, tex[4], "front"), fx._plane([0, -3.5, 0], Y, X, tex[5], "back"),
          fx._plane([-0.6, 1.8, -0.2], -Y, X, tex[6], "board", bounds=(-0.4, 0.4, -0.35, 0.35))]
clean = nf.render(planes, looks128[j][1], W=720, ss=2, noise=0.0)
m = np.nanmean(clean)
ok = np.isfinite(clean)
lat = (0.5 - (np.mgrid[0:360, 0:720][0] + 0.5) / 360) * np.pi
ok &= np.abs(lat) < np.radians(70)
def pair(img, s1024, gain=1.0, seed=7):
    r = np.random.default_rng(seed)
    i1 = img + r.normal(scale=s1024, size=img.shape)
    i8 = gain * (img + r.normal(scale=s1024 * np.sqrt(8.0), size=img.shape))
    return i1, i8
def attrs(i1, i8):
    L1, L8 = list(paths["1024"][0]), list(looks128)
    L1[j] = (L1[j][0], L1[j][1], L1[j][2], i1, None); L8[j] = (L8[j][0], L8[j][1], L8[j][2], i8, None)
    return st.attributes({"128": (L8, truth, fan, refs, None), "1024": (L1, truth, fan, refs, None)}, [j])[j]
edges = [0.02, 0.05, 0.1, 0.2]
for contrast in (0.05, 0.2):
    low = m + contrast * (clean - m)
    T = attrs(low, low)["texture"][ok]                          # D = 0: the clean image's own M4 measure
    A = attrs(*pair(low, 0.005, gain=1.37))                     # a gain between the runs' scalings, as in R3
    for name, v in (("signal (noise removed)", A["texture"][ok]), ("raw M4 measure, 1024", A["texture_raw"][ok])):
        rel = (v - T) / T
        print(f"contrast {contrast:4.2f} clean median {np.median(T):.4f} | {name:24s} median rel. error {np.median(rel):+.3f} "
              f"|rel| p90 {np.quantile(np.abs(rel), 0.9):.3f}; same class as clean {np.mean(np.digitize(v, edges) == np.digitize(T, edges)):.3f}")
    st.SPP_RATIO = 1.0
    v = attrs(*pair(low, 0.005, gain=1.37))["texture"][ok]
    st.SPP_RATIO = 8.0
    print(f"contrast {contrast:4.2f} control (SPP_RATIO 1, must be off): median rel. error {np.median((v - T) / T):+.3f}")
    print(f"           SNR quantiles p10/p50/p90 {np.quantile(A['snr'][ok], [0.1, 0.5, 0.9]).round(2)}")
