import numpy as np

def konvolusi(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = (F_height) // 2
    W = (F_width) // 2

    out = np.zeros_like(X, dtype=np.float32)

    for i in np.arange(H, X_height - H):
        for j in np.arange(W, X_width - W):
            total = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    total += (w * a)
            out[i, j] = total

    return out
